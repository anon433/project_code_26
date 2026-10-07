"""Paired YAML settings: numerical protocol and independent stage selections."""

from copy import deepcopy
import math
from pathlib import Path

import yaml

from . import config

ROOT = Path(__file__).resolve().parents[2]
_ACTIVE = None


def _read(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Stage settings must be a mapping")
    return value


def _merge(template, supplied, path="settings"):
    if not isinstance(supplied, dict) or set(supplied) - set(template):
        raise ValueError(f"Unknown keys or invalid mapping in {path}")
    result = deepcopy(template)
    for key, value in supplied.items():
        expected = template[key]
        if isinstance(expected, dict):
            result[key] = _merge(expected, value, f"{path}.{key}")
        else:
            if isinstance(expected, float):
                valid = type(value) in (float, int) and math.isfinite(value)
            else:
                valid = type(value) is type(expected)
            if key == "mask_seed":
                valid = value == "match" or type(value) is int
            if not valid:
                raise ValueError(f"Invalid type/value for {path}.{key}")
            result[key] = value
    return result


def _selection(stage, number, contract):
    if stage["stage"] != number or stage["budgets"] not in ("headline", "all"):
        raise ValueError("Invalid stage or budgets; use headline or all")
    if stage["seeds"] not in ([3], [3, 7, 11]) or any(
        type(seed) is not int for seed in stage["seeds"]
    ):
        raise ValueError("seeds must be [3] or [3, 7, 11]")
    choices = {"datasets": set(contract["datasets"])}
    if number == 1:
        choices["selectors"] = {"gec", "wagle", "tcus-gd", "tcus-npo"}
    else:
        choices.update(selectors={"gec", "wagle", "tcus"}, updaters={"gd", "npo"})
        if stage["mask_seed"] not in (3, 7, 11, "match"):
            raise ValueError("mask_seed must be 3, 7, 11, or match")
    for key, allowed in choices.items():
        selected = stage[key]
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= allowed
        ):
            raise ValueError(f"Invalid {key} selection")


def load_settings(path):
    path = Path(path).resolve()
    selected = _read(path)
    number = selected.get("stage")
    if number not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    peer_key = f"stage{3-number}_config"
    peer = (path.parent / selected.get(peer_key, f"stage{3-number}.yaml")).resolve()
    other = _read(peer)
    if (peer.parent / other.get(f"stage{number}_config", path.name)).resolve() != path:
        raise ValueError("Stage configuration references must point back to each other")
    values = {number: selected, 3 - number: other}
    contract = config._plain(config.load_contract(config.DEFAULT_CONTRACT_PATH))
    common = dict(datasets=list(contract["datasets"]), seeds=[3], budgets="all")
    templates = {
        1: dict(
            common,
            stage=1,
            stage2_config="stage2.yaml",
            selectors=["gec", "wagle", "tcus-gd", "tcus-npo"],
            tcus=dict(contract["selectors"]["tcus"]),
            gec={k: contract["selectors"]["gec"][k] for k in ("ema_alpha", "eps")},
            wagle={
                d: {k: s["wagle"][k] for k in ("p", "q", "mu")}
                for d, s in contract["datasets"].items()
            },
        ),
        2: dict(
            common,
            stage=2,
            stage1_config="stage1.yaml",
            selectors=["gec", "wagle", "tcus"],
            updaters=["gd", "npo"],
            mask_seed=3,
            training={
                d: {k: v for k, v in s["stage2"].items() if k != "microbatches"}
                for d, s in contract["datasets"].items()
            },
        ),
    }
    stages = {n: _merge(templates[n], values[n]) for n in (1, 2)}
    for n in (1, 2):
        _selection(stages[n], n, contract)
    first, second = stages[1], stages[2]
    tcus = first["tcus"]
    if tcus["score_steps"] <= 0 or tcus["retain_penalty"] < 0:
        raise ValueError("TCUS requires positive steps and nonnegative retain_penalty")
    if not 0 <= first["gec"]["ema_alpha"] <= 1 or first["gec"]["eps"] <= 0:
        raise ValueError("Invalid GEC settings")
    contract["selectors"]["tcus"] = tcus
    contract["selectors"]["gec"].update(first["gec"])
    for dataset, spec in contract["datasets"].items():
        spec["wagle"].update(first["wagle"][dataset])
        if any(first["wagle"][dataset][k] <= 0 for k in ("p", "q", "mu")):
            raise ValueError("WAGLE p, q and mu must be positive")
        recipe = second["training"][dataset]
        if not tcus["score_steps"] <= recipe["max_steps"] <= 500:
            raise ValueError(
                "max_steps must cover TCUS and fit the pinned 500-step stream"
            )
        if (
            recipe["per_device_train_batch_size"] != 1
            or recipe["gradient_accumulation_steps"] != 4
        ):
            raise ValueError(
                "The pinned data protocol requires batch size 1 and accumulation 4"
            )
        if recipe["weight_decay"] != 0:
            raise ValueError("Sparse gradient masking requires weight_decay=0")
        if not 0 <= recipe["warmup_steps"] <= recipe["max_steps"]:
            raise ValueError("Invalid warmup_steps")
        if recipe["lr_scheduler_type"] not in ("constant", "linear") or recipe[
            "optim"
        ] not in ("adamw_torch", "paged_adamw_32bit"):
            raise ValueError("Unsupported scheduler or optimizer")
        if (
            not all(0 <= recipe[k] < 1 for k in ("adam_beta1", "adam_beta2"))
            or recipe["adam_epsilon"] <= 0
            or recipe["max_grad_norm"] <= 0
        ):
            raise ValueError("Invalid Adam or clipping settings")
        for objective in recipe["recipes"].values():
            if any(v <= 0 for v in objective.values()):
                raise ValueError(
                    "Objective coefficients and learning rates must be positive"
                )
        spec["stage2"] = {**recipe, "microbatches": recipe["max_steps"] * 4}
    return {"stage1": first, "stage2": second, "contract": contract}


def activate(settings):
    global _ACTIVE
    _ACTIVE = deepcopy(settings)
    config._ACTIVE = config._freeze(settings["contract"])


def current():
    if _ACTIVE is None:
        activate(load_settings(ROOT / "configs/stage1.yaml"))
    return deepcopy(_ACTIVE)
