"""Executable benchmark recipes using the public Hydra and WAGLE entrypoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .config import load_contract
from .matrix import Stage2Cell


def compose_command(repo: Path, command: list[str]):
    config = next(
        arg.split("=", 1)[1] for arg in command if arg.startswith("--config-name=")
    )
    overrides = command[command.index(f"--config-name={config}") + 1 :]
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        return compose(config_name=config, overrides=overrides)


def _value(value):
    return json.dumps(value, separators=(",", ":"))


def _hydra(repo, dataset, phase, trainer=None):
    benchmark = "wmdp" if dataset == "wmdpall" else "muse"
    script = "eval" if phase == "eval" else "train"
    experiment = (
        f"reproduction/{benchmark}_eval"
        if phase == "eval"
        else f"reproduction/{benchmark}"
    )
    command = [
        sys.executable,
        str(repo / "src" / f"{script}.py"),
        f"--config-name={'eval' if phase == 'eval' else 'unlearn'}.yaml",
        f"experiment={experiment}",
    ]
    if trainer:
        command.append(f"trainer={trainer}")
    return command


def _overrides(repo, command, values, remove=()):
    """Use '+' only for deliberately added fields absent from the selected schema."""
    cfg = OmegaConf.to_container(compose_command(repo, command), resolve=False)

    def exists(key):
        node = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    for key in remove:
        if exists(key):
            command.append(f"~{key}")
    for key, value in values.items():
        prefix = "" if exists(key) else "+"
        command.append(f"{prefix}{key}={_value(value)}")
    return command


def _common(repo, dataset, asset, output):
    return {
        "task_name": f"paper-{dataset}",
        "data_split": "cyber"
        if dataset == "wmdpall"
        else dataset.split("-")[1].capitalize(),
        "model.model_args.pretrained_model_name_or_path": asset["model"],
        "model.tokenizer_args.pretrained_model_name_or_path": asset["tokenizer"],
        "model.model_args.device_map": "balanced",
        "model.model_args.attn_implementation": "sdpa",
        "model.model_args.torch_dtype": "bfloat16",
        "model.tokenizer_args.local_files_only": True,
        "model.tokenizer_args.use_fast": False,
        "paths.output_dir": str(output),
        "paths.work_dir": str(repo),
    }


def stage1_command(repo, cell, asset, output):
    spec = load_contract()["datasets"][cell.dataset]
    if cell.selector == "wagle":
        command = [sys.executable, "-m", "src.reproduction.wagle", "--phase", "masks"]
        flags = {
            "domain": "cyber"
            if cell.dataset == "wmdpall"
            else cell.dataset.split("-")[1],
            "pair-stream": asset["stage1"],
            "expected-pair-stream-sha256": asset["stage1_sha256"],
            "model-path": asset["model"],
            "tokenizer-path": asset["tokenizer"],
            "cache-dir": str(output / "cache"),
            "output-dir": str(output),
            "seed": cell.seed,
            "score-steps": asset["score_steps"],
            "forget-score-batches": asset["forget_score_batches"],
            "retain-score-batches": asset["retain_score_batches"],
            "support-fracs": 0.1,
            **{
                f"wagle-{key}": value
                for key, value in spec["wagle"].items()
                if key in ("p", "q", "mu")
            },
        }
        for key, value in flags.items():
            command.extend([f"--{key}", str(value)])
        command.extend(
            [
                "--scores-only",
                "--save-selector-logits",
                "--selector-logits-storage-dtype",
                "float32",
            ]
        )
        return command
    if cell.selector.startswith("tcus-"):
        updater = cell.selector.split("-")[1]
        paired = Stage2Cell(
            cell.dataset, cell.support, updater, cell.selector, cell.seed
        )
        command = stage2_command(repo, paired, asset, output)
        trainer = "TCUSTrajectoryGradDiff" if updater == "gd" else "TCUSTrajectoryNPO"
        inherited = {
            arg.lstrip("+").split("=", 1)[0]: json.loads(arg.split("=", 1)[1])
            for arg in command[5:]
            if not arg.startswith(("experiment=", "trainer="))
        }
        command = _hydra(repo, cell.dataset, "mask_search", trainer)
        values = {
            **inherited,
            "paths.output_dir": str(output),
            "active_mask_path": None,
            "active_mask_expected_version": None,
            "trainer.method_args.budget_frac": 0.1,
            "trainer.method_args.score_steps": load_contract()["selectors"]["tcus"][
                "score_steps"
            ],
            "trainer.method_args.retain_penalty": load_contract()["selectors"]["tcus"][
                "retain_penalty"
            ],
        }
        return _overrides(repo, command, values)
    if cell.selector != "gec":
        raise ValueError(f"Unknown selector: {cell.selector}")
    trainer = "GradSaliencyMask"
    command = _hydra(repo, cell.dataset, "mask_search", trainer)
    values = _common(repo, cell.dataset, asset, output)
    values.update(
        {
            "trainer.args.seed": cell.seed,
            "trainer.args.num_train_epochs": 1,
            "trainer.args.max_steps": max(
                asset["forget_score_batches"], asset["retain_score_batches"]
            ),
            "trainer.args.per_device_train_batch_size": 1,
            "trainer.args.gradient_accumulation_steps": 1,
            "trainer.args.gradient_checkpointing": False,
            "trainer.args.do_eval": False,
            "trainer.args.eval_on_start": False,
            "trainer.args.eval_strategy": "no",
            "trainer.args.save_strategy": "no",
            "trainer.method_args.param_filter": "all",
            "data.fair_stage": "stage1",
            "data.fair_pair_stream": asset["stage1"],
            "data.expected_pair_stream_sha256": asset["stage1_sha256"],
        }
    )
    values.update(
        {
            f"trainer.method_args.{key}": value
            for key, value in {
                "select_frac": 0.1,
                "score_fn": "norm_diff",
                "ema_alpha": load_contract()["selectors"]["gec"]["ema_alpha"],
                "eps": load_contract()["selectors"]["gec"]["eps"],
                "independent_side_passes": True,
                "forget_score_batches": asset["forget_score_batches"],
                "retain_score_batches": asset["retain_score_batches"],
                "save_boolean_active_mask": False,
                "save_selector_logits": True,
                "selector_logits_storage_dtype": "float32",
            }.items()
        }
    )
    return _overrides(repo, command, values)


def stage2_command(repo, cell, asset, directory):
    command = _hydra(
        repo, cell.dataset, "stage2", "GradDiff" if cell.updater == "gd" else "NPO"
    )
    values = _common(repo, cell.dataset, asset, directory / "checkpoint")
    dataset_spec = load_contract()["datasets"][cell.dataset]
    spec = dataset_spec["stage2"]
    for key in (
        "model.model_args.device_map",
        "model.model_args.attn_implementation",
        "model.model_args.torch_dtype",
        "paths.output_dir",
    ):
        values.pop(key)
    values["paths.root_dir"] = str(directory / "work")
    values["model_revision"] = dataset_spec["model"]["revision"]
    if cell.dataset == "wmdpall":
        values["dataset_revision"] = dataset_spec["datasets"]["wmdp_corpora"][
            "revision"
        ]
    else:
        values["dataset_revision"] = dataset_spec["dataset"]["revision"]
        values["tokenizer_revision"] = dataset_spec["tokenizer"]["revision"]
        values.pop("model.tokenizer_args.local_files_only")
        values.pop("model.tokenizer_args.use_fast")
    values.update(
        {
            f"trainer.args.{key}": spec[key]
            for key in (
                "max_steps",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "warmup_steps",
                "lr_scheduler_type",
                "optim",
                "weight_decay",
                "adam_beta1",
                "adam_beta2",
                "adam_epsilon",
                "max_grad_norm",
            )
        }
    )
    values.update(
        {
            "trainer.args.learning_rate": cell.recipe["learning_rate"],
            "trainer.args.seed": cell.seed,
            "trainer.args.data_seed": cell.seed,
            "trainer.args.bf16": True,
            "trainer.args.gradient_checkpointing": False,
            "trainer.args.dataloader_num_workers": 0,
            "trainer.args.do_eval": False,
            "trainer.args.eval_on_start": False,
            "trainer.args.eval_strategy": "no",
            "trainer.args.save_strategy": "no",
            "active_mask_path": str(directory / "mask" / "active_mask.pt"),
            "active_mask_expected_version": "sparse_mask_v1",
            "data.fair_stage": "stage2",
            "data.fair_pair_stream": asset["stage2"],
            "data.expected_pair_stream_sha256": asset["stage2_sha256"],
        }
    )
    if cell.dataset == "wmdpall":
        values.update(
            {"trainer.args.num_train_epochs": 1, "trainer.args.save_only_model": True}
        )
    else:
        values.update(
            {"trainer.args.complete_epochs": False, "trainer.args.warmup_epochs": 0}
        )
    values.update(
        {
            f"trainer.method_args.{key}": value
            for key, value in cell.recipe.items()
            if key != "learning_rate"
        }
    )
    return _overrides(repo, command, values)


def checkpoint_directory(dataset, directory):
    """The inherited Hydra output path used by the training command."""
    return directory / "work" / "saves" / "unlearn" / f"paper-{dataset}"


def eval_commands(repo, dataset, asset, checkpoint, output):
    commands = []
    for task in (
        ("wmdp_bio", "wmdp_cyber", "mmlu")
        if dataset == "wmdpall"
        else ("muse_knowmem", "muse_privacy")
    ):
        command = _hydra(repo, dataset, "eval")
        spec = load_contract()["datasets"][dataset]
        values = {
            "task_name": f"paper-{dataset}-{task}",
            "data_split": ("bio" if task == "wmdp_bio" else "cyber")
            if dataset == "wmdpall"
            else dataset.split("-")[1].capitalize(),
            "model.model_args.pretrained_model_name_or_path": str(checkpoint),
            "model.tokenizer_args.pretrained_model_name_or_path": asset["tokenizer"],
            "model.tokenizer_args.local_files_only": True,
            "model_revision": None,
            "tokenizer_revision": spec.get("tokenizer", spec["model"])["revision"],
            "paths.root_dir": str(output / task / "work"),
            "paths.work_dir": str(repo),
            "seed": 3,
        }
        if dataset == "wmdpall":
            values.update(
                {
                    "eval.lm_eval.tasks": [task],
                    "eval.lm_eval.output_dir": str(output / task),
                    "eval.lm_eval.overwrite": False,
                    "eval.lm_eval.simple_evaluate_args.batch_size": 16,
                    "model.tokenizer_args.use_fast": False,
                }
            )
        else:
            command.append(f"eval={task}")
            values.update(
                {
                    "retain_logs_path": asset["retain_logs"],
                    "eval.muse.overwrite": False,
                    "dataset_revision": spec["dataset"]["revision"],
                }
            )
        command = _overrides(repo, command, values)
        specs = (
            spec["evaluation"]["datasets"]
            if dataset == "wmdpall"
            else {"dataset": spec["dataset"]}
        )
        snapshots = {name: asset["snapshots"][name] for name in specs}
        command = [
            sys.executable,
            "-m",
            "src.reproduction.evaluation",
            "--dataset",
            dataset,
            "--snapshot-records",
            _value(snapshots),
            "--dataset-cache",
            str(output / "dataset_cache"),
            "--",
            *command[1:],
        ]
        commands.append(command)
    return commands
