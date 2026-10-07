#!/usr/bin/env python3
"""WAGLE score backend for the public mask-search runner."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from dataset.fair_wmdp import FairWMDPUnlearnDataset

import numpy as np


WAGLE_ROOT = Path(__file__).resolve().parents[1]
OPENUNLEARNING_ROOT = WAGLE_ROOT.parents[1]
sys.path.insert(0, str(WAGLE_ROOT / "src"))


PROTOCOL = "wagle_scoring_v1"
SEQUENCE_LENGTH = 512
MUSE_MAX_SEQUENCE_LENGTH = 4096
METHODS = {"graddiff": {"lr": 5e-6}}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _set_offline() -> None:
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        existing = os.environ.get(name)
        if existing not in (None, "1"):
            raise RuntimeError(f"{name} must be unset or '1' for the fair protocol.")
        os.environ[name] = "1"


def validate_pair_contract(domain: str, dataset: FairWMDPUnlearnDataset) -> None:
    """Keep the WMDP contract strict while admitting pinned MUSE streams."""
    if domain in {"news", "books"}:
        if dataset.manifest.get("benchmark") != "muse":
            raise ValueError("WAGLE MUSE requires a manifested MUSE pair stream.")
        if str(dataset.manifest.get("domain", "")).lower() != domain:
            raise ValueError("WAGLE MUSE pair manifest has the wrong domain.")
        if dataset.manifest.get("padding_policy") != "right_pad":
            raise ValueError("WAGLE MUSE requires right-padded chunks.")
        if not 1 < dataset.sequence_length <= MUSE_MAX_SEQUENCE_LENGTH:
            raise ValueError("WAGLE MUSE sequence length must be in [2, 4096] tokens.")
        return
    if dataset.manifest.get("artifact_kind") == "paired_wmdpall_block_a_trace":
        if (
            dataset.sequence_length != 1024
            or dataset.manifest.get("dataset") != "WMDPALL"
        ):
            raise ValueError(
                "Paired WMDPALL Block A traces require 1,024-token WMDPALL rows."
            )
        return
    if dataset.sequence_length != SEQUENCE_LENGTH:
        raise ValueError("Fair WAGLE WMDP protocol requires 512-token pairs.")


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _hash_tensor(hasher: "hashlib._Hash", name: str, value: torch.Tensor) -> None:
    import torch

    value = value.detach().cpu().contiguous().to(dtype=torch.uint8)
    hasher.update(name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(json.dumps(list(value.shape)).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(value.numpy().tobytes(order="C"))


def mask_sha256(mask) -> str:
    import torch

    if not mask:
        raise ValueError("WAGLE mask must contain parameters.")
    hasher = hashlib.sha256()
    hasher.update(b"wagle_native_boolean_mask_v1")
    for name in sorted(mask):
        value = mask[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
            raise TypeError(f"WAGLE mask {name!r} must be a boolean tensor.")
        _hash_tensor(hasher, name, value)
    return hasher.hexdigest()


def _mask_stats(mask: dict[str, torch.Tensor]) -> tuple[int, int, str]:
    import torch

    selected = sum(int(torch.count_nonzero(value).item()) for value in mask.values())
    total = sum(int(value.numel()) for value in mask.values())
    return selected, total, mask_sha256(mask)


def _assert_new_or_reusable(directory: Path, expected: dict[str, Any]) -> bool:
    marker = directory / "stage_success.json"
    if marker.is_file():
        if _read_json(marker) != expected:
            raise RuntimeError(
                f"Completed WAGLE stage configuration mismatch: {directory}"
            )
        return True
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite incomplete WAGLE stage state: {directory}"
        )
    return False


def _new_unlearn(
    args: argparse.Namespace,
    *,
    method: str,
    max_steps: int,
    accumulation: int,
    mask_path: Path | None,
):
    from model.unlearn import Unlearn

    return Unlearn(
        model_name=str(args.model_path),
        tokenizer_name=str(args.tokenizer_path or args.model_path),
        cache_dir=str(args.cache_dir),
        fair_pair_stream=str(args.pair_stream),
        expected_pair_stream_sha256=args.expected_pair_stream_sha256,
    )


def _mask_manifest(
    args: argparse.Namespace,
    *,
    support: float,
    selected: int,
    total: int,
    digest: str,
    score_batch_counts: dict[str, int],
    pair_hash: str,
) -> dict[str, Any]:
    expected_count = math.floor(total * support)
    if selected != expected_count:
        raise RuntimeError(
            f"WAGLE support count mismatch for {support}: {selected} != {expected_count}"
        )
    expected_counts = {
        "forget": args.forget_score_batches,
        "retain": args.retain_score_batches,
    }
    if score_batch_counts != expected_counts:
        raise RuntimeError(f"WAGLE score coverage mismatch: {score_batch_counts}")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_kind": "wagle_native_snip_advanced_mask",
        "domain": args.domain,
        "seed": args.seed,
        "pair_stream_sha256": pair_hash,
        "score_type": "snip_advanced",
        "score_pairs_per_side": args.score_steps,
        "forget_score_batches": args.forget_score_batches,
        "retain_score_batches": args.retain_score_batches,
        "score_batch_counts": score_batch_counts,
        "support_frac": support,
        "selected_count": selected,
        "eligible_count": total,
        "selection_rounding": "floor",
        "mask_sha256": digest,
        "p": args.wagle_p,
        "q": args.wagle_q,
        "mu": args.wagle_mu,
    }


def _run_masks(args: argparse.Namespace, dataset) -> None:
    import torch
    import transformers
    from unlearn import GenerateMask
    from unlearn.generate_mask import save_masks_from_scores

    _score_artifact_spec = importlib.util.spec_from_file_location(
        "openunlearning_score_artifacts",
        OPENUNLEARNING_ROOT / "src" / "score_artifacts.py",
    )
    if _score_artifact_spec is None or _score_artifact_spec.loader is None:
        raise ImportError("Cannot load OpenUnlearning selector-score artifact helper.")
    _score_artifact_module = importlib.util.module_from_spec(_score_artifact_spec)
    _score_artifact_spec.loader.exec_module(_score_artifact_module)
    save_selector_logits = _score_artifact_module.save_selector_logits

    expected = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "phase": "wagle_stage1_masks",
        "domain": args.domain,
        "seed": args.seed,
        "pair_stream_sha256": dataset.pair_stream_sha256,
        "score_pairs_per_side": args.score_steps,
        "forget_score_batches": args.forget_score_batches,
        "retain_score_batches": args.retain_score_batches,
        "support_fracs": list(args.support_fracs),
        "scores_only": args.scores_only,
        "selector_logits_saved": args.save_selector_logits,
        "selector_logits_storage_dtype": args.selector_logits_storage_dtype,
        "wagle_p": args.wagle_p,
        "wagle_q": args.wagle_q,
        "wagle_mu": args.wagle_mu,
    }
    if _assert_new_or_reusable(args.output_dir, expected):
        print(f"[reuse] WAGLE masks {args.output_dir}", flush=True)
        return
    if args.preflight_only:
        return

    _seed_everything(args.seed)
    model_run = _new_unlearn(
        args, method="GA+FT", max_steps=args.score_steps, accumulation=1, mask_path=None
    )
    model_run.init_model()
    model_run.init_dataset()
    mask_dir = args.output_dir / "snip_advanced"
    mask_dir.mkdir(parents=True, exist_ok=False)
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=max(1, args.score_steps // 10),
        max_steps=args.score_steps,
        learning_rate=METHODS["graddiff"]["lr"],
        bf16=True,
        bf16_full_eval=False,
        logging_steps=max(1, args.score_steps // 20),
        logging_dir=str(args.output_dir / "logs"),
        optim="adamw_torch",
        save_strategy="no",
        weight_decay=0.0,
        remove_unused_columns=False,
        output_dir=str(mask_dir),
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )
    generator = GenerateMask(
        score_type="snip_advanced",
        ratios=args.support_fracs,
        mask_dir=str(mask_dir),
        model=model_run.model,
        data_collator=model_run.unlearn_collator,
        tokenizer=model_run.tokenizer,
        train_dataset=model_run.unlearn_dataset,
        eval_dataset=None,
        compute_metrics=None,
        args=training_args,
        p=args.wagle_p,
        q=args.wagle_q,
        mu=args.wagle_mu,
        max_score_batches=args.score_steps,
        forget_score_batches=args.forget_score_batches,
        retain_score_batches=args.retain_score_batches,
    )
    # Run native WAGLE scoring first, then retain only the CPU score vector and
    # parameter layout.  The model, Trainer/Accelerator state, tokenizer, and
    # pair dataset are not needed by the original global argsort selector.
    # Budget model state separately from the selector's large int64 workspace.
    generator.snip_advanced()
    score_batch_counts = dict(generator.score_batch_counts)
    scores = generator.scores
    parameter_layout = [
        (key, tensor.shape, tensor.numel())
        for key, tensor in generator.model.named_parameters()
    ]
    generator.scores = None
    generator.accelerator.free_memory()
    generator.model = None
    generator.model_wrapped = None
    generator.train_dataset = None
    generator.data_collator = None
    generator.tokenizer = None
    model_run.model = None
    model_run.unlearn_dataset = None
    model_run.test_dataset = None
    model_run.unlearn_collator = None
    model_run.test_collator = None
    model_run.tokenizer = None
    del generator, model_run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.save_selector_logits:
        save_selector_logits(
            scores,
            parameter_layout,
            mask_dir,
            selection_order="lower_is_selected",
            selector="wagle_snip_advanced",
            storage_dtype=args.selector_logits_storage_dtype,
        )
    if not args.scores_only:
        save_masks_from_scores(
            scores, args.support_fracs, str(mask_dir), parameter_layout
        )
    del scores
    if (mask_dir / "scores.pt").exists():
        raise RuntimeError("Fair WAGLE mode must not persist raw score tensors.")
    for support in () if args.scores_only else args.support_fracs:
        mask_path = mask_dir / f"with_{support}.pt"
        if not mask_path.is_file():
            raise FileNotFoundError(f"WAGLE did not create expected mask: {mask_path}")
        mask = torch.load(mask_path, map_location="cpu", weights_only=True)
        if not isinstance(mask, dict):
            raise TypeError(f"WAGLE mask is not a tensor mapping: {mask_path}")
        selected, total, digest = _mask_stats(mask)
        manifest = _mask_manifest(
            args,
            support=support,
            selected=selected,
            total=total,
            digest=digest,
            score_batch_counts=score_batch_counts,
            pair_hash=dataset.pair_stream_sha256,
        )
        _write_json_atomic(mask_dir / f"with_{support}.manifest.json", manifest)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _write_json_atomic(args.output_dir / "stage_success.json", expected)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("masks",), required=True)
    parser.add_argument(
        "--domain", choices=("bio", "cyber", "news", "books"), required=True
    )
    parser.add_argument("--pair-stream", type=Path, required=True)
    parser.add_argument("--expected-pair-stream-sha256")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-steps", type=int)
    parser.add_argument("--forget-score-batches", type=int)
    parser.add_argument("--retain-score-batches", type=int)
    parser.add_argument("--wagle-p", type=float, default=0.01)
    parser.add_argument("--wagle-q", type=float, default=0.01)
    parser.add_argument("--wagle-mu", type=float, default=1e-6)
    parser.add_argument("--support-fracs", type=float, nargs="+", default=(0.4, 0.6))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--save-selector-logits", action="store_true")
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Persist selector logits without materializing native dense masks.",
    )
    parser.add_argument(
        "--selector-logits-storage-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.wagle_p) or args.wagle_p <= 0:
        parser.error("--wagle-p must be finite and positive")
    if not math.isfinite(args.wagle_q) or args.wagle_q <= 0:
        parser.error("--wagle-q must be finite and positive")
    if not math.isfinite(args.wagle_mu) or args.wagle_mu <= 0:
        parser.error("--wagle-mu must be finite and positive")
    if any(
        not math.isfinite(value) or not 0 < value < 1 for value in args.support_fracs
    ):
        parser.error("--support-fracs values must lie strictly between 0 and 1")
    if args.scores_only and not args.save_selector_logits:
        parser.error("--scores-only requires --save-selector-logits")
    return args


def main() -> None:
    args = parse_args()
    from dataset.fair_wmdp import FairWMDPUnlearnDataset

    _set_offline()
    args.pair_stream = args.pair_stream.resolve()
    args.model_path = args.model_path.resolve()
    if args.tokenizer_path is not None:
        args.tokenizer_path = args.tokenizer_path.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Pinned Zephyr model is incomplete: {args.model_path}")
    dataset = FairWMDPUnlearnDataset(
        pair_stream_path=str(args.pair_stream),
        expected_pair_stream_sha256=args.expected_pair_stream_sha256,
    )
    validate_pair_contract(args.domain, dataset)
    if args.score_steps is None:
        args.score_steps = dataset.score_pair_count
    elif not 0 < args.score_steps <= dataset.score_pair_count:
        raise ValueError(
            "--score-steps must be positive and no greater than available fair pairs."
        )
    for option in ("forget_score_batches", "retain_score_batches"):
        requested = getattr(args, option)
        if requested is None:
            setattr(args, option, args.score_steps)
        elif not 0 < requested <= dataset.score_pair_count:
            raise ValueError(
                f"--{option.replace('_', '-')} must be positive and no greater "
                "than available fair pairs."
            )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "domain": args.domain,
                    "phase": args.phase,
                    "pair_stream_sha256": dataset.pair_stream_sha256,
                    "score_steps": args.score_steps,
                    "forget_score_batches": args.forget_score_batches,
                    "retain_score_batches": args.retain_score_batches,
                    "wagle_p": args.wagle_p,
                    "wagle_q": args.wagle_q,
                    "wagle_mu": args.wagle_mu,
                    "stage2_microbatches": dataset.stage2_microbatches,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _run_masks(args, dataset)


if __name__ == "__main__":
    main()
