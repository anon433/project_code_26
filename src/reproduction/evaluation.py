"""Run the public evaluator with benchmark loads bound to pinned local data.

lm-eval's MMLU group expands into individual tasks with separate dataset kwargs.
Bind the datasets API boundary for this child process so every group member uses
the validated snapshot, never a mutable Hub ref or a pre-existing default cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys


def collect_outputs(repo, dataset, commands, output):
    """Merge the distinct MUSE pass outputs declared by the executed argv."""
    if dataset == "wmdpall" or not any(
        any(arg.startswith("--config-name=") for arg in command) for command in commands
    ):
        return
    from .benchmarks import compose_command
    from .io import read_json, write_json

    logs, summary = {}, {}
    for command in commands:
        cfg = compose_command(Path(repo), command)
        root = Path(cfg.eval.muse.output_dir)
        requested = set(cfg.eval.muse.metrics)
        values = read_json(root / "MUSE_SUMMARY.json")
        detailed = read_json(root / "MUSE_EVAL.json")
        if (
            set(values) != requested
            or set(detailed) != requested
            or set(summary) & requested
        ):
            raise ValueError("MUSE pass output metric membership differs from command")
        for key in requested:
            value = values[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or detailed[key].get("agg_value") != value
            ):
                raise ValueError("MUSE pass summary differs from finite metric detail")
        summary.update(values)
        logs.update(detailed)
    write_json(output / "MUSE_EVAL.json", logs)
    write_json(output / "MUSE_SUMMARY.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("wmdpall", "muse-news", "muse-books"), default="wmdpall"
    )
    parser.add_argument("--snapshot-records", type=json.loads, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a public evaluator script is required")
    # Set offline/cache values before importing HF libraries.
    cache = args.dataset_cache.resolve()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_CACHE=str(cache),
    )
    from .assets import validate_evaluation_snapshot
    from .config import load_contract
    import datasets

    selected = load_contract()["datasets"][args.dataset]
    specs = (
        selected["evaluation"]["datasets"]
        if args.dataset == "wmdpall"
        else {"dataset": selected["dataset"]}
    )
    if set(args.snapshot_records) != set(specs):
        raise ValueError("Incomplete pinned evaluation snapshot set")
    pinned, layouts = {}, {}
    for name, spec in specs.items():
        record = args.snapshot_records[name]
        layouts[spec["repo"]] = validate_evaluation_snapshot(
            record, {**spec, "repo_type": "dataset"}
        )
        pinned[spec["repo"]] = record
    load_dataset = datasets.load_dataset

    def local_dataset(path, name=None, **kwargs):
        if path not in pinned:
            raise ValueError(f"Unpinned evaluation dataset: {path}")
        record = pinned[path]
        revision = kwargs.pop("revision", None)
        if revision is not None and revision != record["revision"]:
            raise ValueError("Evaluation dataset revision differs from pin")
        # Supplying data_files/data_dir could bypass the pinned local snapshot.
        if any(kwargs.get(key) is not None for key in ("data_files", "data_dir")):
            raise ValueError("Evaluation dataset file overrides are forbidden")
        if name not in layouts[path]:
            raise ValueError(f"Unpinned evaluation dataset configuration: {name}")
        kwargs["cache_dir"] = str(cache)
        kwargs["download_config"] = datasets.DownloadConfig(local_files_only=True)
        # Explicit files avoid fsspec wildcard inference treating HF blob symlinks
        # as non-files. The validated card supplies every required split.
        return load_dataset(
            "parquet", name=name, data_files=layouts[path][name], **kwargs
        )

    datasets.load_dataset = local_dataset
    try:
        sys.argv = command
        sys.path.insert(0, str(Path(command[0]).resolve().parent))
        runpy.run_path(command[0], run_name="__main__")
    finally:
        datasets.load_dataset = load_dataset


if __name__ == "__main__":
    main()
