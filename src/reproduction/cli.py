"""Driver for the two public reproduction scripts."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="action", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--kind", choices=("masks", "stage2"), required=True)
    submit.add_argument("--config", type=Path)
    mode = submit.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose commands without loading models, downloading, or launching jobs",
    )
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Download and validate the pinned inputs, then exit",
    )
    run = sub.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    return command


def dry_run(repo, work, kind):
    from .benchmarks import (
        compose_command,
        stage1_command,
        stage2_command,
        eval_commands,
    )
    from .config import contract_hash, load_contract
    from .matrix import stage1_cells, stage2_cells, mask_key

    cells = stage1_cells() if kind == "masks" else stage2_cells()
    checked = set()
    commands = []
    for cell in cells:
        identity = (cell.dataset, cell.selector, getattr(cell, "updater", None))
        if identity in checked:
            continue
        checked.add(identity)
        spec = load_contract()["datasets"][cell.dataset]
        base = work / "assets" / cell.dataset
        counts = {
            "wmdpall": (25453, 25453),
            "muse-news": (407, 803),
            "muse-books": (553, 105),
        }[cell.dataset]
        asset = dict(
            model=str(base / "model"),
            tokenizer=str(base / "tokenizer"),
            stage1=str(base / "stage1/pairs.pt"),
            stage2=str(base / "stage2/pairs.pt"),
            stage1_sha256=spec["traces"]["stage1_sha256"],
            stage2_sha256=spec["traces"]["stage2_sha256"],
            score_steps=max(counts) // 2,
            forget_score_batches=counts[0],
            retain_score_batches=counts[1],
            retain_logs=str(base / "retain.json"),
            snapshots={
                k: dict(path=str(base / k), **v)
                for k, v in spec["evaluation"]
                .get("datasets", {"dataset": spec.get("dataset", {})})
                .items()
            },
        )
        command = (
            stage1_command(repo, cell, asset, base / "scores")
            if kind == "masks"
            else stage2_command(repo, cell, asset, base / "cell")
        )
        if kind != "masks" or cell.selector != "wagle":
            compose_command(repo, command)
        for evaluation in eval_commands(
            repo, cell.dataset, asset, base / "model", base / "eval"
        ):
            compose_command(repo, evaluation)
        commands.append(command)
    return dict(
        status="planned",
        stage=kind,
        cells=len(cells),
        masks=len({mask_key(c) for c in cells}),
        contract_hash=contract_hash(),
        selections=[asdict(c) for c in cells],
        commands=commands,
    )


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo / "src"))
        if args.action == "run":
            from .orchestrator import run_request

            run_request(args.request.resolve())
            return 0
        from .settings import activate, load_settings

        path = (
            args.config
            or repo / f"configs/{'stage1' if args.kind == 'masks' else 'stage2'}.yaml"
        )
        settings = load_settings(path)
        import yaml

        expected_stage = 1 if args.kind == "masks" else 2
        if yaml.safe_load(path.read_text()).get("stage") != expected_stage:
            raise ValueError("The configuration belongs to the other stage")
        activate(settings)
        work = Path(
            os.environ.get(
                "SPARSE_UNLEARN_WORK_ROOT", repo.with_name(repo.name + "-work")
            )
        ).resolve()
        if work == repo or work.is_relative_to(repo):
            raise ValueError("The work root must be outside the source directory")
        if args.dry_run:
            result = dry_run(repo, work, args.kind)
        elif args.prepare:
            from .assets import prepare_assets

            stage = settings["stage1" if args.kind == "masks" else "stage2"]
            record = prepare_assets(
                repo, work / "assets", tuple(stage["datasets"]), local_files_only=False
            )
            result = dict(
                status="inputs prepared", manifest_hash=record["manifest_hash"]
            )
        else:
            from .orchestrator import submit_campaign

            result = submit_campaign(repo, work, args.kind)
        print(json.dumps(result))
    except Exception as error:
        print(f"failed: {error}; no valid result", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
