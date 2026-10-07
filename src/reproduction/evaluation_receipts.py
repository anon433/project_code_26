"""Bind cache-skipping evaluator passes to their inputs before allowing reuse."""

from pathlib import Path

from .benchmarks import compose_command
from .io import json_hash
from .state import seal_boundary, validate_boundary, validate_checkpoint


def _files(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe evaluation output path")
        if path.is_file():
            files[str(path.relative_to(root))] = path
    return files


def evaluation_passes(
    repo,
    directory,
    dataset,
    request_digest,
    commands,
    *,
    execute=None,
    after_process=None,
    checkpoint_binding=None,
):
    """Precheck every pass, then run only fresh passes and seal successful exits.

    A killed or failed child leaves unsealed output requiring explicit recovery.
    Completed passes resume without launching a cache-skipping evaluator. A pass
    receipt binds exact argv, request, checkpoint content and the whole output
    inventory; hashing cached files only after a new child exits is insufficient.
    """
    results = directory / "results"
    plans, checkpoints = [], {}
    for index, command in enumerate(commands):
        root, checkpoint = results, checkpoint_binding
        if any(arg.startswith("--config-name=") for arg in command):
            cfg = compose_command(Path(repo), command)
            settings = cfg.eval.lm_eval if dataset == "wmdpall" else cfg.eval.muse
            root = Path(settings.output_dir)
            model = str(
                Path(cfg.model.model_args.pretrained_model_name_or_path).resolve()
            )
            if checkpoint_binding is None:
                if model not in checkpoints:
                    checkpoints[model] = validate_checkpoint(Path(model))
                checkpoint = {"path": model, "files": checkpoints[model]}
            else:
                checkpoint = {"path": model, "training_receipt": checkpoint_binding}
        if not root.resolve().is_relative_to(results.resolve()):
            raise ValueError("Evaluation output escapes result directory")
        if any(
            root.resolve().is_relative_to(p[0].resolve())
            or p[0].resolve().is_relative_to(root.resolve())
            for p in plans
        ):
            raise ValueError("Overlapping evaluation pass outputs")
        digest = json_hash(
            {"request": request_digest, "command": command, "checkpoint": checkpoint}
        )
        receipt = directory / "eval-passes" / f"pass-{index}.json"
        files = _files(root)
        if receipt.exists():
            sealed = validate_boundary(receipt, digest)
            if {name: item["path"] for name, item in sealed["files"].items()} != {
                name: str(path.absolute()) for name, path in files.items()
            }:
                raise ValueError("Evaluation pass output inventory changed")
        elif files:
            raise ValueError(
                "Unsealed evaluation output exists; explicit recovery required"
            )
        elif execute is None:
            raise ValueError("Missing sealed evaluation pass")
        plans.append((root, receipt, digest, command))
    # Root-level merged outputs also cannot be imported from another run. They
    # are generated only after every pass has been validated below.
    if execute is not None and any(
        path.is_file() and not any(path.is_relative_to(plan[0]) for plan in plans)
        for path in results.glob("*.json")
    ):
        raise ValueError(
            "Unsealed merged evaluation output exists; explicit recovery required"
        )
    for index, (root, receipt, digest, command) in enumerate(plans):
        if receipt.exists():
            continue
        child = execute(command, index)
        seal_boundary(receipt, _files(root), digest)
        if after_process:
            after_process(child)
    return {f"pass:{index}": plan[1] for index, plan in enumerate(plans)}
