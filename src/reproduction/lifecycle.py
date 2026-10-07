"""Durable train/evaluate/cleanup boundaries with content-verified resume."""

from pathlib import Path
import os

from .config import contract_hash
from .evaluation import collect_outputs
from .evaluation_receipts import evaluation_passes
from .io import read_json, write_json, sha256_file
from .process import process_alive, run_logged
from .state import (
    StateStore,
    seal_boundary,
    validate_boundary,
    validate_checkpoint,
    validate_results,
)


def run_score_search(
    directory,
    selector,
    request_digest,
    command,
    expected_layout,
    targets,
    *,
    cwd=None,
    before_process=None,
    after_process=None,
    on_process=None,
    before_materialize=None,
    score_context=None,
):
    """One native score subprocess, then all exact support masks and cleanup."""
    from .masks import (
        load_scores,
        publish_mask,
        select_exact,
        validate_layout,
        selected_count,
    )

    store = StateStore(directory / "state.json", contract_hash(), request_digest)
    state = store.read()
    raw = directory / "raw"
    receipt = directory / "complete.json"
    score_receipt = directory / "scores.json"

    def cleanup():
        removed = []
        for path in raw.rglob("*.pt"):
            if not path.resolve().is_relative_to(raw.resolve()):
                raise ValueError("Unsafe score cleanup target")
            removed.append({"path": str(path), "bytes": path.stat().st_size})
            path.unlink()
        if removed:
            write_json(
                directory / "cleanup.json",
                {"request_hash": request_digest, "removed": removed},
            )

    if receipt.exists():
        validate_boundary(receipt, request_digest)
        for key, destination, digest in targets:
            validate_boundary(destination / "complete.json", digest)
            validate_layout(destination, key, contract_hash(), expected_layout)
        cleanup()
        store.update(status="complete", boundary="cleanup", child=None)
        return
    if state.get("child") and process_alive(state["child"]):
        raise RuntimeError("Score search already has a live process")
    if targets and all(
        (destination / "complete.json").is_file() for _, destination, _ in targets
    ):
        reused = {}
        for key, destination, digest in targets:
            validate_boundary(destination / "complete.json", digest)
            manifest = validate_layout(
                destination, key, contract_hash(), expected_layout
            )
            if (
                manifest.get("score_provenance", {}).get("request_hash")
                != request_digest
            ):
                raise ValueError("Mask score request mismatch")
            for name in ("complete.json", "manifest.json", "mask.packbits"):
                reused[f"{key.support}:{name}"] = destination / name
        seal_boundary(receipt, reused, request_digest)
        store.update(status="complete", boundary="cleanup", child=None)
        return

    try:
        if score_receipt.exists():
            validate_boundary(score_receipt, request_digest)
        else:
            if raw.exists() and any(raw.iterdir()):
                raise ValueError(
                    "Unsealed score output exists; explicit recovery required"
                )
            if before_process:
                before_process()

            def started(identity):
                store.update(status="running", boundary="score", child=identity)
                if on_process:
                    on_process(identity)

            child = run_logged(
                command,
                cwd or directory,
                directory / "search.log",
                on_start=started,
                env={**os.environ, "FAIR_WMDP_SEQUENTIAL": "1"},
            )
            scores, _, files = load_scores(
                raw, selector, expected_layout, **(score_context or {})
            )
            del scores
            seal_boundary(score_receipt, files, request_digest)
            store.update(status="queued", boundary="materialize", child=None)
            if after_process:
                after_process(child)
        if before_materialize:
            before_materialize()
        scores, provenance, _ = load_scores(
            raw, selector, expected_layout, **(score_context or {})
        )
        provenance["request_hash"] = request_digest
        files = {"score_receipt": score_receipt}
        for key, destination, digest in targets:
            mask_receipt = destination / "complete.json"
            if mask_receipt.exists():
                validate_boundary(mask_receipt, digest)
                manifest = validate_layout(
                    destination, key, contract_hash(), expected_layout
                )
                if (
                    manifest.get("score_provenance", {}).get("request_hash")
                    != request_digest
                ):
                    raise ValueError("Mask score request mismatch")
            else:
                if destination.exists() and any(destination.iterdir()):
                    raise ValueError(
                        "Unsealed budget mask exists; explicit recovery required"
                    )
                selected = select_exact(
                    scores,
                    selected_count(provenance["numel"], key),
                    higher=provenance["higher_is_selected"],
                )
                publish_mask(
                    destination, key, selected, contract_hash(), provenance=provenance
                )
                del selected
                validate_layout(destination, key, contract_hash(), expected_layout)
                seal_boundary(
                    mask_receipt,
                    {
                        "mask": destination / "mask.packbits",
                        "manifest": destination / "manifest.json",
                    },
                    digest,
                )
            for name in ("mask.packbits", "manifest.json", "complete.json"):
                files[f"{key.support}:{name}"] = destination / name
        del scores
        seal_boundary(receipt, files, request_digest)
        validate_boundary(receipt, request_digest)
        cleanup()
        store.update(status="complete", boundary="cleanup", child=None)
    except Exception as error:
        store.update(status="failed", error=str(error), child=None)
        raise


def predecessor_ready(path):
    if not path.is_file():
        raise RuntimeError("Required mask predecessor has not been submitted")
    record = read_json(path)
    if record.get("status") == "failed":
        raise RuntimeError("Mask predecessor failed; no valid result")
    if record.get("status") == "complete":
        StateStore(path, contract_hash(), record["request_hash"]).read()
        receipt = validate_boundary(
            path.parent / "complete.json", record["request_hash"]
        )
        for item in receipt["files"].values():
            child = Path(item["path"])
            validate_boundary(child, read_json(child)["request_hash"])
        return True
    if record.get("status") not in {"queued", "running"} or not process_alive(
        record.get("runner", {})
    ):
        raise RuntimeError(
            f"Mask predecessor has a stale session: {record.get('runner')}"
        )
    return False


def run_evaluation(
    directory,
    dataset,
    request_digest,
    commands,
    *,
    cwd=None,
    before_process=None,
    after_process=None,
    on_process=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "complete.json"
    results = directory / "results"
    if receipt.exists():
        validate_boundary(receipt, request_digest)
        evaluation_passes(
            cwd or directory, directory, dataset, request_digest, commands
        )
        validate_results(results, dataset)
        return

    def execute(command, index):
        if before_process:
            before_process()
        return run_logged(
            command,
            cwd or directory,
            directory / f"eval-{index}.log",
            on_start=on_process,
        )

    passes = evaluation_passes(
        cwd or directory,
        directory,
        dataset,
        request_digest,
        commands,
        execute=execute,
        after_process=after_process,
    )
    collect_outputs(cwd or directory, dataset, commands, results)
    validate_results(results, dataset)
    seal_boundary(
        receipt,
        {
            **passes,
            **{
                str(file.relative_to(results)): file for file in results.rglob("*.json")
            },
        },
        request_digest,
    )


def run_cell(
    directory,
    dataset,
    request_digest,
    train_command,
    evaluation_commands,
    *,
    cwd=None,
    before_process=None,
    after_process=None,
    after_boundary=None,
    on_process=None,
    provenance=None,
    checkpoint=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    if provenance is not None:
        if provenance.get("execution", {}).get("cell_request_hash") != request_digest:
            raise ValueError("Cell request differs from execution-time provenance")
    store = StateStore(directory / "state.json", contract_hash(), request_digest)
    state = store.read()
    if state.get("child") and process_alive(state["child"]):
        raise RuntimeError("Cell already has a live child process")
    checkpoint = directory / "checkpoint" if checkpoint is None else Path(checkpoint)
    if (
        checkpoint.resolve() == directory.resolve()
        or not checkpoint.resolve().is_relative_to(directory.resolve())
    ):
        raise ValueError("Unsafe checkpoint path outside cell directory")
    results = directory / "results"
    train_receipt, eval_receipt = directory / "train.json", directory / "eval.json"
    final_receipt = directory / "complete.json"
    if final_receipt.exists():
        validate_boundary(final_receipt, request_digest)
        validate_results(results, dataset)
        store.update(status="complete", boundary="cleanup")
        return
    if provenance is not None:
        execution_path = directory / "execution-request.json"
        if (
            execution_path.exists()
            and read_json(execution_path) != provenance["execution_request"]
        ):
            raise ValueError("Captured execution request drift")
        write_json(execution_path, provenance["execution_request"])

    def execute(command, boundary, index=0):
        if before_process:
            before_process()
        store.update(status="queued", boundary=boundary, reason="starting child")

        def started(identity):
            store.update(status="running", child=identity)
            if on_process:
                on_process(identity)

        child = run_logged(
            command,
            cwd or directory,
            directory / f"{boundary}-{index}.log",
            on_start=started,
            env={
                **os.environ,
                "FAIR_WMDP_SEQUENTIAL": "1",
                "FAIR_WMDP_TRACE_PATH": str((directory / "consumed.tsv").absolute()),
            }
            if provenance is not None and boundary == "train"
            else None,
        )
        store.update(status="queued", child=None, reason="validating durable output")
        return child

    def finished(boundary):
        store.update(
            status="queued", boundary=boundary, reason="durable boundary validated"
        )
        if after_boundary:
            after_boundary(boundary)

    try:
        if not eval_receipt.exists():
            if train_receipt.exists():
                validate_boundary(train_receipt, request_digest)
                validate_checkpoint(checkpoint)
            else:
                if checkpoint.exists() and any(checkpoint.iterdir()):
                    raise ValueError(
                        "Unsealed checkpoint exists; explicit recovery required"
                    )
                child = execute(train_command, "train")
                hashes = validate_checkpoint(checkpoint)
                provenance_files = {}
                if provenance is not None:
                    from .provenance import publish_provenance

                    publish_provenance(
                        directory, provenance, train_command, evaluation_commands
                    )
                    provenance_files = {
                        "provenance": directory / "provenance.json",
                        "consumption": directory / "consumed.tsv",
                        "campaign_request": directory / "execution-request.json",
                        "cell_request": directory / "request.json",
                    }
                seal_boundary(
                    train_receipt,
                    {
                        **{name: checkpoint / name for name in hashes},
                        **provenance_files,
                    },
                    request_digest,
                )
                if after_process:
                    after_process(child)
                finished("train")
            passes = evaluation_passes(
                cwd or directory,
                directory,
                dataset,
                request_digest,
                evaluation_commands,
                execute=lambda command, index: execute(command, "eval", index),
                after_process=after_process,
                checkpoint_binding=sha256_file(train_receipt),
            )
            collect_outputs(cwd or directory, dataset, evaluation_commands, results)
            validate_results(results, dataset)
            seal_boundary(
                eval_receipt,
                {
                    "training": train_receipt,
                    **passes,
                    **{
                        str(file.relative_to(results)): file
                        for file in results.rglob("*.json")
                    },
                },
                request_digest,
            )
            finished("eval")
        validate_boundary(eval_receipt, request_digest)
        validate_results(results, dataset)
        # A sealed evaluation authorizes deletion only of this cell's temporary
        # dense mask and the exact weight files recorded by training.
        training = read_json(train_receipt)
        removed = []
        for name, item in training["files"].items():
            if name in (
                "provenance",
                "consumption",
                "campaign_request",
                "cell_request",
            ):
                continue
            path = Path(item["path"])
            if path.parent.resolve() != checkpoint.resolve():
                raise ValueError("Unsafe checkpoint cleanup target")
            if name.endswith(".safetensors"):
                path.unlink(missing_ok=True)
                removed.append(str(path))
        dense = directory / "mask" / "active_mask.pt"
        dense.unlink(missing_ok=True)
        cleanup = directory / "cleanup.json"
        write_json(
            cleanup,
            {
                "request_hash": request_digest,
                "removed": removed + [str(dense)],
                "recoverable": "checkpoints require retraining; dense masks restore from packed masks",
            },
        )
        seal_boundary(
            final_receipt,
            {
                "training": train_receipt,
                "evaluation": eval_receipt,
                "cleanup": cleanup,
                **(
                    {
                        "provenance": directory / "provenance.json",
                        "consumption": directory / "consumed.tsv",
                        "campaign_request": directory / "execution-request.json",
                        "cell_request": directory / "request.json",
                    }
                    if provenance is not None
                    else {}
                ),
                **{
                    f"result:{file.relative_to(results)}": file
                    for file in results.rglob("*.json")
                },
            },
            request_digest,
        )
        store.update(status="complete", boundary="cleanup", child=None)
    except InterruptedError:
        raise
    except Exception as error:
        store.update(status="failed", error=str(error), child=None)
        raise
