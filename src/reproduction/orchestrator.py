"""Foreground validation and serial, screen-owned paper reproduction jobs."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .io import json_hash, read_json, sha256_file, write_json
from .lifecycle import predecessor_ready, run_cell, run_evaluation, run_score_search
from .process import (
    gpu_lock,
    memory_snapshot,
    process_identity,
    reclaim_cache,
    require_headroom,
    screen_listing,
    screen_present,
    training_environment,
    validate_launch,
)
from .config import contract_hash, load_contract
from .settings import activate, current
from .matrix import Stage1Cell, mask_key, stage1_cells, stage2_cells
from .benchmarks import (
    compose_command,
    eval_commands,
    stage1_command,
    stage2_command,
    checkpoint_directory,
)
from .state import StateStore, seal_boundary, validate_boundary

GIB = 1 << 30


def wait_for_masks(ready, predecessor, store, *, poll=5.0):
    while not check_mask_dependency(ready(), predecessor):
        store.update(status="queued", reason="waiting for validated masks")
        time.sleep(poll)


def check_mask_dependency(ready, predecessor):
    if predecessor.exists() or not ready:
        if predecessor_ready(predecessor) and not ready:
            raise RuntimeError("Completed predecessor is missing required masks")
    return ready


def check_resources(memory, gpus, disk_free, *, available, minimum_disk=100 * GIB):
    if len(gpus) != 2 or any(total < 80000 for total, _ in gpus):
        raise RuntimeError(
            "Exactly two GPUs with at least 80,000 MiB each are required"
        )
    if memory["limit"] < 200 * GIB:
        raise RuntimeError("At least 200 GiB cgroup memory is required")
    if disk_free < minimum_disk:
        raise RuntimeError(
            f"Insufficient free work disk: need {minimum_disk / GIB:.1f} GiB"
        )
    if available:
        require_headroom(memory, 120 * GIB)
        if any(free < 75000 for _, free in gpus):
            raise RuntimeError("Insufficient free GPU memory")


def submit_screen(command, repo, name, ack, log, *, screen="screen", timeout=15):
    if screen_present(screen_listing(screen), name):
        raise RuntimeError(f"A live screen already owns this request: {name}")
    if ack.exists():
        raise RuntimeError(f"Stale session handshake requires recovery: {ack}")
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([screen, "-dmS", name, *command], cwd=repo, check=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ack.exists():
            identity = read_json(ack)
            try:
                validate_launch(identity, name, screen_listing(screen))
                if identity["command"] != command:
                    raise ValueError("Runner command differs from submitted command")
                return identity
            except ValueError:
                pass
        time.sleep(0.1)
    raise RuntimeError(f"Screen launch unverified; no valid result: {name}; log={log}")


def datasets_for(kind):
    return tuple(current()["stage1" if kind == "masks" else "stage2"]["datasets"])


def job_directory(work, kind):
    stage = current()["stage1" if kind == "masks" else "stage2"]
    return work / "jobs" / f"{kind}-{json_hash(stage)[:12]}"


def gpu_memory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        tuple(int(value.strip()) for value in line.split(","))
        for line in result.stdout.splitlines()
    ]


def source_digest(repo):
    paths = [
        *repo.glob("src/**/*.py"),
        *repo.glob("configs/**/*.yaml"),
        *repo.glob("third_party/WAGLE/src/**/*.py"),
        *repo.glob("third_party/WAGLE/scripts/run_fair_wmdp.py"),
    ]
    return json_hash(
        {
            str(path.relative_to(repo)): sha256_file(path)
            for path in sorted(paths)
            if path not in (repo / "configs/stage1.yaml", repo / "configs/stage2.yaml")
        }
    )


def cell_directory(work, cell):
    return (
        work
        / "cells"
        / cell.dataset
        / f"support{round(cell.support*100):02d}"
        / cell.updater
        / cell.selector
        / f"mask-seed{cell.mask_seed}"
        / f"seed{cell.seed}"
    )


def mask_directory(work, cell):
    return (
        work
        / "masks"
        / cell.dataset
        / cell.selector
        / f"seed{mask_key(cell).seed}"
        / f"support{round(cell.support*100):02d}"
    )


def mask_request(repo, work, cell, asset):
    directory = mask_directory(work, cell)
    _, command, source_hash = score_request(repo, work, cell, asset)
    digest = json_hash(
        {"mask": asdict(mask_key(cell)), "score_request_hash": source_hash}
    )
    return directory, command, digest


def score_request(repo, work, cell, asset):
    stage1 = Stage1Cell(cell.dataset, 0.1, cell.selector, mask_key(cell).seed)
    directory = (
        work
        / "scores"
        / cell.dataset
        / cell.selector
        / f"seed{stage1.seed}"
        / current()["stage1"]["budgets"]
    )
    command = stage1_command(repo, stage1, asset, directory / "raw")
    digest = json_hash(
        {
            "cell": asdict(stage1),
            "command": [
                arg.replace(str(directory), "<score-output>") for arg in command
            ],
            "asset": asset,
            "contract_hash": contract_hash(),
            "source_hash": source_digest(repo),
        }
    )
    return directory, command, digest


def masks_ready(repo, work, cells, assets):
    from .masks import validate_mask

    unique = {mask_key(cell): cell for cell in cells}.values()
    if any(
        not (mask_directory(work, cell) / "complete.json").is_file() for cell in unique
    ):
        return False
    for cell in unique:
        directory, _, digest = mask_request(
            repo, work, cell, assets["datasets"][cell.dataset]
        )
        if (directory / "manifest.json").exists():
            validate_mask(directory, mask_key(cell), contract_hash())
        validate_boundary(directory / "complete.json", digest)
        validate_mask(directory, mask_key(cell), contract_hash())
    return True


def required_disk_bytes(work, cells, layouts):
    packed = 0
    for cell in cells:
        if not (mask_directory(work, cell) / "complete.json").exists():
            packed += sum(
                (math.prod(shape) + 7) // 8 for shape in layouts[cell.dataset].values()
            )
    # GEC's trainer may save twice; its atomic writer briefly retains the old
    # and new FP32 score files. Reserve room for that overlap plus one budget.
    return 70 * GIB + packed


def preflight(repo, work, kind, profile):
    from . import assets as preparation

    contract = load_contract()
    datasets = datasets_for(kind)
    if kind not in {"masks", "stage2"}:
        raise ValueError("Invalid stage")
    prepared = preparation.prepare_assets(repo, work / "assets", datasets)
    if not shutil.which("screen"):
        raise RuntimeError("screen is required for long-running work")
    layouts = (
        {
            dataset: _model_layout(prepared["datasets"][dataset]["model"])
            for dataset in datasets
        }
        if kind == "masks"
        else {}
    )
    minimum_disk = (
        required_disk_bytes(work, stage1_cells(), layouts)
        if kind == "masks"
        else 40 * GIB
    )
    check_resources(
        memory_snapshot(),
        gpu_memory(),
        shutil.disk_usage(work).free,
        available=False,
        minimum_disk=minimum_disk,
    )
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], check=True, capture_output=True, text=True
    ).stdout
    if "GPU0" not in topology or "GPU1" not in topology:
        raise RuntimeError("Cannot verify two-GPU topology")
    # Import the actual execution stacks in foreground: missing optional
    # packages must fail before any screen session is submitted.
    probe = [sys.executable, str(repo / "src/train.py"), "--help"]
    subprocess.run(
        probe,
        cwd=repo,
        env=training_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, str(repo / "src/eval.py"), "--help"],
        cwd=repo,
        env=training_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    if kind == "masks":
        subprocess.run(
            [sys.executable, "-m", "src.reproduction.wagle", "--help"],
            cwd=repo,
            env=training_environment(),
            check=True,
            capture_output=True,
            text=True,
        )
    cells = (
        stage1_cells()
        if kind == "masks"
        else tuple(cell for cell in stage2_cells(profile) if cell.dataset in datasets)
    )
    # Compose every selected recipe, including each updater-conditioned TCUS
    # selector. Budgets and seeds use the same tested schema.
    checked = set()
    for cell in cells:
        identity = (
            (cell.dataset, cell.selector)
            if kind == "masks"
            else (cell.dataset, cell.updater)
        )
        if identity in checked:
            continue
        checked.add(identity)
        asset = prepared["datasets"][cell.dataset]
        command = (
            stage1_command(repo, cell, asset, mask_directory(work, cell) / "raw")
            if kind == "masks"
            else stage2_command(repo, cell, asset, cell_directory(work, cell))
        )
        if cell.selector != "wagle" or kind != "masks":
            compose_command(repo, command)
        for evaluation in eval_commands(
            repo, cell.dataset, asset, Path(asset["model"]), work / "preflight-eval"
        ):
            compose_command(repo, evaluation)
    if kind != "masks":
        predecessor = job_directory(work, "masks") / "state.json"
        check_mask_dependency(masks_ready(repo, work, cells, prepared), predecessor)
    payload = {
        "schema_version": 1,
        "settings": current(),
        "repo": str(repo),
        "work": str(work),
        "kind": kind,
        "profile": profile,
        "contract_hash": contract_hash(contract),
        "source_hash": source_digest(repo),
        "assets_hash": asset_identity(prepared, kind),
    }
    return seal_request(payload)


def validate_campaign(directory, digest):
    record = validate_boundary(directory / "complete.json", digest)
    for item in record["files"].values():
        path = Path(item["path"])
        nested = read_json(path)
        validate_boundary(path, nested["request_hash"])


def request_identity(request):
    unsigned = {
        k: v for k, v in request.items() if k not in ("request_hash", "content_hash")
    }
    settings = unsigned["settings"]
    own = "stage1" if unsigned["kind"] == "masks" else "stage2"
    unsigned["settings"] = {"contract": settings["contract"], own: settings[own]}
    return json_hash(unsigned)


def seal_request(payload):
    result = {
        k: v for k, v in payload.items() if k not in ("request_hash", "content_hash")
    }
    result["request_hash"] = request_identity(result)
    result["content_hash"] = json_hash(result)
    return result


def asset_identity(prepared, kind):
    return json_hash(
        {dataset: prepared["datasets"][dataset] for dataset in datasets_for(kind)}
    )


def validate_request(request):
    unsigned = {k: v for k, v in request.items() if k != "content_hash"}
    if request.get("content_hash") != json_hash(unsigned):
        raise ValueError("Request content hash mismatch")
    if request.get("request_hash") != request_identity(request):
        raise ValueError("Request identity mismatch")
    activate(request["settings"])
    repo = Path(request["repo"])
    if request["contract_hash"] != contract_hash():
        raise ValueError("Request contract drift")
    if request["source_hash"] != source_digest(repo):
        raise ValueError("Request source/config drift")


class ResourceGate:
    """Wait for available resources and clean cache only while GPUs are idle."""

    def __init__(
        self,
        work,
        store,
        prepared,
        *,
        cgroup=Path("/sys/fs/cgroup"),
        poll=5.0,
        drop_caches=Path("/proc/sys/vm/drop_caches"),
        minimum_disk=40 * GIB,
    ):
        self.work, self.store = work, store
        self.cgroup, self.poll, self.drop_caches = cgroup, poll, drop_caches
        self.minimum_disk = minimum_disk
        self.roots = {work}
        for asset in prepared["datasets"].values():
            for snapshot in asset["snapshots"].values():
                blobs = Path(snapshot["path"]).parent.parent / "blobs"
                if blobs.is_dir():
                    self.roots.add(blobs)

    def __call__(self, *, minimum_disk=None):
        self.store.update(
            status="queued",
            reason="waiting for idle GPUs and cgroup headroom",
            child=None,
        )
        before = memory_snapshot(self.cgroup)
        reclaimed = False
        while True:
            gpus = gpu_memory()
            processes = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if processes:
                time.sleep(self.poll)
                continue
            if not reclaimed:
                for root in self.roots:
                    reclaim_cache(root, drop_caches=self.drop_caches)
                reclaimed = True
            snapshot = memory_snapshot(self.cgroup)
            self.store.update(memory=snapshot)
            try:
                check_resources(
                    snapshot,
                    gpus,
                    shutil.disk_usage(self.work).free,
                    available=True,
                    minimum_disk=self.minimum_disk
                    if minimum_disk is None
                    else minimum_disk,
                )
                if snapshot["file"] > max(8 * GIB, before["file"] - 1):
                    raise RuntimeError("File cache has not fallen yet")
                return
            except RuntimeError as error:
                self.store.update(status="queued", reason=str(error))
                reclaimed = False
                time.sleep(self.poll)


def _model_layout(path):
    from safetensors import safe_open

    layout = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            layout.update(
                {
                    name: list(handle.get_slice(name).get_shape())
                    for name in handle.keys()
                }
            )
    if not layout:
        import torch

        for shard in sorted(Path(path).glob("pytorch_model*.bin")):
            tensors = torch.load(
                shard, weights_only=True, map_location="cpu", mmap=True
            )
            layout.update({name: list(value.shape) for name, value in tensors.items()})
    if not layout:
        raise ValueError("Cannot read target model parameter layout")
    return layout


def execute_request(request, *, cells=None, gate=None):
    """Internal Python entry permits a validation-cell subset; shells do not."""
    from trainer.mask_artifacts import save_active_mask_artifact
    from .assets import validate_assets
    from .masks import restore_mask

    validate_request(request)
    repo, work = Path(request["repo"]), Path(request["work"])
    kind, profile = request["kind"], request["profile"]
    if cells is not None:
        cells = tuple(cells)
        allowed = stage1_cells() if kind == "masks" else stage2_cells()
        if (
            not cells
            or not set(cells).issubset(allowed)
            or any(cell.dataset not in datasets_for(kind) for cell in cells)
        ):
            raise ValueError(
                "Internal validation cells must belong to the paper matrix"
            )
        request = {
            key: value for key, value in request.items() if key != "request_hash"
        }
        profile = request["profile"]
        request.update(
            profile=profile, validation_cells=[asdict(cell) for cell in cells]
        )
        request = seal_request(request)
    job = job_directory(work, kind)
    request_path = job / "request.json"
    if request_path.exists() and read_json(request_path) != request:
        original = read_json(request_path)
        if (
            original.get("request_hash") != request["request_hash"]
            or not (job / "complete.json").exists()
        ):
            raise ValueError("Durable execution campaign request drift")
        validate_request(original)
        request = original
    write_json(request_path, request)
    store = StateStore(
        job / "state.json", request["contract_hash"], request["request_hash"]
    )
    store.read()
    prepared = validate_assets(work / "assets", datasets_for(kind))
    if asset_identity(prepared, kind) != request["assets_hash"]:
        raise ValueError("Request asset manifest drift")
    if (job / "complete.json").exists():
        validate_campaign(job, request["request_hash"])
        store.update(status="complete")
        return
    selected = (
        cells
        if cells is not None
        else (
            stage1_cells()
            if kind == "masks"
            else tuple(
                cell
                for cell in stage2_cells(profile)
                if cell.dataset in datasets_for(kind)
            )
        )
    )
    gate = (
        ResourceGate(
            work, store, prepared, minimum_disk=(70 if kind == "masks" else 40) * GIB
        )
        if gate is None
        else gate
    )

    def after_process(child):
        if child.poll() is None:
            raise RuntimeError("Cannot reclaim cache while a child is live")
        if isinstance(gate, ResourceGate):
            gate(minimum_disk=0)
        else:
            gate()

    callbacks = {
        "cwd": repo,
        "before_process": gate,
        "after_process": after_process,
        "on_process": lambda identity: store.update(status="running", child=identity),
    }
    receipts = {}
    try:
        if kind != "masks":
            wait_for_masks(
                lambda: masks_ready(repo, work, selected, prepared),
                job_directory(work, "masks") / "state.json",
                store,
            )
        with gpu_lock(
            work / ".gpu.lock",
            on_wait=lambda: store.update(status="queued", reason="shared GPU lock"),
        ):
            validate_request(request)
            prepared = validate_assets(work / "assets", datasets_for(kind))
            if asset_identity(prepared, kind) != request["assets_hash"]:
                raise ValueError("Queued asset manifest drift")
            if kind != "masks":
                # The predecessor may fail after our initial mask check while
                # another process owns the GPU lock. Revalidate its full receipt
                # before touching masks or starting Original/evaluation/training.
                predecessor = job_directory(work, "masks") / "state.json"
                if predecessor.exists():
                    predecessor_ready(predecessor)
                if not check_mask_dependency(
                    masks_ready(repo, work, selected, prepared), predecessor
                ):
                    raise ValueError("Required masks disappeared while queued")
            if kind == "masks":
                layouts = {}
                groups = {}
                for item in selected:
                    groups.setdefault(
                        (item.dataset, item.selector, item.seed), []
                    ).append(item)
                for group in groups.values():
                    cell = group[0]
                    asset = prepared["datasets"][cell.dataset]
                    if cell.dataset not in layouts:
                        layouts[cell.dataset] = _model_layout(asset["model"])
                    directory, command, digest = score_request(repo, work, cell, asset)
                    targets = [
                        (
                            mask_key(item),
                            mask_directory(work, item),
                            mask_request(repo, work, item, asset)[2],
                        )
                        for item in group
                    ]
                    run_score_search(
                        directory,
                        cell.selector,
                        digest,
                        command,
                        layouts[cell.dataset],
                        targets,
                        score_context=dict(
                            dataset=cell.dataset,
                            seed=cell.seed,
                            pair_stream=asset["stage2"],
                        ),
                        before_materialize=(lambda: gate(minimum_disk=8 * GIB))
                        if isinstance(gate, ResourceGate)
                        else gate,
                        **callbacks,
                    )
                    receipts[str(directory)] = directory / "complete.json"
            else:
                for dataset in sorted({cell.dataset for cell in selected}):
                    asset = prepared["datasets"][dataset]
                    original = work / "original" / dataset
                    commands = eval_commands(
                        repo, dataset, asset, Path(asset["model"]), original / "results"
                    )
                    digest = json_hash(
                        {
                            "commands": commands,
                            "asset": asset,
                            "contract": contract_hash(),
                        }
                    )
                    run_evaluation(original, dataset, digest, commands, **callbacks)
                    receipts[f"original:{dataset}"] = original / "complete.json"
                for cell in selected:
                    asset = prepared["datasets"][cell.dataset]
                    directory = cell_directory(work, cell)
                    maskdir = mask_directory(work, cell)
                    train = stage2_command(repo, cell, asset, directory)
                    evaluations = eval_commands(
                        repo,
                        cell.dataset,
                        asset,
                        checkpoint_directory(cell.dataset, directory),
                        directory / "results",
                    )
                    cell_request = {
                        "cell": asdict(cell),
                        "train": train,
                        "eval": evaluations,
                        "asset": asset,
                        "mask": read_json(maskdir / "manifest.json"),
                        "contract": contract_hash(),
                    }
                    digest = json_hash(cell_request)
                    directory.mkdir(parents=True, exist_ok=True)
                    cell_request_path = directory / "request.json"
                    if (
                        cell_request_path.exists()
                        and read_json(cell_request_path) != cell_request
                    ):
                        raise ValueError("Durable execution cell request drift")
                    write_json(cell_request_path, cell_request)
                    # Only training needs the temporary dense mask. An already
                    # sealed train boundary resumes evaluation directly.
                    if (
                        not (directory / "train.json").exists()
                        and not (directory / "complete.json").exists()
                    ):
                        gate()
                        tensors = restore_mask(maskdir, mask_key(cell), contract_hash())
                        save_active_mask_artifact(
                            directory / "mask",
                            tensors,
                            {
                                "algorithm_version": "sparse_mask_v1",
                                "artifact_kind": "active_mask",
                                "budget_frac": cell.support,
                                "budget_scope": "global",
                                "budget_mode": "exact",
                                "param_filter": "all",
                                "selection_rounding": "round"
                                if cell.selector.startswith("tcus-")
                                else "floor",
                            },
                        )
                        del tensors
                    cell_callbacks = dict(callbacks)
                    if isinstance(gate, ResourceGate):
                        cell_callbacks["before_process"] = lambda: gate(
                            minimum_disk=16 * GIB
                        )
                    from .provenance import training_context

                    run_cell(
                        directory,
                        cell.dataset,
                        digest,
                        train,
                        evaluations,
                        checkpoint=checkpoint_directory(cell.dataset, directory),
                        provenance=training_context(
                            work, cell, asset, maskdir, request_path, cell_request_path
                        ),
                        **cell_callbacks,
                    )
                    receipts[str(directory)] = directory / "complete.json"
            seal_boundary(job / "complete.json", receipts, request["request_hash"])
            validate_campaign(job, request["request_hash"])
            store.update(status="complete", child=None)
    except Exception as error:
        store.update(status="failed", error=str(error), child=None)
        raise


def session_name(request):
    return (
        f"paper-{request['kind']}-{request['profile']}-{request['request_hash'][:10]}"
    )


def submit_campaign(repo, work, kind, profile=None):
    profile = profile or current()["stage1" if kind == "masks" else "stage2"]["budgets"]
    request = preflight(repo, work, kind, profile)
    job = job_directory(work, kind)
    with gpu_lock(work / ".submission.lock", poll=0.1):
        store = StateStore(
            job / "state.json", request["contract_hash"], request["request_hash"]
        )
        previous = store.read()
        if previous["status"] == "complete":
            validate_campaign(job, request["request_hash"])
            return {"status": "complete", "state": str(store.path)}
        if previous["status"] in {"running", "queued", "failed"}:
            raise RuntimeError(
                f"Existing {previous['status']} request requires inspection: {store.path}"
            )
        store.update(status="planned")
        path = job / "request.json"
        write_json(path, request)
        command = [
            sys.executable,
            "-m",
            "src.reproduction.cli",
            "run",
            "--request",
            str(path),
        ]
        identity = submit_screen(
            command,
            repo,
            session_name(request),
            job / "runner.json",
            job / "runner.log",
        )
        state = store.read()
        if state["status"] not in {"queued", "running"}:
            raise RuntimeError(f"Runner did not enter a live state: {state['status']}")
        return {
            "status": state["status"],
            "session": identity["session"],
            "pid": identity["pid"],
            "log": identity["log"],
        }


def run_request(path):
    request = read_json(path)
    validate_request(request)
    name = session_name(request)
    if os.environ.get("STY", "").split(".", 1)[-1] != name or not screen_present(
        screen_listing(), name
    ):
        raise RuntimeError(
            "The runner must execute inside its exact submitted screen session"
        )
    job = path.parent
    log = job / "runner.log"
    with log.open("a", buffering=1) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        handle.write("Runner started; validating request before GPU ownership\n")
        identity = {**process_identity(os.getpid()), "session": name, "log": str(log)}
        store = StateStore(
            job / "state.json", request["contract_hash"], request["request_hash"]
        )
        store.update(
            status="queued",
            runner=identity,
            reason="validating assets and waiting for GPU lock",
        )
        write_json(job / "runner.json", identity)
        try:
            execute_request(request)
        except Exception as error:
            store.update(status="failed", error=str(error))
            raise
