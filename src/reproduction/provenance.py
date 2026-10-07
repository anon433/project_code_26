"""Runner evidence adapter for the strict reproduction gate."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import yaml

from .config import contract_hash, load_contract
from .io import json_hash, read_json, sha256_file, write_json
from .traces import consumption_hash

REQUIRED_REPO = Path(__file__).resolve().parents[2]
INPUT_ROLES = {
    "mask_manifest",
    "mask",
    "stage1_manifest",
    "stage1",
    "stage2_manifest",
    "stage2",
}


def validate_consumption(path, rows, expected_sha256):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    if len(lines) not in (rows, rows + 1):
        raise ValueError("Observed consumption row count differs from effective plan")
    for index, line in enumerate(lines[:rows]):
        values = line.decode().rstrip("\n").split("\t")
        if len(values) != 3 or int(values[0]) != index:
            raise ValueError("Observed consumption order differs from sequential plan")
    digest = hashlib.sha256(b"".join(lines[:rows])).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Observed effective consumption checksum differs from plan")
    return {
        "effective_rows": rows,
        "observed_rows": len(lines),
        "effective_sha256": digest,
        "unused_prefetch_rows": len(lines) - rows,
    }


def _prefix_consumption_hash(manifest, rows):
    """Derive observed-row identity without weakening the full stream digest."""
    return consumption_hash(
        {
            key: manifest["plan"][key][:rows]
            for key in ("forget_indices", "retain_indices")
        }
    )


def training_context(work, cell, asset, maskdir, request_path, cell_request_path):
    mask = read_json(maskdir / "manifest.json")
    trace = read_json(Path(asset["stage2"]).parent / "manifest.json")
    revisions = {
        name: snap["revision"]
        for name, snap in asset["snapshots"].items()
        if name not in ("eval_logs", "eval_mmlu", "eval_wmdp")
    }
    if cell.dataset == "wmdpall":
        revisions.pop("tokenizer", None)
    request = read_json(request_path)
    rows = load_contract()["datasets"][cell.dataset]["stage2"]["microbatches"]
    consumption_digest = _prefix_consumption_hash(trace, rows)
    return {
        "contract_hash": contract_hash(),
        "execution_request": request,
        "cell": asdict(cell),
        "expected_rows": rows,
        "expected_consumption_sha256": consumption_digest,
        "invariants": {
            "revisions": revisions,
            "traces": {
                **{
                    f"{stage}_sha256": asset[f"{stage}_sha256"]
                    for stage in ("stage1", "stage2")
                },
                "stage2_effective_consumption_sha256": consumption_digest,
            },
            "mask_sha256": mask["sha256"],
            "selected_count": mask["selected_count"],
            "eligible_count": mask["numel"],
            "recipe": dict(cell.recipe),
        },
        "input_sources": [
            {**source_reference(work, p), "role": role}
            for role, p in {
                "mask_manifest": maskdir / "manifest.json",
                "mask": maskdir / "mask.packbits",
                "stage1_manifest": Path(asset["stage1"]).parent / "manifest.json",
                "stage1": Path(asset["stage1"]),
                "stage2_manifest": Path(asset["stage2"]).parent / "manifest.json",
                "stage2": Path(asset["stage2"]),
            }.items()
        ],
        "execution": {
            "request": source_reference(work, request_path),
            "cell_request": source_reference(work, cell_request_path),
            "request_hash": request["request_hash"],
            "source_hash": request["source_hash"],
            "cell_request_hash": json_hash(read_json(cell_request_path)),
        },
    }


def validate_input_sources(work, record):
    from .masks import validate_mask
    from .matrix import Stage2Cell, mask_key
    from .traces import validate_stream

    sources = record.get("input_sources", [])
    if (
        len(sources) != len(INPUT_ROLES)
        or {item.get("role") for item in sources} != INPUT_ROLES
    ):
        raise ValueError("Exact input_sources membership is required")
    paths = {item["role"]: verify_source(work, item) for item in sources}
    if len(set(paths.values())) != len(INPUT_ROLES):
        raise ValueError("Duplicate input_sources paths")
    cell = Stage2Cell(**record["cell"])
    inv = record["invariants"]
    if not isinstance(inv.get("traces"), dict) or not {
        "stage1_sha256",
        "stage2_sha256",
    }.issubset(inv["traces"]):
        raise ValueError("Missing trace invariant identity")
    if (
        paths["mask_manifest"].name != "manifest.json"
        or paths["mask"] != paths["mask_manifest"].parent / "mask.packbits"
    ):
        raise ValueError("Mask source paths do not identify the same artifact")
    mask = validate_mask(paths["mask"].parent, mask_key(cell), contract_hash())
    if (
        mask["sha256"] != inv["mask_sha256"]
        or mask["selected_count"] != inv["selected_count"]
        or mask["numel"] != inv["eligible_count"]
    ):
        raise ValueError("Mask input content differs from provenance")
    for stage in ("stage1", "stage2"):
        path = paths[stage + "_manifest"]
        manifest = read_json(path)
        if path.name != "manifest.json" or paths[stage] != path.parent / "pairs.pt":
            raise ValueError("Trace source paths do not identify the same artifact")
        revisions = dict(manifest["revisions"])
        if cell.dataset == "wmdpall":
            revisions.pop("tokenizer", None)
        if (
            revisions != inv["revisions"]
            or manifest["stage"] != stage
            or manifest["dataset"].lower() != cell.dataset
        ):
            raise ValueError("Trace source identity/revisions mismatch")
        validate_stream(
            path.parent,
            expected_hash=inv["traces"][stage + "_sha256"],
            revisions=manifest["revisions"],
            contract_digest=contract_hash(),
        )
        if stage == "stage2":
            rows = record["expected_rows"]
            if (
                type(rows) is not int
                or rows
                != load_contract()["datasets"][cell.dataset]["stage2"]["microbatches"]
                or not 0 < rows <= manifest["pair_count"]
            ):
                raise ValueError("Effective trace source differs from provenance")
            digest = _prefix_consumption_hash(manifest, rows)
            if digest != record["expected_consumption_sha256"] or digest != inv[
                "traces"
            ].get("stage2_effective_consumption_sha256"):
                raise ValueError(
                    "Effective trace consumption invariant differs from source"
                )
    return paths


def validate_execution(work, directory, record, *, complete):
    from .orchestrator import validate_request, validate_campaign, datasets_for
    from .matrix import stage2_cells

    execution = record.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Missing execution-time campaign identity")
    request = read_source(work, execution["request"])
    cell_request = read_source(work, execution["cell_request"])
    validate_request(request)
    if read_json(directory / "execution-request.json") != request:
        raise ValueError("Captured execution request differs from campaign authority")
    if (
        Path(request["repo"]).resolve() != REQUIRED_REPO.resolve()
        or Path(request["work"]).resolve() != Path(work).resolve()
    ):
        raise ValueError("Execution workflow/work root differs from required identity")
    if (
        execution["source_hash"] != request["source_hash"]
        or execution["request_hash"] != request["request_hash"]
    ):
        raise ValueError("Execution-time source/request identity mismatch")
    if (
        json_hash(cell_request) != execution["cell_request_hash"]
        or cell_request["cell"] != record["cell"]
        or cell_request["contract"] != contract_hash()
    ):
        raise ValueError("Execution-time cell request mismatch")
    if verify_source(work, execution["cell_request"]) != directory / "request.json":
        raise ValueError("Cell request path mismatch")
    allowed = (
        request.get("validation_cells")
        if "validation_cells" in request
        else [
            asdict(c)
            for c in stage2_cells(request["profile"])
            if c.dataset in datasets_for(request["kind"])
        ]
    )
    if not allowed or record["cell"] not in allowed:
        raise ValueError("Cell not authorized by execution campaign request")
    if "commands" in record["invariants"] and (
        record["invariants"]["commands"] != cell_request["train"]
        or record["evaluation_commands"] != cell_request["eval"]
    ):
        raise ValueError("Commands differ from execution-time request")
    if complete:
        campaign = verify_source(work, execution["request"]).parent
        validate_campaign(campaign, request["request_hash"])
        receipt = read_json(campaign / "complete.json")
        if str((directory / "complete.json").absolute()) not in {
            item["path"] for item in receipt["files"].values()
        }:
            raise ValueError("Campaign receipt does not bind the completed cell")
    return request, cell_request


def publish_provenance(directory, context, train_command, evaluation_commands):
    if context.get("contract_hash") != contract_hash():
        raise ValueError("Training provenance contract drift")
    request_path = Path(context["execution"]["request"]["path"])
    # The campaign request is a portable reference; its absolute work root is in
    # the execution request that the caller has durably placed next to the cell.
    execution_request = read_json(directory / "execution-request.json")
    work = Path(execution_request["work"])
    request, cell_request = validate_execution(work, directory, context, complete=False)
    if (
        request != execution_request
        or request_path.is_absolute()
        or cell_request["train"] != train_command
        or cell_request["eval"] != evaluation_commands
    ):
        raise ValueError("Executed command/request differs from captured identity")
    validate_input_sources(work, context)
    consumption = validate_consumption(
        directory / "consumed.tsv",
        context["expected_rows"],
        context["expected_consumption_sha256"],
    )
    record = {
        **context,
        "invariants": {
            **context["invariants"],
            "commands": train_command,
            "traces": {
                **context["invariants"]["traces"],
                "stage2_effective_consumption_sha256": consumption["effective_sha256"],
            },
        },
        "evaluation_commands": evaluation_commands,
        "consumption": consumption,
    }
    record.pop("execution_request", None)
    record["provenance_hash"] = json_hash(record)
    write_json(directory / "provenance.json", record)
    return record


def relative_file(root, name):
    root, path = Path(root).resolve(), Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected safe relative path: {name}")
    result = root / path
    if not result.resolve().is_relative_to(root) or any(
        (root / Path(*path.parts[:i])).is_symlink()
        for i in range(1, len(path.parts) + 1)
    ):
        raise ValueError(f"Unsafe relative path: {name}")
    return result


def source_reference(root, path):
    name = Path(path).absolute().relative_to(Path(root).absolute()).as_posix()
    file = relative_file(root, name)
    return {"path": name, "sha256": sha256_file(file)}


def verify_source(root, reference):
    path = relative_file(root, reference["path"])
    if not path.is_file() or sha256_file(path) != reference["sha256"]:
        raise ValueError(f"Source checksum mismatch: {reference['path']}")
    return path


def read_source(root, reference):
    path = verify_source(root, reference)
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text())
    return read_json(path)
