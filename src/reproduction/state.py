"""Atomic lifecycle records and strict validation of durable boundaries."""

from __future__ import annotations

import math
from pathlib import Path

from .config import load_contract
from .io import json_hash, read_json, sha256_file, write_json

STATUSES = frozenset({"planned", "queued", "running", "complete", "failed"})


def seal_boundary(path: Path, files: dict[str, Path], request_digest: str) -> dict:
    if not files:
        raise ValueError("Cannot seal an empty boundary")
    record = {
        "request_hash": request_digest,
        "files": {
            name: {"path": str(file.absolute()), "sha256": sha256_file(file)}
            for name, file in files.items()
        },
    }
    record["receipt_hash"] = json_hash(record)
    write_json(path, record)
    return record


def validate_boundary(path: Path, request_digest: str) -> dict:
    record = read_json(path)
    if record.get("request_hash") != request_digest:
        raise ValueError("Boundary config drift")
    unsigned = {k: v for k, v in record.items() if k != "receipt_hash"}
    if record.get("receipt_hash") != json_hash(unsigned) or not record.get("files"):
        raise ValueError("Boundary receipt content mismatch")
    for item in record["files"].values():
        file = Path(item["path"])
        if not file.is_file() or sha256_file(file) != item["sha256"]:
            raise ValueError(f"Boundary content changed: {file}")
    return record


class StateStore:
    def __init__(self, path: Path, contract_digest: str, request_digest: str):
        self.path = Path(path)
        self.contract_digest = contract_digest
        self.request_digest = request_digest

    def read(self) -> dict:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "status": "planned",
                "contract_hash": self.contract_digest,
                "request_hash": self.request_digest,
            }
        record = read_json(self.path)
        if record.get("contract_hash") != self.contract_digest:
            raise ValueError(f"State contract drift: {self.path}")
        if record.get("request_hash") != self.request_digest:
            raise ValueError(f"State request/config drift: {self.path}")
        if record.get("status") not in STATUSES:
            raise ValueError("Unknown state status")
        return record

    def update(self, **changes) -> dict:
        if changes.get("status", "planned") not in STATUSES:
            raise ValueError("Unknown state status")
        record = self.read()
        record.update(changes)
        write_json(self.path, record)
        return record


def validate_results(directory: Path, dataset: str) -> dict[str, float]:
    keys = load_contract()["datasets"][dataset]["evaluation"]["metric_keys"]
    if dataset.startswith("muse-"):
        payload = read_json(directory / "MUSE_SUMMARY.json")
    else:
        payload = {}
        for key in keys.values():
            task = key.split("/")[0]
            summary = read_json(directory / task / "LMEval_SUMMARY.json")
            payload.update(summary)
    metrics = {}
    for key in keys.values():
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Missing/non-finite result metric {key}: {directory}")
        metrics[key] = float(value)
    return metrics


def validate_checkpoint(directory: Path) -> dict[str, str]:
    """Validate safetensors headers, all shards, finite tensor data, and hashes."""
    from safetensors import safe_open
    import torch

    try:
        config = read_json(directory / "config.json")
        if not config.get("model_type"):
            raise ValueError("Missing model_type")
        index_path = directory / "model.safetensors.index.json"
        if index_path.exists():
            mapping = read_json(index_path)["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Empty or invalid weight shard index")
            if any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in mapping.items()
            ):
                raise ValueError("Invalid weight shard mapping")
            names = set(mapping.values())
        else:
            mapping = None
            names = {"model.safetensors"}
        observed = set()
        hashes = {"config.json": sha256_file(directory / "config.json")}
        for name in sorted(names):
            path = directory / name
            if path.parent.resolve() != directory.resolve():
                raise ValueError("Unsafe shard path")
            with safe_open(path, framework="pt", device="cpu") as handle:
                if not handle.keys():
                    raise ValueError("Empty weights")
                for key in handle.keys():
                    tensor = handle.get_tensor(key).reshape(-1)
                    if not tensor.numel():
                        raise ValueError("Empty weight tensor")
                    for start in range(0, tensor.numel(), 1 << 20):
                        if not torch.isfinite(tensor[start : start + (1 << 20)]).all():
                            raise ValueError("Non-finite weights")
                    if mapping is not None and mapping.get(key) != name:
                        raise ValueError("Shard index mismatch")
                    observed.add(key)
            hashes[name] = sha256_file(path)
        if mapping is not None:
            if set(mapping) != observed:
                raise ValueError("Missing indexed weights")
            hashes[index_path.name] = sha256_file(index_path)
        return hashes
    except Exception as error:
        raise ValueError(f"Invalid checkpoint {directory}: {error}") from error
