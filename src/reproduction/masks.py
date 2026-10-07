"""Portable bit-packed masks, independent of campaign launchers."""

from __future__ import annotations

import math
import json
import hashlib
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .matrix import MaskKey
from .io import json_hash, read_json, sha256_file, write_json


def selected_count(total, key):
    return (
        max(1, round(total * key.support))
        if key.selector.startswith("tcus-")
        else math.floor(total * key.support)
    )


def publish_mask(
    directory: Path,
    key: MaskKey,
    tensors: dict,
    contract_digest: str,
    *,
    provenance=None,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    total = selected = 0
    layout = []
    for name, tensor in tensors.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.bool
            or not tensor.numel()
        ):
            raise ValueError("Mask must contain nonempty boolean tensors")
        count = tensor.numel()
        selected += int(tensor.sum())
        total += count
        layout.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "numel": count,
                "bytes": (count + 7) // 8,
            }
        )
    if not total or selected != selected_count(total, key):
        raise ValueError("Mask selected count violates exact support budget")
    temporary = directory / ".mask.packbits"
    with temporary.open("wb") as handle:
        for tensor in tensors.values():
            handle.write(
                np.packbits(
                    tensor.cpu().numpy().reshape(-1), bitorder="little"
                ).tobytes()
            )
        handle.flush()
        import os

        os.fsync(handle.fileno())
    temporary.replace(directory / "mask.packbits")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "bitpacked_boolean_parameter_mask",
        "key": asdict(key),
        "contract_hash": contract_digest,
        "numel": total,
        "selected_count": selected,
        "layout": layout,
        "sha256": sha256_file(directory / "mask.packbits"),
    }
    if provenance is not None:
        manifest["score_provenance"] = provenance
    manifest["manifest_hash"] = json_hash(manifest)
    write_json(directory / "manifest.json", manifest)
    return validate_mask(directory, key, contract_digest)


def validate_mask(directory: Path, key: MaskKey, contract_digest: str) -> dict:
    manifest = read_json(directory / "manifest.json")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("manifest_hash") != json_hash(unsigned):
        raise ValueError("Mask manifest hash mismatch")
    if (
        manifest.get("key") != asdict(key)
        or manifest.get("contract_hash") != contract_digest
    ):
        raise ValueError("Mask identity/contract mismatch")
    packed = directory / "mask.packbits"
    if sha256_file(packed) != manifest.get("sha256"):
        raise ValueError("Mask checksum mismatch")
    names = set()
    total = selected = length = 0
    with packed.open("rb") as handle:
        for item in manifest["layout"]:
            count = item["numel"]
            if (
                item["name"] in names
                or count <= 0
                or math.prod(item["shape"]) != count
                or item["bytes"] != (count + 7) // 8
            ):
                raise ValueError("Invalid mask tensor layout")
            names.add(item["name"])
            raw = handle.read(item["bytes"])
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
            if len(raw) != item["bytes"] or bits[count:].any():
                raise ValueError("Truncated mask or nonzero padding")
            selected += int(bits[:count].sum())
            total += count
            length += len(raw)
    if (
        not total
        or packed.stat().st_size != length
        or total != manifest["numel"]
        or selected != manifest["selected_count"]
        or selected != selected_count(total, key)
    ):
        raise ValueError("Mask content violates exact support budget")
    return manifest


def validate_layout(directory, key, contract_digest, expected):
    manifest = validate_mask(directory, key, contract_digest)
    observed = {item["name"]: item["shape"] for item in manifest["layout"]}
    if observed != expected:
        raise ValueError("Mask parameter layout differs from target model")
    return manifest


def restore_mask(directory: Path, key: MaskKey, contract_digest: str) -> dict:
    manifest = validate_mask(directory, key, contract_digest)
    tensors = {}
    with (directory / "mask.packbits").open("rb") as handle:
        for item in manifest["layout"]:
            bits = np.unpackbits(
                np.frombuffer(handle.read(item["bytes"]), dtype=np.uint8),
                bitorder="little",
            )
            tensors[item["name"]] = torch.from_numpy(
                bits[: item["numel"]].astype(bool).reshape(item["shape"])
            )
    return tensors


def _score_chunks(scores, chunk_size=8_000_000):
    for tensor in scores.values():
        flat = tensor.reshape(-1)
        for start in range(0, flat.numel(), chunk_size):
            yield flat[start : start + chunk_size]


def _validate_scores(scores):
    if not isinstance(scores, dict) or not scores:
        raise ValueError("Scores must be a nonempty parameter mapping")
    for tensor in scores.values():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or not tensor.numel()
        ):
            raise ValueError("Scores must be nonempty CPU float32 tensors")
    for chunk in _score_chunks(scores):
        if not torch.isfinite(chunk).all():
            raise ValueError("Scores must be finite")


def select_exact(scores, selected_count, *, higher):
    """Exact FP32 global ranking with bounded workspace and coordinate ties.

    Binary search the ordered IEEE-754 key space instead of allocating a
    model-sized argsort or kthvalue workspace. Peak temporary storage is one
    bounded comparison chunk plus the output boolean mapping.
    """
    _validate_scores(scores)
    total = sum(value.numel() for value in scores.values())
    if type(selected_count) is not int or not 0 <= selected_count <= total:
        raise ValueError("Selected count is outside the score universe")
    if selected_count in (0, total):
        return {
            name: torch.full_like(value, selected_count == total, dtype=torch.bool)
            for name, value in scores.items()
        }

    def key(value):
        bits = struct.unpack("I", struct.pack("f", value))[0]
        return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000

    def number(value):
        bits = value ^ 0x80000000 if value & 0x80000000 else ~value & 0xFFFFFFFF
        return struct.unpack("f", struct.pack("I", bits))[0]

    minimum = min(float(value.min()) for value in _score_chunks(scores))
    maximum = max(float(value.max()) for value in _score_chunks(scores))
    low, high = key(minimum), key(maximum)
    rank = selected_count if higher else total - selected_count + 1
    while low < high:
        middle = (low + high + 1) // 2
        candidate = number(middle)
        count = sum(
            int(torch.count_nonzero(value >= candidate))
            for value in _score_chunks(scores)
        )
        if count >= rank:
            low = middle
        else:
            high = middle - 1
    threshold = number(low)
    masks = {
        name: value > threshold if higher else value < threshold
        for name, value in scores.items()
    }
    remaining = selected_count - sum(int(value.sum()) for value in masks.values())
    for name, tensor in scores.items():
        flat, mask = tensor.reshape(-1), masks[name].reshape(-1)
        for start in range(0, flat.numel(), 8_000_000):
            if not remaining:
                break
            ties = torch.nonzero(flat[start : start + 8_000_000] == threshold).reshape(
                -1
            )
            take = min(remaining, ties.numel())
            mask[start + ties[:take]] = True
            remaining -= take
    if remaining:
        raise ValueError("Exact score threshold could not fill the budget")
    return masks


def load_scores(
    raw, selector, expected_layout, *, dataset=None, seed=3, pair_stream=None
):
    """Validate and load native reusable scores without importing launchers."""
    if selector.startswith("tcus-"):
        from .config import load_contract

        manifest_path = raw / "mask_manifest.json"
        manifest = read_json(manifest_path)
        updater = selector.removeprefix("tcus-")
        contract = load_contract()
        tcus = contract["selectors"]["tcus"]
        expected = {
            "algorithm": "TCUS-Trajectory-" + updater.upper(),
            "algorithm_version": "tcus_trajectory_v1",
            "artifact_kind": "active_mask",
            "score_steps": tcus["score_steps"],
            "optimizer_steps": tcus["score_steps"],
            "microbatches": tcus["score_steps"] * 4,
            "retain_penalty": tcus["retain_penalty"],
            "score_formula": "sum(gf*delta - lambda*relu(gr*delta))",
            "budget_frac": 0.1,
            "budget_scope": "global",
            "budget_mode": "exact",
            "param_filter": "all",
            "scores_file": "trajectory_scores.pt",
            "model_mode": "train",
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"TCUS trajectory {key} differs from requested {selector}"
                )
        if dataset is not None:
            training = contract["datasets"][dataset]["stage2"]
            objective = dict(training["recipes"][updater])
            lr = objective.pop("learning_rate")
            objective["retain_loss_type"] = "NLL"
            if updater == "npo":
                objective["reference_model"] = "frozen_initial_model"
            if manifest.get("objective_config") != objective:
                raise ValueError("TCUS objective differs from downstream training")
            args = manifest.get("training_args", {})
            wanted = {
                k: v
                for k, v in training.items()
                if k not in ("recipes", "microbatches")
            }
            wanted.update(learning_rate=lr, seed=seed, data_seed=seed)
            if any(args.get(k) != v for k, v in wanted.items()):
                raise ValueError(
                    "TCUS optimizer/schedule differs from downstream training"
                )
        score_path = raw / "trajectory_scores.pt"
        if sha256_file(score_path) != manifest.get("scores_sha256"):
            raise ValueError("TCUS score checksum mismatch")
        consumed = json.loads((raw / "consumption.json").read_text())
        if len(consumed) != expected["microbatches"]:
            raise ValueError("TCUS consumption length mismatch")
        if pair_stream is not None:
            pairs = torch.load(
                pair_stream, weights_only=True, map_location="cpu", mmap=True
            )
            fingerprints = []
            for row in range(expected["microbatches"]):
                digest = hashlib.sha256()
                for side in ("forget", "retain"):
                    for field in ("input_ids", "attention_mask", "labels"):
                        digest.update(f"{side}/{field}:".encode())
                        digest.update(
                            pairs[side][field][row]
                            .to(torch.int64)
                            .contiguous()
                            .numpy()
                            .tobytes()
                        )
                fingerprints.append(digest.hexdigest())
            if consumed != fingerprints:
                raise ValueError("TCUS did not consume the original downstream prefix")
        scores = torch.load(
            score_path, weights_only=True, map_location="cpu", mmap=True
        )
        higher = True
    else:
        directory = raw / "snip_advanced" if selector == "wagle" else raw
        manifest_path = directory / "selector_logits.manifest.json"
        manifest = read_json(manifest_path)
        higher = selector == "gec"
        if (
            manifest.get("schema_version") != 1
            or manifest.get("artifact_kind") != "full_parameter_selector_logits"
            or manifest.get("dtype") != "float32"
            or manifest.get("selection_order")
            != ("higher_is_selected" if higher else "lower_is_selected")
            or manifest.get("selector")
            != ("normdiff" if higher else "wagle_snip_advanced")
        ):
            raise ValueError("Native score identity/dtype/order mismatch")
        filename = manifest["logits_file"]
        if Path(filename).name != filename:
            raise ValueError("Unsafe score filename")
        score_path = directory / filename
        if sha256_file(score_path) != manifest["logits_sha256"]:
            raise ValueError("Score checksum mismatch")
        flat = torch.load(score_path, weights_only=True, map_location="cpu", mmap=True)
        if (
            not isinstance(flat, torch.Tensor)
            or flat.ndim != 1
            or flat.numel() != manifest["numel"]
        ):
            raise ValueError("Score vector dimensions changed")
        scores, offset = {}, 0
        for item in manifest["layout"]:
            count = item["numel"]
            if item["name"] in scores or count != math.prod(item["shape"]):
                raise ValueError("Invalid score layout")
            scores[item["name"]] = flat[offset : offset + count].reshape(item["shape"])
            offset += count
        if offset != flat.numel():
            raise ValueError("Score layout does not cover the vector")
    _validate_scores(scores)
    if {name: list(value.shape) for name, value in scores.items()} != expected_layout:
        raise ValueError("Score parameter layout differs from target model")
    total = sum(value.numel() for value in scores.values())
    if selector.startswith("tcus-") and manifest.get("maskable_params") != total:
        raise ValueError("TCUS score universe differs from native manifest")
    provenance = {
        "score_sha256": sha256_file(score_path),
        "manifest_sha256": sha256_file(manifest_path),
        "higher_is_selected": higher,
        "numel": total,
    }
    return (
        scores,
        provenance,
        {
            "scores": score_path,
            "manifest": manifest_path,
            **(
                {"consumption": raw / "consumption.json"}
                if selector.startswith("tcus-")
                else {}
            ),
        },
    )
