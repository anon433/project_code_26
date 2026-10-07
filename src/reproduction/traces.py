"""Exact paper trace plans and immutable, portable pair-stream artifacts.

The RNG and padding rules match the original paper campaigns. WMDP pools
must first receive the official dataset-seed-1000 forget permutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from .io import read_json, sha256_file, write_json


def wmdp_plan(forget_count: int, retain_count: int, seed: int = 0) -> dict:
    if forget_count <= 0 or retain_count <= 0:
        raise ValueError("Trace pools must be nonempty")
    return {
        "seed": seed,
        "pair_count": forget_count,
        "forget_chunks": forget_count,
        "retain_chunks": retain_count,
        "forget_indices": torch.randperm(
            forget_count, generator=torch.Generator().manual_seed(seed)
        ).tolist(),
        "retain_indices": torch.randint(
            retain_count,
            (forget_count,),
            generator=torch.Generator().manual_seed(seed + 10000),
        ).tolist(),
    }


def muse_plan(
    forget_count: int, retain_count: int, stage: str, *, rows: int = 2000
) -> dict:
    if min(forget_count, retain_count, rows) <= 0 or stage not in {"stage1", "stage2"}:
        raise ValueError("Invalid MUSE pool counts or stage")
    count = max(forget_count, retain_count) if stage == "stage1" else rows
    return {
        "seed": 3,
        "pair_count": count,
        "forget_chunks": forget_count,
        "retain_chunks": retain_count,
        "forget_indices": [i % forget_count for i in range(count)],
        "retain_indices": [i % retain_count for i in range(count)]
        if stage == "stage1"
        else torch.randint(
            retain_count, (count,), generator=torch.Generator().manual_seed(3)
        ).tolist(),
    }


def build_payload(
    forget, retain, plan: dict, *, pad_token_id: int | None = None
) -> dict:
    """Stack WMDP fixed rows, or right-pad MUSE rows with ignored labels."""
    count = plan["pair_count"]
    selected = {}
    lengths = set()
    for side, samples in (("forget", forget), ("retain", retain)):
        indices = plan[f"{side}_indices"]
        if len(indices) != count or count <= 0:
            raise ValueError("Invalid plan length")
        selected[side] = []
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(samples):
                raise ValueError("Source index outside pool")
            sample = samples[index]
            tokens = sample["input_ids"]
            labels = sample.get("labels", sample.get("label"))
            if (
                not isinstance(tokens, torch.Tensor)
                or not isinstance(labels, torch.Tensor)
                or tokens.ndim != 1
                or labels.shape != tokens.shape
                or not tokens.numel()
            ):
                raise ValueError("Expected aligned nonempty 1-D tokens and labels")
            selected[side].append((tokens, labels, sample.get("attention_mask")))
            lengths.add(tokens.numel())
    if pad_token_id is None and len(lengths) != 1:
        raise ValueError("WMDP rows must have fixed sequence length")
    width = max(lengths)
    payload = {"schema_version": 1}
    for side in ("forget", "retain"):
        tokens = torch.full((count, width), pad_token_id or 0, dtype=torch.int32)
        labels = torch.full((count, width), -100, dtype=torch.int32)
        attention = torch.zeros((count, width), dtype=torch.bool)
        for row, (ids, targets, source_attention) in enumerate(selected[side]):
            length = ids.numel()
            tokens[row, :length] = ids.to(torch.int32)
            labels[row, :length] = targets.to(torch.int32)
            if pad_token_id is None:
                if (
                    not isinstance(source_attention, torch.Tensor)
                    or source_attention.shape != ids.shape
                ):
                    raise ValueError("WMDP attention shape mismatch")
                attention[row] = source_attention.to(torch.bool)
            else:
                attention[row, :length] = True
        payload[side] = {
            "input_ids": tokens,
            "attention_mask": attention,
            "labels": labels,
            "source_indices": torch.tensor(plan[f"{side}_indices"], dtype=torch.int64),
        }
    return payload


def pair_stream_hash(payload: dict) -> str:
    """Schema v1 content digest used by both existing training readers."""
    digest = hashlib.sha256(b"fair_wmdp_pair_stream_v1")
    for side in ("forget", "retain"):
        for field in ("input_ids", "attention_mask", "labels", "source_indices"):
            value = payload[side][field].detach().cpu().contiguous()
            digest.update(f"{side}.{field}".encode() + b"\0")
            digest.update(str(value.dtype).encode() + b"\0")
            digest.update(json.dumps(list(value.shape)).encode() + b"\0")
            array = value.numpy().reshape(-1)
            for start in range(0, len(array), 1 << 20):
                digest.update(array[start : start + (1 << 20)].tobytes())
    return digest.hexdigest()


def consumption_hash(plan: dict) -> str:
    digest = hashlib.sha256()
    for row, (forget, retain) in enumerate(
        zip(plan["forget_indices"], plan["retain_indices"], strict=True)
    ):
        digest.update(f"{row}\t{forget}\t{retain}\n".encode())
    return digest.hexdigest()


def publish_stream(
    directory: Path,
    payload: dict,
    plan: dict,
    *,
    dataset: str,
    stage: str,
    revisions: dict,
    contract_digest: str,
    expected_hash: str | None = None,
) -> dict:
    if stage not in {"stage1", "stage2"}:
        raise ValueError("Unknown stream stage")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing to overwrite immutable stream: {directory}")
    digest = pair_stream_hash(payload)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(
            f"Trace hash differs from paper contract for {dataset}/{stage}: {digest}"
        )
    count, width = payload["forget"]["input_ids"].shape
    manifest = {
        "schema_version": 1,
        "artifact_kind": "fair_muse_pair_stream"
        if dataset.startswith("muse-")
        else "paired_wmdpall_block_a_trace",
        "benchmark": "muse" if dataset.startswith("muse-") else "wmdp",
        "domain": dataset.split("-")[-1].capitalize(),
        "dataset": "WMDPALL" if dataset == "wmdpall" else dataset,
        "stage": stage,
        "contract_hash": contract_digest,
        "revisions": revisions,
        "pair_count": count,
        "sequence_length": width,
        "padding_policy": "right_pad",
        "pair_stream_sha256": digest,
        "plan": plan,
        "effective_consumption_trace_sha256": consumption_hash(plan),
        "stage1_score_steps": count // 2,
        "stage1_score_rows": 2 * (count // 2),
        "stage1_wagle_score_pairs": count,
        "stage1_wagle_forget_batches": plan["forget_chunks"],
        "stage1_wagle_retain_batches": plan["retain_chunks"]
        if dataset.startswith("muse-")
        else count,
        "stage2_microbatches": count,
        "stage2_gradient_accumulation_steps": 4 if stage == "stage2" else 1,
    }
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pairs-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory / "pairs.pt")
    finally:
        Path(temporary).unlink(missing_ok=True)
    manifest["pair_file_sha256"] = sha256_file(directory / "pairs.pt")
    write_json(directory / "manifest.json", manifest)
    validate_stream(
        directory,
        expected_hash=digest,
        revisions=revisions,
        contract_digest=contract_digest,
    )
    return manifest


def validate_stream(
    directory: Path, *, expected_hash: str, revisions: dict, contract_digest: str
) -> dict:
    manifest = read_json(directory / "manifest.json")
    if (
        manifest.get("revisions") != revisions
        or manifest.get("contract_hash") != contract_digest
    ):
        raise ValueError("Trace revision/contract drift")
    path = directory / "pairs.pt"
    if manifest.get("pair_file_sha256") != sha256_file(path):
        raise ValueError("Pair file hash mismatch")
    payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Invalid trace schema")
    count, width = manifest["pair_count"], manifest["sequence_length"]
    expected_dimensions = {
        "stage1_score_steps": count // 2,
        "stage1_score_rows": 2 * (count // 2),
        "stage1_wagle_score_pairs": count,
        "stage2_microbatches": count,
        "stage1_wagle_forget_batches": manifest["plan"]["forget_chunks"],
        "stage1_wagle_retain_batches": manifest["plan"]["retain_chunks"]
        if manifest["dataset"].startswith("muse-")
        else count,
        "stage2_gradient_accumulation_steps": 4 if manifest["stage"] == "stage2" else 1,
    }
    if (
        type(count) is not int
        or type(width) is not int
        or count <= 0
        or width < 2
        or any(
            type(manifest.get(key)) is not int or manifest[key] != value
            for key, value in expected_dimensions.items()
        )
    ):
        raise ValueError("Invalid trace reader dimensions")
    for side in ("forget", "retain"):
        for field, dtype, shape in (
            ("input_ids", torch.int32, (count, width)),
            ("labels", torch.int32, (count, width)),
            ("attention_mask", torch.bool, (count, width)),
            ("source_indices", torch.int64, (count,)),
        ):
            tensor = payload[side][field]
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError("Trace tensor shape/dtype mismatch")
        indices = payload[side]["source_indices"].tolist()
        if indices != manifest["plan"][f"{side}_indices"]:
            raise ValueError("Trace source indices differ from plan")
        attention = payload[side]["attention_mask"]
        if (
            not attention.any(dim=1).all()
            or ((~attention[:, :-1]) & attention[:, 1:]).any()
        ):
            raise ValueError("Trace has invalid right padding")
    digest = pair_stream_hash(payload)
    if digest != expected_hash or digest != manifest["pair_stream_sha256"]:
        raise ValueError("Trace content hash mismatch")
    if manifest["effective_consumption_trace_sha256"] != consumption_hash(
        manifest["plan"]
    ):
        raise ValueError("Trace consumption plan mismatch")
    return manifest
