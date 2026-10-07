"""Read validated sequential forget/retain pairs for scoring and training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


PAIR_SCHEMA_VERSION = 1
_SIDE_FIELDS = ("input_ids", "attention_mask", "labels", "source_indices")
_MODEL_FIELDS = ("input_ids", "attention_mask", "labels")


def _record_stage2_consumption(payload: dict[str, Any], index: int) -> None:
    trace_path = os.environ.get("FAIR_WMDP_TRACE_PATH")
    if not trace_path:
        return
    if os.environ.get("FAIR_WMDP_SEQUENTIAL") != "1":
        raise RuntimeError("FAIR_WMDP_TRACE_PATH requires FAIR_WMDP_SEQUENTIAL=1.")
    forget_source = int(payload["forget"]["source_indices"][index].item())
    retain_source = int(payload["retain"]["source_indices"][index].item())
    with Path(trace_path).open("a", encoding="utf-8") as handle:
        handle.write(f"{index}\t{forget_source}\t{retain_source}\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _hash_tensor(hasher: "hashlib._Hash", name: str, value: torch.Tensor) -> None:
    value = value.detach().cpu().contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(json.dumps(list(value.shape)).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(value.numpy().tobytes(order="C"))


def pair_stream_sha256(payload: dict[str, Any]) -> str:
    """Hash the schema-defined tensor content, not pickle serialization bytes."""
    hasher = hashlib.sha256()
    hasher.update(f"fair_wmdp_pair_stream_v{PAIR_SCHEMA_VERSION}".encode("ascii"))
    for side in ("forget", "retain"):
        values = payload.get(side)
        if not isinstance(values, dict):
            raise TypeError(f"Pair stream {side!r} payload is missing.")
        for field in _SIDE_FIELDS:
            value = values.get(field)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Pair stream {side}.{field} is not a tensor.")
            _hash_tensor(hasher, f"{side}.{field}", value)
    return hasher.hexdigest()


def _validate_payload(payload: dict[str, Any], manifest: dict[str, Any]) -> int:
    if payload.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise ValueError("Unsupported fair WMDP pair-stream schema version.")
    pair_count = manifest.get("pair_count")
    sequence_length = manifest.get("sequence_length")
    if not isinstance(pair_count, int) or pair_count <= 0:
        raise ValueError("Fair WMDP pair manifest has invalid pair_count.")
    if not isinstance(sequence_length, int) or sequence_length <= 1:
        raise ValueError("Fair WMDP pair manifest has invalid sequence_length.")
    for side in ("forget", "retain"):
        values = payload.get(side)
        if not isinstance(values, dict):
            raise ValueError(f"Fair WMDP pair stream is missing {side}.")
        expected = {
            "input_ids": (torch.int32, (pair_count, sequence_length)),
            "attention_mask": (torch.bool, (pair_count, sequence_length)),
            "labels": (torch.int32, (pair_count, sequence_length)),
            "source_indices": (torch.int64, (pair_count,)),
        }
        for field, (dtype, shape) in expected.items():
            value = values.get(field)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or tuple(value.shape) != shape
            ):
                raise ValueError(
                    f"Invalid fair WMDP tensor {side}.{field}; expected {dtype} {shape}."
                )
        attention = values["attention_mask"]
        padding_policy = manifest.get("padding_policy", "none")
        if padding_policy == "none" and not bool(attention.all()):
            raise ValueError("Unpadded fair streams must contain only full chunks.")
        if padding_policy == "right_pad":
            if not bool(attention.any(dim=1).all()):
                raise ValueError("Right-padded fair streams cannot contain empty rows.")
            if bool((~attention[:, :-1] & attention[:, 1:]).any()):
                raise ValueError("Fair-stream padding must be contiguous on the right.")
        elif padding_policy != "none":
            raise ValueError(
                f"Unsupported fair-stream padding policy: {padding_policy!r}."
            )
    expected_hash = manifest.get("pair_stream_sha256")
    actual_hash = pair_stream_sha256(payload)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise RuntimeError(
            "Fair WMDP pair-stream content hash does not match manifest."
        )
    return pair_count


class FairWMDPPairDataset(Dataset):
    """Full baseline scoring streams or the downstream training stream."""

    def __init__(
        self,
        pair_stream_path: str,
        fair_stage: str,
        expected_pair_stream_sha256: str | None = None,
    ) -> None:
        self.path = Path(pair_stream_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Fair WMDP pair stream is missing: {self.path}")
        manifest_path = self.path.with_name("manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Fair WMDP pair stream is missing its manifest: {manifest_path}"
            )
        self.manifest = _read_json(manifest_path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("Fair WMDP pair stream payload must be a dictionary.")
        self.pair_count = _validate_payload(payload, self.manifest)
        self.pair_stream_sha256 = str(self.manifest["pair_stream_sha256"])
        if (
            expected_pair_stream_sha256 is not None
            and expected_pair_stream_sha256 != self.pair_stream_sha256
        ):
            raise RuntimeError(
                "Configured fair WMDP pair hash does not match the artifact manifest."
            )
        if fair_stage not in {"stage1", "stage2"}:
            raise ValueError("fair_stage must be stage1 or stage2")
        self.fair_stage = fair_stage
        self.payload = payload
        self.stage2_microbatches = int(self.manifest["stage2_microbatches"])
        if not 0 < self.stage2_microbatches <= self.pair_count:
            raise ValueError("Fair WMDP Stage-2 microbatch count exceeds the stream.")
        self.stage1_pairs = int(
            self.manifest.get("stage1_wagle_score_pairs", self.pair_count)
        )
        if not 0 < self.stage1_pairs <= self.pair_count:
            raise ValueError("Fair WMDP Stage-1 coverage exceeds the stream.")

    def __len__(self) -> int:
        if self.fair_stage == "stage1":
            return self.stage1_pairs
        return self.stage2_microbatches

    def _example(self, side: str, index: int) -> dict[str, torch.Tensor]:
        values = self.payload[side]
        return {
            field: values[field][index].to(dtype=torch.long) for field in _MODEL_FIELDS
        }

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        _record_stage2_consumption(self.payload, index)
        return {
            "forget": self._example("forget", index),
            "retain": self._example("retain", index),
        }
