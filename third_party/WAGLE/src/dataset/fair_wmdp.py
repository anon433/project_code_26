"""WAGLE-local reader for the neutral fair WMDP pair artifact.

This module deliberately does not import OpenUnlearning.  It reads the shared
tensor artifact directly and emits WAGLE's existing ``unlearncollector``
sample structure.
"""

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


def _record_consumption(payload: dict[str, Any], index: int) -> None:
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


class FairWMDPUnlearnDataset(Dataset):
    """Return aligned WAGLE forget/retain samples without random pairing."""

    def __init__(
        self,
        pair_stream_path: str,
        expected_pair_stream_sha256: str | None = None,
    ) -> None:
        self.path = Path(pair_stream_path)
        manifest_path = self.path.with_name("manifest.json")
        if not self.path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Fair WMDP pair stream requires both pairs.pt and manifest.json."
            )
        self.manifest = _read_json(manifest_path)
        if self.manifest.get("schema_version") != PAIR_SCHEMA_VERSION:
            raise ValueError("Unsupported fair WMDP pair-stream schema version.")
        self.payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if not isinstance(self.payload, dict):
            raise TypeError("Fair WMDP pair stream payload must be a dictionary.")
        self.pair_count = int(self.manifest.get("pair_count", 0))
        self.sequence_length = int(self.manifest.get("sequence_length", 0))
        self.score_pair_count = int(self.manifest.get("stage1_wagle_score_pairs", 0))
        self.stage2_microbatches = int(self.manifest.get("stage2_microbatches", 0))
        if (
            self.pair_count <= 0
            or self.sequence_length <= 1
            or self.score_pair_count <= 0
            or self.stage2_microbatches <= 0
        ):
            raise ValueError("Fair WMDP pair manifest has invalid dimensions.")
        if (
            self.score_pair_count > self.stage2_microbatches
            or self.stage2_microbatches > self.pair_count
        ):
            raise ValueError("WAGLE fair-pair protocol exceeds the pair stream.")
        for side in ("forget", "retain"):
            values = self.payload.get(side)
            if not isinstance(values, dict):
                raise ValueError(f"Fair WMDP stream is missing {side}.")
            expected = {
                "input_ids": (torch.int32, (self.pair_count, self.sequence_length)),
                "attention_mask": (torch.bool, (self.pair_count, self.sequence_length)),
                "labels": (torch.int32, (self.pair_count, self.sequence_length)),
                "source_indices": (torch.int64, (self.pair_count,)),
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
            padding_policy = self.manifest.get("padding_policy", "none")
            if padding_policy == "none" and not bool(attention.all()):
                raise ValueError("Unpadded fair streams must contain only full chunks.")
            if padding_policy == "right_pad":
                if not bool(attention.any(dim=1).all()):
                    raise ValueError(
                        "Right-padded fair streams cannot contain empty rows."
                    )
                if bool((~attention[:, :-1] & attention[:, 1:]).any()):
                    raise ValueError(
                        "Fair-stream padding must be contiguous on the right."
                    )
            elif padding_policy != "none":
                raise ValueError(
                    f"Unsupported fair-stream padding policy: {padding_policy!r}."
                )
        self.pair_stream_sha256 = pair_stream_sha256(self.payload)
        if self.pair_stream_sha256 != self.manifest.get("pair_stream_sha256"):
            raise RuntimeError("Fair WMDP pair-stream hash does not match manifest.")
        if (
            expected_pair_stream_sha256 is not None
            and expected_pair_stream_sha256 != self.pair_stream_sha256
        ):
            raise RuntimeError(
                "Configured fair WMDP pair hash does not match manifest."
            )

    def __len__(self) -> int:
        return self.stage2_microbatches

    def _example(self, side: str, index: int) -> dict[str, torch.Tensor]:
        values = self.payload[side]
        labels = values["labels"][index].to(dtype=torch.long)
        return {
            "input_ids": values["input_ids"][index].to(dtype=torch.long),
            "attention_mask": values["attention_mask"][index].to(dtype=torch.long),
            "label": labels,
            "refused_label": labels.clone(),
            "question_length": torch.tensor(self.sequence_length, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        if index < 0 or index >= self.stage2_microbatches:
            raise IndexError(index)
        _record_consumption(self.payload, index)
        return {
            "forget": self._example("forget", index),
            "retain": self._example("retain", index),
        }
