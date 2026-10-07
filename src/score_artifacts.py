"""Compact, visualization-ready artifacts for full-parameter selector scores."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_selector_logits(
    logits: torch.Tensor,
    parameter_layout: Sequence[tuple[str, torch.Size, int]],
    output_dir: str | Path,
    *,
    selection_order: str,
    selector: str,
    storage_dtype: str = "float16",
) -> dict[str, object]:
    """Persist one flat score vector plus its ordered parameter layout."""
    if selection_order not in {"higher_is_selected", "lower_is_selected"}:
        raise ValueError(f"Unsupported selection order: {selection_order}")
    if logits.ndim != 1 or not logits.dtype.is_floating_point:
        raise TypeError("Selector logits must be a one-dimensional floating tensor.")
    storage_types = {
        "float16": (torch.float16, "fp16"),
        "bfloat16": (torch.bfloat16, "bf16"),
        "float32": (torch.float32, "fp32"),
    }
    if storage_dtype not in storage_types:
        raise ValueError(f"Unsupported selector-logit storage dtype: {storage_dtype}")
    # A single isfinite() over a 7B vector creates a multi-gigabyte temporary.
    # Validate in bounded chunks so persistence itself cannot trigger cgroup OOM.
    finite_chunk_size = 16 * 1024 * 1024
    for offset in range(0, logits.numel(), finite_chunk_size):
        if not bool(torch.isfinite(logits[offset : offset + finite_chunk_size]).all()):
            raise FloatingPointError("Selector logits contain NaN or Inf.")
    layout_count = sum(int(numel) for _, _, numel in parameter_layout)
    if layout_count != logits.numel():
        raise ValueError(
            f"Selector logit layout has {layout_count} entries for {logits.numel()} values."
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch_dtype, filename_dtype = storage_types[storage_dtype]
    logits_path = destination / f"selector_logits.{filename_dtype}.pt"
    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
        temporary_logits = Path(handle.name)
    try:
        torch.save(logits.detach().cpu().contiguous().to(torch_dtype), temporary_logits)
        os.replace(temporary_logits, logits_path)
    finally:
        temporary_logits.unlink(missing_ok=True)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "full_parameter_selector_logits",
        "selector": selector,
        "dtype": storage_dtype,
        "numel": logits.numel(),
        "selection_order": selection_order,
        "logits_file": logits_path.name,
        "logits_sha256": _sha256(logits_path),
        "layout": [
            {
                "name": name,
                "shape": [int(value) for value in shape],
                "numel": int(numel),
            }
            for name, shape, numel in parameter_layout
        ],
    }
    manifest_path = destination / "selector_logits.manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination, delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_manifest = Path(handle.name)
    os.replace(temporary_manifest, manifest_path)
    return manifest
