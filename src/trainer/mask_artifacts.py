"""Content-verified Boolean masks for fixed-support unlearning."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch

MASK_FILENAME = "active_mask.pt"
MANIFEST_FILENAME = "mask_manifest.json"
VERSIONS = {"tcus_trajectory_v1", "sparse_mask_v1"}


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(mask, manifest):
    if manifest.get("algorithm_version") not in VERSIONS:
        raise ValueError("Unsupported active-mask version")
    if (
        manifest.get("artifact_kind") != "active_mask"
        or manifest.get("budget_scope") != "global"
        or manifest.get("budget_mode", "exact") != "exact"
        or manifest.get("param_filter") != "all"
    ):
        raise ValueError("Masks require exact global selection over all parameters")
    fraction = manifest.get("budget_frac")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("Invalid mask budget")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Invalid mask budget")
    if not isinstance(mask, dict) or not mask:
        raise ValueError("Mask must be a nonempty mapping")
    for name, value in mask.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Mask entries must map parameter names to tensors")
        if value.dtype != torch.bool or not value.numel():
            raise TypeError("Masks must contain nonempty Boolean tensors")
    total = sum(value.numel() for value in mask.values())
    selected = sum(int(value.sum()) for value in mask.values())
    rounding = manifest.get("selection_rounding", "round")
    if rounding not in {"round", "floor"}:
        raise ValueError("Unknown mask count rounding")
    expected = (
        round(total * fraction) if rounding == "round" else math.floor(total * fraction)
    )
    if rounding == "round":
        expected = max(1, expected)
    if selected != expected or selected == 0:
        raise ValueError(
            f"Mask selected count {selected} differs from budget {expected}"
        )
    return total, selected


def save_active_mask_artifact(output_dir, active_mask, manifest):
    payload = dict(manifest)
    total, selected = _validate(active_mask, payload)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / MASK_FILENAME
    temporary = output / f".{MASK_FILENAME}.{os.getpid()}"
    torch.save(
        {name: value.detach().cpu() for name, value in active_mask.items()}, temporary
    )
    os.replace(temporary, path)
    payload.update(
        mask_file=MASK_FILENAME,
        mask_sha256=_checksum(path),
        selected_params=selected,
        maskable_params=total,
        selected_frac=selected / total,
    )
    temporary = output / f".{MANIFEST_FILENAME}.{os.getpid()}"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / MANIFEST_FILENAME)
    return path


def load_active_mask_artifact(mask_path, *, expected_version, mmap=False):
    path = Path(mask_path)
    if path.name != MASK_FILENAME:
        raise ValueError(f"Expected {MASK_FILENAME}")
    manifest = json.loads((path.parent / MANIFEST_FILENAME).read_text())
    if manifest.get("algorithm_version") != expected_version:
        raise ValueError("Active-mask version mismatch")
    if manifest.get("mask_file") != MASK_FILENAME or manifest.get(
        "mask_sha256"
    ) != _checksum(path):
        raise ValueError("Mask content checksum mismatch")
    mask = torch.load(path, weights_only=True, map_location="cpu", mmap=bool(mmap))
    total, selected = _validate(mask, manifest)
    if (
        manifest.get("maskable_params") != total
        or manifest.get("selected_params") != selected
    ):
        raise ValueError("Manifest selected counts disagree with mask content")
    return mask, manifest


def apply_active_gradient_mask(model, mask_path, *, expected_version):
    mask, manifest = load_active_mask_artifact(
        mask_path, expected_version=expected_version
    )
    base = model.module if hasattr(model, "module") else model
    parameters = dict(base.named_parameters())
    if set(parameters) != set(mask):
        raise ValueError("Mask parameter coverage differs from model")
    for name, parameter in parameters.items():
        if parameter.shape != mask[name].shape:
            raise ValueError(f"Mask shape mismatch: {name}")
    if getattr(base, "_active_gradient_mask_applied", False):
        raise ValueError("Model already has an active gradient mask")
    for name, parameter in parameters.items():
        selected = mask[name]
        parameter.requires_grad_(bool(selected.any()))
        if parameter.requires_grad:
            device_mask = selected.to(device=parameter.device)
            parameter.register_hook(
                lambda gradient, active=device_mask: gradient
                * active.to(device=gradient.device, dtype=gradient.dtype)
            )
    base._active_gradient_mask_applied = True
    return manifest
