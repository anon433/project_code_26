"""Validation helpers for single-process model-parallel checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from omegaconf import OmegaConf


OFFLOAD_DEVICES = {"cpu", "disk", "meta"}


def _device_label(value) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    if isinstance(value, torch.device):
        if value.type == "cuda":
            return f"cuda:{0 if value.index is None else value.index}"
        return value.type
    text = str(value).lower()
    if text.isdigit():
        return f"cuda:{int(text)}"
    if text == "cuda":
        return "cuda:0"
    return text


def _cuda_indices(labels: set[str]) -> set[int]:
    indices = set()
    for label in labels:
        if label.startswith("cuda:"):
            indices.add(int(label.split(":", 1)[1]))
    return indices


def validate_device_mapped_model(model, requirements) -> dict:
    """Require a real two-GPU map and reject CPU/disk/meta offload.

    Both ``hf_device_map`` and actual parameter placement are checked. This
    prevents a syntactically valid ``balanced`` request from silently fitting
    the entire model on one 96 GiB GPU.
    """
    if OmegaConf.is_config(requirements):
        requirements = OmegaConf.to_container(requirements, resolve=True)
    requirements = dict(requirements or {})
    required = {int(index) for index in requirements.get("required_cuda_devices", [])}
    forbid_offload = bool(requirements.get("forbid_cpu_disk_offload", True))
    layer_prefix = requirements.get("model_layer_prefix")

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, Mapping) or not device_map:
        raise ValueError(
            "Model-parallel validation requires a non-empty model.hf_device_map."
        )
    mapped_labels = {_device_label(value) for value in device_map.values()}
    parameter_labels = {
        _device_label(parameter.device) for parameter in model.parameters()
    }
    mapped_cuda = _cuda_indices(mapped_labels)
    parameter_cuda = _cuda_indices(parameter_labels)
    mapped_layer_cuda = set()
    parameter_layer_cuda = set()
    if layer_prefix is not None:
        if not isinstance(layer_prefix, str) or not layer_prefix:
            raise ValueError("model_layer_prefix must be a non-empty string.")
        mapped_layer_cuda = _cuda_indices(
            {
                _device_label(device)
                for name, device in device_map.items()
                if str(name).startswith(layer_prefix)
            }
        )
        parameter_layer_cuda = _cuda_indices(
            {
                _device_label(parameter.device)
                for name, parameter in model.named_parameters()
                if name.startswith(layer_prefix)
            }
        )

    if forbid_offload:
        mapped_offload = mapped_labels & OFFLOAD_DEVICES
        parameter_offload = parameter_labels & OFFLOAD_DEVICES
        if mapped_offload or parameter_offload:
            raise ValueError(
                "CPU/disk/meta offload is forbidden for this run: "
                f"map={sorted(mapped_offload)}, parameters={sorted(parameter_offload)}."
            )
    if not required.issubset(mapped_cuda):
        raise ValueError(
            "hf_device_map does not use every required GPU: "
            f"required={sorted(required)}, mapped={sorted(mapped_cuda)}."
        )
    if not required.issubset(parameter_cuda):
        raise ValueError(
            "Model parameters do not actually reside on every required GPU: "
            f"required={sorted(required)}, actual={sorted(parameter_cuda)}."
        )
    if layer_prefix is not None and not required.issubset(mapped_layer_cuda):
        raise ValueError(
            "hf_device_map does not place model layers on every required GPU: "
            f"prefix={layer_prefix!r}, required={sorted(required)}, "
            f"mapped={sorted(mapped_layer_cuda)}."
        )
    if layer_prefix is not None and not required.issubset(parameter_layer_cuda):
        raise ValueError(
            "Model layers do not actually reside on every required GPU: "
            f"prefix={layer_prefix!r}, required={sorted(required)}, "
            f"actual={sorted(parameter_layer_cuda)}."
        )
    return {
        "required_cuda_devices": sorted(required),
        "mapped_cuda_devices": sorted(mapped_cuda),
        "parameter_cuda_devices": sorted(parameter_cuda),
        "mapped_model_layer_cuda_devices": sorted(mapped_layer_cuda),
        "parameter_model_layer_cuda_devices": sorted(parameter_layer_cuda),
        "mapped_devices": sorted(mapped_labels),
        "parameter_devices": sorted(parameter_labels),
        "cpu_disk_offload": False,
    }
