"""Loading and hashing for the immutable paper reproduction contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "configs/reproduction/paper.yaml"
)

_ACTIVE = None


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def load_contract(path=None) -> Mapping[str, Any]:
    """Return the resolved numerical protocol, excluding run selection filters."""
    if path is None and _ACTIVE is not None:
        return _ACTIVE
    payload = yaml.safe_load(Path(path or DEFAULT_CONTRACT_PATH).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction protocol")

    def validate_pins(node, location="protocol"):
        if isinstance(node, dict):
            for key, value in node.items():
                name = f"{location}.{key}"
                if key == "revision" and (
                    not isinstance(value, str)
                    or not re.fullmatch(r"[0-9a-f]{40}", value)
                ):
                    raise ValueError(f"Immutable revision required at {name}")
                if key.endswith("sha256") and (
                    not isinstance(value, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value)
                ):
                    raise ValueError(f"SHA256 required at {name}")
                validate_pins(value, name)

    validate_pins(payload)
    return _freeze(payload)


def contract_hash(contract: Mapping[str, Any] | None = None) -> str:
    """Return a canonical SHA-256 identity independent of YAML formatting/path."""
    selected = load_contract() if contract is None else contract
    encoded = json.dumps(
        _plain(selected), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
