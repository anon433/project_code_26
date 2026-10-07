"""Scoring primitives for Trajectory-Conditioned Utility Scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GlobalSelection:
    active_mask: dict[str, torch.Tensor]
    threshold: float
    positive_count: int
    selected_count: int
    total_count: int


def _float32_to_ordered_key(value: float) -> int:
    raw = int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item())
    raw &= (1 << 32) - 1
    sign = 1 << 31
    return ((~raw) & ((1 << 32) - 1)) if raw & sign else raw ^ sign


def _ordered_key_to_float32(key: int) -> float:
    sign = 1 << 31
    raw = key ^ sign if key & sign else (~key) & ((1 << 32) - 1)
    signed_raw = raw if raw < sign else raw - (1 << 32)
    return float(torch.tensor(signed_raw, dtype=torch.int32).view(torch.float32).item())


def _global_count(
    scores: Mapping[str, torch.Tensor],
    threshold: float,
    *,
    strict: bool,
    chunk_size: int,
) -> int:
    count = 0
    for score in scores.values():
        flat_score = score.reshape(-1)
        for start in range(0, flat_score.numel(), chunk_size):
            values = flat_score[start : start + chunk_size]
            comparison = values > threshold if strict else values >= threshold
            count += int(torch.count_nonzero(comparison).item())
    return count


def _global_kth_threshold(
    scores: Mapping[str, torch.Tensor],
    selected_count: int,
    *,
    chunk_size: int,
) -> float:
    minimum = min(float(score.amin().item()) for score in scores.values())
    maximum = max(float(score.amax().item()) for score in scores.values())
    low = _float32_to_ordered_key(minimum)
    high = _float32_to_ordered_key(maximum)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = _ordered_key_to_float32(middle)
        if (
            _global_count(scores, candidate, strict=False, chunk_size=chunk_size)
            >= selected_count
        ):
            low = middle
        else:
            high = middle - 1
    return _ordered_key_to_float32(low)


@torch.no_grad()
def select_global_topk(
    scores: Mapping[str, torch.Tensor],
    *,
    budget_frac: float,
    chunk_size: int = 8_000_000,
) -> GlobalSelection:
    """Select an exact global budget with stable parameter/coordinate ties."""
    if not 0.0 < budget_frac <= 1.0:
        raise ValueError(f"TCUS budget_frac must be in (0, 1], got {budget_frac}.")
    total_count = sum(score.numel() for score in scores.values())
    selected_count = max(
        1,
        min(total_count, int(round(float(budget_frac) * total_count))),
    )
    return select_global_topk_count(
        scores,
        selected_count=selected_count,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def select_global_topk_count(
    scores: Mapping[str, torch.Tensor],
    *,
    selected_count: int,
    chunk_size: int = 8_000_000,
) -> GlobalSelection:
    """Select an exact shared integer budget with stable global tie breaking."""
    if not scores:
        raise ValueError("TCUS requires at least one score tensor.")
    if chunk_size <= 0:
        raise ValueError("TCUS selection chunk_size must be positive.")

    total_count = 0
    for name, score in scores.items():
        if score.device.type != "cpu" or score.dtype != torch.float32:
            raise TypeError(f"TCUS score {name!r} must be CPU float32.")
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError(f"TCUS score {name!r} contains NaN/Inf.")
        total_count += score.numel()
    if not isinstance(selected_count, int) or not 0 < selected_count <= total_count:
        raise ValueError("TCUS selected_count must be an integer in [1, total_count].")
    threshold = _global_kth_threshold(
        scores,
        selected_count,
        chunk_size=chunk_size,
    )
    remaining = selected_count - _global_count(
        scores,
        threshold,
        strict=True,
        chunk_size=chunk_size,
    )
    if remaining <= 0:
        raise RuntimeError("TCUS global threshold did not expose kth-value ties.")

    active_mask = {}
    for name, score in scores.items():
        selected = score > threshold
        if remaining:
            selected = selected.clone()
            flat_score = score.reshape(-1)
            flat_selected = selected.reshape(-1)
            for start in range(0, flat_score.numel(), chunk_size):
                if remaining == 0:
                    break
                values = flat_score[start : start + chunk_size]
                ties = torch.nonzero(values == threshold, as_tuple=False).flatten()
                take = min(remaining, ties.numel())
                if take:
                    flat_selected[start + ties[:take]] = True
                    remaining -= take
        active_mask[name] = selected
    if remaining:
        raise RuntimeError(f"TCUS could not fill {remaining} global threshold ties.")

    return GlobalSelection(
        active_mask=active_mask,
        threshold=threshold,
        positive_count=_global_count(
            scores,
            0.0,
            strict=True,
            chunk_size=chunk_size,
        ),
        selected_count=selected_count,
        total_count=total_count,
    )


@torch.no_grad()
def summarize_scores(scores: Mapping[str, torch.Tensor]) -> dict[str, float | int]:
    count = sum(score.numel() for score in scores.values())
    total = sum(float(score.sum().item()) for score in scores.values())
    square_total = sum(float(score.square().sum().item()) for score in scores.values())
    mean = total / count
    return {
        "count": count,
        "positive_count": sum(
            int(torch.count_nonzero(score > 0).item()) for score in scores.values()
        ),
        "mean": mean,
        "std": math.sqrt(max(square_total / count - mean * mean, 0.0)),
        "min": min(float(score.amin().item()) for score in scores.values()),
        "max": max(float(score.amax().item()) for score in scores.values()),
    }
