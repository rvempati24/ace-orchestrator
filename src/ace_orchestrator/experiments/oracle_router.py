"""Helpers for estimating the best available contract from offline outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OfflineOutcome:
    domain: str
    expert_id: str
    policy_id: str
    success_rate: float
    latency_s: float
    cost_usd: float


def best_contract(
    outcomes: tuple[OfflineOutcome, ...], *, latency_weight: float = 0.0, cost_weight: float = 0.0
) -> OfflineOutcome:
    if not outcomes:
        raise ValueError("oracle needs at least one outcome")
    return max(
        outcomes,
        key=lambda row: (
            row.success_rate - latency_weight * row.latency_s - cost_weight * row.cost_usd
        ),
    )
