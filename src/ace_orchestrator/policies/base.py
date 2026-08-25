from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    reasoning_budget: str
    max_output_tokens: int
    context_window: int
    model_id: str
    suggested_action_horizon: int
    success_modifier: float = 0.0
    simulated_latency_s: float = 0.0
    simulated_cost_usd: float = 0.0


class Policy(ABC):
    """Compute allocation, deliberately separate from expert capability."""

    policy_id: str
    config: PolicyConfig

    def __init__(self, policy_id: str, config: PolicyConfig) -> None:
        self.policy_id = policy_id
        self.config = config
