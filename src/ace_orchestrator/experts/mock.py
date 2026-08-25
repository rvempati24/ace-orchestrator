from __future__ import annotations

import random

from ace_orchestrator.core.models import (
    ActionRecord,
    AutonomyHorizon,
    ExecutionState,
    ExpertResult,
    Subgoal,
    Usage,
)
from ace_orchestrator.experts.base import Expert
from ace_orchestrator.policies.base import Policy


class MockExpert(Expert):
    """Cheap simulation seam for routing experiments; never makes external calls."""

    def __init__(
        self,
        expert_id: str,
        domain: str,
        *,
        base_success: float = 0.7,
        off_domain_penalty: float = 0.15,
        success_by_domain: dict[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(expert_id, f"Mock {domain} specialist", (domain,))
        self.domain = domain
        self.base_success = base_success
        self.off_domain_penalty = off_domain_penalty
        self.success_by_domain = success_by_domain or {}
        self._random = random.Random(seed)

    def success_probability(
        self, subgoal: Subgoal, policy: Policy, horizon: AutonomyHorizon
    ) -> float:
        base = self.success_by_domain.get(subgoal.domain)
        if base is None:
            base = self.base_success
            if self.domain not in ("general", subgoal.domain):
                base -= self.off_domain_penalty
        horizon_modifier = (
            -0.08 if horizon.action_limit < 3 else 0.02 if horizon.action_limit >= 8 else 0
        )
        return min(0.99, max(0.01, base + policy.config.success_modifier + horizon_modifier))

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
    ) -> ExpertResult:
        probability = self.success_probability(subgoal, policy, horizon)
        success = self._random.random() < probability
        actions_count = min(horizon.action_limit, 2 if success else 1)
        actions = tuple(
            ActionRecord("mock_step", f"{subgoal.subgoal_id}:{index + 1}", success)
            for index in range(actions_count)
        )
        usage = Usage(
            wall_clock_latency_s=policy.config.simulated_latency_s,
            model_latency_s=policy.config.simulated_latency_s,
            input_tokens=100,
            output_tokens=policy.config.max_output_tokens // 8,
            estimated_cost_usd=policy.config.simulated_cost_usd,
            model_calls=1,
        )
        return ExpertResult(
            success=success,
            output={
                "expert_id": self.expert_id,
                "domain": subgoal.domain,
                "success_probability": probability,
            },
            actions=actions,
            usage=usage,
            error=None if success else "simulated execution failure",
        )
