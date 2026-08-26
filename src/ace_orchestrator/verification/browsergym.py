"""Reward-based verification for BrowserGym benchmark episodes."""

from __future__ import annotations

from ace_orchestrator.core.models import (
    ExecutionState,
    ExpertResult,
    Subgoal,
    VerificationResult,
)
from ace_orchestrator.execution.environment import EnvironmentSession
from ace_orchestrator.verification.base import Verifier


class BrowserGymVerifier(Verifier):
    """Treat benchmark reward/termination as truth, independently of expert claims."""

    async def verify(
        self,
        goal: Subgoal,
        before: ExecutionState,
        after: ExecutionState,
        expert_result: ExpertResult,
        environment: EnvironmentSession,
    ) -> VerificationResult:
        metrics = environment.metrics_snapshot()
        reward = float(metrics.get("last_reward", 0.0))
        checks = {
            "expert_reported_success": expert_result.success,
            "actions_completed": bool(expert_result.actions)
            and all(record.success for record in expert_result.actions),
            "positive_benchmark_reward": reward > 0,
            "episode_terminated": bool(metrics.get("terminated")),
            "episode_not_truncated": not bool(metrics.get("truncated")),
        }
        success = all(
            checks[key]
            for key in (
                "actions_completed",
                "positive_benchmark_reward",
                "episode_terminated",
                "episode_not_truncated",
            )
        )
        return VerificationResult(
            success=success,
            score=max(0.0, min(1.0, reward)),
            checks=checks,
            feedback=None if success else "BrowserGym reward/termination checks did not pass",
        )
