from ace_orchestrator.core.models import (
    ExecutionState,
    ExpertResult,
    Subgoal,
    VerificationResult,
)
from ace_orchestrator.execution.environment import EnvironmentSession
from ace_orchestrator.verification.base import Verifier


class ResultVerifier(Verifier):
    """V0 verifier that trusts an environment/expert success signal, but logs explicit checks."""

    async def verify(
        self,
        goal: Subgoal,
        before: ExecutionState,
        after: ExecutionState,
        expert_result: ExpertResult,
        environment: EnvironmentSession,
    ) -> VerificationResult:
        checks = {
            "expert_reported_success": expert_result.success,
            "actions_completed": bool(expert_result.actions)
            and all(record.success for record in expert_result.actions),
            "no_execution_error": expert_result.error is None,
        }
        score = sum(checks.values()) / len(checks)
        return VerificationResult(all(checks.values()), score, checks, expert_result.error)
