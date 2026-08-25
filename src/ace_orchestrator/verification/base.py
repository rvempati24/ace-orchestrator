from __future__ import annotations

from abc import ABC, abstractmethod

from ace_orchestrator.core.models import (
    ExecutionState,
    ExpertResult,
    Subgoal,
    VerificationResult,
)


class Verifier(ABC):
    @abstractmethod
    async def verify(
        self,
        goal: Subgoal,
        before: ExecutionState,
        after: ExecutionState,
        expert_result: ExpertResult,
    ) -> VerificationResult:
        raise NotImplementedError
