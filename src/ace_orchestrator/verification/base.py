from __future__ import annotations

from abc import ABC, abstractmethod

from ace_orchestrator.core.models import (
    ExecutionState,
    ExpertResult,
    Subgoal,
    VerificationResult,
)
from ace_orchestrator.execution.environment import EnvironmentSession


class Verifier(ABC):
    @abstractmethod
    async def verify(
        self,
        goal: Subgoal,
        before: ExecutionState,
        after: ExecutionState,
        expert_result: ExpertResult,
        environment: EnvironmentSession,
    ) -> VerificationResult:
        raise NotImplementedError
