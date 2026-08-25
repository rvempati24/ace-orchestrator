from __future__ import annotations

from abc import ABC, abstractmethod

from ace_orchestrator.core.models import AutonomyHorizon, ExecutionState, ExpertResult, Subgoal
from ace_orchestrator.execution.environment import EnvironmentSession
from ace_orchestrator.policies.base import Policy


class Expert(ABC):
    expert_id: str
    description: str
    capabilities: tuple[str, ...]

    def __init__(self, expert_id: str, description: str, capabilities: tuple[str, ...]) -> None:
        self.expert_id = expert_id
        self.description = description
        self.capabilities = capabilities

    @abstractmethod
    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        raise NotImplementedError
