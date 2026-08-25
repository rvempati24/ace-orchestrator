from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.core.models import Subgoal, Usage
from ace_orchestrator.execution.environment import EnvironmentObservation
from ace_orchestrator.policies.base import Policy


@dataclass(frozen=True)
class CUAProposal:
    actions: tuple[ComputerAction, ...]
    done: bool
    summary: str
    usage: Usage = field(default_factory=Usage)


class CUABackend(ABC):
    @abstractmethod
    async def propose(
        self,
        *,
        system_prompt: str,
        subgoal: Subgoal,
        observation: EnvironmentObservation,
        policy: Policy,
    ) -> CUAProposal:
        raise NotImplementedError
