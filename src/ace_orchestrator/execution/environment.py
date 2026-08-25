from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ace_orchestrator.core.models import ActionRecord


@dataclass(frozen=True)
class EnvironmentObservation:
    values: dict[str, Any]
    screenshot_base64: str | None = None


class ComputerEnvironment(ABC):
    """Replaceable GUI/browser/API execution surface for a real CUA."""

    @abstractmethod
    async def observe(self) -> EnvironmentObservation:
        raise NotImplementedError

    @abstractmethod
    async def apply(self, action: ActionRecord) -> EnvironmentObservation:
        raise NotImplementedError


class MockEnvironment(ComputerEnvironment):
    def __init__(self) -> None:
        self.actions: list[ActionRecord] = []

    async def observe(self) -> EnvironmentObservation:
        return EnvironmentObservation({"action_count": len(self.actions)})

    async def apply(self, action: ActionRecord) -> EnvironmentObservation:
        self.actions.append(action)
        return await self.observe()
