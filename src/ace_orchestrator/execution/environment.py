"""Task-owned, concurrency-safe environment sessions."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ace_orchestrator.benchmarks.base import BenchmarkTask
from ace_orchestrator.core.actions import ActionRecord, ComputerAction


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class EnvironmentObservation:
    observation_id: str
    values: dict[str, Any]
    captured_at: str = field(default_factory=_now)
    screenshot_base64: str | None = None
    available_action_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepOutcome:
    record: ActionRecord
    observation: EnvironmentObservation


class EnvironmentSession(ABC):
    """A mutable task world that may be handed between stateless experts."""

    def __init__(self, task: BenchmarkTask, *, session_id: str | None = None) -> None:
        self.task = task
        self.session_id = session_id or uuid4().hex
        self._lock = asyncio.Lock()
        self._closed = False
        self._last_observation_id: str | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> EnvironmentSession:
        self._ensure_open()
        return self

    async def __aexit__(
        self, exc_type: object, exc: BaseException | None, traceback: object
    ) -> None:
        await asyncio.shield(self.close())

    async def observe(self) -> EnvironmentObservation:
        async with self._lock:
            self._ensure_open()
            observation = await self._observe()
            self._last_observation_id = observation.observation_id
            return observation

    async def act(
        self,
        action: ComputerAction,
        *,
        before: EnvironmentObservation | None = None,
    ) -> StepOutcome:
        """Serialize mutations and reject commands based on stale observations."""

        async with self._lock:
            self._ensure_open()
            if before is None:
                before = await self._observe()
                self._last_observation_id = before.observation_id
            elif before.observation_id != self._last_observation_id:
                raise StaleObservationError(
                    f"action used stale observation {before.observation_id}; "
                    f"latest is {self._last_observation_id}"
                )

            started_at = _now()
            try:
                after = await self._apply(action)
                success = True
                error = None
            except Exception as execution_error:  # Environment failures become trajectory data.
                success = False
                error = str(execution_error)
                try:
                    after = await self._observe()
                except Exception:
                    after = before
            self._last_observation_id = after.observation_id
            record = ActionRecord(
                action=action,
                started_at=started_at,
                finished_at=_now(),
                observation_before_id=before.observation_id,
                observation_after_id=after.observation_id,
                success=success,
                error=error,
                metadata={"environment_session_id": self.session_id},
            )
            return StepOutcome(record, after)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            try:
                await self._close()
            finally:
                self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"environment session {self.session_id} is closed")

    @abstractmethod
    async def _observe(self) -> EnvironmentObservation:
        raise NotImplementedError

    @abstractmethod
    async def _apply(self, action: ComputerAction) -> EnvironmentObservation:
        raise NotImplementedError

    @abstractmethod
    async def _close(self) -> None:
        raise NotImplementedError


class EnvironmentFactory(ABC):
    @abstractmethod
    async def create(self, task: BenchmarkTask) -> EnvironmentSession:
        raise NotImplementedError


class MockEnvironmentSession(EnvironmentSession):
    def __init__(self, task: BenchmarkTask, *, session_id: str | None = None) -> None:
        super().__init__(task, session_id=session_id)
        self.actions: list[ComputerAction] = []
        self.observation_count = 0

    async def _observe(self) -> EnvironmentObservation:
        self.observation_count += 1
        return EnvironmentObservation(
            observation_id=f"{self.session_id}:observation:{self.observation_count}",
            values={"action_count": len(self.actions), "task_id": self.task.task_id},
        )

    async def _apply(self, action: ComputerAction) -> EnvironmentObservation:
        self.actions.append(action)
        return await self._observe()

    async def _close(self) -> None:
        return None


class MockEnvironmentFactory(EnvironmentFactory):
    """Creates a fresh observable session for every task run or experiment trial."""

    def __init__(self) -> None:
        self.sessions: list[MockEnvironmentSession] = []

    async def create(self, task: BenchmarkTask) -> MockEnvironmentSession:
        session = MockEnvironmentSession(task)
        self.sessions.append(session)
        return session


class StaleObservationError(RuntimeError):
    pass
