import pytest

from ace_orchestrator.benchmarks.base import BenchmarkTask
from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.execution.environment import (
    MockEnvironmentFactory,
    MockEnvironmentSession,
    StaleObservationError,
)


class PartiallyFailingSession(MockEnvironmentSession):
    async def _apply(self, action: ComputerAction):
        self.actions.append(action)
        raise RuntimeError("application failed after mutation")


@pytest.mark.asyncio
async def test_environment_records_commands_separately_from_results() -> None:
    session = MockEnvironmentSession(BenchmarkTask("task-1", "Click the button"))
    async with session:
        before = await session.observe()
        command = ComputerAction("click", target_ref="button-1")
        outcome = await session.act(command, before=before)

    assert outcome.record.action is command
    assert outcome.record.success
    assert outcome.record.observation_before_id == before.observation_id
    assert outcome.record.observation_after_id == outcome.observation.observation_id
    assert outcome.record.metadata["environment_session_id"] == session.session_id
    assert session.closed


@pytest.mark.asyncio
async def test_environment_rejects_actions_from_stale_observations() -> None:
    session = MockEnvironmentSession(BenchmarkTask("task-1", "Click the button"))
    async with session:
        stale = await session.observe()
        await session.observe()
        with pytest.raises(StaleObservationError):
            await session.act(ComputerAction("click", target_ref="button-1"), before=stale)


@pytest.mark.asyncio
async def test_factory_creates_fresh_isolated_sessions() -> None:
    factory = MockEnvironmentFactory()
    first = await factory.create(BenchmarkTask("first", "First task"))
    second = await factory.create(BenchmarkTask("second", "Second task"))

    async with first, second:
        first_before = await first.observe()
        await first.act(ComputerAction("type", target="field", value="one"), before=first_before)
        second_observation = await second.observe()

    assert first.session_id != second.session_id
    assert len(first.actions) == 1
    assert second.actions == []
    assert second_observation.values["action_count"] == 0
    assert first.closed and second.closed


@pytest.mark.asyncio
async def test_failed_action_records_observable_partial_state() -> None:
    session = PartiallyFailingSession(BenchmarkTask("task-1", "Mutating failure"))
    async with session:
        before = await session.observe()
        outcome = await session.act(ComputerAction("type", target="field"), before=before)

    assert not outcome.record.success
    assert outcome.record.error == "application failed after mutation"
    assert outcome.observation.values["action_count"] == 1
    assert outcome.record.observation_after_id != before.observation_id
