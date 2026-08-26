import pytest

from ace_orchestrator.benchmarks.base import BenchmarkTask
from ace_orchestrator.core.actions import ActionRecord, ComputerAction
from ace_orchestrator.core.models import ExecutionState, ExpertResult, Subgoal
from ace_orchestrator.execution.environment import MockEnvironmentSession
from ace_orchestrator.verification import BrowserGymVerifier


class CompletedBenchmarkEnvironment(MockEnvironmentSession):
    def metrics_snapshot(self):
        return {"last_reward": 1.0, "terminated": True, "truncated": False}


@pytest.mark.asyncio
async def test_browsergym_reward_is_authoritative_over_expert_done_bit() -> None:
    environment = CompletedBenchmarkEnvironment(BenchmarkTask("task", "Click", "browser"))
    record = ActionRecord(
        ComputerAction("click", target_ref="12"),
        "start",
        "finish",
        "before",
        "after",
        True,
    )
    result = ExpertResult(False, {}, (record,), error="autonomy action horizon exhausted")

    verification = await BrowserGymVerifier().verify(
        Subgoal("goal", "Click", domain="browser"),
        ExecutionState(),
        ExecutionState(),
        result,
        environment,
    )

    assert not verification.checks["expert_reported_success"]
    assert verification.success
