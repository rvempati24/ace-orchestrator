import asyncio
from pathlib import Path

import pytest

from ace_orchestrator.benchmarks import BrowserGymTask
from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.core.models import (
    AutonomyHorizon,
    ExecutionState,
    ExpertResult,
    Subgoal,
)
from ace_orchestrator.execution.browsergym import BrowserGymEnvironmentFactory
from ace_orchestrator.execution.environment import EnvironmentSession, StaleObservationError
from ace_orchestrator.experts import Expert, ExpertRegistry, ScriptedMiniWoBExpert
from ace_orchestrator.orchestration.expert_router import StaticRouter
from ace_orchestrator.orchestration.orchestrator import Orchestrator
from ace_orchestrator.orchestration.planner import Planner
from ace_orchestrator.orchestration.policy_router import FixedPolicyRouter
from ace_orchestrator.policies.base import Policy
from ace_orchestrator.policies.configured import default_policies
from ace_orchestrator.telemetry.logger import JsonlTrajectoryLogger
from ace_orchestrator.verification import BrowserGymVerifier

pytestmark = pytest.mark.browsergym


class OneSubgoalPlanner(Planner):
    async def plan(self, task, state):
        return [Subgoal("browser-task", task.user_goal, domain="browser")]


class ObserveThenRaiseExpert(Expert):
    def __init__(self) -> None:
        super().__init__("browser", "cleanup test expert", ("browser",))

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        await environment.observe()
        raise RuntimeError("forced post-launch crash")


class ObserveThenFailExpert(Expert):
    def __init__(self) -> None:
        super().__init__("browser", "reroute test expert", ("browser",))
        self.session_ids: list[str] = []

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        self.session_ids.append(environment.session_id)
        await environment.observe()
        return ExpertResult(False, {}, error="force reroute")


class ObserveThenBlockExpert(Expert):
    def __init__(self) -> None:
        super().__init__("browser", "cancellation test expert", ("browser",))
        self.started = asyncio.Event()

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        await environment.observe()
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def build_orchestrator(
    factory: BrowserGymEnvironmentFactory,
    *,
    expert: Expert | None = None,
    log_path: Path | None = None,
) -> Orchestrator:
    registry = ExpertRegistry()
    registry.register(expert or ScriptedMiniWoBExpert())
    return Orchestrator(
        planner=OneSubgoalPlanner(),
        expert_router=StaticRouter(),
        policy_router=FixedPolicyRouter("fast"),
        experts=registry,
        policies=default_policies(),
        verifier=BrowserGymVerifier(),
        environment_factory=factory,
        logger=JsonlTrajectoryLogger(log_path),
    )


@pytest.mark.parametrize(
    "task_name",
    ["click-test", "click-button", "enter-text", "choose-list", "copy-paste"],
)
@pytest.mark.asyncio
async def test_five_representative_miniwob_tasks_pass(
    task_name: str, miniwob_url: str, tmp_path: Path
) -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url=miniwob_url)
    orchestrator = build_orchestrator(factory, log_path=tmp_path / f"{task_name}.jsonl")

    result = await orchestrator.run_benchmark(BrowserGymTask.miniwob(task_name, seed=1))

    assert result.success
    assert result.trajectory["final_verification"]["success"]
    metrics = result.trajectory["environment_metrics"]
    assert metrics["last_reward"] == 1.0
    assert metrics["terminated"] is True
    assert metrics["step_count"] in {1, 2}
    assert factory.sessions[0].closed
    assert result.trajectory["schema_version"] == "1.1.0"


@pytest.mark.asyncio
async def test_two_browsergym_tasks_run_in_isolated_processes(miniwob_url: str) -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url=miniwob_url)
    orchestrator = build_orchestrator(factory)

    first, second = await asyncio.gather(
        orchestrator.run_benchmark(BrowserGymTask.miniwob("click-test", seed=2)),
        orchestrator.run_benchmark(BrowserGymTask.miniwob("click-button", seed=3)),
    )

    assert first.success and second.success
    assert len(factory.sessions) == 2
    assert factory.sessions[0].session_id != factory.sessions[1].session_id
    assert (
        factory.sessions[0].metrics_snapshot()["worker_pid"]
        != factory.sessions[1].metrics_snapshot()["worker_pid"]
    )
    assert all(session.closed for session in factory.sessions)


@pytest.mark.asyncio
async def test_real_browser_closes_after_expert_crash(miniwob_url: str) -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url=miniwob_url)
    orchestrator = build_orchestrator(factory, expert=ObserveThenRaiseExpert())

    with pytest.raises(RuntimeError, match="forced post-launch crash"):
        await orchestrator.run_benchmark(BrowserGymTask.miniwob("click-test", seed=4))

    assert len(factory.sessions) == 1
    assert factory.sessions[0].metrics_snapshot()["worker_pid"] is not None
    assert factory.sessions[0].closed


@pytest.mark.asyncio
async def test_reroute_preserves_the_live_browser_session(miniwob_url: str) -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url=miniwob_url)
    failing = ObserveThenFailExpert()
    registry = ExpertRegistry()
    registry.register(failing)
    registry.register(ScriptedMiniWoBExpert("fallback"))
    orchestrator = Orchestrator(
        planner=OneSubgoalPlanner(),
        expert_router=StaticRouter(),
        policy_router=FixedPolicyRouter("fast"),
        experts=registry,
        policies=default_policies(),
        verifier=BrowserGymVerifier(),
        environment_factory=factory,
        logger=JsonlTrajectoryLogger(),
    )

    result = await orchestrator.run_benchmark(BrowserGymTask.miniwob("click-test", seed=5))

    session_id = result.trajectory["environment"]["session_id"]
    assert result.success
    assert failing.session_ids == [session_id, session_id]
    assert len(factory.sessions) == 1
    assert {
        attempt["environment_session_id"]
        for attempt in result.trajectory["subgoals"][0]["attempts"]
    } == {session_id}
    assert any(event["action"] == "reroute_expert" for event in result.trajectory["escalations"])


@pytest.mark.asyncio
async def test_real_session_rejects_stale_observations_and_unknown_refs(miniwob_url: str) -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url=miniwob_url)
    session = await factory.create(BrowserGymTask.miniwob("click-test", seed=6))

    async with session:
        stale = await session.observe()
        current = await session.observe()
        with pytest.raises(StaleObservationError):
            await session.act(ComputerAction("click", target_ref="13"), before=stale)
        outcome = await session.act(
            ComputerAction("click", target_ref="missing-ref"), before=current
        )

    assert not outcome.record.success
    assert "unknown action ref" in str(outcome.record.error)
    assert session.metrics_snapshot()["step_count"] == 0
    assert session.closed


@pytest.mark.asyncio
async def test_real_browser_closes_after_cancellation(miniwob_url: str) -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url=miniwob_url)
    expert = ObserveThenBlockExpert()
    orchestrator = build_orchestrator(factory, expert=expert)
    run = asyncio.create_task(
        orchestrator.run_benchmark(BrowserGymTask.miniwob("click-test", seed=7))
    )
    await expert.started.wait()

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert factory.sessions[0].closed


@pytest.fixture
def miniwob_url() -> str:
    import os

    value = os.getenv("MINIWOB_URL")
    if not value:
        pytest.skip("set MINIWOB_URL to the pinned MiniWoB++ html/miniwob directory")
    return value
