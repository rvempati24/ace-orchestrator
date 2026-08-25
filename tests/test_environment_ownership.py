import asyncio
from pathlib import Path

import pytest

from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.core.models import (
    AutonomyHorizon,
    ExecutionState,
    ExpertResult,
    Subgoal,
)
from ace_orchestrator.execution.environment import EnvironmentSession, MockEnvironmentFactory
from ace_orchestrator.experts.base import Expert
from ace_orchestrator.experts.registry import ExpertRegistry
from ace_orchestrator.orchestration.expert_router import StaticRouter
from ace_orchestrator.orchestration.orchestrator import Orchestrator
from ace_orchestrator.orchestration.planner import DeterministicPlanner
from ace_orchestrator.orchestration.policy_router import FixedPolicyRouter
from ace_orchestrator.policies.base import Policy
from ace_orchestrator.policies.configured import default_policies
from ace_orchestrator.telemetry.logger import JsonlTrajectoryLogger
from ace_orchestrator.verification.simple import ResultVerifier


class SessionExpert(Expert):
    def __init__(self, expert_id: str, capability: str, success: bool) -> None:
        super().__init__(expert_id, "session test expert", (capability,))
        self.success = success
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
        before = await environment.observe()
        outcome = await environment.act(
            ComputerAction("test", target=f"{self.expert_id}:{subgoal.subgoal_id}"),
            before=before,
        )
        return ExpertResult(
            self.success,
            {"session_id": environment.session_id},
            (outcome.record,),
            error=None if self.success else "forced failure",
        )


class RaisingExpert(Expert):
    def __init__(self) -> None:
        super().__init__("web", "raising expert", ("web",))

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        raise RuntimeError("forced expert crash")


class BlockingExpert(Expert):
    def __init__(self) -> None:
        super().__init__("web", "blocking expert", ("web",))
        self.started = asyncio.Event()

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def build_orchestrator(
    experts: tuple[Expert, ...], factory: MockEnvironmentFactory, path: Path
) -> Orchestrator:
    registry = ExpertRegistry()
    for expert in experts:
        registry.register(expert)
    return Orchestrator(
        planner=DeterministicPlanner(),
        expert_router=StaticRouter(),
        policy_router=FixedPolicyRouter("fast"),
        experts=registry,
        policies=default_policies(),
        verifier=ResultVerifier(),
        environment_factory=factory,
        logger=JsonlTrajectoryLogger(path),
    )


@pytest.mark.asyncio
async def test_each_task_run_gets_a_fresh_environment(tmp_path: Path) -> None:
    factory = MockEnvironmentFactory()
    expert = SessionExpert("web", "web", True)
    orchestrator = build_orchestrator((expert,), factory, tmp_path / "runs.jsonl")
    shared_initial_state = ExecutionState(values={"seed": {"value": 1}})

    first, second = await asyncio.gather(
        orchestrator.run("Research the first company", shared_initial_state),
        orchestrator.run("Research the second company", shared_initial_state),
    )

    assert len(factory.sessions) == 2
    assert (
        first.trajectory["environment"]["session_id"]
        != second.trajectory["environment"]["session_id"]
    )
    assert all(session.closed for session in factory.sessions)
    assert [len(session.actions) for session in factory.sessions] == [1, 1]
    assert factory.sessions[0].task.task_id != factory.sessions[1].task.task_id
    assert first.state is not second.state
    assert shared_initial_state == ExecutionState(values={"seed": {"value": 1}})


@pytest.mark.asyncio
async def test_dependent_subgoals_handoff_one_task_environment(tmp_path: Path) -> None:
    factory = MockEnvironmentFactory()
    web = SessionExpert("web", "web", True)
    spreadsheet = SessionExpert("spreadsheet", "spreadsheet", True)
    orchestrator = build_orchestrator((web, spreadsheet), factory, tmp_path / "handoff.jsonl")

    result = await orchestrator.run(
        "Research the company and then enter the result into the spreadsheet"
    )

    session_id = result.trajectory["environment"]["session_id"]
    assert result.success
    assert web.session_ids == [session_id]
    assert spreadsheet.session_ids == [session_id]
    assert result.state.completed_subgoals == ["subgoal-1", "subgoal-2"]
    assert len(factory.sessions) == 1


@pytest.mark.asyncio
async def test_rerouting_hands_the_same_environment_to_the_next_expert(tmp_path: Path) -> None:
    factory = MockEnvironmentFactory()
    failing = SessionExpert("web", "web", False)
    succeeding = SessionExpert("general", "general", True)
    orchestrator = build_orchestrator((failing, succeeding), factory, tmp_path / "reroute.jsonl")

    result = await orchestrator.run("Research the company")

    session_id = result.trajectory["environment"]["session_id"]
    assert result.success
    assert failing.session_ids == [session_id, session_id]
    assert succeeding.session_ids == [session_id]
    assert len(factory.sessions) == 1
    assert len(factory.sessions[0].actions) == 3
    reroute = next(
        event for event in result.trajectory["escalations"] if event["action"] == "reroute_expert"
    )
    assert reroute["environment_session_id"] == session_id
    assert reroute["state_strategy"] == "continue_in_place"


@pytest.mark.asyncio
async def test_environment_closes_when_an_expert_crashes(tmp_path: Path) -> None:
    factory = MockEnvironmentFactory()
    orchestrator = build_orchestrator((RaisingExpert(),), factory, tmp_path / "crash.jsonl")

    with pytest.raises(RuntimeError, match="forced expert crash"):
        await orchestrator.run("Research the company")

    assert len(factory.sessions) == 1
    assert factory.sessions[0].closed


@pytest.mark.asyncio
async def test_environment_closes_when_a_run_is_cancelled(tmp_path: Path) -> None:
    factory = MockEnvironmentFactory()
    expert = BlockingExpert()
    orchestrator = build_orchestrator((expert,), factory, tmp_path / "cancel.jsonl")
    run = asyncio.create_task(orchestrator.run("Research the company"))
    await expert.started.wait()

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert len(factory.sessions) == 1
    assert factory.sessions[0].closed
