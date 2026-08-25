from pathlib import Path

import pytest

from ace_orchestrator.core.models import (
    ActionRecord,
    AutonomyHorizon,
    ExecutionState,
    ExpertResult,
    Subgoal,
)
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


class AlwaysExpert(Expert):
    def __init__(self, expert_id: str, capability: str, outcomes: list[bool]) -> None:
        super().__init__(expert_id, "test expert", (capability,))
        self.outcomes = iter(outcomes)

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
    ) -> ExpertResult:
        success = next(self.outcomes)
        return ExpertResult(
            success,
            {"ok": success},
            (ActionRecord("test", subgoal.subgoal_id),),
            error=None if success else "failed",
        )


def make_orchestrator(expert: Expert, path: Path) -> Orchestrator:
    registry = ExpertRegistry()
    registry.register(expert)
    return Orchestrator(
        planner=DeterministicPlanner(),
        expert_router=StaticRouter(),
        policy_router=FixedPolicyRouter("fast"),
        experts=registry,
        policies=default_policies(),
        verifier=ResultVerifier(),
        logger=JsonlTrajectoryLogger(path),
    )


@pytest.mark.asyncio
async def test_run_logs_complete_contract_and_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    result = await make_orchestrator(AlwaysExpert("web", "web", [True]), path).run(
        "Research the company"
    )
    assert result.success
    attempt = result.trajectory["subgoals"][0]["attempts"][0]
    assert attempt["selected_expert"] == "web"
    assert attempt["selected_policy"] == "fast"
    assert attempt["verification_result"]["success"] is True
    assert path.read_text().count("\n") == 1


@pytest.mark.asyncio
async def test_recovery_retries_with_stronger_policy(tmp_path: Path) -> None:
    result = await make_orchestrator(
        AlwaysExpert("web", "web", [False, True]), tmp_path / "r.jsonl"
    ).run("Research the company")
    step = result.trajectory["subgoals"][0]
    assert result.success
    assert step["retry_count"] == 1
    assert [attempt["selected_policy"] for attempt in step["attempts"]] == ["fast", "medium"]
    assert result.trajectory["escalations"][0]["action"] == "retry_stronger_policy"
    assert result.state.failed_subgoals == []
