import pytest

from ace_orchestrator.core.models import ExecutionState, Subgoal, Task
from ace_orchestrator.experts.mock import MockExpert
from ace_orchestrator.orchestration.expert_router import StaticRouter
from ace_orchestrator.orchestration.planner import DeterministicPlanner


@pytest.mark.asyncio
async def test_deterministic_planner_builds_ordered_dependencies() -> None:
    plan = await DeterministicPlanner().plan(
        Task("Research the company and then update the spreadsheet"), ExecutionState()
    )
    assert [goal.domain for goal in plan] == ["web", "spreadsheet"]
    assert plan[1].dependencies == (plan[0].subgoal_id,)


@pytest.mark.asyncio
async def test_static_router_uses_capability_metadata() -> None:
    experts = (
        MockExpert("general", "general"),
        MockExpert("spreadsheet", "spreadsheet"),
    )
    selection = await StaticRouter().route(
        Subgoal("sheet", "Update cells", domain="spreadsheet"), ExecutionState(), experts
    )
    assert selection.expert_id == "spreadsheet"
    assert selection.candidates == ("general", "spreadsheet")
