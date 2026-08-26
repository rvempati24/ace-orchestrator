import pytest

from ace_orchestrator.benchmarks.base import BenchmarkTask
from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.core.models import AutonomyHorizon, ExecutionState, Subgoal
from ace_orchestrator.execution.environment import MockEnvironmentSession
from ace_orchestrator.experts.prompted_cua import PromptedCUAExpert
from ace_orchestrator.inference.base import CUABackend, CUAProposal
from ace_orchestrator.policies.configured import FastPolicy


class OneStepBackend(CUABackend):
    async def propose(self, *, system_prompt, subgoal, observation, policy) -> CUAProposal:
        return CUAProposal(
            (ComputerAction("click", target_ref="button-1"),),
            done=True,
            summary="completed",
        )


class OversizedBackend(CUABackend):
    async def propose(self, *, system_prompt, subgoal, observation, policy) -> CUAProposal:
        return CUAProposal(
            (
                ComputerAction("click", target_ref="one"),
                ComputerAction("click", target_ref="two"),
            ),
            done=False,
            summary="too many",
        )


class RaisingBackend(CUABackend):
    async def propose(self, *, system_prompt, subgoal, observation, policy) -> CUAProposal:
        raise RuntimeError("endpoint unavailable")


class HistoryBackend(CUABackend):
    def __init__(self) -> None:
        self.observations = []

    async def propose(self, *, system_prompt, subgoal, observation, policy) -> CUAProposal:
        self.observations.append(observation)
        if len(self.observations) == 1:
            return CUAProposal(
                (ComputerAction("click", target_ref="button-1"),),
                done=False,
                summary="clicked",
            )
        return CUAProposal((), done=True, summary="complete")


@pytest.mark.asyncio
async def test_prompted_expert_receives_environment_per_execution() -> None:
    expert = PromptedCUAExpert(
        "web",
        "web expert",
        ("web",),
        "Use the browser",
        OneStepBackend(),
    )
    environment = MockEnvironmentSession(BenchmarkTask("task", "Click the button", "web"))

    async with environment:
        result = await expert.execute(
            Subgoal("click", "Click the button", domain="web"),
            ExecutionState(),
            FastPolicy(),
            AutonomyHorizon(max_actions=1),
            environment,
        )

    assert not hasattr(expert, "environment")
    assert result.success
    assert len(environment.actions) == 1
    assert result.actions[0].metadata["environment_session_id"] == environment.session_id


@pytest.mark.asyncio
async def test_prompted_expert_enforces_action_horizon_before_execution() -> None:
    expert = PromptedCUAExpert("web", "web expert", ("web",), "Use the browser", OversizedBackend())
    environment = MockEnvironmentSession(BenchmarkTask("task", "Click", "web"))

    async with environment:
        result = await expert.execute(
            Subgoal("click", "Click", domain="web"),
            ExecutionState(),
            FastPolicy(),
            AutonomyHorizon(max_actions=1),
            environment,
        )

    assert not result.success
    assert result.error == "proposal exceeds the remaining autonomy action budget"
    assert environment.actions == []


@pytest.mark.asyncio
async def test_prompted_expert_returns_inference_failures_for_bounded_recovery() -> None:
    expert = PromptedCUAExpert("web", "web expert", ("web",), "Use the browser", RaisingBackend())
    environment = MockEnvironmentSession(BenchmarkTask("task", "Click", "web"))

    async with environment:
        result = await expert.execute(
            Subgoal("click", "Click", domain="web"),
            ExecutionState(),
            FastPolicy(),
            AutonomyHorizon(max_actions=1),
            environment,
        )

    assert not result.success
    assert result.error == "inference failed: endpoint unavailable"
    assert result.output["model_turns"] == 0
    assert environment.closed


@pytest.mark.asyncio
async def test_prompted_expert_exposes_recent_action_history_to_later_turns() -> None:
    backend = HistoryBackend()
    expert = PromptedCUAExpert("web", "web expert", ("web",), "Use the browser", backend)
    environment = MockEnvironmentSession(BenchmarkTask("task", "Click", "web"))

    async with environment:
        result = await expert.execute(
            Subgoal("click", "Click", domain="web"),
            ExecutionState(),
            FastPolicy(),
            AutonomyHorizon(max_actions=2),
            environment,
        )

    assert result.success
    assert backend.observations[0].values["recent_actions"] == []
    assert backend.observations[1].values["recent_actions"] == [
        {
            "kind": "click",
            "target_ref": "button-1",
            "target": None,
            "value": None,
            "success": True,
            "error": None,
        }
    ]
