import re

import pytest

from ace_orchestrator.benchmarks import BrowserGymTask
from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.core.models import (
    AutonomyHorizon,
    ExecutionState,
    Subgoal,
    Usage,
)
from ace_orchestrator.execution.browsergym import BrowserGymEnvironmentFactory
from ace_orchestrator.experts.prompted_cua import PromptedCUAExpert
from ace_orchestrator.inference.base import CUABackend, CUAProposal
from ace_orchestrator.policies.configured import FastPolicy
from ace_orchestrator.verification import BrowserGymVerifier

pytestmark = pytest.mark.browsergym


def _attribute(node, key):
    value = node.get(key)
    return value.get("value") if isinstance(value, dict) else value


class ObservationDrivenBackend(CUABackend):
    """Model-shaped test double that sees only the public CUA observation contract."""

    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, *, system_prompt, subgoal, observation, policy) -> CUAProposal:
        self.calls += 1
        assert observation.screenshot_base64
        assert observation.screenshot_mime_type == "image/png"
        nodes = observation.values["semantic_state"]["nodes"]
        if self.calls == 1:
            textbox = next(
                node
                for node in nodes
                if node.get("browsergym_id") and _attribute(node, "role") == "textbox"
            )
            value = re.search(r'"([^"]+)"', observation.values["goal"]).group(1)
            return CUAProposal(
                (ComputerAction("fill", str(textbox["browsergym_id"]), value=value),),
                done=False,
                summary="filled observed text",
                usage=Usage(model_calls=1, input_tokens=100, output_tokens=10),
            )
        submit = next(
            node
            for node in nodes
            if node.get("browsergym_id")
            and _attribute(node, "role") == "button"
            and _attribute(node, "name") == "Submit"
        )
        return CUAProposal(
            (ComputerAction("click", str(submit["browsergym_id"])),),
            done=True,
            summary="submitted",
            usage=Usage(model_calls=1, input_tokens=100, output_tokens=10),
        )


@pytest.mark.asyncio
async def test_prompted_cua_completes_a_real_multistep_browser_episode(miniwob_url: str) -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url=miniwob_url)
    session = await factory.create(BrowserGymTask.miniwob("enter-text", seed=11))
    backend = ObservationDrivenBackend()
    expert = PromptedCUAExpert(
        "browser", "prompted browser expert", ("browser",), "Use observed refs", backend
    )
    state = ExecutionState()
    goal = Subgoal("browser-task", "Complete the observed browser task", domain="browser")

    async with session:
        result = await expert.execute(
            goal, state, FastPolicy(), AutonomyHorizon(max_actions=3), session
        )
        verification = await BrowserGymVerifier().verify(
            goal, ExecutionState(), state, result, session
        )

    assert result.success
    assert verification.success
    assert backend.calls == 2
    assert len(result.actions) == 2
    assert result.usage.model_calls == 2
    assert session.closed


@pytest.fixture
def miniwob_url() -> str:
    import os

    value = os.getenv("MINIWOB_URL")
    if not value:
        pytest.skip("set MINIWOB_URL to the pinned MiniWoB++ html/miniwob directory")
    return value
