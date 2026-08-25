import base64
import json

import pytest

from ace_orchestrator.core.models import Subgoal
from ace_orchestrator.execution.environment import EnvironmentObservation
from ace_orchestrator.inference.openai_compatible import OpenAICompatibleCUA
from ace_orchestrator.policies.configured import FastPolicy


class StubOpenAICompatibleCUA(OpenAICompatibleCUA):
    def __init__(self, decoded: dict) -> None:
        super().__init__("https://example.invalid/v1", "test-model")
        self.decoded = decoded

    def _request(self, system_prompt, subgoal, observation, policy):
        return {
            "choices": [{"message": {"content": json.dumps(self.decoded)}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }


def browser_observation() -> EnvironmentObservation:
    return EnvironmentObservation(
        "observation-1",
        {
            "goal": "Click Submit",
            "url": "https://example.test/form",
            "semantic_state": {
                "nodes": [
                    {
                        "browsergym_id": "12",
                        "role": {"value": "button"},
                        "name": {"value": "Submit"},
                        "properties": [],
                    }
                ]
            },
            "dom_state": {"large": "not sent to the model"},
        },
        screenshot_base64=base64.b64encode(b"fake-png").decode(),
        screenshot_mime_type="image/png",
        available_action_refs=("12",),
    )


def test_extract_json_accepts_fenced_model_output() -> None:
    assert OpenAICompatibleCUA._extract_json('```json\n{"actions": [], "done": true}\n```')["done"]


def test_extract_json_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleCUA._extract_json("[]")


def test_request_compacts_browser_state_and_preserves_png_mime_type() -> None:
    client = StubOpenAICompatibleCUA({})
    body = client._build_request_body(
        "browser specialist",
        Subgoal("task", "Click Submit", domain="browser"),
        browser_observation(),
        FastPolicy(),
    )

    text = json.loads(body["messages"][1]["content"][0]["text"])
    assert "semantic_state" not in text["observation"]["page"]
    assert "dom_state" not in text["observation"]["page"]
    assert text["observation"]["elements"] == [{"ref": "12", "role": "button", "name": "Submit"}]
    image_url = body["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert "Return exactly one JSON object" in body["messages"][0]["content"]


@pytest.mark.asyncio
async def test_proposal_parses_grounded_actions_and_finish() -> None:
    client = StubOpenAICompatibleCUA(
        {
            "actions": [
                {"kind": "click", "target_ref": "12"},
                {"kind": "finish"},
            ],
            "done": False,
            "summary": "submitted",
        }
    )

    proposal = await client.propose(
        system_prompt="browser specialist",
        subgoal=Subgoal("task", "Click Submit", domain="browser"),
        observation=browser_observation(),
        policy=FastPolicy(),
    )

    assert proposal.done
    assert [action.kind for action in proposal.actions] == ["click"]
    assert proposal.actions[0].target_ref == "12"
    assert proposal.usage.model_calls == 1
    assert proposal.usage.input_tokens == 20


@pytest.mark.asyncio
async def test_proposal_rejects_unknown_refs_and_oversized_batches() -> None:
    unknown = StubOpenAICompatibleCUA(
        {"actions": [{"kind": "click", "target_ref": "missing"}], "done": False}
    )
    with pytest.raises(ValueError, match="unknown target_ref"):
        await unknown.propose(
            system_prompt="browser specialist",
            subgoal=Subgoal("task", "Click Submit", domain="browser"),
            observation=browser_observation(),
            policy=FastPolicy(),
        )

    oversized = StubOpenAICompatibleCUA({"actions": [{"kind": "noop"}] * 4, "done": False})
    with pytest.raises(ValueError, match="maximum is 3"):
        await oversized.propose(
            system_prompt="browser specialist",
            subgoal=Subgoal("task", "Click Submit", domain="browser"),
            observation=browser_observation(),
            policy=FastPolicy(),
        )


@pytest.mark.asyncio
async def test_proposal_requires_boolean_completion_state() -> None:
    client = StubOpenAICompatibleCUA({"actions": [], "done": "yes"})
    with pytest.raises(ValueError, match="must be a boolean"):
        await client.propose(
            system_prompt="browser specialist",
            subgoal=Subgoal("task", "Click Submit", domain="browser"),
            observation=browser_observation(),
            policy=FastPolicy(),
        )
