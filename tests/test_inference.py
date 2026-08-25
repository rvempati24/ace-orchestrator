import pytest

from ace_orchestrator.inference.openai_compatible import OpenAICompatibleCUA


def test_extract_json_accepts_fenced_model_output() -> None:
    assert OpenAICompatibleCUA._extract_json('```json\n{"actions": [], "done": true}\n```')["done"]


def test_extract_json_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleCUA._extract_json("[]")
