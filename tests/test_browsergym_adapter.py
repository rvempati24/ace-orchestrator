import pytest

from ace_orchestrator.benchmarks import BenchmarkTask, BrowserGymTask
from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.execution.browsergym import (
    BrowserGymEnvironmentFactory,
    computer_action_to_browsergym,
)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (ComputerAction("click", target_ref="12"), "click('12')"),
        (ComputerAction("fill", target_ref="9", value="O'Reilly"), "fill('9', \"O'Reilly\")"),
        (ComputerAction("select", target_ref="4", value="blue"), "select_option('4', 'blue')"),
        (ComputerAction("press", target_ref="5", value="Enter"), "press('5', 'Enter')"),
        (
            ComputerAction("scroll", metadata={"delta_x": 0, "delta_y": 250}),
            "scroll(0.0, 250.0)",
        ),
        (ComputerAction("navigate", target="https://example.com"), "goto('https://example.com')"),
    ],
)
def test_computer_actions_map_to_browsergym_calls(action: ComputerAction, expected: str) -> None:
    assert computer_action_to_browsergym(action) == expected


def test_browsergym_action_mapping_rejects_unsupported_or_unsafe_actions() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        computer_action_to_browsergym(ComputerAction("shell", value="whoami"))
    with pytest.raises(ValueError, match="http"):
        computer_action_to_browsergym(ComputerAction("navigate", target="file:///etc/passwd"))


def test_miniwob_task_has_reproducible_environment_identity() -> None:
    task = BrowserGymTask.miniwob("click-test", seed=7)
    assert task.environment_id == "browsergym/miniwob.click-test"
    assert task.task_id == "miniwob-click-test-seed-7"
    assert task.metadata["benchmark"] == "miniwob"


@pytest.mark.asyncio
async def test_browsergym_factory_requires_explicit_environment_metadata() -> None:
    factory = BrowserGymEnvironmentFactory(miniwob_url="file:///unused/")
    with pytest.raises(TypeError, match="BrowserGymTask"):
        await factory.create(BenchmarkTask("task", "instruction"))
