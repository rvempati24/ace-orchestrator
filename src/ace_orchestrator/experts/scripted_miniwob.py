"""Deterministic MiniWoB expert used to acceptance-test the real browser harness."""

from __future__ import annotations

import re
from typing import Any

from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.core.models import (
    AutonomyHorizon,
    ExecutionState,
    ExpertResult,
    Subgoal,
)
from ace_orchestrator.execution.environment import EnvironmentObservation, EnvironmentSession
from ace_orchestrator.experts.base import Expert
from ace_orchestrator.policies.base import Policy


def _attribute(node: dict[str, Any], key: str) -> Any:
    value = node.get(key)
    return value.get("value") if isinstance(value, dict) else value


def _nodes(observation: EnvironmentObservation) -> list[dict[str, Any]]:
    semantic_state = observation.values.get("semantic_state", {})
    rows = semantic_state.get("nodes", []) if isinstance(semantic_state, dict) else []
    return [row for row in rows if isinstance(row, dict) and row.get("browsergym_id")]


def _find_node(
    nodes: list[dict[str, Any]], role: str, *, name: str | None = None
) -> dict[str, Any]:
    for node in nodes:
        if _attribute(node, "role") != role:
            continue
        node_name = str(_attribute(node, "name") or "")
        if name is None or node_name.casefold() == name.casefold():
            return node
    raise ValueError(f"could not find observed {role!r} element named {name!r}")


def _quoted_value(goal: str) -> str:
    match = re.search(r'"([^"]+)"', goal)
    if not match:
        raise ValueError(f"goal has no quoted value: {goal}")
    return match.group(1)


class ScriptedMiniWoBExpert(Expert):
    """Solve five representative primitives from accessibility-tree observations only."""

    supported_tasks = frozenset(
        {"click-test", "click-button", "enter-text", "choose-list", "copy-paste"}
    )

    def __init__(self, expert_id: str = "browser") -> None:
        super().__init__(expert_id, "Deterministic MiniWoB acceptance expert", ("browser",))

    def _plan_actions(
        self, task_name: str, observation: EnvironmentObservation
    ) -> list[ComputerAction]:
        nodes = _nodes(observation)
        goal = str(observation.values.get("goal") or "")

        if task_name == "click-test":
            button = _find_node(nodes, "button")
            return [ComputerAction("click", target_ref=str(button["browsergym_id"]))]
        if task_name == "click-button":
            button = _find_node(nodes, "button", name=_quoted_value(goal))
            return [ComputerAction("click", target_ref=str(button["browsergym_id"]))]
        if task_name == "enter-text":
            field = _find_node(nodes, "textbox")
            submit = _find_node(nodes, "button", name="Submit")
            return [
                ComputerAction(
                    "fill", target_ref=str(field["browsergym_id"]), value=_quoted_value(goal)
                ),
                ComputerAction("click", target_ref=str(submit["browsergym_id"])),
            ]
        if task_name == "choose-list":
            selection = re.match(r"Select (.+) from the list", goal)
            if not selection:
                raise ValueError(f"could not parse list choice from goal: {goal}")
            field = _find_node(nodes, "combobox")
            submit = _find_node(nodes, "button", name="Submit")
            return [
                ComputerAction(
                    "select_option",
                    target_ref=str(field["browsergym_id"]),
                    value=selection.group(1),
                ),
                ComputerAction("click", target_ref=str(submit["browsergym_id"])),
            ]
        if task_name == "copy-paste":
            textboxes = [node for node in nodes if _attribute(node, "role") == "textbox"]
            source = next((node for node in textboxes if _attribute(node, "value")), None)
            destination = next((node for node in textboxes if not _attribute(node, "value")), None)
            if source is None or destination is None:
                raise ValueError(
                    "copy-paste observation did not expose source and destination fields"
                )
            submit = _find_node(nodes, "button", name="Submit")
            return [
                ComputerAction(
                    "fill",
                    target_ref=str(destination["browsergym_id"]),
                    value=str(_attribute(source, "value")),
                ),
                ComputerAction("click", target_ref=str(submit["browsergym_id"])),
            ]
        raise ValueError(f"unsupported scripted MiniWoB task: {task_name}")

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        task_name = str(environment.task.metadata.get("task_name", ""))
        if task_name not in self.supported_tasks:
            return ExpertResult(False, {}, error=f"unsupported scripted MiniWoB task: {task_name}")
        observation = await environment.observe()
        try:
            actions = self._plan_actions(task_name, observation)
        except ValueError as error:
            return ExpertResult(False, {}, error=str(error))
        if len(actions) > horizon.action_limit:
            return ExpertResult(False, {}, error="autonomy horizon is shorter than scripted plan")

        records = []
        for action in actions:
            outcome = await environment.act(action, before=observation)
            records.append(outcome.record)
            observation = outcome.observation
            if not outcome.record.success:
                return ExpertResult(False, {}, tuple(records), error=outcome.record.error)
        return ExpertResult(
            True,
            {
                "task_name": task_name,
                "goal": observation.values.get("goal"),
                "environment_metrics": environment.metrics_snapshot(),
            },
            tuple(records),
        )
