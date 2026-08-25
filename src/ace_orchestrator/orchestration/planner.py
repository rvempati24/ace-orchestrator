from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Protocol

from ace_orchestrator.core.models import ExecutionState, Subgoal, Task


class StructuredTextModel(Protocol):
    async def complete_json(self, system: str, user: str, *, max_tokens: int) -> dict: ...


class Planner(ABC):
    @abstractmethod
    async def plan(self, task: Task, state: ExecutionState) -> list[Subgoal]:
        raise NotImplementedError


def infer_domain(text: str) -> str:
    lowered = text.lower()
    keyword_domains = {
        "spreadsheet": ("spreadsheet", "sheet", "excel", "column", "cell"),
        "web": ("research", "web", "find", "search", "company", "funding"),
        "crm": ("crm", "salesforce", "hubspot", "lead", "contact"),
        "document": ("document", "doc", "report", "write", "draft"),
    }
    for domain, keywords in keyword_domains.items():
        if any(keyword in lowered for keyword in keywords):
            return domain
    return "general"


class DeterministicPlanner(Planner):
    """Dependency-aware local baseline with no API calls."""

    async def plan(self, task: Task, state: ExecutionState) -> list[Subgoal]:
        parts = [
            part.strip(" .")
            for part in re.split(r"\b(?:and then|then|and)\b", task.user_goal, flags=re.I)
        ]
        parts = [part for part in parts if part] or [task.user_goal]
        subgoals: list[Subgoal] = []
        for index, description in enumerate(parts):
            subgoal_id = f"subgoal-{index + 1}"
            dependencies = (subgoals[-1].subgoal_id,) if subgoals else ()
            subgoals.append(
                Subgoal(subgoal_id, description, dependencies, infer_domain(description))
            )
        return subgoals


class LLMPlanner(Planner):
    def __init__(self, model: StructuredTextModel) -> None:
        self.model = model

    async def plan(self, task: Task, state: ExecutionState) -> list[Subgoal]:
        payload = await self.model.complete_json(
            "Decompose the task into semantic, dependency-ordered subgoals. Return JSON only.",
            json.dumps({"task": task.user_goal, "state": state.snapshot()}),
            max_tokens=1_000,
        )
        rows = payload.get("subgoals")
        if not isinstance(rows, list) or not rows:
            raise ValueError("planner response must contain non-empty subgoals")
        return [
            Subgoal(
                str(row["id"]),
                str(row["description"]),
                tuple(str(item) for item in row.get("dependencies", [])),
                str(row.get("domain") or infer_domain(str(row["description"]))),
            )
            for row in rows
        ]
