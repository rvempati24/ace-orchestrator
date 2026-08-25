"""Benchmark-neutral task descriptions used to provision execution environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ace_orchestrator.core.models import Task


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    instruction: str
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: Task) -> BenchmarkTask:
        return cls(
            task.task_id, task.user_goal, str(task.metadata.get("domain", "general")), task.metadata
        )
