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


@dataclass(frozen=True)
class BrowserGymTask(BenchmarkTask):
    """A benchmark episode backed by a registered BrowserGym Gymnasium environment."""

    environment_id: str = "browsergym/miniwob.click-test"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.environment_id.startswith("browsergym/"):
            raise ValueError("BrowserGym environment ids must start with 'browsergym/'")
        if self.seed < 0:
            raise ValueError("BrowserGym seed cannot be negative")

    @classmethod
    def miniwob(cls, name: str, *, seed: int = 0, task_id: str | None = None) -> BrowserGymTask:
        environment_id = f"browsergym/miniwob.{name}"
        return cls(
            task_id=task_id or f"miniwob-{name}-seed-{seed}",
            instruction=f"Complete the {name} MiniWoB task",
            domain="browser",
            metadata={"benchmark": "miniwob", "task_name": name},
            environment_id=environment_id,
            seed=seed,
        )
