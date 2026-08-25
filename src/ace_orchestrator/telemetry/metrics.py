from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class AggregateMetrics:
    task_success_rate: float
    subgoal_success_rate: float
    mean_latency_s: float
    mean_estimated_cost_usd: float
    mean_actions: float
    mean_retries: float


def aggregate(trajectories: Iterable[dict]) -> AggregateMetrics:
    runs = list(trajectories)
    steps = [step for run in runs for step in run.get("subgoals", [])]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return AggregateMetrics(
        mean([float(run.get("final_task_success", False)) for run in runs]),
        mean([float(step.get("success", False)) for step in steps]),
        mean([float(run.get("usage", {}).get("wall_clock_latency_s", 0)) for run in runs]),
        mean([float(run.get("usage", {}).get("estimated_cost_usd", 0)) for run in runs]),
        mean(
            [
                float(sum(len(attempt.get("actions", [])) for attempt in step.get("attempts", [])))
                for step in steps
            ]
        ),
        mean([float(step.get("retry_count", 0)) for step in steps]),
    )
