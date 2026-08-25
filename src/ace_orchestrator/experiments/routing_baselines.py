from __future__ import annotations

from dataclasses import asdict, dataclass

from ace_orchestrator.orchestration.orchestrator import Orchestrator
from ace_orchestrator.telemetry.metrics import AggregateMetrics, aggregate


@dataclass(frozen=True)
class BaselineResult:
    name: str
    metrics: AggregateMetrics
    trajectories: tuple[dict, ...]

    def summary(self) -> dict:
        return {"name": self.name, "metrics": asdict(self.metrics)}


async def run_baseline(
    name: str, tasks: tuple[str, ...], orchestrator: Orchestrator
) -> BaselineResult:
    trajectories = tuple((await orchestrator.run(task)).trajectory for task in tasks)
    return BaselineResult(name, aggregate(trajectories), trajectories)


async def compare_routing_baselines(
    tasks: tuple[str, ...], orchestrators: dict[str, Orchestrator]
) -> tuple[BaselineResult, ...]:
    """Compare generalist/static/LLM/oracle orchestrators built with identical executors."""

    results = []
    for name, orchestrator in orchestrators.items():
        results.append(await run_baseline(name, tasks, orchestrator))
    return tuple(results)
