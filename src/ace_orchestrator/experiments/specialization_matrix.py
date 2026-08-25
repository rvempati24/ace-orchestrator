from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ace_orchestrator.benchmarks.base import BenchmarkTask
from ace_orchestrator.core.models import (
    AutonomyHorizon,
    ExecutionState,
    ExperimentBudget,
    Subgoal,
    Usage,
)
from ace_orchestrator.execution.environment import EnvironmentFactory, MockEnvironmentFactory
from ace_orchestrator.experts.base import Expert
from ace_orchestrator.policies.base import Policy


@dataclass(frozen=True)
class SpecializationMatrix:
    domains: tuple[str, ...]
    expert_ids: tuple[str, ...]
    scores: dict[str, dict[str, float]]
    trials_per_cell: int

    def to_dict(self) -> dict:
        return {
            "domains": self.domains,
            "expert_ids": self.expert_ids,
            "scores": self.scores,
            "trials_per_cell": self.trials_per_cell,
        }

    def diagonal_advantage(self) -> float:
        differences = []
        for domain in self.domains:
            if domain not in self.expert_ids:
                continue
            diagonal = self.scores[domain][domain]
            off_diagonal = [
                score for expert_id, score in self.scores[domain].items() if expert_id != domain
            ]
            if off_diagonal:
                differences.append(diagonal - sum(off_diagonal) / len(off_diagonal))
        return sum(differences) / len(differences) if differences else 0.0

    def export_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def export_csv(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task_domain", *self.expert_ids])
            for domain in self.domains:
                writer.writerow(
                    [domain, *(self.scores[domain][expert] for expert in self.expert_ids)]
                )


async def run_specialization_matrix(
    *,
    experts: tuple[Expert, ...],
    domains: tuple[str, ...],
    policy: Policy,
    trials_per_cell: int = 10,
    budget: ExperimentBudget | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> SpecializationMatrix:
    if trials_per_cell < 1:
        raise ValueError("trials_per_cell must be positive")
    guard = budget or ExperimentBudget()
    factory = environment_factory if environment_factory is not None else MockEnvironmentFactory()
    usage = Usage()
    scores: dict[str, dict[str, float]] = {}
    horizon = AutonomyHorizon(max_actions=policy.config.suggested_action_horizon)
    for domain in domains:
        scores[domain] = {}
        for expert in experts:
            successes = 0
            for trial in range(trials_per_cell):
                if usage.model_calls >= guard.max_model_calls:
                    guard.check(Usage(model_calls=usage.model_calls + 1))
                projected_cost = usage.estimated_cost_usd + policy.config.simulated_cost_usd
                if projected_cost > guard.max_estimated_cost_usd:
                    guard.check(Usage(estimated_cost_usd=projected_cost))
                subgoal = Subgoal(
                    f"{domain}-{trial}",
                    f"Complete a {domain} benchmark task",
                    domain=domain,
                )
                task = BenchmarkTask(
                    f"matrix:{domain}:{trial}",
                    subgoal.description,
                    domain,
                    {"trial": trial},
                )
                environment = await factory.create(task)
                async with environment:
                    result = await expert.execute(
                        subgoal,
                        ExecutionState(),
                        policy,
                        horizon,
                        environment,
                    )
                usage = usage + result.usage
                guard.check(usage)
                successes += int(result.success)
            scores[domain][expert.expert_id] = successes / trials_per_cell
    return SpecializationMatrix(
        domains,
        tuple(expert.expert_id for expert in experts),
        scores,
        trials_per_cell,
    )
