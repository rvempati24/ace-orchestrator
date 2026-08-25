from __future__ import annotations

from pathlib import Path

from ace_orchestrator.experts.mock import MockExpert
from ace_orchestrator.experts.registry import ExpertRegistry
from ace_orchestrator.orchestration.expert_router import StaticRouter
from ace_orchestrator.orchestration.orchestrator import Orchestrator
from ace_orchestrator.orchestration.planner import DeterministicPlanner
from ace_orchestrator.orchestration.policy_router import HeuristicPolicyRouter
from ace_orchestrator.policies.configured import default_policies
from ace_orchestrator.telemetry.logger import JsonlTrajectoryLogger
from ace_orchestrator.verification.simple import ResultVerifier

DOMAINS = ("general", "web", "spreadsheet", "crm")


def mock_experts(*, seed: int = 7) -> tuple[MockExpert, ...]:
    return tuple(
        MockExpert(
            domain,
            domain,
            success_by_domain={
                candidate: (0.82 if domain == candidate else 0.74 if domain == "general" else 0.58)
                for candidate in DOMAINS
            },
            seed=seed + index,
        )
        for index, domain in enumerate(DOMAINS)
    )


def build_mock_orchestrator(
    trajectory_path: str | Path | None = None, *, seed: int = 7
) -> Orchestrator:
    registry = ExpertRegistry()
    for expert in mock_experts(seed=seed):
        registry.register(expert)
    return Orchestrator(
        planner=DeterministicPlanner(),
        expert_router=StaticRouter(),
        policy_router=HeuristicPolicyRouter(),
        experts=registry,
        policies=default_policies(),
        verifier=ResultVerifier(),
        logger=JsonlTrajectoryLogger(trajectory_path),
    )
