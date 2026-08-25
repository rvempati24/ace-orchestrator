"""Run one real prompted-CUA MiniWoB episode against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ace_orchestrator.benchmarks import BrowserGymTask
from ace_orchestrator.core.models import ExecutionModality, Subgoal
from ace_orchestrator.execution.browsergym import BrowserGymEnvironmentFactory
from ace_orchestrator.experts import ExpertRegistry
from ace_orchestrator.experts.prompted_cua import PromptedCUAExpert
from ace_orchestrator.inference import OpenAICompatibleCUA
from ace_orchestrator.orchestration.expert_router import StaticRouter
from ace_orchestrator.orchestration.orchestrator import Orchestrator
from ace_orchestrator.orchestration.planner import Planner
from ace_orchestrator.orchestration.policy_router import FixedPolicyRouter
from ace_orchestrator.orchestration.recovery import RecoveryPolicy
from ace_orchestrator.policies.configured import default_policies
from ace_orchestrator.telemetry.logger import JsonlTrajectoryLogger
from ace_orchestrator.verification import BrowserGymVerifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Hcompany/Holo-3.1-35B-A3B"


class BrowserTaskPlanner(Planner):
    async def plan(self, task, state):
        return [Subgoal("browser-task", task.user_goal, domain="browser")]


def required_environment(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"set {name} before running the prompted BrowserGym example")
    return value


async def run(task_name: str, seed: int, output: Path) -> dict:
    base_url = required_environment("ACE_CUA_BASE_URL")
    miniwob_url = required_environment("MINIWOB_URL")
    backend = OpenAICompatibleCUA(
        base_url=base_url,
        model_id=os.getenv("ACE_CUA_MODEL_ID", DEFAULT_MODEL_ID),
        api_key=os.getenv("ACE_CUA_API_KEY"),
    )
    prompt = (PROJECT_ROOT / "prompts" / "generalist.md").read_text(encoding="utf-8")
    expert = PromptedCUAExpert(
        "browser-generalist",
        "General prompted browser CUA",
        ("browser",),
        prompt,
        backend,
    )
    registry = ExpertRegistry()
    registry.register(expert)
    orchestrator = Orchestrator(
        planner=BrowserTaskPlanner(),
        expert_router=StaticRouter(),
        policy_router=FixedPolicyRouter("medium"),
        experts=registry,
        policies=default_policies(backend.model_id),
        verifier=BrowserGymVerifier(),
        environment_factory=BrowserGymEnvironmentFactory(miniwob_url=miniwob_url),
        recovery=RecoveryPolicy(retries_with_stronger_policy=0, reroutes=0),
        logger=JsonlTrajectoryLogger(output),
        modality=ExecutionModality.BROWSER,
    )
    result = await orchestrator.run_benchmark(BrowserGymTask.miniwob(task_name, seed=seed))
    return {
        "task_id": result.task_id,
        "success": result.success,
        "usage": result.trajectory["usage"],
        "environment_metrics": result.trajectory["environment_metrics"],
        "trajectory": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="click-test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("trajectories/phase3-smoke.jsonl"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.task, args.seed, args.output)), indent=2))


if __name__ == "__main__":
    main()
