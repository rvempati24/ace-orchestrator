from __future__ import annotations

from ace_orchestrator.core.models import (
    AutonomyHorizon,
    ExecutionState,
    ExpertResult,
    Subgoal,
    Usage,
)
from ace_orchestrator.execution.environment import ComputerEnvironment
from ace_orchestrator.experts.base import Expert
from ace_orchestrator.inference.base import CUABackend
from ace_orchestrator.policies.base import Policy


class PromptedCUAExpert(Expert):
    """Same CUA backend, specialized by prompt and registered capability metadata."""

    def __init__(
        self,
        expert_id: str,
        description: str,
        capabilities: tuple[str, ...],
        system_prompt: str,
        backend: CUABackend,
        environment: ComputerEnvironment,
    ) -> None:
        super().__init__(expert_id, description, capabilities)
        self.system_prompt = system_prompt
        self.backend = backend
        self.environment = environment

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
    ) -> ExpertResult:
        actions = []
        usage = Usage()
        summary = "autonomy horizon exhausted"
        for _ in range(horizon.action_limit):
            observation = await self.environment.observe()
            proposal = await self.backend.propose(
                system_prompt=self.system_prompt,
                subgoal=subgoal,
                observation=observation,
                policy=policy,
            )
            usage = usage + proposal.usage
            summary = proposal.summary
            for action in proposal.actions:
                await self.environment.apply(action)
                actions.append(action)
            if proposal.done:
                return ExpertResult(True, {"summary": summary}, tuple(actions), usage)
            if not proposal.actions:
                return ExpertResult(
                    False,
                    {"summary": summary},
                    tuple(actions),
                    usage,
                    "model neither completed the subgoal nor proposed an action",
                )
        return ExpertResult(False, {"summary": summary}, tuple(actions), usage, summary)
