from __future__ import annotations

from ace_orchestrator.core.models import (
    AutonomyHorizon,
    ExecutionState,
    ExpertResult,
    Subgoal,
    Usage,
)
from ace_orchestrator.execution.environment import EnvironmentSession
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
    ) -> None:
        super().__init__(expert_id, description, capabilities)
        self.system_prompt = system_prompt
        self.backend = backend

    async def execute(
        self,
        subgoal: Subgoal,
        state: ExecutionState,
        policy: Policy,
        horizon: AutonomyHorizon,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        records = []
        usage = Usage()
        summary = "autonomy horizon exhausted"
        model_turns = 0
        while len(records) < horizon.action_limit:
            observation = await environment.observe()
            try:
                proposal = await self.backend.propose(
                    system_prompt=self.system_prompt,
                    subgoal=subgoal,
                    observation=observation,
                    policy=policy,
                )
            except Exception as error:
                return ExpertResult(
                    False,
                    {"summary": summary, "model_turns": model_turns},
                    tuple(records),
                    usage,
                    f"inference failed: {error}",
                )
            model_turns += 1
            usage = usage + proposal.usage
            summary = proposal.summary
            remaining_actions = horizon.action_limit - len(records)
            if len(proposal.actions) > remaining_actions:
                return ExpertResult(
                    False,
                    {"summary": summary, "model_turns": model_turns},
                    tuple(records),
                    usage,
                    "proposal exceeds the remaining autonomy action budget",
                )
            for action in proposal.actions:
                outcome = await environment.act(action, before=observation)
                records.append(outcome.record)
                observation = outcome.observation
                if not outcome.record.success:
                    return ExpertResult(
                        False,
                        {"summary": summary, "model_turns": model_turns},
                        tuple(records),
                        usage,
                        outcome.record.error,
                    )
            if proposal.done:
                return ExpertResult(
                    True,
                    {"summary": summary, "model_turns": model_turns},
                    tuple(records),
                    usage,
                )
            if not proposal.actions:
                return ExpertResult(
                    False,
                    {"summary": summary, "model_turns": model_turns},
                    tuple(records),
                    usage,
                    "model neither completed the subgoal nor proposed an action",
                )
        return ExpertResult(
            False,
            {"summary": summary, "model_turns": model_turns},
            tuple(records),
            usage,
            "autonomy action horizon exhausted",
        )
