from __future__ import annotations

from ace_orchestrator.core.models import ExecutionContract, ExecutionState, ExpertResult
from ace_orchestrator.execution.environment import EnvironmentSession
from ace_orchestrator.experts.registry import ExpertRegistry
from ace_orchestrator.policies.base import Policy


class Executor:
    def __init__(self, experts: ExpertRegistry, policies: dict[str, Policy]) -> None:
        self.experts = experts
        self.policies = policies

    async def execute(
        self,
        contract: ExecutionContract,
        state: ExecutionState,
        environment: EnvironmentSession,
    ) -> ExpertResult:
        expert = self.experts.get(contract.expert_id)
        try:
            policy = self.policies[contract.policy_id]
        except KeyError as error:
            raise KeyError(f"unknown policy: {contract.policy_id}") from error
        return await expert.execute(
            contract.subgoal,
            state,
            policy,
            contract.autonomy_horizon,
            environment,
        )
