from __future__ import annotations

from abc import ABC, abstractmethod

from ace_orchestrator.core.models import ExecutionState, PolicySelection, Subgoal
from ace_orchestrator.experts.base import Expert
from ace_orchestrator.policies.base import Policy


class PolicyRouter(ABC):
    @abstractmethod
    async def route_policy(
        self,
        expert: Expert,
        subgoal: Subgoal,
        state: ExecutionState,
        policies: dict[str, Policy],
    ) -> PolicySelection:
        raise NotImplementedError


class FixedPolicyRouter(PolicyRouter):
    def __init__(self, policy_id: str = "medium") -> None:
        self.policy_id = policy_id

    async def route_policy(
        self, expert: Expert, subgoal: Subgoal, state: ExecutionState, policies: dict[str, Policy]
    ) -> PolicySelection:
        if self.policy_id not in policies:
            raise KeyError(f"unknown fixed policy: {self.policy_id}")
        return PolicySelection(
            self.policy_id,
            tuple(policies),
            "fixed-policy baseline",
        )


class HeuristicPolicyRouter(PolicyRouter):
    async def route_policy(
        self, expert: Expert, subgoal: Subgoal, state: ExecutionState, policies: dict[str, Policy]
    ) -> PolicySelection:
        if subgoal.subgoal_id in state.failed_subgoals:
            choice, reason = "deep", "failure/recovery uses more compute"
        elif not state.history:
            choice, reason = "medium", "unknown initial state uses medium compute"
        else:
            choice, reason = "fast", "routine continuation uses fast compute"
        if choice not in policies:
            choice = next(iter(policies))
            reason += "; requested tier unavailable"
        return PolicySelection(choice, tuple(policies), reason)


class OraclePolicyRouter(PolicyRouter):
    def __init__(self, outcomes: dict[tuple[str, str, str], float]) -> None:
        self.outcomes = outcomes

    async def route_policy(
        self, expert: Expert, subgoal: Subgoal, state: ExecutionState, policies: dict[str, Policy]
    ) -> PolicySelection:
        choice = max(
            policies,
            key=lambda policy_id: self.outcomes.get(
                (subgoal.domain, expert.expert_id, policy_id), 0.0
            ),
        )
        return PolicySelection(choice, tuple(policies), "offline oracle policy", {})
