from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from ace_orchestrator.benchmarks.base import BenchmarkTask
from ace_orchestrator.core.models import (
    AutonomyHorizon,
    BudgetExceeded,
    ExecutionContract,
    ExecutionModality,
    ExecutionState,
    ExperimentBudget,
    OrchestrationResult,
    Task,
    Usage,
    to_primitive,
)
from ace_orchestrator.execution.environment import (
    EnvironmentFactory,
    EnvironmentSession,
    MockEnvironmentFactory,
)
from ace_orchestrator.execution.executor import Executor
from ace_orchestrator.experts.registry import ExpertRegistry
from ace_orchestrator.orchestration.expert_router import ExpertRouter
from ace_orchestrator.orchestration.planner import Planner
from ace_orchestrator.orchestration.policy_router import PolicyRouter
from ace_orchestrator.orchestration.recovery import RecoveryAction, RecoveryPolicy, stronger_policy
from ace_orchestrator.policies.base import Policy
from ace_orchestrator.telemetry.logger import JsonlTrajectoryLogger
from ace_orchestrator.verification.base import Verifier


class Orchestrator:
    def __init__(
        self,
        *,
        planner: Planner,
        expert_router: ExpertRouter,
        policy_router: PolicyRouter,
        experts: ExpertRegistry,
        policies: dict[str, Policy],
        verifier: Verifier,
        environment_factory: EnvironmentFactory | None = None,
        recovery: RecoveryPolicy | None = None,
        budget: ExperimentBudget | None = None,
        logger: JsonlTrajectoryLogger | None = None,
        modality: ExecutionModality = ExecutionModality.MOCK,
    ) -> None:
        self.planner = planner
        self.expert_router = expert_router
        self.policy_router = policy_router
        self.experts = experts
        self.policies = policies
        self.verifier = verifier
        self.environment_factory = (
            environment_factory if environment_factory is not None else MockEnvironmentFactory()
        )
        self.recovery = recovery or RecoveryPolicy()
        self.budget = budget or ExperimentBudget()
        self.logger = logger or JsonlTrajectoryLogger()
        self.modality = modality
        self.executor = Executor(experts, policies)

    async def run(
        self, task: str, initial_state: ExecutionState | None = None
    ) -> OrchestrationResult:
        """Run one isolated task episode and always close its environment."""

        started = perf_counter()
        root_task = Task(task)
        benchmark_task = BenchmarkTask.from_task(root_task)
        environment = await self.environment_factory.create(benchmark_task)
        async with environment:
            return await self._run_in_environment(
                root_task,
                deepcopy(initial_state) if initial_state is not None else ExecutionState(),
                environment,
                started,
            )

    async def _run_in_environment(
        self,
        root_task: Task,
        state: ExecutionState,
        environment: EnvironmentSession,
        started: float,
    ) -> OrchestrationResult:
        created_at = datetime.now(UTC).isoformat()
        subgoals = await self.planner.plan(root_task, state)
        self._validate_plan(subgoals)
        trajectory: dict[str, Any] = {
            "schema_version": self.logger.schema_version,
            "task_id": root_task.task_id,
            "user_goal": root_task.user_goal,
            "created_at": created_at,
            "environment": {
                "session_id": environment.session_id,
                "factory": type(self.environment_factory).__name__,
                "benchmark_task": to_primitive(environment.task),
            },
            "planner_decision": {
                "planner": type(self.planner).__name__,
                "subgoals": to_primitive(subgoals),
            },
            "available_experts": [
                {
                    "expert_id": expert.expert_id,
                    "description": expert.description,
                    "capabilities": list(expert.capabilities),
                }
                for expert in self.experts.all()
            ],
            "subgoals": [],
            "escalations": [],
        }
        usage = Usage()

        for subgoal in subgoals:
            if any(
                dependency not in state.completed_subgoals for dependency in subgoal.dependencies
            ):
                state.failed_subgoals.append(subgoal.subgoal_id)
                trajectory["subgoals"].append(
                    {
                        "subgoal_id": subgoal.subgoal_id,
                        "subgoal_description": subgoal.description,
                        "success": False,
                        "reason": "dependency not completed",
                        "environment_session_id": environment.session_id,
                        "attempts": [],
                        "retry_count": 0,
                        "reroute_count": 0,
                    }
                )
                continue

            selection = await self.expert_router.route(subgoal, state, self.experts.all())
            expert_id = selection.expert_id
            policy_selection = await self.policy_router.route_policy(
                self.experts.get(expert_id), subgoal, state, self.policies
            )
            policy_id = policy_selection.policy_id
            retry_count = 0
            reroute_count = 0
            attempts: list[dict[str, Any]] = []
            subgoal_success = False

            while True:
                self._check_before_call(usage, policy_id)
                horizon = AutonomyHorizon(
                    max_actions=self.policies[policy_id].config.suggested_action_horizon
                )
                contract = ExecutionContract(
                    subgoal,
                    expert_id,
                    policy_id,
                    horizon,
                    self.modality,
                    {
                        "retry_count": retry_count,
                        "reroute_count": reroute_count,
                        "environment_session_id": environment.session_id,
                    },
                )
                before = deepcopy(state)
                result = await self.executor.execute(contract, state, environment)
                usage = usage + result.usage
                self.budget.check(usage)
                if result.success:
                    state.values[subgoal.subgoal_id] = result.output
                verification = await self.verifier.verify(
                    subgoal,
                    before,
                    state,
                    result,
                    environment,
                )
                attempt = {
                    "execution_contract": to_primitive(contract),
                    "environment_session_id": environment.session_id,
                    "expert_candidates": list(selection.candidates),
                    "selected_expert": expert_id,
                    "expert_reasoning": (
                        selection.reasoning
                        if expert_id == selection.expert_id
                        else "bounded recovery reroute after failed verification"
                    ),
                    "policy_candidates": list(policy_selection.candidates),
                    "selected_policy": policy_id,
                    "policy_reasoning": (
                        policy_selection.reasoning
                        if policy_id == policy_selection.policy_id
                        else "bounded recovery escalation to a stronger policy"
                    ),
                    "autonomy_horizon": to_primitive(horizon),
                    "modality": self.modality.value,
                    "start_state": before.snapshot(),
                    "end_state": state.snapshot(),
                    "actions": to_primitive(result.actions),
                    "usage": to_primitive(result.usage),
                    "verification_result": to_primitive(verification),
                    "error": result.error,
                }
                attempts.append(attempt)
                state.history.append(
                    {
                        "subgoal_id": subgoal.subgoal_id,
                        "expert_id": expert_id,
                        "policy_id": policy_id,
                        "environment_session_id": environment.session_id,
                        "verified": verification.success,
                    }
                )
                if verification.success:
                    if subgoal.subgoal_id in state.failed_subgoals:
                        state.failed_subgoals.remove(subgoal.subgoal_id)
                    state.completed_subgoals.append(subgoal.subgoal_id)
                    subgoal_success = True
                    break

                if subgoal.subgoal_id not in state.failed_subgoals:
                    state.failed_subgoals.append(subgoal.subgoal_id)
                recovery_action = self.recovery.decide(retry_count, reroute_count)
                escalation = {
                    "subgoal_id": subgoal.subgoal_id,
                    "action": recovery_action.value,
                    "from_expert": expert_id,
                    "from_policy": policy_id,
                    "environment_session_id": environment.session_id,
                    "state_strategy": "continue_in_place",
                }
                trajectory["escalations"].append(escalation)
                if recovery_action is RecoveryAction.RETRY_STRONGER:
                    retry_count += 1
                    policy_id = stronger_policy(policy_id, tuple(self.policies))
                    continue
                if recovery_action is RecoveryAction.REROUTE:
                    alternatives = self.experts.alternatives(expert_id)
                    if not alternatives:
                        break
                    reroute_count += 1
                    expert_id = alternatives[0].expert_id
                    policy_id = "deep" if "deep" in self.policies else policy_id
                    escalation["to_expert"] = expert_id
                    escalation["to_policy"] = policy_id
                    continue
                # V0 returns control to the planner as a logged terminal escalation.
                break

            trajectory["subgoals"].append(
                {
                    "subgoal_id": subgoal.subgoal_id,
                    "subgoal_description": subgoal.description,
                    "domain": subgoal.domain,
                    "environment_session_id": environment.session_id,
                    "success": subgoal_success,
                    "attempts": attempts,
                    "retry_count": retry_count,
                    "reroute_count": reroute_count,
                }
            )

        success = bool(subgoals) and len(state.completed_subgoals) == len(subgoals)
        measured_latency = perf_counter() - started
        usage = Usage(
            wall_clock_latency_s=max(measured_latency, usage.wall_clock_latency_s),
            model_latency_s=usage.model_latency_s,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            model_calls=usage.model_calls,
        )
        trajectory.update(
            {
                "final_task_success": success,
                "final_verification": {
                    "success": success,
                    "completed": list(state.completed_subgoals),
                    "failed": list(state.failed_subgoals),
                },
                "usage": to_primitive(usage),
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        self.logger.write(trajectory)
        return OrchestrationResult(root_task.task_id, success, state, usage, trajectory)

    def _check_before_call(self, usage: Usage, policy_id: str) -> None:
        if usage.model_calls >= self.budget.max_model_calls:
            raise BudgetExceeded("model-call budget exhausted before next execution")
        projected_cost = (
            usage.estimated_cost_usd + self.policies[policy_id].config.simulated_cost_usd
        )
        if projected_cost > self.budget.max_estimated_cost_usd:
            raise BudgetExceeded("estimated-cost budget exhausted before next execution")

    @staticmethod
    def _validate_plan(subgoals: list) -> None:
        ids = [subgoal.subgoal_id for subgoal in subgoals]
        if len(ids) != len(set(ids)):
            raise ValueError("planner emitted duplicate subgoal ids")
        seen: set[str] = set()
        for subgoal in subgoals:
            if any(dependency not in seen for dependency in subgoal.dependencies):
                raise ValueError("subgoal dependencies must refer to earlier subgoals")
            seen.add(subgoal.subgoal_id)
