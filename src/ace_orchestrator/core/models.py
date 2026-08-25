"""Stable, serializable models shared across orchestration components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from typing import Any
from uuid import uuid4


class ExecutionModality(StrEnum):
    MOCK = "mock"
    GUI = "gui"
    BROWSER = "browser"
    DOM = "dom"
    API = "api"
    CODE = "code"


@dataclass(frozen=True)
class AutonomyHorizon:
    """An explicit limit on how long an expert may retain control."""

    max_actions: int | None = None
    until_subgoal_complete: bool = False

    def __post_init__(self) -> None:
        if self.max_actions is None and not self.until_subgoal_complete:
            raise ValueError("set max_actions or until_subgoal_complete")
        if self.max_actions is not None and self.max_actions < 1:
            raise ValueError("max_actions must be positive")

    @property
    def action_limit(self) -> int:
        """A safety cap even for semantic horizons."""

        return self.max_actions or 25


@dataclass(frozen=True)
class Task:
    user_goal: str
    task_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Subgoal:
    subgoal_id: str
    description: str
    dependencies: tuple[str, ...] = ()
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    values: dict[str, Any] = field(default_factory=dict)
    completed_subgoals: list[str] = field(default_factory=list)
    failed_subgoals: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class ExecutionContract:
    subgoal: Subgoal
    expert_id: str
    policy_id: str
    autonomy_horizon: AutonomyHorizon
    modality: ExecutionModality
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionRecord:
    kind: str
    target: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    wall_clock_latency_s: float = 0.0
    model_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model_calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            wall_clock_latency_s=self.wall_clock_latency_s + other.wall_clock_latency_s,
            model_latency_s=self.model_latency_s + other.model_latency_s,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
            model_calls=self.model_calls + other.model_calls,
        )


@dataclass(frozen=True)
class ExpertResult:
    success: bool
    output: dict[str, Any]
    actions: tuple[ActionRecord, ...] = ()
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    feedback: str | None = None


@dataclass(frozen=True)
class ExpertSelection:
    expert_id: str
    candidates: tuple[str, ...]
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySelection:
    policy_id: str
    candidates: tuple[str, ...]
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentBudget:
    max_model_calls: int = 100
    max_estimated_cost_usd: float = 5.0

    def __post_init__(self) -> None:
        if self.max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        if self.max_estimated_cost_usd < 0:
            raise ValueError("max_estimated_cost_usd cannot be negative")

    def check(self, usage: Usage) -> None:
        if usage.model_calls > self.max_model_calls:
            raise BudgetExceeded("model-call budget exceeded")
        if usage.estimated_cost_usd > self.max_estimated_cost_usd:
            raise BudgetExceeded("estimated-cost budget exceeded")


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class OrchestrationResult:
    task_id: str
    success: bool
    state: ExecutionState
    usage: Usage
    trajectory: dict[str, Any]


def to_primitive(value: Any) -> Any:
    """Convert the public schema to JSON-safe primitives without a framework dependency."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value
