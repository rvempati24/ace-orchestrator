"""Commands issued by experts and immutable records produced by environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComputerAction:
    """An unexecuted command proposed by a CUA or deterministic expert."""

    kind: str
    target_ref: str | None = None
    target: str | None = None
    value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("action kind cannot be empty")


@dataclass(frozen=True)
class ActionRecord:
    """The environment-assigned result of attempting one command."""

    action: ComputerAction
    started_at: str
    finished_at: str
    observation_before_id: str
    observation_after_id: str
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
