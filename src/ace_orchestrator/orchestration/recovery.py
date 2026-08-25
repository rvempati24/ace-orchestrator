from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RecoveryAction(StrEnum):
    RETRY_STRONGER = "retry_stronger_policy"
    REROUTE = "reroute_expert"
    REPLAN = "return_to_planner"
    STOP = "stop"


@dataclass(frozen=True)
class RecoveryPolicy:
    retries_with_stronger_policy: int = 1
    reroutes: int = 1

    def decide(self, retry_count: int, reroute_count: int) -> RecoveryAction:
        if retry_count < self.retries_with_stronger_policy:
            return RecoveryAction.RETRY_STRONGER
        if reroute_count < self.reroutes:
            return RecoveryAction.REROUTE
        return RecoveryAction.REPLAN


def stronger_policy(current: str, available: tuple[str, ...]) -> str:
    order = ("fast", "medium", "deep")
    if current not in order:
        return current
    for candidate in order[order.index(current) + 1 :]:
        if candidate in available:
            return candidate
    return current
