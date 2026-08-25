from __future__ import annotations

import json
from abc import ABC, abstractmethod

from ace_orchestrator.core.models import ExecutionState, ExpertSelection, Subgoal
from ace_orchestrator.experts.base import Expert
from ace_orchestrator.orchestration.planner import StructuredTextModel


class ExpertRouter(ABC):
    @abstractmethod
    async def route(
        self, subgoal: Subgoal, state: ExecutionState, experts: tuple[Expert, ...]
    ) -> ExpertSelection:
        raise NotImplementedError


class StaticRouter(ExpertRouter):
    async def route(
        self, subgoal: Subgoal, state: ExecutionState, experts: tuple[Expert, ...]
    ) -> ExpertSelection:
        if not experts:
            raise ValueError("no experts registered")
        candidates = tuple(expert.expert_id for expert in experts)
        exact = next(
            (expert for expert in experts if subgoal.domain in expert.capabilities),
            None,
        )
        generalist = next((expert for expert in experts if "general" in expert.capabilities), None)
        chosen = exact or generalist or experts[0]
        return ExpertSelection(
            chosen.expert_id,
            candidates,
            f"static capability match for domain={subgoal.domain}",
        )


class LLMRouter(ExpertRouter):
    def __init__(self, model: StructuredTextModel) -> None:
        self.model = model

    async def route(
        self, subgoal: Subgoal, state: ExecutionState, experts: tuple[Expert, ...]
    ) -> ExpertSelection:
        candidates = tuple(expert.expert_id for expert in experts)
        payload = await self.model.complete_json(
            "Select exactly one expert. Return expert_id and reasoning as JSON.",
            json.dumps(
                {
                    "subgoal": subgoal.description,
                    "domain": subgoal.domain,
                    "experts": [
                        {
                            "id": expert.expert_id,
                            "description": expert.description,
                            "capabilities": expert.capabilities,
                        }
                        for expert in experts
                    ],
                    "failures": state.failed_subgoals,
                }
            ),
            max_tokens=400,
        )
        expert_id = str(payload.get("expert_id", ""))
        if expert_id not in candidates:
            raise ValueError(f"router selected unknown expert: {expert_id}")
        return ExpertSelection(
            expert_id, candidates, str(payload.get("reasoning", "LLM selection")), payload
        )


class OracleRouter(ExpertRouter):
    """Offline-only router over precomputed expected success/cost outcomes."""

    def __init__(self, outcomes: dict[tuple[str, str], float]) -> None:
        self.outcomes = outcomes

    async def route(
        self, subgoal: Subgoal, state: ExecutionState, experts: tuple[Expert, ...]
    ) -> ExpertSelection:
        candidates = tuple(expert.expert_id for expert in experts)
        chosen = max(
            candidates, key=lambda expert_id: self.outcomes.get((subgoal.domain, expert_id), 0.0)
        )
        return ExpertSelection(
            chosen,
            candidates,
            "offline oracle chose the highest precomputed outcome",
            {"scores": {key: self.outcomes.get((subgoal.domain, key), 0.0) for key in candidates}},
        )
