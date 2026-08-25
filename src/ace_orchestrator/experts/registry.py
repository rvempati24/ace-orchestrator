from ace_orchestrator.experts.base import Expert


class ExpertRegistry:
    def __init__(self) -> None:
        self._experts: dict[str, Expert] = {}

    def register(self, expert: Expert) -> None:
        if expert.expert_id in self._experts:
            raise ValueError(f"expert already registered: {expert.expert_id}")
        self._experts[expert.expert_id] = expert

    def get(self, expert_id: str) -> Expert:
        try:
            return self._experts[expert_id]
        except KeyError as error:
            raise KeyError(f"unknown expert: {expert_id}") from error

    def all(self) -> tuple[Expert, ...]:
        return tuple(self._experts.values())

    def alternatives(self, expert_id: str) -> tuple[Expert, ...]:
        return tuple(expert for key, expert in self._experts.items() if key != expert_id)
