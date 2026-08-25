import pytest

from ace_orchestrator.experts.mock import MockExpert
from ace_orchestrator.experts.registry import ExpertRegistry


def test_registry_rejects_duplicate_ids() -> None:
    registry = ExpertRegistry()
    registry.register(MockExpert("web", "web"))
    with pytest.raises(ValueError):
        registry.register(MockExpert("web", "web"))


def test_registry_exposes_alternatives() -> None:
    registry = ExpertRegistry()
    registry.register(MockExpert("web", "web"))
    registry.register(MockExpert("general", "general"))
    assert [expert.expert_id for expert in registry.alternatives("web")] == ["general"]
