import pytest

from ace_orchestrator.core.models import AutonomyHorizon, ExperimentBudget, Usage


def test_autonomy_horizon_requires_a_limit() -> None:
    with pytest.raises(ValueError):
        AutonomyHorizon()


def test_semantic_horizon_has_a_safety_cap() -> None:
    assert AutonomyHorizon(until_subgoal_complete=True).action_limit == 25


def test_usage_addition_preserves_all_accounting() -> None:
    total = Usage(model_calls=1, input_tokens=10, estimated_cost_usd=0.1) + Usage(
        model_calls=2, output_tokens=5, estimated_cost_usd=0.2
    )
    assert total.model_calls == 3
    assert total.input_tokens == 10
    assert total.output_tokens == 5
    assert total.estimated_cost_usd == pytest.approx(0.3)


def test_budget_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        ExperimentBudget(max_model_calls=0)
    with pytest.raises(ValueError):
        ExperimentBudget(max_estimated_cost_usd=-0.01)
