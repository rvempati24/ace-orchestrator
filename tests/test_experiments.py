import pytest

from ace_orchestrator.experiments.oracle_router import OfflineOutcome, best_contract
from ace_orchestrator.experiments.specialization_matrix import run_specialization_matrix
from ace_orchestrator.mock_setup import DOMAINS, mock_experts
from ace_orchestrator.policies.configured import MediumPolicy


@pytest.mark.asyncio
async def test_specialization_matrix_shape_and_export(tmp_path) -> None:
    matrix = await run_specialization_matrix(
        experts=mock_experts(seed=100),
        domains=DOMAINS,
        policy=MediumPolicy(),
        trials_per_cell=2,
    )
    assert set(matrix.scores) == set(DOMAINS)
    assert all(set(row) == set(DOMAINS) for row in matrix.scores.values())
    matrix.export_json(tmp_path / "matrix.json")
    matrix.export_csv(tmp_path / "matrix.csv")
    assert (tmp_path / "matrix.json").is_file()
    assert (tmp_path / "matrix.csv").read_text().startswith("task_domain")


def test_oracle_can_trade_success_for_cost() -> None:
    outcomes = (
        OfflineOutcome("web", "large", "deep", 0.95, 4, 2),
        OfflineOutcome("web", "small", "fast", 0.85, 1, 0.1),
    )
    assert best_contract(outcomes).expert_id == "large"
    assert best_contract(outcomes, cost_weight=0.2).expert_id == "small"
