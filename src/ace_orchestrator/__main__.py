from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ace_orchestrator.experiments.specialization_matrix import run_specialization_matrix
from ace_orchestrator.mock_setup import DOMAINS, build_mock_orchestrator, mock_experts
from ace_orchestrator.policies.configured import MediumPolicy


async def demo(output: Path) -> None:
    orchestrator = build_mock_orchestrator(output)
    result = await orchestrator.run(
        "Research these companies and then enter the results into the spreadsheet."
    )
    print(json.dumps(result.trajectory, indent=2))


async def specialization(json_path: Path, csv_path: Path, trials: int) -> None:
    matrix = await run_specialization_matrix(
        experts=mock_experts(),
        domains=DOMAINS,
        policy=MediumPolicy(),
        trials_per_cell=trials,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.export_json(json_path)
    matrix.export_csv(csv_path)
    print(
        json.dumps(
            {**matrix.to_dict(), "diagonal_advantage": matrix.diagonal_advantage()}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ace Orchestrator research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="run one local mock trajectory")
    demo_parser.add_argument("--output", type=Path, default=Path("trajectories/demo.jsonl"))
    matrix_parser = subparsers.add_parser(
        "specialization", help="run the mock specialization matrix"
    )
    matrix_parser.add_argument("--json", type=Path, default=Path("results/specialization.json"))
    matrix_parser.add_argument("--csv", type=Path, default=Path("results/specialization.csv"))
    matrix_parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(demo(args.output))
    elif args.command == "specialization":
        asyncio.run(specialization(args.json, args.csv, args.trials))


if __name__ == "__main__":
    main()
