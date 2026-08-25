import asyncio
import json

from ace_orchestrator.mock_setup import build_mock_orchestrator


async def main() -> None:
    result = await build_mock_orchestrator("trajectories/example.jsonl").run(
        "Research the companies and then enter the results into the spreadsheet."
    )
    print(json.dumps(result.trajectory, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
