from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlTrajectoryLogger:
    """Append one complete run per line. The schema version makes evolution explicit."""

    schema_version = "0.1.0"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None

    def write(self, trajectory: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory, sort_keys=True) + "\n")
