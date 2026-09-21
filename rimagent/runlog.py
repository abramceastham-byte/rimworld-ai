"""Save state snapshots and decisions as JSON lines under logs/.

Each run gets its own file, logs/run-YYYYMMDD-HHMMSS.jsonl. Every line is one
JSON object with a "kind" ("state" or "decision"), a UTC timestamp, and the
payload. One object per line means you can read a log with a few lines of
Python, or with tools like jq, even if a run crashed halfway through.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RunLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"run-{started}.jsonl"

    def _write(self, kind: str, payload: Any) -> None:
        if isinstance(payload, BaseModel):
            payload = payload.model_dump()
        record = {
            "kind": kind,
            "time": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        }
        # Open, append, close on every write so each line reaches disk immediately.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_state(self, state: Any) -> None:
        """Record a game state snapshot (a GameState or any JSON-friendly value)."""
        self._write("state", state)

    def log_decision(self, decision: Any) -> None:
        """Record what the model decided (raw text, a dict, or a pydantic model)."""
        self._write("decision", decision)


def read_log(path: str | Path) -> list[dict]:
    """Load every record from a .jsonl log file."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
