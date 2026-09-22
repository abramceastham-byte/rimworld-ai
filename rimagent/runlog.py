"""Save state snapshots and decisions as JSON lines under logs/.

Each run gets its own folder, named by local start time and a label such as
the model and think setting:

    logs/runs/2026-09-22_0630_gpt-oss-20b_think-medium/run.jsonl

Every line of run.jsonl is one JSON object with a "kind" (such as "run",
"state", "event", "decision", or "model_failure"), a UTC timestamp, and the
payload. One object per line means you can read a log with a few lines of
Python, or with tools like jq, even if a run crashed halfway through.
Dry runs go to logs/dry-runs/ with the same kind of name.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def stamped_name(label: str | None = None, when: datetime | None = None) -> str:
    """"2026-09-22_0630_<label>": sorts by time, readable at a glance."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    if not label:
        return stamp
    # Keep names safe for any filesystem: "gpt-oss:20b" -> "gpt-oss-20b".
    return f"{stamp}_{re.sub(r'[^A-Za-z0-9._-]+', '-', label).strip('-')}"


class RunLogger:
    def __init__(self, log_dir: str | Path = "logs", label: str | None = None):
        runs = Path(log_dir) / "runs"
        folder = runs / stamped_name(label)
        n = 2
        while folder.exists():  # two runs started in the same second
            folder = runs / f"{stamped_name(label)}_{n}"
            n += 1
        folder.mkdir(parents=True)
        self.folder = folder
        self.path = folder / "run.jsonl"

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

    def log_run_info(self, info: Any) -> None:
        """Record how a run was set up (model, thinking, context size), once at the start."""
        self._write("run", info)

    def log_state(self, state: Any) -> None:
        """Record a game state snapshot (a GameState or any JSON-friendly value)."""
        self._write("state", state)

    def log_event(self, event: Any) -> None:
        """Record a letter or message the game sent (a GameEvent or dict)."""
        self._write("event", event)

    def log_decision(self, decision: Any) -> None:
        """Record what the model decided (raw text, a dict, or a pydantic model)."""
        self._write("decision", decision)

    def log_model_failure(self, failure: Any) -> None:
        """Record a failed model call or an invalid model response."""
        self._write("model_failure", failure)


def read_log(path: str | Path) -> list[dict]:
    """Load every record from a .jsonl log file."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
