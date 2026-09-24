"""Rock the agent marked for mining, and what became of it.

RIMAPI can't list designations and reports success whether or not a cell was
already marked, so a model re-marked the same few steel cells eleven times in
one run. This keeps the rock cells it marked (memory/mining.json) and checks
them against the ore map each turn: a cell no longer rock was mined.
"""

import json
from pathlib import Path


class MineTracker:
    def __init__(self, path: str | Path | None = None, read_only: bool = False):
        self.path = Path(path) if path else None
        self.read_only = read_only
        self.cells: dict[tuple[int, int], str] = {}  # cell -> what rock it is
        self.mined_since_last = 0
        if self.path and self.path.exists():
            self.cells = {(c["x"], c["z"]): c["name"]
                          for c in json.loads(self.path.read_text(encoding="utf-8"))}

    def save(self) -> None:
        if self.path and not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([{"x": x, "z": z, "name": n} for (x, z), n in self.cells.items()],
                                            indent=2), encoding="utf-8")

    def add(self, targets: list[dict]) -> None:
        self.cells.update({(t["x"], t["z"]): t["name"] for t in targets})
        self.save()

    def reconcile(self, rock: set[tuple[int, int]]) -> None:
        """Forget marked cells that are no longer rock: those were mined."""
        before = len(self.cells)
        self.cells = {c: n for c, n in self.cells.items() if c in rock}
        self.mined_since_last = before - len(self.cells)
        if self.mined_since_last:
            self.save()

    def describe(self, miners: list[str]) -> str | None:
        if not self.cells and not self.mined_since_last:
            return None
        kinds: dict[str, int] = {}
        for name in self.cells.values():
            kinds[name] = kinds.get(name, 0) + 1
        still = ", ".join(f"{n} {name}" for name, n in sorted(kinds.items(), key=lambda kv: -kv[1])) or "none"
        return (f"Mining you ordered: {self.mined_since_last} cells mined since last turn; still marked: {still}. "
                "Marking them again changes nothing. "
                + (f"Mining is on for: {', '.join(miners)}." if miners else
                   "Nobody has Mining on, so nothing gets mined: enable_work Mining."))
