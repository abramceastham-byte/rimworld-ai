"""Remember where the agent's zones are; RIMAPI only reports their size.

GET /map/zones gives a zone's name, type and cell count but never its
location, so a growing zone or stockpile is invisible to the model once
created. It then picks spots that overlap its own fields and is refused. The
agent therefore records each zone it makes (memory/zones.json, written by code,
not by the model) and shows the rectangles in the prompt.

Zones the player draws by hand are still unknown, so free spots are also
checked against the game before being offered.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rimagent.construction import Rect

ZoneKind = Literal["growing", "stockpile"]


class TrackedZone(BaseModel):
    kind: ZoneKind
    zone_id: int
    label: str            # e.g. "Plant_Potato field" or "stockpile"
    cells: tuple[int, int, int, int]

    def area(self) -> Rect:
        return Rect(*self.cells)

    def describe(self) -> str:
        return f"{self.label} {self.area()}"


class ZoneTracker:
    def __init__(self, path: str | Path = Path("memory") / "zones.json"):
        self.path = Path(path)
        self.items: list[TrackedZone] = []
        if self.path.exists():
            self.items = [TrackedZone.model_validate(z)
                          for z in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([z.model_dump() for z in self.items], indent=2),
                             encoding="utf-8")

    def add(self, kind: ZoneKind, zone_id: int, label: str, area: Rect) -> None:
        self.items.append(TrackedZone(kind=kind, zone_id=zone_id, label=label,
                                      cells=(area.x1, area.z1, area.x2, area.z2)))
        self.save()

    def cells(self) -> set[tuple[int, int]]:
        """Every cell the agent's zones cover, to keep new work off them."""
        return {cell for zone in self.items for cell in zone.area()}
