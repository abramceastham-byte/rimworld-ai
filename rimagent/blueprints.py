"""Keep track of the blueprints the agent placed, and check on them in the game.

RIMAPI can't list the blueprints on a map, but it can say what is on one cell.
So the agent remembers *where* it placed each blueprint (memory/blueprints.json,
written by code, not by the model) and asks the game each step *what is there
now*. Status never comes from memory:

    waiting             a blueprint: nobody has started building it
    under construction  a frame: building has started
    built               the finished building is there
    gone                nothing: cancelled, or destroyed

Built and gone blueprints are reported once, then dropped from the list.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from rimagent.construction import Rect

if TYPE_CHECKING:
    from rimagent.rimapi import RimApiClient

Status = Literal["waiting", "under construction", "built", "gone"]


class TrackedBlueprint(BaseModel):
    def_name: str
    stuff: str | None = None
    x: int
    z: int
    rotation: int = 0
    floor: bool = False
    cells: tuple[int, int, int, int]  # x1, z1, x2, z2 the building covers

    def area(self) -> Rect:
        return Rect(*self.cells)

    def label(self) -> str:
        material = f" ({self.stuff})" if self.stuff else ""
        return f"{self.def_name}{material} at ({self.x},{self.z})"


def status_from_things(def_name: str, things_here: list[str]) -> Status:
    """What a cell's contents say about one blueprint."""
    if f"Blueprint_{def_name}" in things_here:
        return "waiting"
    if f"Frame_{def_name}" in things_here:
        return "under construction"
    if def_name in things_here:
        return "built"
    return "gone"


class BlueprintTracker:
    def __init__(self, path: str | Path = Path("memory") / "blueprints.json"):
        self.path = Path(path)
        self.items: list[TrackedBlueprint] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = [TrackedBlueprint.model_validate(b) for b in data]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([b.model_dump() for b in self.items], indent=2), encoding="utf-8"
        )

    def add(self, def_name: str, stuff: str | None, x: int, z: int, rotation: int, area: Rect,
            floor: bool = False) -> None:
        self.items = [b for b in self.items if (b.def_name, b.x, b.z) != (def_name, x, z)]
        self.items.append(TrackedBlueprint(
            def_name=def_name, stuff=stuff, x=x, z=z, rotation=rotation,
            cells=(area.x1, area.z1, area.x2, area.z2), floor=floor,
        ))
        self.save()

    def check(
        self, client: "RimApiClient", map_id: int = 0, forget_finished: bool = True
    ) -> list[tuple[TrackedBlueprint, Status]]:
        """Ask the game about each tracked blueprint. Finished ones (built or
        gone) are returned this time and then forgotten, unless forget_finished
        is False (for dry runs, which must not change memory)."""
        report = []
        for bp in self.items:
            here = [t.def_name for t in client.get_things_at(bp.x, bp.z, map_id)]
            if bp.floor and client.get_terrain(map_id).at(bp.x, bp.z) == bp.def_name:
                here.append(bp.def_name)
            report.append((bp, status_from_things(bp.def_name, here)))
        still_open = [bp for bp, status in report if status in ("waiting", "under construction")]
        if forget_finished and len(still_open) != len(self.items):
            self.items = still_open
            self.save()
        return report
