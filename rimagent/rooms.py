"""Rooms the agent built, with the purpose it gave them, and what each still needs.

RimWorld decides a room's role from what is inside it, and RIMAPI can't say
which walls belong together. So the agent records each build_room with the
purpose the model chose (memory/rooms.json) and reports, every turn, how far
the walls are and what the purpose still calls for: a bedroom without a bed,
a kitchen without a stove. Without this a model walled in a potato field,
which the game then roofed, and nothing grew.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rimagent.construction import Rect

Purpose = Literal["bedroom", "barracks", "kitchen", "dining", "workshop", "storage", "hospital", "other"]

WORK_TABLES = ("TableButcher", "FueledStove", "ElectricStove", "SimpleResearchBench",
               "HandTailoringBench", "TableStonecutter")

# What each purpose needs inside: (label, any of these buildings, how many).
NEEDS: dict[str, list[tuple[str, tuple[str, ...], int]]] = {
    "bedroom": [("a bed", ("Bed", "DoubleBed"), 1)],
    "barracks": [("beds", ("Bed", "DoubleBed"), 2)],
    "hospital": [("a bed", ("Bed", "DoubleBed"), 1)],
    "kitchen": [("a stove", ("FueledStove", "ElectricStove"), 1)],
    "dining": [("a table", ("Table2x2c",), 1), ("chairs", ("DiningChair", "Stool"), 1)],
    "workshop": [("a work table", WORK_TABLES, 1)],
    "storage": [],   # a stockpile, which RIMAPI can't place by room; see NEXT
    "other": [],
}
NEXT = {
    "bedroom": "place a Bed inside",
    "barracks": "place several Beds inside",
    "hospital": "place a Bed inside",
    "kitchen": "place a FueledStove inside (and a TableButcher nearby for meat)",
    "dining": "place a Table2x2c and DiningChairs inside",
    "workshop": "place work tables inside",
    "storage": "create_stockpile over the inside, then Shelf blueprints along the walls",
    "other": "furnish it as you intended",
}


def inside_of(rect: Rect) -> Rect:
    return Rect(rect.x1 + 1, rect.z1 + 1, rect.x2 - 1, rect.z2 - 1)


def perimeter(rect: Rect) -> set[tuple[int, int]]:
    return {(x, z) for x, z in rect if x in (rect.x1, rect.x2) or z in (rect.z1, rect.z2)}


class TrackedRoom(BaseModel):
    purpose: Purpose
    door_side: str
    cells: tuple[int, int, int, int]  # outer rectangle, walls included

    def area(self) -> Rect:
        return Rect(*self.cells)


class RoomTracker:
    def __init__(self, path: str | Path | None = None, read_only: bool = False):
        self.path = Path(path) if path else None
        self.read_only = read_only
        self.items: list[TrackedRoom] = []
        if self.path and self.path.exists():
            self.items = [TrackedRoom.model_validate(r)
                          for r in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self) -> None:
        if self.path and not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([r.model_dump() for r in self.items], indent=2),
                                 encoding="utf-8")

    def add(self, rect: Rect, purpose: str, door_side: str) -> None:
        self.items = [r for r in self.items if r.area() != rect]
        self.items.append(TrackedRoom(purpose=purpose, door_side=door_side,
                                      cells=(rect.x1, rect.z1, rect.x2, rect.z2)))
        self.save()

    def describe(self, walls: dict[tuple[int, int], str], contents: list[tuple[str, Rect, str]]) -> list[str]:
        """One line per room, forgetting rooms whose walls are all gone.

        walls: cell -> "built" / "planned" for walls and doors (built,
        under construction or blueprint); contents: (def name, area, state)
        for buildings and blueprints anywhere, to find what is inside.
        """
        lines, kept = [], []
        for room in self.items:
            rect = room.area()
            ring = perimeter(rect)
            built = sum(1 for c in ring if walls.get(c) == "built")
            planned = sum(1 for c in ring if walls.get(c) == "planned")
            if not built and not planned:
                continue  # cancelled or destroyed
            kept.append(room)
            inside = inside_of(rect)
            here = [(name, state) for name, area, state in contents if area.overlaps(inside)]
            text = (f"- {room.purpose} {rect} (inside {inside}): {built}/{len(ring)} wall cells built"
                    + (f", {planned} planned" if planned else "")
                    + (", closed, so the game roofs it" if built == len(ring) else ""))
            if here:
                text += "; inside: " + ", ".join(f"{name} ({state})" for name, state in here)
            missing = [label for label, defs, count in NEEDS[room.purpose]
                       if sum(1 for name, _ in here if name in defs) < count]
            if missing or room.purpose in ("storage", "other"):
                text += f"; next: {NEXT[room.purpose]}" if missing or not here else ""
            lines.append(text + ".")
        if kept != self.items:
            self.items = kept
            self.save()
        return lines
