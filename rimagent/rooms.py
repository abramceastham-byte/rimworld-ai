"""Rooms the agent built, with the purpose it gave them, and what each still needs.

RimWorld decides a room's role from what is inside it, and RIMAPI can't say
which walls belong together. So the agent records each build_room with the
purpose the model chose (memory/rooms.json) and reports, every turn, how far
the walls are and what the purpose still calls for: a bedroom without a bed,
a kitchen without a stove. Without this a model walled in a potato field,
which the game then roofed, and nothing grew.
"""

import json
from collections.abc import Callable
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


# Why a floor matters in each kind of room. Bare ground is dirty and ugly.
FLOOR_WHY = {
    "kitchen": "dirt in a kitchen gives food poisoning",
    "hospital": "dirt in a hospital infects wounds",
    "dining": "a bare dining room is ugly and dirty, which hurts mood at every meal",
    "bedroom": "a bare bedroom is ugly, which hurts mood every night",
    "barracks": "a bare barracks is ugly, which hurts mood every night",
    "workshop": "dirt slows work and lowers what it makes",
}


# About 12 cells inside per colonist: RimWorld calls a room under about 12.5
# space "cramped", which colonists dislike, and a 3x4 inside with a bed clears it.
BEDROOM_CELLS_PER_COLONIST = 12
SLEEPERS = {"Bed": 1, "DoubleBed": 2}


def housing_advice(colonists: int, rooms: list["TrackedRoom"], contents: list[tuple[str, Rect, str]]) -> str:
    """Beds and bedroom space against one bed and about 12 cells inside per colonist.

    contents: (def name, area, state) for buildings and blueprints, as for describe.
    """
    target = BEDROOM_CELLS_PER_COLONIST * colonists
    built = sum(SLEEPERS[name] for name, _, state in contents if name in SLEEPERS and state == "built")
    planned = sum(SLEEPERS[name] for name, _, state in contents if name in SLEEPERS and state != "built")
    bedrooms = [r for r in rooms if r.purpose in ("bedroom", "barracks")]
    space = sum(inside_of(r.area()).cells for r in bedrooms)
    text = (f"Recommended housing: one bed and about {BEDROOM_CELLS_PER_COLONIST} cells of bedroom inside per "
            f"colonist, {colonists} beds and {target} cells for {colonists}: a bedroom each or one barracks "
            "with that much room per bed. You have "
            f"beds for {built}" + (f" (+{planned} planned)" if planned else "")
            + f" and {space} cells inside {len(bedrooms)} bedroom{'s' if len(bedrooms) != 1 else ''}"
            + (" (" + "; ".join(f"{r.purpose} {r.area()}, {inside_of(r.area()).cells} cells inside"
                                for r in bedrooms) + ")" if bedrooms else ""))
    short = []
    if built + planned < colonists:
        short.append(f"{colonists - built - planned} more bed{'s' if colonists - built - planned > 1 else ''}")
    if space < target:
        short.append(f"about {target - space} more cells of bedroom")
    if short:
        return text + "; still needed: " + " and ".join(short) + "."
    return text + ", which is enough: don't add bedrooms or beds until you have more colonists."


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

    def describe(self, walls: dict[tuple[int, int], str], contents: list[tuple[str, Rect, str]],
                 floor_of: Callable[[Rect], tuple[int, int, int]] | None = None,
                 extra: Callable[["TrackedRoom"], str | None] | None = None) -> list[str]:
        """One line per room, forgetting rooms whose walls are all gone.

        walls: cell -> "built" / "planned" for walls and doors (built,
        under construction or blueprint); contents: (def name, area, state)
        for buildings and blueprints anywhere, to find what is inside.
        floor_of: (cells floored, cells with a floor planned, cells in all) for a room's inside, when known.
        extra: a further line for a room (ready-made furniture places), or None.
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
            # Counts, not just "planned": with only a "no floor yet" line, a laid floor
            # went unmentioned and a model spent eight turns checking on it.
            laid, planned, total = floor_of(inside) if floor_of else (0, 0, 0)
            if total and laid >= total:
                text += f"; floor laid ({laid}/{total} cells): done"
            elif laid or planned:
                text += f"; floor: {laid} of {total} cells laid" + (f", {planned} more planned" if planned else "")
            elif floor_of and built == len(ring) and room.purpose in FLOOR_WHY:
                text += (f"; no floor yet: build_floor over {inside} (e.g. WoodPlankFloor, 3 WoodLog a cell), "
                         f"since {FLOOR_WHY[room.purpose]}")
            lines.append(text + ".")
            more = extra(room) if extra else None
            if more:
                lines.append(more)
        if kept != self.items:
            self.items = kept
            self.save()
        return lines
