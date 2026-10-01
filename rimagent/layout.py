"""Keeping the base organised: the model's layout plan, and room slots.

Left alone, a model places each building wherever the biggest clear rectangle
happens to be that turn: a research bench off to the west, a butcher table
15 cells east of everything, beds outdoors by the tailoring bench. Two helps:

- A layout plan the model owns (memory/layout.json): with plan_area it
  reserves an area for each kind of building (bedrooms, workshop, fields...).
  Later prompts list the areas and what stands in them, and an order that
  puts something outside its area gets a HINT. Nothing is refused: the model
  can move an area by planning it again.
- Room slots: rectangles right beside the rooms already built, sharing a
  wall, checked clear. A room there costs fewer walls and keeps the base
  compact.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rimagent.construction import BUILDINGS, Rect, footprint

AreaPurpose = Literal["bedrooms", "kitchen_dining", "storage", "workshop", "research", "fields", "power",
                      "recreation", "hospital"]
MAX_AREA_SIDE = 30

# Which areas each building belongs in (any of them). Walls, doors, lamps,
# conduits and floors belong anywhere.
BUILDING_AREAS: dict[str, tuple[str, ...]] = {
    "Bed": ("bedrooms", "hospital"), "DoubleBed": ("bedrooms", "hospital"),
    "FueledStove": ("kitchen_dining",), "ElectricStove": ("kitchen_dining",), "Campfire": ("kitchen_dining",),
    "TableButcher": ("kitchen_dining",),
    "Table2x2c": ("kitchen_dining", "recreation"), "DiningChair": ("kitchen_dining", "recreation"),
    "Stool": ("kitchen_dining", "recreation"),
    "Shelf": ("storage",),
    "HandTailoringBench": ("workshop",), "TableStonecutter": ("workshop",),
    "SimpleResearchBench": ("research",),
    "WoodFiredGenerator": ("power",), "Battery": ("power",), "SolarGenerator": ("power",),
    "WindTurbine": ("power",),
    "HorseshoesPin": ("recreation",), "ChessTable": ("recreation",),
}
ROOM_AREAS = {"bedroom": ("bedrooms",), "barracks": ("bedrooms",), "kitchen": ("kitchen_dining",),
              "dining": ("kitchen_dining",), "storage": ("storage",), "workshop": ("workshop",),
              "hospital": ("hospital",)}


class Area(BaseModel):
    purpose: AreaPurpose
    cells: tuple[int, int, int, int]

    def area(self) -> Rect:
        return Rect(*self.cells)


class LayoutTracker:
    """The areas the model reserved, one per purpose, saved between runs."""

    def __init__(self, path: Path | str, read_only: bool = False):
        self.path, self.read_only = Path(path), read_only
        self.items: list[Area] = []
        if self.path.exists():
            self.items = [Area.model_validate(a) for a in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self) -> None:
        if not self.read_only:
            self.path.write_text(json.dumps([a.model_dump() for a in self.items], indent=2), encoding="utf-8")

    def get(self, purpose: str) -> Area | None:
        return next((a for a in self.items if a.purpose == purpose), None)

    def plan(self, purpose: str, rect: Rect) -> str:
        """Reserve (or move) the area for a purpose. Refuses overlapping another purpose's area."""
        if rect.x2 - rect.x1 + 1 > MAX_AREA_SIDE or rect.z2 - rect.z1 + 1 > MAX_AREA_SIDE:
            raise ValueError(f"an area is at most {MAX_AREA_SIDE} cells a side; {rect} is bigger")
        for other in self.items:
            if other.purpose != purpose and overlaps(other.area(), rect):
                raise ValueError(f"{rect} overlaps your {other.purpose} area {other.area()}: choose another "
                                 f"place, or plan_area {other.purpose} somewhere else first")
        old = self.get(purpose)
        if old is not None and old.area() == rect:
            return f"unchanged: your {purpose} area is already {rect}"
        self.items = [a for a in self.items if a.purpose != purpose]
        self.items.append(Area(purpose=purpose, cells=(rect.x1, rect.z1, rect.x2, rect.z2)))
        self.save()
        return (f"{purpose} area moved from {old.area()} to {rect}" if old else f"{purpose} area reserved: {rect}") \
            + "; nothing is built by this, it is your plan for where such things go"

    def describe(self, buildings: list[tuple[str, Rect, str]]) -> list[str]:
        """The layout section: each area and what stands in it, or how to start one."""
        if not self.items:
            return ["Layout plan: none yet. Reserve an area for each kind of building with plan_area (bedrooms, "
                    "kitchen_dining, storage, workshop, research, fields, power, recreation, hospital) so related "
                    "things stay together: kitchen beside storage and dining, bedrooms together, the workshop "
                    "beside storage, fields in the open, power close to what it powers."]
        lines = ["Your layout plan (areas you reserved; put each kind of building in its area):"]
        for a in self.items:
            inside = [f"{name} ({state})" for name, rect, state in buildings if contains(a.area(), rect)]
            lines.append(f"- {a.purpose} {a.area()}: " + (", ".join(inside) if inside else "nothing yet"))
        return lines


def overlaps(a: Rect, b: Rect) -> bool:
    return a.x1 <= b.x2 and b.x1 <= a.x2 and a.z1 <= b.z2 and b.z1 <= a.z2


def contains(outer: Rect, inner: Rect) -> bool:
    return outer.x1 <= inner.x1 and inner.x2 <= outer.x2 and outer.z1 <= inner.z1 and inner.z2 <= outer.z2


def order_areas(action) -> tuple[tuple[str, ...], Rect | None, str]:
    """(areas it belongs in, the cells it covers, what it is) for an order that places something."""
    kind = action.action
    if kind == "place_blueprint":
        spec = BUILDINGS.get(action.building_def)
        size = spec.size if spec else (1, 1)
        return (BUILDING_AREAS.get(action.building_def, ()), footprint(action.x, action.z, size, action.rotation),
                action.building_def)
    if kind == "build_room":
        return ROOM_AREAS.get(action.purpose, ()), action.rect(), f"{action.purpose} room"
    if kind == "create_growing_zone":
        return ("fields",), action.rect(), "growing zone"
    if kind == "create_stockpile":
        return ("storage",), action.rect(), "stockpile"
    return (), None, ""


def room_refusal(action, rooms: list, contents: list[tuple[str, Rect, str]],
                 layout: LayoutTracker | None, colonists: int = 0) -> str | None:
    """Refuse a room that would cover another room, shut in buildings that don't
    belong in it, or take over an area reserved for something else.

    rooms: TrackedRoom; contents: (def name, area, state) for buildings and
    blueprints. A model walled a storage room around its own generator and
    campfire, then failed to fit a stockpile in it for three turns.
    """
    from rimagent.rooms import inside_of
    if action.action != "build_room":
        return None
    rect = action.rect()
    if rect.x2 - rect.x1 < 3 or rect.z2 - rect.z1 < 3:
        return None  # too small; execute says why
    inside = inside_of(rect)
    if action.purpose in ("bedroom", "barracks") and colonists:
        # One run walled a 12x12 bedroom (100 cells inside) for 3 colonists: 240 wood
        # of walls, and with its beds more wood than the colony had.
        from math import ceil, sqrt

        from rimagent.rooms import BEDROOM_CELLS_PER_COLONIST
        need = BEDROOM_CELLS_PER_COLONIST * colonists
        if inside.cells > 2 * need:
            side = ceil(sqrt(need)) + 2
            return (f"a {action.purpose} {rect} is {inside.cells} cells inside, more than twice the {need} "
                    f"recommended for {colonists} colonists, and every wall costs material. Make it about "
                    f"{side}x{side} outer ({side - 2}x{side - 2} inside).")
    for room in rooms:
        other = room.area()
        if other == rect:
            return None  # the same room again: execute reports it unchanged
        if overlaps(inside, other) or overlaps(inside_of(other), rect):
            return (f"{rect} overlaps your {room.purpose} room {other} (see the site plan); a new room may "
                    "share a wall with it, not cover it. Use a room slot or open ground.")
    homes = set(ROOM_AREAS.get(action.purpose, ()))
    if homes:
        misfits = [f"{name} at {area}" for name, area, _ in contents
                   if overlaps(inside, area) and name in BUILDING_AREAS and not homes & set(BUILDING_AREAS[name])]
        if misfits:
            return (f"a {action.purpose} room at {rect} would shut in {', '.join(misfits[:3])}, which "
                    f"doesn't belong in a {action.purpose}. Build it on open ground, or where it holds only its "
                    "own kind of thing.")
    if layout is not None:
        taken = next((a for a in layout.items if a.purpose not in homes and overlaps(inside, a.area())), None)
        if taken is not None and action.purpose != "other":
            return (f"a {action.purpose} room at {rect} would take over your {taken.purpose} area {taken.area()} "
                    f"(mark on the detail map). Put it in "
                    + (" or ".join(f"your {h} area {layout.get(h).area()}" for h in homes if layout.get(h))
                       or "open ground outside the reserved areas")
                    + f", or plan_area {taken.purpose} somewhere else first.")
    return None


def field_area_refusal(action, layout: LayoutTracker | None) -> str | None:
    """Refuse a growing zone inside an area reserved for buildings.

    Rooms go in those areas, and a closed room is roofed, so crops there get no
    sun. A model reserved storage and sowed potatoes over the same cells in one
    turn, then spent ten turns failing to put storage around its own field.
    """
    if layout is None or action.action != "create_growing_zone":
        return None
    rect = action.rect()
    hit = next((a for a in layout.items if a.purpose != "fields" and overlaps(a.area(), rect)), None)
    if hit is None:
        return None
    fields = layout.get("fields")
    return (f"{rect} overlaps your {hit.purpose} area {hit.area()}, which is kept for buildings; a field there "
            "would be in the way of its rooms. Put the field "
            + (f"in your fields area {fields.area()}" if fields else "outside your reserved areas")
            + f", or plan_area {hit.purpose} somewhere else first.")


def layout_hint(action, layout: LayoutTracker | None) -> str:
    """A nudge when an order puts something outside its area or inside another's."""
    if layout is None or not layout.items:
        return ""
    homes, rect, what = order_areas(action)
    if rect is None:
        return ""
    planned = [layout.get(h) for h in homes if layout.get(h)]
    if planned and not any(contains(a.area(), rect) for a in planned):
        where = " or ".join(f"{a.purpose} area {a.area()}" for a in planned)
        return (f"The {what} at {rect} is outside your {where}: keep such things there, or plan_area again if "
                f"your plan changed.")
    intruded = [a for a in layout.items if a.purpose not in homes and overlaps(a.area(), rect)]
    if intruded and homes:
        return f"The {what} at {rect} is in your {intruded[0].purpose} area {intruded[0].area()}."
    if homes and not planned:
        return f"You have no {homes[0]} area yet: plan_area {homes[0]} around it so later ones join it."
    return ""


# Outer depths tried for a new room beside another, deepest first, walls
# included. The slot says how far clear ground reaches; the model picks the size.
SLOT_DEPTHS = tuple(range(15, 3, -1))


def room_slots(rooms: list[tuple[str, Rect]], is_free: Callable[[int, int], bool],
               width: int, height: int, is_clear: Callable[[Rect], bool] | None = None,
               limit: int = 4, max_checks: int = 8) -> list[str]:
    """Rooms that could be built right beside existing ones, sharing a wall.

    rooms: (purpose, outer rectangle). is_free: a cell is buildable ground with
    nothing on it (known locally). is_clear: asks the game about a rectangle
    (zones, buildings, rock it knows better); only candidates it passes are offered.
    """
    slots: list[tuple[int, str]] = []
    checks = 0
    taken: list[Rect] = []
    for purpose, r in rooms:
        for side in ("north", "east", "south", "west"):
            for depth in SLOT_DEPTHS:
                if side == "north":
                    outer, fresh = Rect(r.x1, r.z2, r.x2, r.z2 + depth - 1), Rect(r.x1, r.z2 + 1, r.x2, r.z2 + depth - 1)
                elif side == "south":
                    outer, fresh = Rect(r.x1, r.z1 - depth + 1, r.x2, r.z1), Rect(r.x1, r.z1 - depth + 1, r.x2, r.z1 - 1)
                elif side == "east":
                    outer, fresh = Rect(r.x2, r.z1, r.x2 + depth - 1, r.z2), Rect(r.x2 + 1, r.z1, r.x2 + depth - 1, r.z2)
                else:
                    outer, fresh = Rect(r.x1 - depth + 1, r.z1, r.x1, r.z2), Rect(r.x1 - depth + 1, r.z1, r.x1 - 1, r.z2)
                if outer.x1 < 0 or outer.z1 < 0 or outer.x2 >= width or outer.z2 >= height:
                    continue
                if any(overlaps(fresh, t) for t in taken) or not all(is_free(x, z) for x, z in fresh):
                    continue
                if is_clear is not None:
                    if checks >= max_checks:
                        break
                    checks += 1
                    if not is_clear(fresh):
                        continue
                shared = (r.x2 - r.x1 + 1) if side in ("north", "south") else (r.z2 - r.z1 + 1)
                wall = {"north": Rect(r.x1, r.z2, r.x2, r.z2), "south": Rect(r.x1, r.z1, r.x2, r.z1),
                        "east": Rect(r.x2, r.z1, r.x2, r.z2), "west": Rect(r.x1, r.z1, r.x1, r.z2)}[side]
                slots.append((shared, f"- {side} of your {purpose} room {r}: its {side} wall {wall} can be "
                                      f"shared ({shared} cells, {5 * shared} material saved). Clear ground reaches "
                                      f"{depth} cells out, walls included, as far as {outer}; choose the size "
                                      f"within that. door_side {side} keeps the door off the shared wall."))
                taken.append(fresh)
                break
    slots.sort(key=lambda s: -s[0])
    return [line for _, line in slots[:limit]]
