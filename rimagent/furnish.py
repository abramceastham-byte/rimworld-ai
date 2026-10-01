"""Ready-made places for furniture inside the agent's rooms.

Models copy coordinates they are given, but they get a building's anchor and
rotation wrong when they work out a footprint themselves: a shelf turned east
at a room's inner edge reached into the wall, and the fix-up hint sent it to
open ground 15 cells away. So each room lists orders that already fit: inside
the walls, clear of what stands there, off the doorway, and with the working
spot in front of a work table left free.
"""

from rimagent.construction import BUILDINGS, Rect, footprint
from rimagent.rooms import inside_of

# What each kind of room is furnished with, in the order worth offering.
ROOM_FURNITURE: dict[str, tuple[str, ...]] = {
    "storage": ("Shelf",),
    "bedroom": ("Bed",),
    "barracks": ("Bed",),
    "hospital": ("Bed",),
    "kitchen": ("FueledStove", "Campfire", "TableButcher"),
    "dining": ("Table2x2c", "DiningChair", "Stool"),
    "workshop": ("HandTailoringBench", "TableStonecutter", "TableButcher"),
}
# Buildings a colonist works at from the cell in front (RimWorld's
# interactionCellOffset (0,0,-1), turned with the building).
WORKED_FROM_FRONT = {"TableButcher", "FueledStove", "ElectricStove", "HandTailoringBench", "TableStonecutter",
                     "SimpleResearchBench"}
FRONT = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}
SPOTS_PER_BUILDING = 3


def doorway(rect: Rect, door_side: str) -> tuple[int, int]:
    """The inside cell in front of build_room's centred door, kept free."""
    mx, mz = (rect.x1 + rect.x2) // 2, (rect.z1 + rect.z2) // 2
    return {"north": (mx, rect.z2 - 1), "south": (mx, rect.z1 + 1),
            "east": (rect.x2 - 1, mz), "west": (rect.x1 + 1, mz)}[door_side]


def work_cell(def_name: str, x: int, z: int, rotation: int) -> tuple[int, int] | None:
    """The cell a colonist stands on to use it, or None if it has none."""
    if def_name not in WORKED_FROM_FRONT:
        return None
    dx, dz = FRONT[rotation]
    # The offset is from the building's centre cell, which for these odd-width
    # tables is the anchor.
    return x + dx, z + dz


def fits(def_name: str, room_rect: Rect, door_side: str, taken: set[tuple[int, int]],
         limit: int = SPOTS_PER_BUILDING) -> list[tuple[int, int, int, Rect]]:
    """Up to `limit` non-overlapping (x, z, rotation, cells) for def_name inside the room.

    Places against the walls come first, as RimWorld players furnish; the
    doorway and every chosen working spot stay free.
    """
    spec = BUILDINGS.get(def_name)
    if spec is None:
        return []
    inside = inside_of(room_rect)
    blocked = set(taken) | {doorway(room_rect, door_side)}
    candidates = []
    seen = set()
    for x, z in inside:
        for rotation in range(4):
            cells = footprint(x, z, spec.size, rotation)
            if (cells, rotation) in seen or not all(inside.contains(*c) for c in cells):
                continue
            seen.add((cells, rotation))
            front = work_cell(def_name, x, z, rotation)
            if front is not None and (not inside.contains(*front) or front in cells):
                continue
            against_wall = sum(1 for cx, cz in cells if cx in (inside.x1, inside.x2) or cz in (inside.z1, inside.z2))
            candidates.append((-against_wall, -z, x, rotation, cells, front, z))
    candidates.sort(key=lambda c: c[:4])
    chosen = []
    for _, _, x, rotation, cells, front, z in candidates:
        if any(c in blocked for c in cells) or (front is not None and front in blocked):
            continue
        chosen.append((x, z, rotation, cells))
        blocked |= set(cells)
        if front is not None:
            blocked.add(front)
        if len(chosen) >= limit:
            break
    return chosen


def room_taken(room_rect: Rect, contents: list[tuple[str, Rect, str]]) -> set[tuple[int, int]]:
    """Cells inside the room already holding a building or blueprint."""
    inside = inside_of(room_rect)
    return {c for _, area, _ in contents for c in area if inside.contains(*c)}


def furnish_line(purpose: str, room_rect: Rect, door_side: str, contents: list[tuple[str, Rect, str]],
                 materials: dict[str, list], stockpile: bool | None = None,
                 notes: dict[str, str] | None = None) -> str | None:
    """One line of ready-made orders for this room's furniture, or None.

    materials: building def -> stuff it can be built in now (None for fixed cost),
    as Choices.materials; buildings missing from it aren't offered. Spots for
    different buildings don't overlap, so all of them can be ordered at once.
    stockpile: for a storage room, whether a stockpile already covers its inside.
    notes: building def -> how many the colony has against what it needs, e.g.
    "beds for 3 of 3 colonists: enough". How many to put in one room is the
    model's call (a barracks or a bedroom each); the note says when to stop.
    """
    taken = room_taken(room_rect, contents)
    note = ""
    if purpose == "storage" and stockpile is not None:
        note = ("  stockpile: it has one already; don't create another." if stockpile else
                "  stockpile: none yet; create_stockpile over the clear part of the inside.")
    parts = []
    for name in ROOM_FURNITURE.get(purpose, ()):
        if name not in materials:
            continue
        # One place per work table, so a small workshop holds one of each.
        limit = 1 if name in WORKED_FROM_FRONT else SPOTS_PER_BUILDING
        spots = fits(name, room_rect, door_side, taken, limit=limit)
        stuff = materials[name][0]
        if not spots:
            parts.append(f"no room left for a {name}")
            continue
        for _, _, _, cells in spots:
            taken |= set(cells)
        orders = "; ".join(f"x={x} z={z} rotation={r} (covers {cells})" for x, z, r, cells in spots)
        count = f" ({notes[name]})" if notes and name in notes else ""
        parts.append(f"{name}" + (f" stuff={stuff}" if stuff else "") + f"{count} fits at {orders}")
    if parts:
        line = "  place_blueprint spots checked just now (copy the values): " + " | ".join(parts)
        return f"{note}\n{line}" if note else line
    return note or None


def room_hint(def_name: str, x: int, z: int, rotation: int, rooms: list,
              contents: list[tuple[str, Rect, str]]) -> str | None:
    """For a placement that failed inside one of the rooms: a spot that fits in that same room."""
    spec = BUILDINGS.get(def_name)
    if spec is None:
        return None
    cells = footprint(x, z, spec.size, rotation)
    for room in rooms:
        rect = room.area()
        if not rect.contains(x, z) and not any(rect.contains(*c) for c in cells):
            continue
        spots = fits(def_name, rect, room.door_side, room_taken(rect, contents), limit=1)
        if spots:
            sx, sz, sr, scells = spots[0]
            return (f"In that {room.purpose} room, {def_name} fits at x={sx} z={sz} rotation={sr} "
                    f"(covers {scells}).")
        return f"No {def_name} fits in that {room.purpose} room any more; it is full."
    return None
