"""Two views of the colony for the prompt, plus ready-made places to build.

A single 25x25 grid was too small for a colony and too easy to misread: models
spent their reasoning counting characters along rows, and read "R" (rich soil)
as rock. So there are two scales here:

  overview  one character per 5x5 block, covering a wide area
  detail    one character per cell, following the base as it grows

Both carry a coordinate ruler, and free_spots() hands the model rectangles that
are already checked to be clear, so it can pick a place instead of doing
arithmetic on the grid.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rimagent.construction import Rect, TerrainInfo

# Cell marks, chosen so none suggests something else (no "R" for rich soil).
RICH_SOIL, SOIL, POOR_SOIL, BARE, WATER = "*", ".", ",", "_", "~"
COLONIST, TREE, FORBIDDEN = "@", "T", "f"
WALL, DOOR, BUILDING, FRAME, BLUEPRINT = "#", "+", "n", "F", "b"
GROWING, STOCKPILE = "g", "s"

DETAIL_LEGEND = (
    f"  {RICH_SOIL} rich soil  {SOIL} soil  {POOR_SOIL} poor soil  "
    f"{BARE} stone or floor, nothing grows  {WATER} water or marsh\n"
    f"  {COLONIST} colonist  {TREE} tree  {FORBIDDEN} forbidden item  "
    f"{WALL} wall  {DOOR} door  {BUILDING} building  {FRAME} being built  "
    f"{BLUEPRINT} your blueprint  {GROWING} growing zone  {STOCKPILE} stockpile"
)
OVERVIEW_LEGEND = (
    f"  {BUILDING} your base  {TREE} forest  {WATER} water  {BARE} stone, often mountain  "
    f"{RICH_SOIL} rich soil  {SOIL} open ground"
)

MAX_DETAIL_SPAN = 21   # cells across; bigger costs too many tokens
DETAIL_MARGIN = 3      # keep this much space around the base in view
OVERVIEW_BLOCK = 5     # cells per overview character
OVERVIEW_BLOCKS = 12   # overview is this many characters square


def terrain_mark(info: TerrainInfo | None) -> str:
    if info is None:
        return "?"
    if "Heavy" not in info.affordances:
        return WATER
    if info.fertility >= 1.1:
        return RICH_SOIL
    if info.fertility >= 0.7:
        return SOIL
    if info.fertility > 0:
        return POOR_SOIL
    return BARE


@dataclass(frozen=True)
class MapView:
    """Everything the prompt needs about the map, already laid out."""

    anchor: tuple[int, int]
    detail: Rect
    lines: list[str]


def centre_of(points: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not points:
        return None
    return (round(sum(x for x, _ in points) / len(points)),
            round(sum(z for _, z in points) / len(points)))


def detail_bounds(anchor: tuple[int, int], keep_in_view: list[tuple[int, int]],
                  width: int, height: int) -> Rect:
    """A window around the base: big enough for it, capped, inside the map."""
    cx, cz = anchor
    if keep_in_view:
        xs = [x for x, _ in keep_in_view] + [cx]
        zs = [z for _, z in keep_in_view] + [cz]
        wanted = Rect(min(xs) - DETAIL_MARGIN, min(zs) - DETAIL_MARGIN,
                      max(xs) + DETAIL_MARGIN, max(zs) + DETAIL_MARGIN)
        cx = (wanted.x1 + wanted.x2) // 2
        cz = (wanted.z1 + wanted.z2) // 2
        span = max(wanted.x2 - wanted.x1 + 1, wanted.z2 - wanted.z1 + 1)
    else:
        span = MAX_DETAIL_SPAN
    half = min(span, MAX_DETAIL_SPAN) // 2
    x1, z1 = max(0, cx - half), max(0, cz - half)
    x2, z2 = min(width - 1, cx + half), min(height - 1, cz + half)
    return Rect(x1, z1, x2, z2)


def ruler(view: Rect, label_width: int) -> str:
    """An x-axis scale, so the model reads columns instead of counting them."""
    # Full x values every 5 columns; they fit because the numbers are shorter
    # than the gap. Last-two-digit labels read as different coordinates.
    marks = [" "] * (view.x2 - view.x1 + 1)
    for x in range(view.x1, view.x2 + 1):
        if x % 5:
            continue
        for offset, digit in enumerate(str(x)):
            at = x - view.x1 + offset
            if at < len(marks):
                marks[at] = digit
    return " " * label_width + "".join(marks) + "  <- x"


def detail_rows(view: Rect, terrain, defs_terrain: dict[str, TerrainInfo],
                marks: dict[tuple[int, int], str]) -> list[str]:
    """The detail grid, north at the top, with a ruler above and z labels."""
    rows = [f"Detail, one character per cell, x {view.x1}-{view.x2}, z {view.z2} (top) "
            f"down to {view.z1}:", DETAIL_LEGEND, ruler(view, len("  z=123 "))]
    for z in range(view.z2, view.z1 - 1, -1):
        row = "".join(
            marks.get((x, z)) or terrain_mark(defs_terrain.get(terrain.at(x, z)))
            for x in range(view.x1, view.x2 + 1)
        )
        rows.append(f"  z={z:3d} {row}")
    return rows


def overview_rows(anchor: tuple[int, int], terrain, defs_terrain: dict[str, TerrainInfo],
                  trees: set[tuple[int, int]], built: set[tuple[int, int]]) -> list[str]:
    """A coarse picture of the surroundings: one character per 5x5 block."""
    span = OVERVIEW_BLOCK * OVERVIEW_BLOCKS
    cx, cz = anchor
    x1 = max(0, min(terrain.width - span, cx - span // 2))
    z1 = max(0, min(terrain.height - span, cz - span // 2))
    rows = [f"Overview, one character per {OVERVIEW_BLOCK}x{OVERVIEW_BLOCK} cells, "
            f"x {x1}-{x1 + span - 1}, z {z1 + span - 1} (top) down to {z1}:", OVERVIEW_LEGEND]
    for bz in range(OVERVIEW_BLOCKS - 1, -1, -1):
        row = ""
        for bx in range(OVERVIEW_BLOCKS):
            block = Rect(x1 + bx * OVERVIEW_BLOCK, z1 + bz * OVERVIEW_BLOCK,
                         x1 + (bx + 1) * OVERVIEW_BLOCK - 1, z1 + (bz + 1) * OVERVIEW_BLOCK - 1)
            row += block_mark(block, terrain, defs_terrain, trees, built)
        rows.append(f"  z={z1 + bz * OVERVIEW_BLOCK:3d} {row}")
    return rows


def block_mark(block: Rect, terrain, defs_terrain: dict[str, TerrainInfo],
               trees: set[tuple[int, int]], built: set[tuple[int, int]]) -> str:
    """One character summarising a block: the most important thing in it."""
    cells = [(x, z) for x, z in block if x < terrain.width and z < terrain.height]
    if any(cell in built for cell in cells):
        return BUILDING
    if sum(cell in trees for cell in cells) >= 3:
        return TREE
    kinds = [terrain_mark(defs_terrain.get(terrain.at(x, z))) for x, z in cells]
    for mark in (WATER, BARE, RICH_SOIL):  # whichever dominates, in this order
        if kinds.count(mark) > len(kinds) / 2:
            return mark
    return SOIL


MAX_SPOT_SIDE = 15   # the most one order can use (a room's side, a 225-cell zone)
MIN_SPOT_SIDE = 3


def free_spots(anchor: tuple[int, int], terrain, defs_terrain: dict[str, TerrainInfo],
               occupied: set[tuple[int, int]], radius: int = 20, count: int = 5,
               blocked_in: Callable[[Rect], set[tuple[int, int]]] | None = None,
               max_checks: int = 20) -> list[tuple[str, Rect, str]]:
    """The biggest open rectangles near the base, as lettered options.

    Saves the model from picking coordinates off the grid, which is where it
    goes wrong most often. Each is the largest clear rectangle left (at most
    MAX_SPOT_SIDE a side), favouring ones nearer the base; the model may use
    all of one or any part. blocked_in, when given, asks the game what is in a
    candidate (zones not known locally, rock, buildings); cells it returns are
    ruled out (they may reach beyond the candidate) and the search runs again,
    up to max_checks asks in all. Only a candidate it finds nothing in is offered.
    """
    cx, cz = anchor
    x1, z1 = max(0, cx - radius), max(0, cz - radius)
    x2, z2 = min(terrain.width - 1, cx + radius), min(terrain.height - 1, cz + radius)
    info_at: dict[tuple[int, int], TerrainInfo] = {}
    for x in range(x1, x2 + 1):
        for z in range(z1, z2 + 1):
            info = defs_terrain.get(terrain.at(x, z))
            if info is not None and "Heavy" in info.affordances and (x, z) not in occupied:
                info_at[(x, z)] = info

    found: list[tuple[str, Rect, str]] = []
    checks = 0
    while len(found) < count:
        rect = biggest_rect(set(info_at), (x1, z1, x2, z2), anchor)
        if rect is None:
            break
        if blocked_in is not None:
            if checks >= max_checks:
                break
            checks += 1
            in_the_way = blocked_in(rect)
            if in_the_way:
                if not any(rect.contains(*c) for c in in_the_way):
                    in_the_way = set(rect)  # reported, but nowhere inside: don't ask about it again
                for cell in in_the_way:
                    info_at.pop(cell, None)
                continue
        fertile = sum(1 for c in rect if info_at[c].fertility >= 0.7) / rect.cells
        soil = ("fertile throughout, crops can grow" if fertile == 1 else
                f"{fertile:.0%} fertile" if fertile >= 0.5 else "poor soil, no crops")
        steps = abs((rect.x1 + rect.x2) // 2 - cx) + abs((rect.z1 + rect.z2) // 2 - cz)
        found.append((chr(ord("A") + len(found)), rect,
                      f"{rect.x2 - rect.x1 + 1}x{rect.z2 - rect.z1 + 1} open, {soil}, {steps} cells from the base"))
        for cell in rect:
            info_at.pop(cell, None)
    return found


def biggest_rect(free: set[tuple[int, int]], bounds: tuple[int, int, int, int],
                 anchor: tuple[int, int]) -> Rect | None:
    """The best clear rectangle within bounds: the largest area (sides capped
    at MAX_SPOT_SIDE), less a little for distance from the anchor so a big spot
    next to the base beats a slightly bigger one far away."""
    x1, z1, x2, z2 = bounds
    # up[x, z]: free cells in a column starting at (x, z) going north, capped.
    up: dict[tuple[int, int], int] = {}
    for x in range(x1, x2 + 1):
        run = 0
        for z in range(z2, z1 - 1, -1):
            run = min(run + 1, MAX_SPOT_SIDE) if (x, z) in free else 0
            up[(x, z)] = run
    best, best_score = None, None
    for z in range(z1, z2 + 1):
        for x in range(x1, x2 + 1):
            height = MAX_SPOT_SIDE
            for width in range(1, MAX_SPOT_SIDE + 1):
                if x + width - 1 > x2:
                    break
                height = min(height, up[(x + width - 1, z)])
                if height < MIN_SPOT_SIDE:
                    break
                if width < MIN_SPOT_SIDE:
                    continue
                rect = Rect(x, z, x + width - 1, z + height - 1)
                centre = ((rect.x1 + rect.x2) / 2, (rect.z1 + rect.z2) / 2)
                distance = abs(centre[0] - anchor[0]) + abs(centre[1] - anchor[1])
                score = rect.cells - 2 * distance
                if best_score is None or score > best_score:
                    best, best_score = rect, score
    return best
