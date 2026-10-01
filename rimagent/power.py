"""Route power conduits between two buildings: the model says what to connect,
this works out where the conduits go.

Generators, batteries and conduits carry power to whatever touches them on a
side; a building that uses power (lamp, stove) links to any of them within
about 6 cells. So a route starts touching the power source (or its conduits)
and only has to end near the building that needs power. A regular PowerConduit
(1 steel, quick) can go under walls and other buildings, where the building
hides it; on open ground a HiddenConduit (2 steel, slow, tougher) is used, so
nothing is left lying in the open to trip raiders or look bad. Existing
conduits are reused for free.
"""

import heapq
from collections.abc import Callable

from rimagent.construction import Rect

REACH = 5              # cells from a powered building to a conduit it links to (the game allows 6)
UNDER_BUILDING = 1     # cost of a conduit under a wall/building: 1 steel, quick
OPEN_GROUND = 4        # a hidden conduit: 2 steel and eight times the work
CONDUITS = ("PowerConduit", "HiddenConduit")


def within_reach(cells: set[tuple[int, int]], bounds: Rect) -> set[tuple[int, int]]:
    """Every cell in bounds within REACH (straight-line) of any of cells."""
    return {(x, z) for x, z in bounds
            if any((x - cx) ** 2 + (z - cz) ** 2 <= REACH * REACH for cx, cz in cells)}


def components(conduits: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Existing conduit cells grouped into connected lines (side neighbours)."""
    left, groups = set(conduits), []
    while left:
        group, todo = set(), [left.pop()]
        while todo:
            x, z = todo.pop()
            group.add((x, z))
            for n in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                if n in left:
                    left.discard(n)
                    todo.append(n)
        groups.append(group)
    return groups


def route(source: set[tuple[int, int]], target: set[tuple[int, int]], bounds: Rect,
          conduits: set[tuple[int, int]], covered: set[tuple[int, int]],
          blocked: Callable[[int, int], bool]) -> list[tuple[int, int, str]] | None:
    """The cheapest new conduits linking source to target, as (x, z, def).

    source: the cells of the power source (generator, battery or a conduit);
    target: the cells of what needs power. conduits: cells that already have
    a conduit (built or planned). covered: cells under a wall or other
    building, where a plain conduit is hidden. blocked: cells no conduit can
    go (rock, water). Returns [] when they are already linked, None when no
    route exists inside bounds.
    """
    # The source's grid: the source plus every conduit line touching it.
    grid = set(source)
    for group in components(conduits):
        if any((x + dx, z + dz) in grid for x, z in group for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            grid |= group
    goal = within_reach(target, bounds)
    if grid & goal:
        return []

    def cost(cell: tuple[int, int]) -> int:
        if cell in conduits:
            return 0
        return UNDER_BUILDING if cell in covered else OPEN_GROUND

    frontier: list[tuple[int, tuple[int, int]]] = []
    best: dict[tuple[int, int], int] = {}
    came: dict[tuple[int, int], tuple[int, int] | None] = {}
    for cell in grid:
        best[cell] = 0
        came[cell] = None
        heapq.heappush(frontier, (0, cell))
    end = None
    while frontier:
        spent, cell = heapq.heappop(frontier)
        if spent > best.get(cell, spent):
            continue
        if cell in goal:
            end = cell
            break
        x, z = cell
        for n in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
            if not bounds.contains(*n) or n in grid or (blocked(*n) and n not in conduits):
                continue
            total = spent + cost(n)
            if total < best.get(n, total + 1):
                best[n] = total
                came[n] = cell
                heapq.heappush(frontier, (total, n))
    if end is None:
        return None
    path = []
    while end is not None:
        if end not in conduits and end not in grid:
            path.append((end[0], end[1], "PowerConduit" if end in covered else "HiddenConduit"))
        end = came[end]
    return list(reversed(path))
