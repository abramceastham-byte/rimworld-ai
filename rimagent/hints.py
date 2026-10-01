"""A concrete next step with every order's result, failed or not.

A bare reason ("would cover (117,140), where TableStonecutter is already
planned") left a model to guess the fix, and it guessed wrong: it read its own
blueprint as a conflict, "repositioned" it by placing another, and in one run
planned ten stonecutter's tables in 25 turns. So each result carries a hint
drawn from what this turn observed: free spots to use, the nearest targets,
who can do the work, what a build costs against what is left.
"""

from dataclasses import dataclass, field

from rimagent.observation import FIELD_CELLS_PER_COLONIST

from rimagent.choices import incapable_work
from rimagent.construction import BUILDINGS, FLOORS, Rect, footprint
from rimagent.furnish import room_hint
from rimagent.mapview import biggest_rect
from rimagent.observation import PLENTY_OF_STORAGE
from rimagent.rooms import NEXT, inside_of
from rimagent.supplies import commitments, how_to_get, stock_by_def

# Buildings a colony wants one of at a time: a second copy while the first is
# still unbuilt only doubles the material bill (see execution's guard).
ONE_AT_A_TIME = ("TableButcher", "FueledStove", "ElectricStove", "SimpleResearchBench",
                 "HandTailoringBench", "TableStonecutter", "WoodFiredGenerator")


@dataclass
class TurnFacts:
    """What this turn observed, for hints. Built once in prepare_turn."""

    spots: list = field(default_factory=list)       # (letter, Rect, note) from free_spots
    targets: dict = field(default_factory=dict)     # designation -> [{name, x, z}]
    trees: list = field(default_factory=list)
    items: list = field(default_factory=list)
    stored: list | None = None
    report: list = field(default_factory=list)      # (TrackedBlueprint, status)
    buildings: list = field(default_factory=list)
    colonists: list = field(default_factory=list)
    anchor: tuple[int, int] | None = None           # middle of the base
    storage: dict | None = None                     # /resources/storages/summary
    rooms: list = field(default_factory=list)       # TrackedRoom, from rooms.RoomTracker
    learned: dict = field(default_factory=dict)     # work RIMAPI refused, by colonist
    field_cells: int = 0                            # growing zone cells before this turn's orders
    zone_cells: set = field(default_factory=set)    # cells of the zones the agent made

    def contents(self) -> list[tuple[str, Rect, str]]:
        """(def name, area, state) for buildings and blueprints other than walls, doors and floors."""
        out = [(b.def_name.removeprefix("Frame_"), b.area(), "built") for b in self.buildings
               if b.def_name.removeprefix("Frame_") not in ("Wall", "Door")]
        out += [(bp.def_name, bp.area(), "planned") for bp in self.unfinished()
                if not bp.floor and bp.def_name not in ("Wall", "Door")]
        return out

    def unfinished(self, def_name: str | None = None) -> list:
        return [bp for bp, s in self.report if s in ("waiting", "under construction")
                and (def_name is None or bp.def_name == def_name)]

    def built(self, def_name: str) -> list:
        return [b for b in self.buildings if b.def_name == def_name and not b.under_construction]

    def with_work(self, work_type: str) -> list[str]:
        return [c.name for c in self.colonists if work_type in self.with_work_of(c)]

    @staticmethod
    def with_work_of(colonist) -> set[str]:
        """The work a colonist has switched on."""
        return {w.work_type for w in colonist.work_priorities if w.priority > 0}

    def near(self, x: int, z: int) -> int:
        return abs(x - self.anchor[0]) + abs(z - self.anchor[1]) if self.anchor else 0


# Jobs that mean a colonist has nothing to do. Recreation (GoForWalk,
# SocialRelax, Skygaze...) is left out: colonists need it. Across the logged
# runs GotoWander and Wait_Wander were the two most common jobs of all.
IDLE_JOBS = {"", "GotoWander", "Wait_Wander", "Wait"}
IDLE_AFTER = 2  # consecutive turns idle before the prompt says so

# The work type that runs each kind of work table's bills.
TABLE_WORK = {"FueledStove": "Cooking", "ElectricStove": "Cooking", "TableButcher": "Cooking",
              "HandTailoringBench": "Tailoring", "TableStonecutter": "Crafting",
              "SimpleResearchBench": "Research"}


class IdleTracker:
    """How many turns in a row each colonist was seen idle at the start of a turn."""

    def __init__(self):
        self.turns: dict[str, int] = {}

    def update(self, colonists) -> dict[str, int]:
        self.turns = {c.name: self.turns.get(c.name, 0) + 1 if c.current_job in IDLE_JOBS else 0
                      for c in colonists}
        return self.turns


def idle_hint(idle: dict[str, int], facts: TurnFacts, pending: list[tuple[str, str, bool]],
              capable: dict[str, set[str]]) -> str | None:
    """When colonists have idled for IDLE_AFTER+ turns: what work is pending,
    who has that work on, and what the idle ones could take on.

    pending: (work type, what is waiting, blocked) where blocked means the work
    can't go ahead for now (e.g. short of material), so idling is expected.
    capable: work each colonist can do (see choices.incapable_work).
    """
    idlers = [name for name, n in idle.items() if n >= IDLE_AFTER]
    if not idlers:
        return None
    text = ("HINT: " + ", ".join(f"{name} ({idle[name]} turns)" for name in idlers)
            + " wandering or idle. Pending work and who has it on:")
    if not pending:
        return text + (" nothing is pending. Give them work: blueprints, a growing zone, trees to cut, "
                       "rock to mine, forbidden items to allow.")
    takers: dict[str, list[str]] = {name: [] for name in idlers}
    for work, what, blocked in pending:
        on = facts.with_work(work)
        text += (f" {work}: {what}" + (" (blocked for now)" if blocked else "")
                 + f", on for {', '.join(on) or 'nobody'};")
        for name in idlers:
            if not blocked and name not in on and work in capable.get(name, set()):
                takers[name].append(work)
    offers = [f"{name} could take {', '.join(works)} (enable_work)" for name, works in takers.items() if works]
    if offers:
        text += " " + "; ".join(offers) + "."
    elif all(blocked for _, _, blocked in pending):
        text += " All pending work is blocked: unblock it (materials) or give new work."
    else:
        text += (" The idle ones already have the unblocked work on, or can't do it: they may be unable "
                 "to reach it, or it needs time advanced.")
    return text


STORAGE_FULL = 0.9  # share of storage cells used before the prompt says storage is full


def storage_hint(storage: dict | None, shelf_buildable: bool) -> str | None:
    """When the stockpiles are (nearly) full: what that costs, and two ways out."""
    if not storage or not storage.get("total_cells"):
        return None
    used, total = storage.get("used_cells", 0), storage["total_cells"]
    if used < STORAGE_FULL * total:
        return None
    text = (f"HINT: Storage is {'full' if used >= total else 'nearly full'} ({used}/{total} cells used in "
            f"{storage.get('total_stockpiles', '?')} stockpiles): new items stay where they fall, and outdoors "
            "they deteriorate. Check existing storage rooms and pending shelves first. Add only enough "
            "capacity to address the observed shortage. Once sufficient capacity exists or is queued, stop "
            "expanding storage and reassess the colony's needs.")
    return text + (" Shelf (2x1) stores several stacks per cell. Prefer furnishing a suitable existing storage "
                   "room; add a new room only if needed. A stockpile on open ground works too, but items there weather."
                   if shelf_buildable else
                   " Shelves hold more in less space once ComplexFurniture is researched.")


# Games played from a distance (RimWorld's JoyGiver_WatchBuilding): the
# player stands in a strip `width` cells wide, exactly `distance` cells from
# the building, on any of its four sides, in the same room and in sight of it
# (Buildings_Joy.xml: watchBuildingStandDistanceRange, watchBuildingStandRectWidth).
WATCH_BUILDINGS = {"HorseshoesPin": (5, 3)}
SIGHT_BLOCKERS = {"Wall", "Door"}
# Buildings a colonist can stand on; anything else on a cell blocks it.
STAND_ON = {"Bed", "DoubleBed", "PowerConduit", "HiddenConduit"}


def watch_cells(x: int, z: int, distance: int, width: int) -> list[tuple[int, int]]:
    half = width // 2
    cells = []
    for dx, dz in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        for off in range(-half, half + 1):
            cells.append((x + dx * distance + (off if dx == 0 else 0), z + dz * distance + (off if dz == 0 else 0)))
    return cells


def between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """The cells on the straight line from a to b, ends left out."""
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + round((b[0] - a[0]) * i / steps), a[1] + round((b[1] - a[1]) * i / steps))
            for i in range(1, steps)]


def watch_spot_refusal(action, facts: TurnFacts, client) -> str | None:
    """Refuse a horseshoes pin nobody could play: every cell it is played from
    is blocked, off the map, or cut off from it by a wall or door.

    A model put one in the middle of a 3x3 storage room; its throwing spots, 5
    cells away, were all outside the walls.
    """
    if action.action != "place_blueprint" or action.building_def not in WATCH_BUILDINGS:
        return None
    distance, width = WATCH_BUILDINGS[action.building_def]
    x, z = action.x, action.z
    terrain = client.get_terrain()
    reach = Rect(max(0, x - distance), max(0, z - distance),
                 min(terrain.width - 1, x + distance), min(terrain.height - 1, z + distance))
    sight: set[tuple[int, int]] = set()
    stand: set[tuple[int, int]] = set()

    def mark(name: str, cells) -> None:
        cells = set(cells)
        if name in SIGHT_BLOCKERS:
            sight.update(cells)
        if name not in STAND_ON and name not in FLOORS:
            stand.update(cells)

    for b in facts.buildings:
        mark(b.def_name.removeprefix("Frame_"), b.area())
    for bp in facts.unfinished():
        if not bp.floor:
            mark(bp.def_name, bp.area())
    check = client.check_area(reach)
    for issue in check.ores:  # rock: nobody stands in it or sees through it
        sight.add((issue.x, issue.z))
    for issue in check.buildings:
        mark((issue.def_name or "").removeprefix("Frame_"), [(issue.x, issue.z)])
    stand |= sight
    stand.discard((x, z))
    usable = [c for c in watch_cells(x, z, distance, width)
              if 0 <= c[0] < terrain.width and 0 <= c[1] < terrain.height
              and c not in stand and "Deep" not in terrain.at(*c)
              and not any(m in sight for m in between((x, z), c))]
    if usable:
        return None
    return (f"{action.building_def} at ({x},{z}) could never be played: colonists play it standing exactly "
            f"{distance} cells away on one of its four sides ({width} cells wide), in the same room with a clear "
            f"line to it, and all {4 * width} of those cells are blocked or behind walls. Put it on open ground "
            f"with {distance} clear cells around it, or in a room at least {2 * distance + 1} cells across inside.")


def stockpile_refusal(action, facts: TurnFacts) -> str | None:
    """Refuse a new stockpile while storage has plenty of room, unless it is the
    inside of a storage room: more open-air stockpiles only spread items out."""
    if action.action != "create_stockpile" or not facts.storage:
        return None
    used, total = facts.storage.get("used_cells", 0), facts.storage.get("total_cells", 0)
    if not total or used >= PLENTY_OF_STORAGE * total:
        return None
    rect = action.rect()
    for room in facts.rooms:
        if room.purpose == "storage" and inside_of(room.area()).contains(rect.x1, rect.z1) \
                and inside_of(room.area()).contains(rect.x2, rect.z2):
            return None
    return (f"storage already has plenty of room ({used}/{total} cells used), so no new stockpile outside a "
            "storage room")


FIELD_LIMIT = 2  # total field cells may reach this many times the recommendation, no more


def field_refusal(action, facts: TurnFacts) -> str | None:
    """Refuse a growing zone that would take the fields past FIELD_LIMIT times the
    recommendation. Below that, extra fields are the model's call; the prompt
    says how much is recommended.

    Advice alone didn't work: models copied a whole 15x15 open-ground spot as
    their first field (225 cells for 3 colonists) with the advice in view. The
    offered replacement is sized to what is still needed, not to the limit: a
    refusal that offered "at most 30 more" was taken as an invitation, and a
    colony with its 30 recommended cells got a second field.
    """
    if action.action != "create_growing_zone" or not facts.colonists:
        return None
    recommended = FIELD_CELLS_PER_COLONIST * len(facts.colonists)
    allowed = FIELD_LIMIT * recommended - facts.field_cells
    rect = action.rect()
    if rect.cells <= allowed:
        return None
    need = recommended - facts.field_cells
    if need < 4:
        return (f"{rect} is {rect.cells} cells, which would take your fields past {FIELD_LIMIT * recommended} "
                f"({FIELD_LIMIT}x the {recommended} recommended for {len(facts.colonists)} colonists); you already "
                f"have {facts.field_cells}, which is enough")
    width = min(rect.x2 - rect.x1 + 1, max(2, round(need ** 0.5)))
    depth = min(rect.z2 - rect.z1 + 1, max(1, need // width))
    fit = Rect(rect.x1, rect.z1, rect.x1 + width - 1, rect.z1 + depth - 1)
    return (f"{rect} is {rect.cells} cells; you have {facts.field_cells} of the {recommended} recommended for "
            f"{len(facts.colonists)} colonists, so about {need} more is enough. Use {fit} ({fit.cells} cells) "
            f"instead: x1={fit.x1} z1={fit.z1} x2={fit.x2} z2={fit.z2}")


def pending_work(facts: TurnFacts, research, tables, live_zones, chop_orders: int,
                 short: set[str], mine_orders: int = 0) -> list[tuple[str, str, bool]]:
    """(work type, what is waiting, blocked) for everything already ordered."""
    pending = []
    waiting = facts.unfinished()
    if waiting:
        stuck = {bp.stuff for bp in waiting if bp.stuff in short}
        pending.append(("Construction", f"{len(waiting)} blueprints or frames"
                        + (f", short of {', '.join(sorted(stuck))}" if stuck else ""),
                        bool(stuck) and all(bp.stuff in short for bp in waiting)))
    if chop_orders:
        pending.append(("PlantCutting", f"{chop_orders} trees ordered cut", False))
    if mine_orders:
        pending.append(("Mining", f"{mine_orders} rock cells marked", False))
    fields = sum(1 for z in live_zones if "Growing" in z.type)
    if fields:
        pending.append(("Growing", f"{fields} growing zones", False))
    if research and research.current:
        pending.append(("Research", research.current.name, not research.has_bench))
    for table in tables:
        if any(not b.suspended for b in table.bills) and table.thing_def in TABLE_WORK:
            pending.append((TABLE_WORK[table.thing_def], f"bills at {table.label} {table.id}", False))
    return pending


def spot_list(facts: TurnFacts, min_side: int = 1) -> str:
    fits = [(letter, rect, note) for letter, rect, note in facts.spots
            if min(rect.x2 - rect.x1, rect.z2 - rect.z1) + 1 >= min_side]
    return "; ".join(f"{letter} {rect} ({note})" for letter, rect, note in fits)


def anchor_in_spots(def_name: str, facts: TurnFacts) -> str | None:
    """A worked place_blueprint for this building inside a free spot, or None."""
    spec = BUILDINGS.get(def_name)
    if not spec:
        return None
    for letter, rect, _ in facts.spots:
        for x, z in rect:
            if all(rect.contains(*c) for c in footprint(x, z, spec.size, 0)):
                return f"{def_name} at ({x},{z}) facing north covers {footprint(x, z, spec.size, 0)}, inside free spot {letter}"
    return None


def failure_hint(action, error: str, facts: TurnFacts) -> tuple[str, bool]:
    """(hint, keep): keep is False when there is nothing to fix, so the failure
    shouldn't linger in the prompt as "unresolved" (that is what fed the loop)."""
    kind = action.action
    if kind == "place_blueprint":
        own = f"{action.building_def} is already planned" in error or f"{action.building_def} is already being built" in error
        if own or "still unbuilt" in error:
            return (f"That {action.building_def} is your own, from an earlier turn. Blueprints can't be moved "
                    "or cancelled from here, and placing another makes a second one, not a move. Nothing to "
                    "fix: leave it, and it gets built once its materials are there.", False)
        in_room = room_hint(action.building_def, action.x, action.z, action.rotation, facts.rooms, facts.contents())
        if in_room:
            return in_room, False
        spot = anchor_in_spots(action.building_def, facts)
        if spot:
            return f"Try {spot}.", False
        return "No free spot fits it this turn; try somewhere else near the base.", False
    if kind == "build_room":
        if "4..15" in error:
            return ("Give the OUTER rectangle, walls included, 4 to 15 cells a side. Free spots big enough: "
                    + (spot_list(facts, 4) or "none this turn") + ".", False)
        spots = spot_list(facts, 4)
        if spots:
            return (f"Put it on open ground checked clear this turn (all of an area or any part, 4..15 a "
                    f"side): {spots}.", False)
        return "No free spot is 4x4 or more this turn; try a clear area away from buildings and fields.", False
    if kind == "build_wall" and "taken" in error:
        return ("Those cells already have walls or other things; nothing to add there. build_wall only adds "
                "walls, it never removes anything.", False)
    if kind == "build_floor":
        return "Those cells are floored or planned already, or rock; nothing to add there.", False
    if kind == "designate":
        targets = sorted(facts.targets.get(action.designation, []), key=lambda t: facts.near(t["x"], t["z"]))[:3]
        if targets:
            listed_targets = "; ".join(f"{t['name']} at ({t['x']},{t['z']})" for t in targets)
            return f"Nearest {action.designation} targets: {listed_targets}.", True
        return f"There are no observed {action.designation} targets now.", False
    if kind == "allow_items":
        forbidden = sorted({i.def_name for i in facts.items if i.is_forbidden})
        if forbidden:
            return (f"Still forbidden: {', '.join(forbidden[:8])}. allow_items what=all allows everything at once.",
                    True)
        return "Nothing is forbidden any more.", False
    if kind == "chop_trees":
        trees = sorted(facts.trees, key=lambda t: facts.near(t.position.x, t.position.z))[:5]
        if trees:
            box = Rect.from_corners(min(t.position.x for t in trees), min(t.position.z for t in trees),
                                    max(t.position.x for t in trees), max(t.position.z for t in trees))
            return f"The nearest trees are within {box}.", True
        return "", True
    if kind == "enable_work" and "incapable" in error:
        work = action.work_type
        others = [c for c in facts.colonists if c.name != action.colonist]
        doing = [c.name for c in others if work in facts.with_work_of(c)]
        able = [c.name for c in others if work not in facts.with_work_of(c) and work not in incapable_work(c, facts.learned)]
        text = f"{action.colonist} can never do {work}, so don't ask again."
        if doing:
            text += f" {work} is already on for {', '.join(doing)}."
        if able:
            text += (f" {', '.join(able)} can do it and has it off: e.g. enable_work colonist={able[0]} "
                     f"work_type={work}.")
        elif not doing:
            text += f" Nobody else can do {work} either."
        return text, False
    if kind == "create_growing_zone":
        fertile = [f"{letter} {rect}" for letter, rect, note in facts.spots if "fertile" in note]
        return ("Fertile free spots this turn: " + ("; ".join(fertile) or "none") + ".", False)
    if kind == "create_stockpile" and "plenty of room" in error:
        return ("Nothing to do: there's room to spare. Stop expanding storage and reassess the colony's needs.",
                False)
    if kind == "create_stockpile" and "blocked by" in error:
        return clear_part_hint(action.rect(), facts), False
    if kind == "create_stockpile":
        return f"Free spots this turn: {spot_list(facts) or 'none'}.", False
    return "", True


def clear_part_hint(rect: Rect, facts: TurnFacts) -> str:
    """For a stockpile over buildings or zones: the largest clear rectangle inside the area.

    A model built a storage room, put a shelf in it, then asked three times for
    a stockpile over the whole inside. A shelf can't be inside a stockpile: it
    stores things by itself. Zones count too: a hint that left them out named
    the very rectangle a growing zone was blocking, and the model tried it
    again and again.
    """
    taken = {c for b in facts.buildings for c in b.area()}
    taken |= {c for bp in facts.unfinished() for c in bp.area()}
    taken |= facts.zone_cells
    free = {c for c in rect if c not in taken}
    best = biggest_rect(free, (rect.x1, rect.z1, rect.x2, rect.z2), ((rect.x1 + rect.x2) // 2, (rect.z1 + rect.z2) // 2))
    text = ("Creating a stockpile over an existing shelf or other building is refused. A shelf stores things "
            "by itself and can be built over stockpile cells, taking over those cells; the remaining "
            "stockpile does not need replacing. ")
    if best and best != rect:
        text += f"The largest clear part of that area is {best}: create_stockpile there. "
    elif best:
        # Nothing known blocks it, yet the game refused: don't send it back.
        return (text + "Something there that this prompt doesn't show is in the way; put the stockpile "
                "somewhere else. Nothing to fix here.")
    else:
        text += "No clear 3x3 part is left there, so the shelves are that room's storage. Nothing to fix here. "
        return text.rstrip()
    return text + "Next time make the stockpile first: a shelf built on it then takes over its own cells."


def construction_hint(action, submitted: list[dict], facts: TurnFacts) -> str:
    """After a construction order: which copy this is, what it costs against
    what is left, who builds it; for a room, what to put in it next."""
    names = sorted({bp["def_name"] for bp in submitted})
    parts = []
    for name in names:
        if name in ("Wall", "Door") or name in FLOORS:
            continue
        others = len(facts.unfinished(name))
        built = len(facts.built(name))
        if others or built:
            parts.append(f"You now have {others + 1} {name} planned and {built} built"
                         + (": more than one is rarely needed early" if name in ONE_AT_A_TIME else ""))
    builds = [dict(def_name=bp.def_name, stuff=bp.stuff, floor=bp.floor) for bp in facts.unfinished()]
    this = [dict(def_name=bp["def_name"], stuff=bp.get("stuff"), floor=bp.get("floor", False)) for bp in submitted]
    _, _, before = commitments(facts.items, facts.stored, builds)
    _, _, after = commitments(facts.items, facts.stored, builds + this)
    usable = stock_by_def(facts.items, False)
    costs = {m: n - before.get(m, 0) for m, n in after.items() if n > before.get(m, 0)}
    if costs:
        parts.append("It needs " + ", ".join(f"{n} {m}" for m, n in costs.items()) + "; with your other "
                     "unfinished builds that leaves " + ", ".join(
                         f"{m} {usable.get(m, 0) - after[m]}" if usable.get(m, 0) >= after[m]
                         else f"{m} short by {after[m] - usable.get(m, 0)} ({how_to_get(m)})" for m in costs))
    builders = facts.with_work("Construction")
    parts.append((f"Construction is on for {', '.join(builders)}: they haul the materials and build it while "
                  "time advances; RIMAPI doesn't report construction progress, so watch it go from waiting "
                  "to under construction to built") if builders else
                 "Nobody has Construction on: enable_work Construction or it never gets built")
    if action.action == "build_room":
        inside = inside_of(action.rect())
        parts.append(f"The inside is {inside}; next for a {action.purpose}: {NEXT[action.purpose]} once the walls stand")
    return ". ".join(parts) + "."


def success_hint(action, result: str, submitted: list[dict], facts: TurnFacts) -> str:
    if result.startswith("unchanged") or not facts:
        return ""
    if action.action in ("place_blueprint", "build_room", "build_wall", "build_floor") and submitted:
        return construction_hint(action, submitted, facts)
    if action.action == "create_growing_zone":
        growers = facts.with_work("Growing")
        text = (f"Colonists with Growing on sow and harvest it by themselves: {', '.join(growers)}." if growers
                else "Nobody has Growing on: enable_work Growing or nothing gets sown.")
        recommended = FIELD_CELLS_PER_COLONIST * max(1, len(facts.colonists))
        total = facts.field_cells + action.rect().cells
        if total > 2 * recommended:
            text += (f" You now have {total} field cells, {total / recommended:.0f} times the {recommended} "
                     f"recommended for {len(facts.colonists)} colonists: sowing and harvesting it all will take "
                     "much of their time. Don't add more fields for now.")
        return text
    if action.action == "set_research":
        researchers = facts.with_work("Research")
        return (f"It carries on by itself while time advances; Research is on for {', '.join(researchers)}."
                if researchers else "Nobody has Research on: enable_work Research or it stays stalled.")
    return ""



# Orders worth retrying when they fail: placement near misses, mostly off by a tile or two.
RETRY_ACTIONS = ("place_blueprint", "build_room", "build_wall", "build_floor", "create_growing_zone",
                 "create_stockpile", "connect_power", "set_bill")
MAX_FIX_TURNS = 3  # turns in a row a failed order may stay unfixed before it is dropped
# Failures no corrected order can fix: they stay off the retry list.
NOTHING_TO_FIX = ("Nothing to fix", "nothing to add there", "no observed", "Nothing to do", "which is enough")


def task_key(order) -> str:
    """What an order is for, ignoring where: "place_blueprint DiningChair". A
    corrected order (another spot) counts as fixing the failed one."""
    get = order.get if isinstance(order, dict) else (lambda name: getattr(order, name, None))
    detail = next((get(f) for f in ("building_def", "purpose", "plant", "terrain", "recipe_def_name") if get(f)), "")
    return f"{get('action')} {detail}".strip()


@dataclass
class Retry:
    order: dict      # the failed order, without its reason
    error: str       # what went wrong, with its HINT
    turns: int = 0   # turns it has ended unfixed


class RetryList:
    """Failed orders the next prompts ask to fix first, until fixed or MAX_FIX_TURNS.

    A model placing a chair one tile off used to give the task up after one
    failure. Now the order stays at the top of the feedback, with the hint's
    suggestion, until an order for the same thing succeeds. Not fixing it in a
    turn (trying and failing, or not trying) counts too, so it can't linger.
    """

    def __init__(self) -> None:
        self.items: dict[str, Retry] = {}

    def failed(self, action, error: str) -> None:
        if action.action not in RETRY_ACTIONS or any(s in error for s in NOTHING_TO_FIX):
            return
        key = task_key(action)
        old = self.items.get(key)
        order = action.model_dump(exclude={"reason"})
        if old is not None and old.order == order:
            # The same order again, failing again: repeating it won't fix it.
            del self.items[key]
            return
        self.items[key] = Retry(order, error, old.turns if old else 0)

    def succeeded(self, action) -> None:
        self.items.pop(task_key(action), None)

    def end_turn(self) -> list[str]:
        """Count a turn for everything still unfixed; drop what has run out. Returns notes."""
        notes = []
        for key, retry in list(self.items.items()):
            retry.turns += 1
            if retry.turns >= MAX_FIX_TURNS:
                del self.items[key]
                notes.append(f"Dropped {key}: still not placed after {MAX_FIX_TURNS} turns. Do something else, "
                             "or order it somewhere quite different.")
        return notes

    def lines(self) -> list[str]:
        if not self.items:
            return []
        out = ["RETRY IF STILL NEEDED: retry a failed order only if its intended result is still missing and "
               "needed for the current plan. A suggested placement is an option, not a new task. If another "
               "successful order already meets the need, leave the failed order behind. Each is "
               f"dropped after {MAX_FIX_TURNS} turns without a fix:"]
        for key, retry in self.items.items():
            out.append(f"  - {key}, unfixed for {retry.turns} of {MAX_FIX_TURNS} turns: {retry.order} failed: "
                       f"{retry.error[:400]}")
        return out
