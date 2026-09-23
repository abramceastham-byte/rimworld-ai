"""A concrete next step with every order's result, failed or not.

A bare reason ("would cover (117,140), where TableStonecutter is already
planned") left a model to guess the fix, and it guessed wrong: it read its own
blueprint as a conflict, "repositioned" it by placing another, and in one run
planned ten stonecutter's tables in 25 turns. So each result carries a hint
drawn from what this turn observed: free spots to use, the nearest targets,
who can do the work, what a build costs against what is left.
"""

from dataclasses import dataclass, field

from rimagent.choices import incapable_work
from rimagent.construction import BUILDINGS, FLOORS, Rect, footprint
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
            "they deteriorate. Best: a storage room (build_room purpose storage), then create_stockpile over "
            "its inside")
    return text + ("; place_blueprint Shelf (2x1) inside it: a shelf stores several stacks per cell, so the room "
                   "holds more in less space. A stockpile on open ground works too, but items there weather."
                   if shelf_buildable else
                   "; shelves hold more in less space once ComplexFurniture is researched.")


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


def pending_work(facts: TurnFacts, research, tables, live_zones, chop_orders: int,
                 short: set[str]) -> list[tuple[str, str, bool]]:
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
        forbidden = sorted((i for i in facts.items if i.is_forbidden),
                           key=lambda i: facts.near(i.position.x, i.position.z))[:3]
        if forbidden:
            return ("Nearest forbidden items: " + "; ".join(
                f"{i.def_name} x{i.stack_count} at ({i.position.x},{i.position.z})" for i in forbidden)
                + ". A rectangle around them allows them all.", True)
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
        return ("Nothing to do: there's room to spare. For better storage, build_room with purpose storage, "
                "place Shelf blueprints inside, and create_stockpile over its inside (that is always allowed).",
                False)
    if kind == "create_stockpile":
        return f"Free spots this turn: {spot_list(facts) or 'none'}.", False
    return "", True


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
        return (f"Colonists with Growing on sow and harvest it by themselves: {', '.join(growers)}." if growers
                else "Nobody has Growing on: enable_work Growing or nothing gets sown.")
    if action.action == "set_research":
        researchers = facts.with_work("Research")
        return (f"It carries on by itself while time advances; Research is on for {', '.join(researchers)}."
                if researchers else "Nobody has Research on: enable_work Research or it stays stalled.")
    return ""

