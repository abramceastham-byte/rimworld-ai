"""The agent loop: observe the game, ask a model what to do, act, and log it."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from rimagent.actions import Turn
from rimagent.blueprints import BlueprintTracker, Status, TrackedBlueprint
from rimagent.choices import Choices, FailureMemory, incapable_work
from rimagent.construction import (
    BUILDINGS,
    CROPS,
    FLOORS,
    STUFF_BY_CATEGORY,
    Rect,
    TerrainMap,
    available_buildings,
    footprint,
)
from rimagent.execution import execute
from rimagent.observation import read_operations
from rimagent.mapview import (
    BLUEPRINT,
    BUILDING,
    COLONIST,
    DOOR,
    FORBIDDEN,
    FRAME,
    GROWING,
    STOCKPILE,
    TREE,
    WALL,
    centre_of,
    detail_bounds,
    detail_rows,
    free_spots,
    overview_rows,
)
from rimagent.events import EventListener, GameEvent
from rimagent.memory import AgentMemory, MemoryUpdate
from rimagent.rimapi import (
    NORMAL,
    Alert,
    Building,
    Colonist,
    GameDefs,
    GameState,
    MapThing,
    RimApiClient,
    RimApiError,
    ResearchState,
    Threat,
    WorkTable,
)
from rimagent.runlog import RunLogger
from rimagent.zones import TrackedZone, ZoneTracker


GAME_FACTS = """Rules:
- Orders queue work; construction, chopping, growing and production need advancing time.
- Allowed supplies are not necessarily reachable. Chunks require stonecutting; use logs/steel/blocks to build.
- Growing needs fertile ground, suitable temperature and Growing work; no seeds are needed.
- Construction needs an enabled capable worker and all listed materials. Fuelled devices also need fuel;
  electrical devices need a connected power supply. A bill needs ingredients and a capable worker.
- A blueprint is an order, not a finished building. A room is a closed ring of walls with a door;
  colonists roof it automatically once it is closed.
"""

INSTRUCTIONS = """Manage this colony; choose your own priorities and strategy.
Reply as JSON: {"actions":[0 to 5 orders], "time":"advance" or "hold", "memory_update":optional}.
All orders execute while paused. Choose advance to let queued work progress after this turn;
hold keeps this turn paused. Choose afresh every turn. Empty actions + advance means wait.
Orders execute in order against refreshed state. If one fails, remaining orders are skipped;
repair the problem next turn. API acceptance is not proof of completed work.
Use exact names/IDs below. Never include display labels in def names. Each order has action FIRST,
a short reason, and only its fields. Already-satisfied settings are unchanged, not failures.
Fields (inclusive rectangles use x1,z1,x2,z2; x east, z north):
- enable_work / disable_work: colonist, work_type.
- set_bill: building_id, recipe_def_name, target_count (1..500); creates/updates Do until X.
- set_research: project (available name); no bills on research benches.
- create_growing_zone: plant, rectangle (<=225 cells).
- create_stockpile: rectangle (<=225); normal priority, accepts normal items.
- place_blueprint: building_def, x,z, rotation (0=N,1=E,2=S,3=W), stuff.
  x,z is RimWorld's anchor, not always a corner; use checked candidates. Fixed-cost buildings: stuff=null.
- build_room: outer rectangle (4..15 cells per side), stuff, door_side (north/east/south/west).
  One centred door in that side. Only the perimeter must be clear; walls or rock already on it are kept,
  anything else refuses the whole room. The inside is untouched; no floor included.
- build_wall: straight rectangle (<=20 cells), stuff; blocked cells are skipped and reported.
- build_floor: rectangle (<=225), terrain (listed floor name); existing/pending floors are skipped.
- designate: designation (mine/harvest/hunt), rectangle (<=400). Only use observed targets;
  hunt requires exactly ONE cell: x1=x2 and z1=z2, at an observed animal.
  harvest readiness is unknown unless stated; hunting risks retaliation and needs an armed hunter.
- allow_items: rectangle (<=400); makes forbidden items usable.
- chop_trees: rectangle (<=400), up to 30 trees; submits harvest requests, young/already-marked trees may be ignored.
Actions without usable targets are excluded from this turn's schema; unavailable capabilities are listed.
Keep memory brief: objective/steps/policies are intentions, observation is durable knowledge.
Do not copy live facts into memory. Mark completed only after observation confirms it.
Plan/policy changes are discarded on failed or partially verified orders.
"""


def describe_colonist(c: Colonist, work_types: list[str],
                      learned_incapable: dict[str, set[str]] | None = None) -> str:
    """One line per colonist: needs, job, skills, and work that is on, off or impossible."""
    passion = {0: "", 1: " (interested)", 2: " (burning)"}
    skilled = sorted(
        (s for s in c.skills if not s.totally_disabled), key=lambda s: -s.level
    )
    skills = ", ".join(f"{s.name} {s.level}{passion.get(s.passion, '')}" for s in skilled)
    on = [w.work_type for w in c.work_priorities if w.priority > 0]
    incapable = incapable_work(c, learned_incapable)
    off = [w for w in work_types if w not in on and w not in incapable]
    return (
        f"- {c.name} (age {c.age}): mood {c.mood:.0%}, health {c.health:.0%}, "
        f"fed {c.hunger:.0%}, doing {c.current_job or 'nothing'}. "
        f"Skills: {skills}. Works: {', '.join(on) or 'nothing'}. "
        f"Off: {', '.join(off) or 'none'}."
        + (f" Incapable of: {', '.join(sorted(incapable))}." if incapable else "")
    )


TICKS_PER_HOUR = 2500      # RimWorld: 60,000 ticks per in-game day
TICKS_PER_SECOND = 60      # at speed 1; speed 2 is 3x that, speed 3 is 6x
PAUSED_NUDGE_AFTER = 4     # consecutive held turns


@dataclass
class TimeWindow:
    """What happened while the loop let the game run after a decision."""

    seconds: float                 # real seconds the game ran
    ticks: int                     # game ticks that passed
    stopped_by: str | None         # None if it ran the full time, else why it stopped early
    events: list[GameEvent] = field(default_factory=list)  # letters/messages meanwhile
    game_paused: bool = False      # stopped because the game itself paused (RimWorld or the player)
    pause_reason: str = ""         # short phrase for the prompt, e.g. "the player paused it"

    def to_log(self) -> dict:
        return {"seconds": self.seconds, "ticks": self.ticks, "stopped_by": self.stopped_by,
                "game_paused": self.game_paused}


@dataclass
class TimeMode:
    """How time stood after the last turn, for the next prompt.

    The game is always paused while the model decides, and every reply chooses
    advance or hold afresh; nothing carries over except this report. paused is
    True after a hold, or when the game paused itself while time ran.
    """

    paused: bool
    reason: str = ""           # why it's paused, for the prompt
    paused_decisions: int = 0  # consecutive turns the model chose hold

    def to_log(self) -> dict:
        return {"paused": self.paused, "reason": self.reason,
                "paused_decisions": self.paused_decisions}


def mode_after_decision(mode: TimeMode, choice: str) -> TimeMode:
    """Count complete held turns; every reply explicitly chooses time again."""
    if choice == "advance":
        return TimeMode(paused=False)
    if choice != "hold":
        raise ValueError("time must be advance or hold")
    return TimeMode(True, "you chose hold", mode.paused_decisions + 1)


def mode_after_window(mode: TimeMode, window: TimeWindow) -> TimeMode:
    """If the game paused itself while time ran, stay paused until the model resumes.

    That's RimWorld's own auto-pause on a major threat, or the player pressing
    pause; either way the model should look before time runs again.
    """
    if window.game_paused:
        return TimeMode(paused=True, reason=window.pause_reason or "the game paused")
    return mode


def let_time_run(
    client: RimApiClient,
    events: EventListener | None,
    ticks: int,
    speed: int = NORMAL,
    poll_seconds: float = 0.5,
) -> TimeWindow:
    """Unpause until `ticks` of game time have passed, then pause again.

    Measured in game ticks, not real seconds, so a decision covers the same
    amount of colony time at any speed (RimWorld runs 60 ticks/s at speed 1,
    180 at speed 2, 360 at speed 3): a faster speed just shortens the wait.

    Stops early, so the model can respond straight away, when a threat letter
    arrives or when the game pauses itself (RimWorld can auto-pause on major
    threats) or is paused by the player.
    """
    start_tick = client.get_state().game_tick
    collected: list[GameEvent] = []
    stopped_by = None
    game_paused = False
    pause_reason = ""
    start = time.monotonic()
    # However slow the game runs, never block a whole turn on it.
    deadline = start + ticks / TICKS_PER_SECOND * 4 + 5
    client.resume(speed)
    try:
        while time.monotonic() < deadline:
            time.sleep(poll_seconds)
            new = events.drain() if events else []
            collected += new
            threat = next((e for e in new if e.category.startswith("Threat")), None)
            try:
                state = client.get_state()
                paused, passed = state.is_paused, state.game_tick - start_tick
            except (RimApiError, httpx.HTTPError):
                paused, passed = False, 0  # a missed check is fine; try again
            if passed >= ticks and not threat:
                break
            if threat:
                # RimWorld may have auto-paused for it at the same moment.
                game_paused = paused
                stopped_by = (
                    f"the game paused itself for a threat ({threat.kind}: {threat.text})"
                    if paused else f"a threat arrived ({threat.kind}: {threat.text})"
                )
                pause_reason = f"RimWorld paused it for a threat: {threat.text}"
                break
            if paused:
                game_paused = True
                stopped_by = "the game was paused (by RimWorld or the player)"
                pause_reason = "the player paused it"
                break
    finally:
        client.pause()
    return TimeWindow(
        seconds=round(time.monotonic() - start, 1),
        ticks=client.get_state().game_tick - start_tick,
        stopped_by=stopped_by,
        events=collected,
        game_paused=game_paused,
        pause_reason=pause_reason,
    )


def describe_time(mode: TimeMode, window: TimeWindow | None,
                  run_ticks: int, threat_run_ticks: int) -> str:
    text = (f"Time: orders execute paused. Choose advance for {run_ticks} game ticks "
            f"({threat_run_ticks} during danger), or hold for no elapsed time. "
            f"Consecutive held turns: {mode.paused_decisions}.")
    if mode.paused_decisions >= PAUSED_NUDGE_AFTER:
        text += " No work can progress while held; choose advance when waiting for results."
    if mode.reason:
        text += f" Last pause: {mode.reason}."
    if window:
        text += f" Last window: {window.ticks} ticks, {window.seconds:g}s real time."
        if window.stopped_by:
            text += f" Interrupted: {window.stopped_by}."
    return text


def honest_memory_update(update: MemoryUpdate | None, any_failed: bool) -> MemoryUpdate | None:
    """Drop plan and policy changes when an action in the same turn failed.

    A failed action changes nothing in the game, but models still report the
    step "completed" in the same reply. Observations are kept: they may well
    be about the failure.
    """
    if update is None or not any_failed:
        return update
    claims_progress = bool(
        update.replace_plan or update.clear_plan or update.status_updates
        or update.clear_statuses or update.policies_to_add or update.policies_to_remove
    )
    if not claims_progress:
        return update
    return MemoryUpdate(observation=update.observation) if update.observation else None


def describe_work_table(table: WorkTable) -> list[str]:
    """Describe one table without making the model invent ids or recipe names."""
    position = f"({table.position.x}, {table.position.z})"
    lines = [
        f"- {table.label} [{table.thing_def}], id {table.id}, at {position}:"
    ]
    if table.bills:
        lines.append("  Current bills:")
        for bill in table.bills:
            count = (
                bill.target_count
                if bill.target_count is not None
                else bill.repeat_count
            )
            count_text = f", count {count}" if count is not None else ""
            suspended = ", suspended" if bill.suspended else ""
            lines.append(
                f"  - bill {bill.load_id}: {bill.recipe_def_name}, "
                f"{bill.repeat_mode or 'unknown mode'}{count_text}{suspended}"
            )
    else:
        lines.append("  Current bills: none.")

    recipes = "; ".join(describe_recipe(r) for r in table.recipes)
    skills = sorted({r.work_skill for r in table.recipes if r.work_skill})
    lines.append(f"  Recipes (def: ingredients -> product){' use ' + '/'.join(skills) if skills else ''}: "
                 + (recipes or "none."))
    return lines


def describe_recipe(recipe) -> str:
    """e.g. "Make_Apparel_Parka: 80 fabric or leather", with the product only
    when the def name doesn't already say it."""
    details = recipe.model_extra or {}
    needs = " + ".join(f"{i.get('count', 0):g} {i.get('filter_label', '?')}"
                       for i in details.get("ingredients") or [])
    made = [p for p in details.get("products") or []
            if p.get("count", 1) != 1 or not recipe.def_name.endswith(p.get("thing_def", ""))]
    out = " -> " + ", ".join(f"{p.get('count', 1)} {p.get('thing_def')}" for p in made) if made else ""
    return f"{recipe.def_name}: {needs or 'no ingredients listed'}{out}"


MINIMAP_RADIUS = 12


def colony_center(colonists: list[Colonist]) -> tuple[int, int] | None:
    spots = [c.position for c in colonists if c.position]
    if not spots:
        return None
    return (round(sum(p.x for p in spots) / len(spots)), round(sum(p.z for p in spots) / len(spots)))


def stock_by_def(items: list[MapThing], forbidden: bool) -> dict[str, int]:
    """How much of each item def is lying around, allowed or forbidden."""
    totals: dict[str, int] = {}
    for i in items:
        if i.is_forbidden == forbidden:
            totals[i.def_name] = totals.get(i.def_name, 0) + i.stack_count
    return totals


def describe_supplies(items: list[MapThing], limit: int = 12) -> list[str]:
    """What the colony can build with now, and what is locked behind allow_items."""
    def listed(totals: dict[str, int]) -> str:
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:limit]
        more = f", and {len(totals) - limit} more kinds" if len(totals) > limit else ""
        return ", ".join(f"{name} {n}" for name, n in top) + more if top else "nothing"

    usable, locked = stock_by_def(items, False), stock_by_def(items, True)
    return [
        f"Map-wide allowed supplies you can use now: {listed(usable)}.",
        f"Map-wide forbidden, unusable until you allow_items: {listed(locked)}.",
    ]


def describe_research(state: ResearchState) -> list[str]:
    """What the colony is researching, and what it could start instead."""
    if state.current:
        now = f"Researching {state.current.label} ({state.current.progress_percent:.0f}% done)."
    else:
        now = "Researching nothing."
    if not state.has_bench:
        return [now + " No research bench is built, so no research happens yet."]
    choices = ", ".join(f"{p.name} ({p.label}, {p.research_points:.0f} points)"
                        for p in state.available) or "none"
    return [now, f"Could start now: {choices}."]


def blocked_by(check: object) -> bool:
    """True when RIMAPI's area check found anything in the way."""
    return bool(getattr(check, "terrain", []) or getattr(check, "ores", [])
                or getattr(check, "buildings", []) or getattr(check, "zones", []))


def describe_map(
    colonists: list[Colonist],
    terrain: TerrainMap,
    defs: GameDefs,
    zones: list[TrackedZone],
    finished_research: set[str],
    items: list[MapThing] | None = None,
    trees: list[MapThing] | None = None,
    blueprints: list[tuple[TrackedBlueprint, Status]] | None = None,
    buildings: list[Building] | None = None,
    check_area: Callable[[Rect], object] | None = None,
) -> list[str]:
    """Everything the model needs to pick a place without reading the grid.

    Two scales (see mapview): a coarse overview of the surroundings, and a
    detail view that follows the base. Exact positions are also listed in text
    below, and free_spots offers rectangles that are already clear.
    """
    forbidden = [i for i in items or [] if i.is_forbidden]
    blueprint_marks = {"waiting": BLUEPRINT, "under construction": FRAME}
    lines = [f"Map: {terrain.width} x {terrain.height} cells. x grows east, z grows north.",
             "Positions below are exact; read them from the text, not by counting the grid."]
    for c in colonists:
        if c.position:
            lines.append(f"- {c.name} is at ({c.position.x}, {c.position.z}).")

    # One mark per cell; the most important thing wins.
    marks: dict[tuple[int, int], str] = {}
    tree_cells = {(t.position.x, t.position.z) for t in trees or []}
    for cell in tree_cells:
        marks[cell] = TREE
    for i in forbidden:
        marks[(i.position.x, i.position.z)] = FORBIDDEN
    built_cells: set[tuple[int, int]] = set()
    for b in buildings or []:
        mark = (FRAME if b.under_construction else
                WALL if b.def_name == "Wall" else
                DOOR if b.def_name == "Door" else BUILDING)
        for cell in b.area():
            marks[cell] = mark
            built_cells.add(cell)
    for bp, status in blueprints or []:
        if status in blueprint_marks:
            for cell in bp.area():
                marks[cell] = blueprint_marks[status]
                built_cells.add(cell)
    zone_cells: set[tuple[int, int]] = set()
    for zone in zones:
        for cell in zone.area():
            marks[cell] = GROWING if zone.kind == "growing" else STOCKPILE
            zone_cells.add(cell)
    for c in colonists:
        if c.position:
            marks[(c.position.x, c.position.z)] = COLONIST

    # The base is the anchor once anything is built; colonists before that.
    base_cells = [(b.position.x, b.position.z) for b in buildings or []]
    base_cells += [(bp.x, bp.z) for bp, status in blueprints or [] if status != "gone"]
    colonist_cells = [(c.position.x, c.position.z) for c in colonists if c.position]
    anchor = centre_of(base_cells) or centre_of(colonist_cells)
    if anchor is None:
        return lines

    view = detail_bounds(anchor, base_cells + colonist_cells, terrain.width, terrain.height)
    lines += ["", *overview_rows(anchor, terrain, defs.terrain, tree_cells, built_cells)]
    lines += ["", *detail_rows(view, terrain, defs.terrain, marks)]

    spots = free_spots(anchor, terrain, defs.terrain, built_cells | zone_cells,
                       is_clear=(lambda rect: not blocked_by(check_area(rect))) if check_area else None)
    if spots:
        lines += ["", "Free spots near the base, checked just now (use these coordinates):"]
        lines += [f"  {letter}: {rect} - {note}" for letter, rect, note in spots]

    lines += ["", *describe_supplies(items or [], limit=24)]
    nearby = stock_by_def([i for i in items or [] if view.contains(i.position.x, i.position.z)], False)
    lines.append("Of those, inside the detail view: "
                 + (", ".join(f"{k} {v}" for k, v in sorted(nearby.items(), key=lambda kv: -kv[1]))
                    or "nothing") + ".")
    # Where to point allow_items: the nearest forbidden items, not a box around
    # every one on the map (which covers everything and helps nobody).
    def steps_away(i: MapThing) -> int:
        return max(abs(i.position.x - anchor[0]), abs(i.position.z - anchor[1]))

    # The nearest cluster, not every nearby item: a box around scattered items
    # is larger than allow_items accepts.
    closest_first = sorted((i for i in forbidden if steps_away(i) <= 30), key=steps_away)
    near = []
    if closest_first:
        first = closest_first[0]
        near = [i for i in closest_first
                if max(abs(i.position.x - first.position.x),
                       abs(i.position.z - first.position.z)) <= 8]
    if not near and forbidden:
        closest = min(forbidden, key=steps_away)
        lines.append(f"No forbidden items near the base; the closest is {closest.def_name} at "
                     f"({closest.position.x},{closest.position.z}), {steps_away(closest)} cells away.")
    elif near:
        box = Rect.from_corners(
            min(i.position.x for i in near), min(i.position.z for i in near),
            max(i.position.x for i in near), max(i.position.z for i in near),
        )
        kinds = ", ".join(sorted({i.def_name for i in near})[:6])
        lines.append(f"Nearest forbidden items ({kinds}) are inside {box}; allow_items only "
                     "works on a rectangle that actually contains some.")
    else:
        lines.append("Nothing on the map is forbidden, so allow_items has nothing to do.")
    lines.append(f"Trees in the detail view: {sum(1 for cell in tree_cells if view.contains(*cell))}.")

    lines += ["", "Your zones: " + ("; ".join(z.describe() for z in zones) or "none") + "."]

    done = [b for b in buildings or [] if not b.under_construction]
    building_lines = [f"{b.def_name} id={b.id} covering {b.area()}" for b in done]
    lines.append("Your buildings: " + ("; ".join(building_lines) or "none yet") + ".")
    started = [b for b in buildings or [] if b.under_construction]
    if started:
        lines.append("Being built now: " + "; ".join(
            f"{b.label} at ({b.position.x},{b.position.z})" for b in started) + ".")

    if blueprints:
        usable = stock_by_def([i for i in items or []], False)
        lines.append("Your blueprints, checked in the game just now:")
        for bp, status in blueprints:
            if status == "waiting":
                note = "waiting to be built"
                if bp.stuff and not usable.get(bp.stuff):
                    note += (f" - no usable {bp.stuff} on the map, so nobody can build it "
                             "(allow_items, or chop trees for wood)")
                else:
                    note += " (needs a colonist with Construction enabled)"
            else:
                note = {"under construction": "under construction", "built": "built",
                        "gone": "gone (cancelled or destroyed)"}[status]
            lines.append(f"- {bp.label()}, covering {bp.area()}: {note}")
    else:
        lines.append("Your blueprints: none.")

    buildable = available_buildings(finished_research)
    heading = ("Buildable (def name, size facing north, cost; 'of A/B' means stuff is one of "
               "those, otherwise stuff=null):")
    lines += ["", heading]
    for name, spec in buildable.items():
        label = "" if spec.label.replace(" ", "").lower() == name.lower() else f" ({spec.label})"
        lines.append(f"- {name}{label}, {spec.size[0]}x{spec.size[1]}, {spec.cost_text()}")

    lines.append(
        "Stuff materials: "
        + "; ".join(f"{cat} = {', '.join(names)}" for cat, names in STUFF_BY_CATEGORY.items())
        + "."
    )
    crops = ", ".join(
        f"{name} ({label}, fertility {defs.crop_fertility_min.get(name, 0):.1f}+)"
        for name, label in CROPS.items()
    )
    lines.append(f"Crops for growing zones: {crops}.")
    lines.append("Floor defs and cost per cell: " + "; ".join(
        f"{n}: {f.cost[1]} {f.cost[0]}" for n, f in FLOORS.items()
        if n in defs.terrain and (not f.research or f.research in finished_research)))
    wall, door = BUILDINGS["Wall"].stuff_count, BUILDINGS["Door"].stuff_count
    lines.append(f"build_room cost: {wall} per wall cell + {door} for the door, e.g. a 6x5 room "
                 f"has 17 walls + 1 door = {17 * wall + door} material; existing walls are reused.")
    nearest_trees = sorted(trees or [], key=steps_away)[:8]
    lines.append("Nearest trees (young ones give no wood yet): " + "; ".join(
        f"{t.def_name} at ({t.position.x},{t.position.z})" for t in nearest_trees))
    if spots:
        # place_blueprint's x,z is RimWorld's anchor, which is not the corner
        # for wider buildings. One worked example per footprint size, computed
        # with the same rule placement is checked against, inside a free spot.
        letter, rect, _ = spots[-1]  # the largest spot fits every size
        examples = []
        for size in sorted({spec.size for spec in buildable.values()} - {(1, 1)}):
            anchor_cell = next(((x, z) for x, z in rect
                                if all(rect.contains(*c) for c in footprint(x, z, size, 0))), None)
            if anchor_cell:
                x, z = anchor_cell
                examples.append(f"{size[0]}x{size[1]} at ({x},{z}) covers {footprint(x, z, size, 0)}")
        if examples:
            lines.append(f"Anchors facing north, e.g. in spot {letter}: " + "; ".join(examples)
                         + ". A 1x1 building covers just its x,z.")
    return lines


def build_prompt(
    state: GameState,
    colonists: list[Colonist],
    alerts: list[Alert],
    threats: list[Threat],
    events: list[GameEvent],
    work_types: list[str],
    work_tables: list[WorkTable],
    research: ResearchState,
    time_status: str,
    memory: AgentMemory,
    map_lines: list[str] | None = None,
    last_results: list[str] | None = None,
    learned_incapable: dict[str, set[str]] | None = None,
) -> str:
    """Build a prompt for the LLM based on the current game state."""
    lines = [
        "Current game state:",
        time_status,
        *(["", "What your last orders actually did, and failures still unresolved:",
           *[f"- {r}" for r in last_results]] if last_results else []),
        "",
        f"Tick {state.game_tick}. {state.colonist_count} colonists.",
        f"Wealth {state.colony_wealth:,.0f}. Storyteller {state.storyteller}.",
        "",
        "Threats:",
        *([t.summary() for t in threats] or ["None."]),
        "",
        "New since last step:",
        *([f"- {e.kind} ({e.category}): {e.text}" for e in events] or ["Nothing."]),
        "",
        "Alerts: " + (", ".join(a.label for a in alerts) or "none") + ".",
        "",
        "Colonists:",
        *[describe_colonist(c, work_types, learned_incapable) for c in colonists],
        "",
        "Research: " + " ".join(describe_research(research)),
        "",
        "Work tables and production bills:",
        *(
            [line for table in work_tables for line in describe_work_table(table)]
            or ["None. Build a work table before trying to set a bill."]
        ),
        "",
        *(map_lines or []),
        "",
    ]
    return "\n".join(lines) + GAME_FACTS + INSTRUCTIONS + "\n" + memory.to_prompt()


def request_turn(
    model: Callable[[str], str],
    prompt: str,
    logger: RunLogger,
    step: int,
    danger_present: bool,
    choices=None,
) -> tuple[Turn, object | None]:
    """Ask twice if needed, logging failures before choosing a safe fallback."""
    attempt_prompt = prompt
    last_reply: object | None = None

    for attempt in range(1, 3):
        try:
            reply = model(attempt_prompt)
            last_reply = reply
        except Exception as error:  # noqa: BLE001 - isolate arbitrary model adapters
            # Model adapters can fail because of timeouts, connection errors, or
            # server errors. KeyboardInterrupt/SystemExit are intentionally not caught.
            logger.log_model_failure(
                {
                    "step": step,
                    "attempt": attempt,
                    "failure_type": "model_call_error",
                    "response": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        try:
            turn = Turn.model_validate_json(reply)
            if choices is not None:
                choices.validate(turn)
            return turn, reply
        except (ValidationError, TypeError, ValueError) as error:
            logger.log_model_failure(
                {
                    "step": step,
                    "attempt": attempt,
                    "failure_type": "invalid_response",
                    "response": reply,
                    "thinking": getattr(model, "last_thinking", None),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if attempt == 1:
                attempt_prompt = (
                    prompt
                    + "\n\nYour previous response was invalid:\n"
                    + str(reply)[:4000]
                    + "\n\nValidation error:\n"
                    + str(error)
                    + "\nReturn one corrected JSON object only."
                )

    # The fallback gives no orders and holds: time never runs on a turn the
    # model didn't choose, which matters most during a threat.
    logger.log_model_failure({
        "step": step, "attempt": None, "failure_type": "fallback_hold",
        "response": None, "error": "model failed twice; holding with no orders"
        + (" (a threat is present)" if danger_present else ""),
    })
    return Turn(time="hold", actions=[]), last_reply


def action_area(target: dict) -> Rect | None:
    """The cells an order is about, widened a little, or None for non-map orders."""
    if "x1" in target:
        rect = Rect.from_corners(target["x1"], target["z1"], target["x2"], target["z2"])
    elif "x" in target:
        rect = Rect(target["x"], target["z"], target["x"], target["z"])
    else:
        return None
    return Rect(rect.x1 - 3, rect.z1 - 3, rect.x2 + 3, rect.z2 + 3)


def prepare_turn(client, memory, blueprints, zones, mode, window=None,
                 run_ticks=600, threat_run_ticks=180, new_events=(), last_results=(), failures=None,
                 read_only=False, learned_incapable=None):
    """Observe the game and build this turn's prompt and choices.

    Shared by live play and dry runs; with read_only=True nothing is saved,
    so a dry run never touches memory, zone or blueprint files.
    """
    client.begin_snapshot()
    state = client.get_state()
    colonists = client.get_colonists()
    alerts, threats = client.get_alerts(), client.get_threats()
    work_types, tables = client.get_work_types(), client.get_work_tables()
    research = client.get_research()  # also seeds completed-research snapshot cache
    terrain, defs = client.get_terrain(), client.get_game_defs()
    items, trees = client.get_items(), client.get_trees()
    buildings, live_zones = client.get_buildings(), client.get_zones()
    live_zone_sizes = {z.id: z.cells_count for z in live_zones}
    known_zones = [z for z in zones.items if live_zone_sizes.get(z.zone_id) == z.area().cells]
    if not read_only and known_zones != zones.items:
        zones.items = known_zones
        zones.save()
    report = blueprints.check(client, forget_finished=not read_only)
    operations = read_operations(client, colonists, defs)
    unknown = [z for z in live_zones if z.id not in {k.zone_id for k in known_zones}]
    if unknown:
        operations.lines.append("Zones with unknown/edited geometry (placement checks still protect them): " +
                                "; ".join(f"{z.label} id={z.id}, {z.cells_count} cells" for z in unknown))
    choices = Choices.observe(colonists, work_types, tables, research, defs, terrain,
                              items, trees, [k for k, v in operations.targets.items() if v],
                              learned_incapable)
    map_lines = describe_map(colonists, terrain, defs, known_zones,
                             {p.name for p in research.all_projects if p.is_finished},
                             items, trees, report, buildings, client.check_area)
    time_status = describe_time(mode, window, run_ticks, threat_run_ticks)

    def local_zones(area):
        if area is None:
            return []
        try:
            check = client.check_area(Rect(max(0, area.x1), max(0, area.z1),
                min(terrain.width - 1, area.x2), min(terrain.height - 1, area.z2)))
            return sorted((i.x, i.z, i.zone_type, i.label) for i in check.zones)
        except (RimApiError, httpx.HTTPError, ValueError):
            # Recording a failed order must not itself abort the agent.
            return ["zone observation unavailable"]

    def fingerprint(target: dict) -> str:
        """The state a failed order depended on. When it changes, the failure
        is dropped from the prompt: the retry might now work."""
        kind = target["action"]
        if kind in ("enable_work", "disable_work"):
            # Only work settings: mood or hunger changing mustn't release it.
            data = [(c.id, [w.model_dump() for w in c.work_priorities])
                    for c in colonists if c.name == target.get("colonist")]
        elif kind == "set_research":
            data = [research.current.name if research.current else None,
                    sorted(p.name for p in research.available)]
        elif kind == "set_bill":
            data = [t.model_dump() for t in tables if t.id == target.get("building_id")]
        else:
            # Map orders: only what lies in and around the order's own area,
            # so a colonist hauling elsewhere doesn't release the failure.
            area = action_area(target)
            near = (lambda x, z: area.contains(x, z)) if area else (lambda x, z: True)
            data = [
                sorted((b.def_name, b.position.x, b.position.z, b.rotation, b.size.x, b.size.z, b.under_construction)
                       for b in buildings if any(near(*cell) for cell in b.area())),
                # Spatial API query includes hand-drawn zones; unrelated zones
                # elsewhere must not expire this failure.
                local_zones(area),
                [(x, z, terrain.at(x, z)) for x, z in area
                 if 0 <= x < terrain.width and 0 <= z < terrain.height] if area else [],
                sorted((b.def_name, b.x, b.z, b.rotation, b.cells, status) for b, status in report if near(b.x, b.z)),
                sorted((i.def_name, i.is_forbidden) for i in items if near(i.position.x, i.position.z)),
                sorted(t.thing_id for t in trees if near(t.position.x, t.position.z)),
                {k: [t for t in v if near(t["x"], t["z"])] for k, v in operations.targets.items()},
                # Materials anywhere count: a build can fail for want of wood.
                stock_by_def(items, False).get(target.get("stuff") or "", 0) > 0,
            ]
        return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)

    if failures:
        failures.reconcile(fingerprint)
    feedback = list(last_results) + (failures.lines() if failures else [])
    unavailable = ["Unavailable now: " + "; ".join(choices.unavailable)] if choices.unavailable else []
    prompt = build_prompt(state, colonists, alerts, threats, list(new_events), work_types,
                          tables, research, time_status, memory,
                          map_lines + operations.lines + unavailable,
                          feedback, learned_incapable)
    return dict(prompt=prompt, choices=choices, operations=operations, state=state,
                threats=threats, time_status=time_status, fingerprint=fingerprint)


def run(client: RimApiClient, model: Callable[[str], str], logger: RunLogger,
        events: EventListener | None = None, max_steps: int = 10,
        run_ticks: int = 10 * TICKS_PER_SECOND, threat_run_ticks: int = 3 * TICKS_PER_SECOND,
        speed: int = NORMAL, memory: AgentMemory | None = None) -> None:
    """Run the agent loop for max_steps turns.

    Each turn: pause, observe, ask the model, carry out its orders in order
    (stopping at the first failure), then follow its time choice. advance lets
    the game run run_ticks (threat_run_ticks while a threat is active),
    stopping early if a threat arrives or the game pauses itself; hold keeps it
    paused. Nothing about time carries over: every reply chooses again.
    """
    if run_ticks <= 0 or threat_run_ticks <= 0:
        raise ValueError("time windows must contain at least one tick")
    memory = memory or AgentMemory()
    blueprints = BlueprintTracker(memory.memory_dir / "blueprints.json")
    zones = ZoneTracker(memory.memory_dir / "zones.json")
    failures = FailureMemory()
    learned_incapable: dict[str, set[str]] = {}  # colonist name -> work RIMAPI refused
    mode = TimeMode(True, "session start")
    window = None
    carried_events, last_results = [], []
    try:
        for step in range(max_steps):
            client.pause()
            new_events = carried_events + (events.drain() if events else [])
            carried_events = []
            try:
                context = prepare_turn(client, memory, blueprints, zones, mode, window,
                                       run_ticks, threat_run_ticks, new_events, last_results, failures,
                                       learned_incapable=learned_incapable)
            except (RimApiError, httpx.HTTPError, ValueError) as error:
                logger.log_model_failure({"step": step, "failure_type": "observation_error", "error": str(error)})
                # Preserve threat events even when this observation fails.
                carried_events = new_events
                print(f"Step {step}: observation failed; held paused: {error}")
                time.sleep(2)
                continue
            prompt, choices = context['prompt'], context['choices']
            if getattr(model, 'schema', None) is not None:
                model.schema = choices.schema()
            logger.log_state(context['state'])
            for event in new_events:
                logger.log_event(event)
            danger = any(t.active for t in context['threats']) or any(e.category.startswith('Threat') for e in new_events)
            turn, reply = request_turn(model, prompt, logger, step, danger, choices)
            results, any_failed = [], False
            # Empty turns still retain prompt/reasoning/reply in logs.
            for index, action in enumerate(turn.actions or [None]):
                error, result = None, "no orders"
                if action is not None:
                    if any_failed:
                        result = "skipped: an earlier order failed; observations must be refreshed"
                    else:
                        try:
                            result = execute(client, action, blueprints, zones, context['operations'],
                                             learned_incapable)
                        except (RimApiError, httpx.HTTPError, ValueError) as exc:
                            error = str(exc)
                            result = f"FAILED {action.model_dump(exclude={'reason'})}: {error}"
                            any_failed = True
                            failures.remember(action, error, context['fingerprint'](action.model_dump()))
                    results.append(result)
                    memory.remember_decision(step, action.action, action.reason, error,
                                             getattr(action, 'colonist', None), getattr(action, 'work_type', None))
                logger.log_decision({
                    "step": step, "action_index": index, "actions_in_turn": len(turn.actions),
                    **(action.model_dump() if action else {"action": "no_orders", "reason": "observe progress"}),
                    "time": turn.time, "result": result, "error": error,
                    "prompt": prompt if index == 0 else None, "reply": reply if index == 0 else None,
                    "time_status": context['time_status'] if index == 0 else None,
                    "thinking": getattr(model, 'last_thinking', None) if index == 0 else None,
                    "model_stats": getattr(model, 'last_stats', None) if index == 0 else None,
                    "choice_schema": choices.schema() if index == 0 else None,
                })
                print(f"Step {step}.{index}: {result}")
            last_results = results
            memory.apply_update(step, context['state'].game_tick,
                                honest_memory_update(turn.memory_update, any_failed))
            mode = mode_after_decision(mode, turn.time)
            if turn.time == 'hold':
                window = None
            else:
                # Catch threat events that arrived while orders were being processed.
                arrived = events.drain() if events else []
                carried_events += arrived
                if any(e.category.startswith('Threat') for e in arrived):
                    window = TimeWindow(0, 0, "new threat during orders; inspect before advancing", events=arrived)
                else:
                    window = let_time_run(client, events, threat_run_ticks if danger else run_ticks, speed)
                    carried_events += window.events
                mode = mode_after_window(mode, window)
            logger.log_decision({"step": step, "action": "turn_end", "reason": "time",
                                 "time": turn.time, "time_after": window.to_log() if window else "stayed paused"})
    finally:
        # Runs even on Ctrl+C or a crash: don't leave the colony unattended,
        # but don't hide the original error if RIMAPI is the thing that's down.
        try:
            client.pause()
        except (RimApiError, httpx.HTTPError):
            pass
