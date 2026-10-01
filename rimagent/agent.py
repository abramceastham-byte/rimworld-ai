"""The agent loop: observe the game, ask a model what to do, act, and log it."""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ValidationError

from rimagent.actions import Turn
from rimagent.blueprints import BlueprintTracker, Status, TrackedBlueprint
from rimagent.choices import Choices, FailureMemory, incapable_work
from rimagent.chopping import ChopTracker
from rimagent.mining import MineTracker
from rimagent.hints import (IdleTracker, RetryList, TurnFacts, failure_hint, field_refusal, idle_hint, pending_work,
                            stockpile_refusal, storage_hint, success_hint, task_key, watch_spot_refusal)
from rimagent.defense import defense_lines
from rimagent.finish import (ALL, FOOD, FinishCheck, FinishState, canonical_item, finish_progress, is_food,
                             unreachable_count)
from rimagent.furnish import furnish_line
from rimagent.rooms import SLEEPERS, RoomTracker, housing_advice, inside_of, perimeter
from rimagent.layout import LayoutTracker, field_area_refusal, layout_hint, room_refusal, room_slots
from rimagent.construction import (
    BUILDING_USE,
    BUILDINGS,
    CROPS,
    FLOORS,
    STUFF_BY_CATEGORY,
    Rect,
    TerrainMap,
    available_buildings,
    footprint,
)
from rimagent.execution import Combat, execute
from rimagent.observation import FIELD_CELLS_PER_COLONIST, read_operations
from rimagent.mapview import (
    AREA_MARKS,
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
    suggested_field,
    overview_rows,
)
from rimagent.events import EventListener, GameEvent, describe_event, is_attack
from rimagent.memory import AgentMemory, CurrentPlan, MemoryUpdate, PlanStep
from rimagent.modes import (COMBAT, FAILED_TURNS_TO_REPLAN, MAX_FOCUS_TURNS, MAX_TOOLKITS, FocusTurn, Overview,
                            ThreatTurn, allowed_actions, designations, focus_prompt, food_emergency, narrowed,
                            overview_prompt, plan_mismatch,
                            toolkit_work,
                            threat_prompt)
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

# What the single-prompt loop offers: the orders it had before the modes loop.
LEGACY_ACTIONS = {"enable_work", "disable_work", "set_bill", "set_research", "create_growing_zone",
                  "create_stockpile", "place_blueprint", "designate", "build_wall", "build_floor", "build_room",
                  "allow_items", "chop_trees"}
# Extra steps past --steps the modes loop may take to finish a plan or a fight.
OVERRUN_STEPS = MAX_FOCUS_TURNS + 6
# Cells a suggested field keeps from the base's middle, where the first rooms go.
FIELD_GAP = 5
POWER_SOURCES = {"WoodFiredGenerator", "SolarGenerator", "WindTurbine", "ChemfuelPoweredGenerator",
                 "GeothermalGenerator", "Battery"}
from rimagent.supplies import describe_supplies, how_to_get, short_materials, stock_by_def, usable_stuff
from rimagent.zones import TrackedZone, ZoneTracker


GAME_FACTS = """Rules:
- Orders queue work; construction, chopping, growing and production need advancing time.
- Allowed supplies are not necessarily reachable.
- Materials: build walls and furniture in WoodLog; trees regrow (chop_trees for more). Stone blocks are
  a fireproof upgrade for later, not a first step. Save Steel: it doesn't regrow, and components,
  machines, power and weapons need it. Use Steel for building only when nothing else will do.
- Growing needs fertile ground, suitable temperature and Growing work; no seeds are needed.
  Colonists with Growing work sow and harvest their growing zones by themselves when crops are ripe.
- Construction needs an enabled capable worker and all listed materials. Fuelled devices also need fuel;
  electrical devices need a connected power supply. A bill needs ingredients and a capable worker.
- A blueprint is an order, not a finished building. Blueprints can't be moved or cancelled from here:
  placing one again makes a second copy. A room is a closed ring of walls with a door; colonists
  roof it automatically once it is closed, so never wall in a growing zone.
- Layout: keep the colony compact and convenient, since colonists walk to everything. Before placing a
  building, look at what surrounds the spot: put it near what it works with (beds inside bedrooms,
  the stove by the dining room and food storage, work tables near their materials' stockpile), keep
  doors and walkways clear, and leave room for the rooms you will add next.
"""

INSTRUCTIONS = """You are the colony's manager, not a colonist. Your orders are standing orders: the research
project, enabled work, placed blueprints, bills, zones and cut orders all stay in effect, and colonists
carry them out by themselves whenever time advances and they are able to.
Never repeat an order that is already in effect; it changes nothing. If one isn't progressing, fix the
cause (missing material, nobody with that work on, no bench or table) or simply advance time.
Most turns need a few new orders, or none, then advance. Choose your own priorities and strategy.
Reply as JSON: {"actions":[0 to 10 orders], "time":"advance" or "hold", "memory_update":optional}.
All orders execute while paused. Choose advance to let queued work progress after this turn;
hold keeps this turn paused. Choose afresh every turn. Empty actions + advance means wait.
Orders execute in order against refreshed state, and every one is tried even if an earlier one
failed, so an order that relied on a failed one may fail too. Next turn each is reported by number
as OK, UNCHANGED or FAILED. API acceptance is not proof of completed work. A result may end
with HINT: an option drawn from this turn's state, only if still needed for your goal, not a new task.
Use exact names/IDs below. Never include display labels in def names. Each order has action FIRST,
a short reason, and only its fields.
Fields (inclusive rectangles use x1,z1,x2,z2; x east, z north):
- enable_work / disable_work: colonist, work_type.
- set_bill: building_id, recipe_def_name, target_count (1..500); creates/updates Do until X.
- set_research: project (available name); no bills on research benches.
- create_growing_zone: plant, rectangle (<=225 cells), in the open: never inside a room or where you
  mean to build one, since a closed room is roofed and crops need sunlight.
- create_stockpile: rectangle (<=225); normal priority, accepts normal items. While storage is under
  half full it is refused except inside a room built with purpose storage: prefer a storage room with
  Shelf blueprints to more open-air stockpiles.
- place_blueprint: building_def, x,z, rotation (0=N,1=E,2=S,3=W), stuff.
  x,z is RimWorld's anchor, not always a corner; use checked candidates. Fixed-cost buildings: stuff=null.
- build_room: OUTER rectangle including the walls (4..15 cells per side; inside is 2 smaller each way,
  so a 3x3 inside needs 5x5 outer), stuff, door_side (north/east/south/west), purpose (bedroom,
  barracks, kitchen, dining, workshop, storage, hospital, other: later prompts say what it still needs).
  One centred door in that side. Only the perimeter must be clear; walls, doors or rock already on it are kept,
  anything else refuses the whole room. The inside is untouched; no floor included.
- build_wall: straight rectangle (<=20 cells), stuff; blocked cells are skipped and reported. It only
  adds walls: nothing here removes a wall or building.
- build_floor: rectangle (<=225), terrain (listed floor name); existing/pending floors are skipped.
- designate: never for trees (use chop_trees). mine = rock only, hunt = one animal, harvest = ripe wild plants.
  Rectangle: mine <=400 cells, harvest any size. Only use observed targets;
  hunt requires exactly ONE cell: x1=x2 and z1=z2, at an observed animal.
  harvest marks ripe plants only (wild ones; fields are harvested anyway); hunting risks retaliation and needs an armed hunter.
- allow_items: what = all, Food (meals and raw food) or one forbidden item def; makes every forbidden item of that kind on the whole map usable at once.
- chop_trees: THE way to get wood from trees. Any rectangle; orders up to 30 trees nearest its centre cut.
  Trees already ordered are skipped; the game ignores trees too young for wood.
Actions without usable targets are excluded from this turn's schema; unavailable capabilities are listed.
memory_update keeps what matters for the whole game, not a diary of this turn:
- observation: a big discovery or idea worth remembering for the rest of the colony's life, e.g.
  "colonists prefer beds inside a room to beds outdoors",
  "colonists are happier with a bedroom each than sharing", "colonists are comfortable between
  16 and 26 C", "raids have come from the east edge". Not what you ordered or what happened this
  turn, and no numbers the prompt shows every turn (stock, jobs, ticks): those go stale.
- replace_plan / policies: your objective, its steps and your standing rules.
A step you mark completed is held as in_progress until a later turn: mark it completed again once
the game shows it done.
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
# Game time per advance: 3,750 ticks (1.5 in-game hours), which is about 10.4
# real seconds at speed 3, 21 at speed 2, 62.5 at speed 1. While a threat is
# active, 30% of that (1,125 ticks). Shorter windows spent most of a run asking
# the model about a colony that had barely moved; with focus prompts keeping
# the model on its plan, colonists get more time to carry it out.
RUN_TICKS = round(1.5 * TICKS_PER_HOUR)
THREAT_RUN_TICKS = round(RUN_TICKS * 0.3)


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
            threat = next((e for e in new if is_attack(e)), None)
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


def describe_research(state: ResearchState, researchers: list[str] | None = None) -> list[str]:
    """What the colony is researching, who works on it, and what it could switch to.

    Says outright that the current project carries on by itself: models kept
    selecting the project already being researched, turn after turn.
    """
    current = state.current.name if state.current else None
    if state.current:
        done = state.current.progress_percent
        points = state.current.model_extra.get("progress") if state.current.model_extra else None
        of = (f", {points:g} of {state.current.research_points:g} points"
              if isinstance(points, (int, float)) and state.current.research_points else "")
        now = (f"Researching {state.current.label} ({done:.0%} done{of}). This is a "
               "standing order: it carries on by itself while time advances")
        if researchers is None:
            now += "."
        elif researchers:
            now += f", worked on by colonists with Research on: {', '.join(researchers)}."
        else:
            now += ", but nobody has Research on, so it is stalled: enable_work Research."
    else:
        now = "Researching nothing: choose a project with set_research."
    if not state.has_bench:
        return [now + " No research bench is built, so no research happens yet."]
    choices = ", ".join(f"{p.name} ({p.label}, {p.research_points:.0f} points)"
                        for p in state.available if p.name != current) or "none"
    return [now, f"{'Could switch to' if current else 'Could start now'}: {choices}."]


def issue_cells(check: object) -> set[tuple[int, int]]:
    """The cells RIMAPI's area check reported something on."""
    return {(i.x, i.z) for kind in ("terrain", "ores", "buildings", "zones")
            for i in getattr(check, kind, [])}


def reach_of_issues(check: object, zone_sizes: dict[str, int]) -> set[tuple[int, int]]:
    """Cells to rule out around what RIMAPI's area check reported.

    It names one cell per zone and a building at its anchor only, so ruling
    out just those cells made the next candidate the same rectangle a cell
    over, and every check went on the same field. Instead, rule out the area
    the thing could cover: for a zone of N cells (from /map/zones, by name) a
    square reaching sqrt(N) cells around the reported cell; for a building, 2
    cells (the biggest the catalog has is 3 wide). Rock and bad terrain are
    reported cell by cell, so those are exact. Offered spots are still checked
    clear, so this only ever skips a little free ground, never offers taken.
    """
    cells = {(i.x, i.z) for i in getattr(check, "terrain", []) + getattr(check, "ores", [])}

    def around(x: int, z: int, reach: int) -> set[tuple[int, int]]:
        return {(x + dx, z + dz) for dx in range(-reach, reach + 1) for dz in range(-reach, reach + 1)}

    for zone in getattr(check, "zones", []):
        size = zone_sizes.get(zone.label or "", 225)
        cells |= around(zone.x, zone.z, max(1, math.ceil(math.sqrt(size)) - 1))
    for building in getattr(check, "buildings", []):
        cells |= around(building.x, building.z, 2)
    return cells


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
    stored: list[MapThing] | None = None,
    tables: list[WorkTable] = (),
    found: dict | None = None,
    zone_sizes: dict[str, int] | None = None,
    sections: dict[str, list[str]] | None = None,
    areas: list[tuple[str, Rect]] = (),
) -> list[str]:
    """Everything the model needs to pick a place without reading the grid.

    Two scales (see mapview): a coarse overview of the surroundings, and a
    detail view that follows the base. Exact positions are also listed in text
    below, and free_spots offers rectangles that are already clear.

    sections, when given, is filled with the same lines split by topic (map,
    open_ground, supplies, forbidden, trees_nearby, zones, buildings,
    blueprints, catalog, crops, floors, room_cost, trees, anchors), for the
    overview/focus prompts to pick from (see rimagent.modes).
    """
    starts: list[tuple[int, str]] = []

    def section(name: str) -> None:
        starts.append((len(lines), name))

    def split() -> None:
        if sections is not None:
            bounds = starts + [(len(lines), "")]
            for (start, name), (end, _) in zip(bounds, bounds[1:]):
                sections[name] = [line for line in lines[start:end] if line]

    forbidden = [i for i in items or [] if i.is_forbidden]
    blueprint_marks = {"waiting": BLUEPRINT, "under construction": FRAME}
    lines: list[str] = []
    section("map")
    lines += [f"Map: {terrain.width} x {terrain.height} cells. x grows east, z grows north.",
              "Positions below are exact; read them from the text, not by counting the grid."]
    for c in colonists:
        if c.position:
            lines.append(f"- {c.name} is at ({c.position.x}, {c.position.z}).")

    # One mark per cell; the most important thing wins. Reserved areas only
    # show where nothing else does.
    marks: dict[tuple[int, int], str] = {}
    shown_areas = list(areas)[:len(AREA_MARKS)]
    for index, (_, rect) in enumerate(shown_areas):
        for cell in rect:
            marks[cell] = AREA_MARKS[index]
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
        split()
        return lines

    area_corners = [c for _, r in shown_areas for c in ((r.x1, r.z1), (r.x2, r.z2))]
    view = detail_bounds(anchor, base_cells + colonist_cells + area_corners, terrain.width, terrain.height)
    lines += ["", *overview_rows(anchor, terrain, defs.terrain, tree_cells, built_cells)]
    lines += ["", *detail_rows(view, terrain, defs.terrain, marks, [name for name, _ in shown_areas])]

    spots = free_spots(anchor, terrain, defs.terrain, built_cells | zone_cells,
                       blocked_in=(lambda rect: reach_of_issues(check_area(rect), zone_sizes or {}))
                       if check_area else None)
    if found is not None:  # for hints (see hints.TurnFacts)
        found.update(spots=spots, anchor=anchor, occupied=built_cells | zone_cells)
    section("open_ground")
    if spots:
        lines += ["", "Open ground near the base, checked clear just now. Each is the biggest clear area found "
                      "there, not a size to fill: use the part you need. For a field, use the Suggested field "
                      "line instead:"]
        lines += [f"  {letter}: {rect} - {note}" for letter, rect, note in spots]

    unfinished = [dict(def_name=bp.def_name, stuff=bp.stuff, floor=bp.floor)
                  for bp, status in blueprints or [] if status in ("waiting", "under construction")]
    section("supplies")
    lines += ["", *describe_supplies(items or [], stored, tables, finished_research, unfinished)]
    def steps_away(i: MapThing) -> int:
        return max(abs(i.position.x - anchor[0]), abs(i.position.z - anchor[1]))

    # allow_items covers the whole map, so no locations: the model once spent
    # 15,000 characters of reasoning reading the grid to find which box held the wood.
    section("forbidden")
    if forbidden:
        food = sum(i.stack_count for i in forbidden if is_food(i.def_name))
        lines.append("allow_items what=all makes everything forbidden usable at once; what=Food"
                     + (f" ({food} forbidden)" if food else "") + " or an item def (e.g. WoodLog) only that.")
    else:
        lines.append("Nothing on the map is forbidden, so allow_items has nothing to do.")
    section("trees_nearby")
    lines.append(f"Trees in the detail view: {sum(1 for cell in tree_cells if view.contains(*cell))}.")

    section("zones")
    lines += ["", "Your zones: " + ("; ".join(z.describe() for z in zones) or "none") + "."]

    section("buildings")
    done = [b for b in buildings or [] if not b.under_construction]
    # Walls, doors and conduits as counts: listing each wall cost ~950 tokens in one colony.
    counted = {name: sum(1 for b in done if b.def_name == name) for name in ("Wall", "Door", "PowerConduit",
                                                                            "HiddenConduit")}
    building_lines = [f"{b.def_name} id={b.id} covering {b.area()}" for b in done if b.def_name not in counted]
    building_lines += [f"{n} {name}{'s' if n > 1 else ''}" for name, n in counted.items() if n]
    lines.append("Your buildings: " + ("; ".join(building_lines) or "none yet") + ".")
    started = [b for b in buildings or [] if b.under_construction]
    if started:
        lines.append("Being built now: " + "; ".join(
            f"{b.label} at ({b.position.x},{b.position.z})" for b in started) + ".")

    section("blueprints")
    if blueprints:
        lines.append("Your blueprints, checked in the game just now (touching ones grouped):")
        builders = [c.name for c in colonists
                    if any(w.work_type == "Construction" and w.priority > 0 for w in c.work_priorities)]
        lines += describe_blueprints(blueprints, usable_stuff(items or []), builders,
                                     short_materials(items or [], stored, unfinished))
    else:
        lines.append("Your blueprints: none.")

    buildable = available_buildings(finished_research)
    section("catalog")
    heading = ("Buildable (def name, size facing north, cost; '(stuff, pick one)' is the stuff field: one of "
               "those materials, not all; without it stuff=null):")
    lines += ["", heading]
    for name, spec in buildable.items():
        label = "" if spec.label.replace(" ", "").lower() == name.lower() else f" ({spec.label})"
        use = f" - {BUILDING_USE[name]}" if name in BUILDING_USE else ""
        lines.append(f"- {name}{label}, {spec.size[0]}x{spec.size[1]}, {spec.cost_text()}{use}")

    lines.append(
        "Stuff materials: "
        + "; ".join(f"{cat} = {', '.join(names)}" for cat, names in STUFF_BY_CATEGORY.items())
        + " (stone blocks, cut from chunks; chunks themselves are not stuff)."
    )
    crops = ", ".join(
        f"{name} ({label}, fertility {defs.crop_fertility_min.get(name, 0):.1f}+)"
        for name, label in CROPS.items()
    )
    section("crops")
    lines.append(f"Crops for growing zones: {crops}.")
    section("floors")
    lines.append("Floor defs and cost per cell: " + "; ".join(
        f"{n}: {f.cost[1]} {f.cost[0]}" for n, f in FLOORS.items()
        if n in defs.terrain and (not f.research or f.research in finished_research)))
    wall, door = BUILDINGS["Wall"].stuff_count, BUILDINGS["Door"].stuff_count
    section("room_cost")
    lines.append(f"build_room cost: {wall} per wall cell + {door} for the door; existing walls are reused.")
    nearest_trees = sorted(trees or [], key=steps_away)[:8]
    section("trees")
    lines.append("Nearest trees (young ones give no wood yet): " + "; ".join(
        f"{t.def_name} at ({t.position.x},{t.position.z})" for t in nearest_trees))
    section("anchors")
    if spots:
        # place_blueprint's x,z is RimWorld's anchor, which is not the corner
        # for wider buildings. One worked example per footprint size, computed
        # with the same rule placement is checked against, inside a free spot.
        letter, rect, _ = max(spots, key=lambda s: s[1].cells)  # the largest fits every size
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
    split()
    return lines


def blueprint_groups(blueprints: list[tuple[TrackedBlueprint, Status]]) -> list[list]:
    """Blueprints whose cells touch (sides or corners) form one group, so a
    room's walls and door read as one thing, not a line per wall."""
    cells = [set(bp.area()) for bp, _ in blueprints]
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(blueprints)):
        if start in seen:
            continue
        group, todo = [], [start]
        seen.add(start)
        while todo:
            i = todo.pop()
            group.append(i)
            near = {(x + dx, z + dz) for x, z in cells[i] for dx in (-1, 0, 1) for dz in (-1, 0, 1)}
            for j in range(len(blueprints)):
                if j not in seen and cells[j] & near:
                    seen.add(j)
                    todo.append(j)
        groups.append([blueprints[i] for i in sorted(group)])
    return groups


def describe_blueprints(blueprints: list[tuple[TrackedBlueprint, Status]],
                        usable: dict[str, int], builders: list[str] | None = None,
                        short: set[str] = frozenset()) -> list[str]:
    """One line per group, e.g. "- 15 Wall + 1 Door (BlocksSandstone) within
    (117,132)-(121,136): 16 waiting - nobody can build them: no usable ..."."""
    lines = []
    for group in blueprint_groups(blueprints):
        kinds: dict[tuple[str, str | None], int] = {}
        statuses: dict[str, int] = {}
        for bp, status in group:
            kinds[(bp.def_name, bp.stuff)] = kinds.get((bp.def_name, bp.stuff), 0) + 1
            statuses[status] = statuses.get(status, 0) + 1
        stuffs = sorted({stuff for _, stuff in kinds if stuff})
        what = " + ".join(f"{n} {name}" if n > 1 or len(group) > 1 else name
                          for (name, _), n in kinds.items())
        if stuffs:
            what += f" ({', '.join(stuffs)})"
        area = Rect.from_corners(min(bp.area().x1 for bp, _ in group), min(bp.area().z1 for bp, _ in group),
                                 max(bp.area().x2 for bp, _ in group), max(bp.area().z2 for bp, _ in group))
        where = f"covering {area}" if len(group) == 1 else f"within {area}"
        state = ", ".join(f"{n} {s}" for s, n in statuses.items()) if len(group) > 1 else group[0][1]
        state = state.replace("gone", "gone (cancelled or destroyed)")
        notes = []
        if statuses.get("waiting"):
            missing = sorted({bp.stuff for bp, s in group if s == "waiting" and bp.stuff
                              and not usable.get(bp.stuff)})
            notes += [f"nobody can build with {m}, none is usable: {how_to_get(m)}" for m in missing]
            scarce = sorted({bp.stuff for bp, s in group if s == "waiting" and bp.stuff in short} - set(missing))
            notes += [f"not enough {m} for all your unfinished builds: {how_to_get(m)}" for m in scarce]
            if not missing and not scarce and builders:
                notes.append(f"materials are there and Construction is on for {', '.join(builders)}: "
                             "they build it while time advances, between more urgent jobs")
            elif not missing and not scarce:
                notes.append("nobody has Construction on: enable_work Construction")
        lines.append(f"- {what} {where}: {state}" + (" - " + "; ".join(notes) if notes else ""))
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
        *(["", "What your last orders actually did, failures still unresolved, and hints:",
           *[f"- {r}" for r in last_results]] if last_results else []),
        "",
        f"Tick {state.game_tick}. {state.colonist_count} colonists.",
        f"Wealth {state.colony_wealth:,.0f}. Storyteller {state.storyteller}.",
        "",
        "Threats:",
        *([t.summary() for t in threats] or ["None."]),
        "",
        "New since last step:",
        *([describe_event(e) for e in events] or ["Nothing."]),
        "",
        "Alerts: " + (", ".join(a.label for a in alerts) or "none") + ".",
        "",
        "Colonists:",
        *[describe_colonist(c, work_types, learned_incapable) for c in colonists],
        "",
        "Research: " + " ".join(describe_research(research, [
            c.name for c in colonists
            if any(w.work_type == "Research" and w.priority > 0 for w in c.work_priorities)])),
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
    turn_model=Turn,
    allowed: set[str] | None = None,
    fallback=None,
):
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
            turn = turn_model.model_validate_json(reply)
            if choices is not None:
                choices.validate(turn, allowed)
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
    return (fallback if fallback is not None else Turn(time="hold", actions=[])), last_reply


def action_area(target: dict) -> Rect | None:
    """The cells an order is about, widened a little, or None for non-map orders."""
    if "x1" in target:
        rect = Rect.from_corners(target["x1"], target["z1"], target["x2"], target["z2"])
    elif "x" in target:
        rect = Rect(target["x"], target["z"], target["x"], target["z"])
    else:
        return None
    return Rect(rect.x1 - 3, rect.z1 - 3, rect.x2 + 3, rect.z2 + 3)


def still_known(tracked: list[TrackedZone], live_sizes: dict[int, int]) -> list[TrackedZone]:
    """The tracked zones the game still has, with their current size.

    A zone only shrinks when something is built on it (a shelf on a
    stockpile), so it keeps its place; dropping it from the prompt made a model
    re-create a stockpile it already had, three times. One that grew was
    edited by hand, and its rectangle is no longer known.
    """
    known = []
    for zone in tracked:
        live = live_sizes.get(zone.zone_id)
        if live and live <= zone.cell_count():
            known.append(zone if live == zone.cell_count() else zone.model_copy(update={"size": live}))
    return known


def site_plan(sections: dict[str, list[str]], zones: list[TrackedZone], layout) -> list[str]:
    """Everything that takes space in one block, so new work goes around it.

    Models laid rooms, fields and stockpiles over each other when the areas,
    rooms and zones sat in three places in the prompt and areas weren't on the
    map at all.
    """
    lines = ["SITE PLAN: what already takes space. Nothing new may overlap any of it: a new room, zone or "
             "area goes on open ground (a new room may share a wall with an existing one). Put things of a "
             "reserved area's kind inside that area, and nothing else there."]
    if layout is not None and layout.items:
        lines.append(f"Reserved areas ({AREA_MARKS[0]}, {AREA_MARKS[1]}... on the detail map):")
        described = sections.get("layout", [])[1:]
        for index, text in enumerate(described):
            mark = AREA_MARKS[index] if index < len(AREA_MARKS) else "?"
            lines.append(f"- [{mark}] {text[2:]}")
    else:
        lines += sections.get("layout", [])
    rooms = sections.get("rooms", [])[1:]
    lines.append("Rooms (# walls and + doors on the map; the inside is theirs):" if rooms else "Rooms: none yet.")
    lines += rooms
    if zones:
        lines.append("Zones (g growing, s stockpile on the map):")
        lines += [f"- {z.describe()}" for z in zones]
    else:
        lines.append("Zones: none yet.")
    return lines


def is_floor(terrain_name: str) -> bool:
    return terrain_name in FLOORS or "Floor" in terrain_name or terrain_name.startswith(("Tile", "Carpet"))


def measure_finish(buildings, report, rooms, live_zones, finished_research, items, terrain=None) -> FinishState:
    """What plan finish checks are measured against this turn."""
    walls, _ = room_state(buildings, report)
    closed: dict[str, int] = {}
    floored: dict[str, int] = {}
    for room in (rooms.items if rooms is not None else []):
        if all(walls.get(c) == "built" for c in perimeter(room.area())):
            closed[room.purpose] = closed.get(room.purpose, 0) + 1
        if terrain is not None and all(is_floor(terrain.at(x, z)) for x, z in inside_of(room.area())):
            floored[room.purpose] = floored.get(room.purpose, 0) + 1
    built: dict[str, int] = {}
    for b in buildings:
        if not b.under_construction:
            built[b.def_name] = built.get(b.def_name, 0) + 1
    cells: dict[str, int] = {}
    for z in live_zones:
        kind = "growing" if "Growing" in z.type else "stockpile" if "Stockpile" in z.type else None
        if kind:
            cells[kind] = cells.get(kind, 0) + z.cells_count
    return FinishState(rooms_closed=closed, rooms_floored=floored, built=built,
                       sleepers=sum(SLEEPERS.get(name, 0) * n for name, n in built.items()),
                       zone_cells=cells, research=set(finished_research), items=stock_by_def(items or [], False),
                       forbidden=stock_by_def(items or [], True))


def room_state(buildings, report) -> tuple[dict, list]:
    """For RoomTracker.describe: which cells hold walls or doors, and every
    building or blueprint with its area and state."""
    walls: dict[tuple[int, int], str] = {}
    contents = []
    for b in buildings:
        state = "being built" if b.under_construction else "built"
        name = b.def_name.removeprefix("Frame_")
        if name in ("Wall", "Door"):
            for cell in b.area():
                walls[cell] = "built" if state == "built" else "planned"
        else:
            contents.append((name, b.area(), state))
    for bp, status in report:
        if status not in ("waiting", "under construction"):
            continue
        if bp.def_name in ("Wall", "Door"):
            for cell in bp.area():
                walls.setdefault(cell, "planned")
        elif not bp.floor:
            contents.append((bp.def_name, bp.area(), "planned" if status == "waiting" else "being built"))
    return walls, contents


def prepare_turn(client, memory, blueprints, zones, mode, window=None,
                 run_ticks=RUN_TICKS, threat_run_ticks=THREAT_RUN_TICKS, new_events=(), last_results=(), failures=None,
                 read_only=False, learned_incapable=None, chops=None, rooms=None, idle=None, mines=None,
                 layout=None, retries=None):
    """Observe the game and build this turn's prompt and choices.

    Shared by live play and dry runs; with read_only=True nothing is saved,
    so a dry run never touches memory, zone or blueprint files.
    """
    client.begin_snapshot()
    state = client.get_state()
    colonists = client.get_colonists()
    alerts, threats = client.get_alerts(), client.get_threats()
    try:
        visitors = client.get_visitors() if hasattr(client, "get_visitors") else []
    except (RimApiError, httpx.HTTPError, ValueError, KeyError) as error:
        print(f"could not read visitors: {error}")
        visitors = []
    work_types, tables = client.get_work_types(), client.get_work_tables()
    research = client.get_research()  # also seeds completed-research snapshot cache
    terrain, defs = client.get_terrain(), client.get_game_defs()
    items, trees = client.get_items(), client.get_trees()
    buildings, live_zones = client.get_buildings(), client.get_zones()
    live_zone_sizes = {z.id: z.cells_count for z in live_zones}
    known_zones = still_known(zones.items, live_zone_sizes)
    if not read_only and known_zones != zones.items:
        zones.items = known_zones
        zones.save()
    report = blueprints.check(client, forget_finished=not read_only)
    tracked = {z.zone_id: z for z in known_zones}
    operations = read_operations(client, colonists, defs, fields=[
        tracked[z.id].describe() if z.id in tracked else f"{z.label}, {z.cells_count} cells, drawn or edited by hand"
        for z in live_zones if "Growing" in z.type])
    if chops is not None:
        chops.reconcile(trees, state.game_tick,
                        sum(i.stack_count for i in items if i.def_name == "WoodLog" and not i.is_forbidden))
        cutters = [c.name for c in colonists
                   if any(w.work_type == "PlantCutting" and w.priority > 0 for w in c.work_priorities)]
        chop_line = chops.describe(state.game_tick, cutters)
        if chop_line:
            operations.add("wood_mining", chop_line)
    if mines is not None and "mine" in operations.targets:
        mines.reconcile({(t["x"], t["z"]) for t in operations.targets["mine"]})
        miners = [c.name for c in colonists
                  if any(w.work_type == "Mining" and w.priority > 0 for w in c.work_priorities)]
        mine_line = mines.describe(miners)
        if mine_line:
            operations.add("wood_mining", mine_line)
    finished = {p.name for p in research.all_projects if p.is_finished}
    field_cells = sum(z.cells_count for z in live_zones if "Growing" in z.type)
    enemies = []
    if threats:
        try:
            enemies = client.get_enemies(threats)
        except (RimApiError, httpx.HTTPError) as error:
            print(f"could not locate enemies: {error}")
    built = {b.def_name.removeprefix("Frame_") for b in buildings} | {bp.def_name for bp, _ in report}
    power_ready = "Electricity" in finished and bool(built & POWER_SOURCES)
    unknown = [z for z in live_zones if z.id not in {k.zone_id for k in known_zones}]
    if unknown:
        operations.add("zones", "Zones with unknown/edited geometry (placement checks still protect them): " +
                                "; ".join(f"{z.label} id={z.id}, {z.cells_count} cells" for z in unknown))
    choices = Choices.observe(colonists, work_types, tables, research, defs, terrain,
                              items, trees, [k for k, v in operations.targets.items() if v],
                              learned_incapable, enemies, power_ready)
    # One bed per colonist: one run placed nine beds for three colonists, though the
    # prompt said the planned ones were enough.
    sleepers = (sum(SLEEPERS.get(b.def_name, 0) for b in buildings if not b.under_construction)
                + sum(SLEEPERS.get(bp.def_name, 0) for bp, status in report
                      if status in ("waiting", "under construction")))
    if colonists and sleepers >= len(colonists):
        choices.withhold(set(SLEEPERS), f"beds for {sleepers} are built or planned, enough for all "
                                        f"{len(colonists)} colonists; wait for the planned ones to be built")
    try:
        stored = client.get_stored_items()
    except (RimApiError, httpx.HTTPError) as error:  # the prompt still works without it
        print(f"could not read stored resources: {error}")
        stored = None
    found: dict = {}
    sections: dict[str, list[str]] = {}
    map_lines = describe_map(colonists, terrain, defs, known_zones, finished,
                             items, trees, report, buildings, client.check_area, stored, tables, found,
                             {z.label: z.cells_count for z in live_zones}, sections,
                             [(a.purpose, a.area()) for a in layout.items] if layout is not None else [])
    facts = TurnFacts(spots=found.get("spots", []), targets=operations.targets, trees=trees, items=items,
                      stored=stored, report=report, buildings=buildings, colonists=colonists,
                      anchor=found.get("anchor"), learned=learned_incapable or {},
                      storage=operations.storage, rooms=list(rooms.items) if rooms is not None else [],
                      field_cells=field_cells, zone_cells={c for z in known_zones for c in z.area()})
    need = FIELD_CELLS_PER_COLONIST * len(colonists) - field_cells
    if need >= 4 and found.get("spots") and found.get("anchor"):
        crop = "Plant_Potato" if "Plant_Potato" in defs.crop_fertility_min else next(iter(defs.crop_fertility_min), None)
        fertility = defs.crop_fertility_min.get(crop, 0.7) if crop else 0.7
        areas = layout.items if layout is not None else []
        field = suggested_field(found["spots"], terrain, defs.terrain, found["anchor"], need, fertility,
                                avoid=[a.area() for a in areas if a.purpose != "fields"],
                                prefer=next((a.area() for a in areas if a.purpose == "fields"), None),
                                gap=FIELD_GAP)
        if field:
            operations.add("farming", f"Suggested field ({field.cells} cells, the size still recommended; checked "
                                      f"clear and fertile enough for {crop}, close to the base): create_growing_zone "
                                      f"plant={crop} x1={field.x1} z1={field.z1} x2={field.x2} z2={field.z2}.")
    if colonists:
        operations.add("shelter", housing_advice(len(colonists), rooms.items if rooms is not None else [],
                                                 room_state(buildings, report)[1]))
    if rooms is not None and rooms.items:
        floors_planned = {c for bp, status in report if bp.floor and status in ("waiting", "under construction")
                          for c in bp.area()}

        def floor_of(inside: Rect) -> tuple[int, int, int]:
            """(cells floored, cells with a floor planned, cells in all)."""
            cells = list(inside)
            laid = {(x, z) for x, z in cells if is_floor(terrain.at(x, z))}
            return len(laid), sum(1 for c in cells if c in floors_planned and c not in laid), len(cells)

        walls_now, contents_now = room_state(buildings, report)

        def has_stockpile(room) -> bool | None:
            """Whether a stockpile covers part of a storage room's inside, asking the game:
            one edited by shelves can drop out of the zone tracker."""
            if room.purpose != "storage":
                return None
            inside = Rect(room.area().x1 + 1, room.area().z1 + 1, room.area().x2 - 1, room.area().z2 - 1)
            try:
                return any("Stockpile" in (i.zone_type or "") or (i.label or "").startswith("Stockpile")
                           for i in client.check_area(inside).zones)
            except (RimApiError, httpx.HTTPError, ValueError):
                return None

        sleepers = sum(SLEEPERS.get(name, 0) for name, _, _ in contents_now)
        short = len(colonists) - sleepers
        bed_note = {"Bed": f"beds for {sleepers} of {len(colonists)} colonists, built or planned: "
                           + (f"{short} more needed" if short > 0 else
                              "enough for everyone; another is only a spare")}

        def furniture(room) -> str | None:
            return furnish_line(room.purpose, room.area(), room.door_side, contents_now, choices.materials,
                                has_stockpile(room), bed_note)

        sections["rooms"] = ["Your rooms (purpose as you tagged it):",
                             *(rooms.describe(walls_now, contents_now, floor_of, furniture)
                               or ["none standing any more"])]
        map_lines += ["", *sections["rooms"]]
    time_status = describe_time(mode, window, run_ticks, threat_run_ticks)
    if layout is not None:
        placed = room_state(buildings, report)[1] + [
            ("growing zone" if z.kind == "growing" else "stockpile", z.area(), "zone") for z in known_zones]
        sections["layout"] = layout.describe(placed)
    sections["site"] = site_plan(sections, known_zones, layout)
    if rooms is not None and rooms.items and "occupied" in found:
        occupied = found["occupied"]

        def is_free(x: int, z: int) -> bool:
            info = defs.terrain.get(terrain.at(x, z))
            return info is not None and "Heavy" in info.affordances and (x, z) not in occupied

        def is_clear(rect: Rect) -> bool:
            try:
                return not issue_cells(client.check_area(rect))
            except (RimApiError, httpx.HTTPError, ValueError):
                return False

        slots = room_slots([(r.purpose, r.area()) for r in rooms.items], is_free, terrain.width, terrain.height,
                           is_clear)
        if slots:
            sections["room_slots"] = ["Room slots beside your rooms, checked clear (sharing a wall keeps the base "
                                      "compact and saves material; size the room for what goes in it):", *slots]

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
    # Orders on the fix-first list aren't repeated as "Unresolved".
    fixing = set(retries.items) if retries else set()
    feedback = (list(last_results) + (retries.lines() if retries else [])
                + ([line for (target, _, _), line in zip(failures.entries, failures.lines())
                    if task_key(target) not in fixing] if failures else []))
    unfinished = [dict(def_name=bp.def_name, stuff=bp.stuff, floor=bp.floor) for bp in facts.unfinished()]
    pending = pending_work(facts, research, tables, live_zones, len(chops.orders) if chops else 0,
                           short_materials(items, stored, unfinished), len(mines.cells) if mines else 0)
    if idle is not None:
        capable = {c.name: set(work_types) - incapable_work(c, learned_incapable) for c in colonists}
        idle_line = idle_hint(idle.update(colonists), facts, pending, capable)
        if idle_line:
            feedback.append(idle_line)
    full = storage_hint(operations.storage, "Shelf" in choices.materials)
    if full:
        feedback.append(full)
    unavailable = ["Unavailable now: " + "; ".join(choices.unavailable)] if choices.unavailable else []
    prompt = build_prompt(state, colonists, alerts, threats, list(new_events), work_types,
                          tables, research, time_status, memory,
                          map_lines + operations.lines + unavailable,
                          feedback, learned_incapable)
    blocks = turn_blocks(client, colonists, work_types, research, tables, items, buildings, enemies,
                         threats, new_events, feedback, learned_incapable, pending)
    blocks["time"] = [time_status]
    door_cells = [(b.position.x, b.position.z) for b in buildings if b.def_name.removeprefix("Frame_") == "Door"]
    door_cells += [(bp.x, bp.z) for bp, status in report
                   if bp.def_name == "Door" and status in ("waiting", "under construction")]
    armed = 0
    for c in colonists:
        try:
            armed += any(i.is_weapon for i in client.get_equipment(c.id))
        except (RimApiError, httpx.HTTPError, AttributeError, ValueError):
            pass
    blocks["defense"] = defense_lines(room_state(buildings, report)[1], door_cells, enemies, found.get("anchor"),
                                      armed, len(colonists), sum(t.pawn_count for t in threats if not t.active))
    finish_state = measure_finish(buildings, report, rooms, live_zones, finished, items, terrain)
    return dict(prompt=prompt, choices=choices, operations=operations, state=state, finish_state=finish_state,
                threats=threats, visitors=visitors, time_status=time_status, fingerprint=fingerprint, facts=facts,
                sections=sections, blocks=blocks, colonists=colonists, items=items, alerts=alerts,
                enemies=enemies, power_ready=power_ready, events=list(new_events), thing_names=defs.thing_names)


def turn_blocks(client, colonists, work_types, research, tables, items, buildings, enemies, threats,
                events, feedback, learned_incapable, pending) -> dict[str, list[str]]:
    """Named prompt blocks for the overview/focus/threat prompts (rimagent.modes)."""
    blocks = {
        "colonists": [describe_colonist(c, work_types, learned_incapable) for c in colonists],
        "research": ["Research: " + " ".join(describe_research(research, [
            c.name for c in colonists
            if any(w.work_type == "Research" and w.priority > 0 for w in c.work_priorities)]))],
        "tables": ["Work tables and production bills:",
                   *([line for t in tables for line in describe_work_table(t)]
                     or ["None. Build a work table before trying to set a bill."])],
        "events": [describe_event(e) for e in events],
        "feedback": list(feedback),
        "work_pending": ["Work waiting for colonists:",
                         *([f"- {what}: needs {work}" + (" (blocked)" if blocked else "")
                            for work, what, blocked in pending] or ["- nothing ordered yet"])],
        "weapons": [f"id={i.thing_id} {i.def_name} at ({i.position.x},{i.position.z})"
                    + (" ranged" if i.is_ranged else " melee") + (", forbidden" if i.is_forbidden else "")
                    for i in items if i.is_weapon][:12],
        "enemies": [f"id={e.id} {e.kind} at ({e.x},{e.z}), health {e.health:.0%}, {e.group}"
                    + ("" if e.active else ", ASLEEP") for e in enemies],
    }
    hurt = [c for c in colonists if c.health < 0.99]
    beds = [b for b in buildings if "Bed" in b.def_name and not b.under_construction]
    blocks["injured"] = ["Injured colonists: " + (", ".join(f"{c.name} {c.health:.0%}" for c in hurt) or "none")
                         + ".", f"Beds built: {len(beds)}."]
    fighters = []
    armed = any(t.active for t in threats)
    for c in colonists:
        weapons = "?"
        if armed:
            try:
                weapons = ", ".join(i.def_name for i in client.get_equipment(c.id)) or "unarmed"
            except (RimApiError, httpx.HTTPError):
                pass
        where = f"({c.position.x},{c.position.z})" if c.position else "(?)"
        fighters.append(f"- {c.name} at {where}, health {c.health:.0%}, weapon: {weapons}, "
                        f"doing {c.current_job or 'nothing'}")
    blocks["fighters"] = fighters
    return blocks


def run(client: RimApiClient, model: Callable[[str], str], logger: RunLogger,
        events: EventListener | None = None, max_steps: int = 10,
        run_ticks: int = RUN_TICKS, threat_run_ticks: int = THREAT_RUN_TICKS,
        speed: int = NORMAL, memory: AgentMemory | None = None, loop: str = "single") -> None:
    """Run the agent loop for max_steps turns.

    Each turn: pause, observe, ask the model, carry out its orders in order
    (every one, even after a failure), then follow its time choice. advance lets
    the game run run_ticks (threat_run_ticks while a threat is active),
    stopping early if a threat arrives or the game pauses itself; hold keeps it
    paused. Nothing about time carries over: every reply chooses again.

    loop="single" asks one big prompt every turn. loop="modes" asks an overview
    for a plan and its toolkits, then focus prompts with only those toolkits,
    and a threat prompt while enemies are active (see rimagent.modes).

    A step is one turn (an overview doesn't take a step of its own). In the
    modes loop the run doesn't stop in the middle of a plan or a fight: past
    max_steps it keeps going until the plan is done, replanned or out of
    turns, or the fight is over, but it never starts a new plan, and never
    runs more than OVERRUN_STEPS extra steps. Drafted colonists are released
    whenever the run ends.
    """
    if run_ticks <= 0 or threat_run_ticks <= 0:
        raise ValueError("time windows must contain at least one tick")
    if loop not in ("single", "modes"):
        raise ValueError("loop must be single or modes")
    memory = memory or AgentMemory()
    blueprints = BlueprintTracker(memory.memory_dir / "blueprints.json")
    zones = ZoneTracker(memory.memory_dir / "zones.json")
    chops = ChopTracker(memory.memory_dir / "chops.json")
    mines = MineTracker(memory.memory_dir / "mining.json")
    rooms = RoomTracker(memory.memory_dir / "rooms.json")
    layout = LayoutTracker(memory.memory_dir / "layout.json")
    idle = IdleTracker()
    failures = FailureMemory()
    retries = RetryList()
    learned_incapable: dict[str, set[str]] = {}  # colonist name -> work RIMAPI refused
    combat = Combat()
    mode = TimeMode(True, "session start")
    window = None
    carried_events, last_results = [], []
    last_counts = ""
    # Modes loop: why the next calm turn starts with an overview (None = keep focusing).
    need_overview: str | None = "the session starts"
    failed_turns, was_danger = 0, False
    food_alarm = False  # a food emergency already sent the loop to the overview
    # The plan a food emergency cut short, offered again by the next calm overview:
    # one run's bedroom (12 of 44 walls) was dropped for good when the colony went hungry.
    interrupted: CurrentPlan | None = None

    step = -1
    try:
        while True:
            step += 1
            if step >= max_steps:
                plan_in_hand = need_overview is None and bool(
                    memory.state.current_plan and memory.state.current_plan.toolkits)
                busy = loop == "modes" and (was_danger or plan_in_hand)
                if not busy or step >= max_steps + OVERRUN_STEPS:
                    break
                if step == max_steps:
                    print(f"Step {step}: past --steps; finishing the current "
                          f"{'fight' if was_danger else 'plan'} before stopping")
            client.pause()
            new_events = carried_events + (events.drain() if events else [])
            carried_events = []
            try:
                context = prepare_turn(client, memory, blueprints, zones, mode, window,
                                       run_ticks, threat_run_ticks, new_events, last_results, failures,
                                       learned_incapable=learned_incapable, chops=chops, rooms=rooms, mines=mines,
                                       idle=idle, layout=layout if loop == "modes" else None,
                                       retries=retries)
            except (RimApiError, httpx.HTTPError, ValueError) as error:
                logger.log_model_failure({"step": step, "failure_type": "observation_error", "error": str(error)})
                # Preserve threat events even when this observation fails.
                carried_events = new_events
                print(f"Step {step}: observation failed; held paused: {error}")
                time.sleep(2)
                continue
            combat.enemies = {e.id: e for e in context['enemies']}
            logger.log_state(context['state'])
            for event in new_events:
                logger.log_event(event)
            danger = any(t.active for t in context['threats']) or any(is_attack(e) for e in new_events)
            choices = context['choices']
            plan = None
            if loop == "single":
                prompt, allowed, turn_model, fallback = context['prompt'], LEGACY_ACTIONS, Turn, None
            elif danger:
                prompt = threat_prompt(context, combat.drafted, memory, last_counts)
                allowed, turn_model = set(COMBAT.actions), ThreatTurn
                fallback = ThreatTurn(actions=[], time="hold")
            else:
                if was_danger:
                    context['blocks']['feedback'] += release_drafted(client, combat, context['colonists'])
                    need_overview = need_overview or "the threat is over"
                plan = memory.state.current_plan
                if plan is None or not plan.toolkits:
                    need_overview = need_overview or "no plan yet"
                elif plan.finish and not need_overview:
                    # The game, not the model, says when a plan is finished.
                    _, met = finish_progress(plan.finish, context.get('finish_state'))
                    if met:
                        need_overview = f"the game shows your plan finished: {plan.done_when}"
                # Once per emergency: the plan the overview makes for it then runs undisturbed.
                emergency = food_emergency(context)
                if emergency and not food_alarm:
                    if not need_overview and plan is not None and plan.toolkits:
                        interrupted = plan
                    need_overview = need_overview or emergency
                food_alarm = emergency is not None
                if need_overview and step >= max_steps:
                    print(f"Step {step}: stopping; the plan in hand is finished ({need_overview})")
                    break
                if need_overview:
                    resume = interrupted if not emergency else None
                    plan = overview_turn(model, context, memory, combat, logger, step, need_overview, plan,
                                         last_counts, resume)
                    if resume is not None:
                        interrupted = None  # offered once; the model decides
                    need_overview, failed_turns = None, 0
                toolkits = plan.toolkits[:MAX_TOOLKITS]
                allowed = allowed_actions(toolkits)
                choices = narrowed(choices, allowed, designations(toolkits), toolkit_work(toolkits))
                prompt = focus_prompt(context, plan, toolkits, combat.drafted, memory, choices, GAME_FACTS,
                                      last_counts)
                turn_model = FocusTurn
                fallback = FocusTurn(actions=[], time="hold", plan="continue")
            if loop == "modes":
                was_danger = danger
            schema = choices.schema(allowed, turn_model)
            if getattr(model, 'schema', None) is not None:
                model.schema = schema
            turn, reply = request_turn(model, prompt, logger, step, danger, choices, turn_model, allowed, fallback)
            results = []
            counts = {"OK": 0, "UNCHANGED": 0, "FAILED": 0}
            # Empty turns still retain prompt/reasoning/reply in logs.
            seen: dict[str, int] = {}  # order -> its number, to catch repeats within this reply
            for index, action in enumerate(turn.actions or [None]):
                error, result = None, "no orders"
                key = json.dumps(action.model_dump(exclude={"reason"}), sort_keys=True) if action else None
                if action is not None and key in seen:
                    # One turn sent enable_work Squint Crafting five times.
                    result = f"unchanged: the same order as #{seen[key]} in this reply, so it wasn't sent again"
                    counts["UNCHANGED"] += 1
                    results.append(f"#{index + 1} UNCHANGED: {result.removeprefix('unchanged: ')}")
                elif action is not None:
                    seen[key] = index + 1
                    # Every order is tried, even after a failure: the game refuses
                    # one that relied on it, and the report says which worked.
                    try:
                        refusal = (stockpile_refusal(action, context['facts'])
                                   or field_refusal(action, context['facts'])
                                   or field_area_refusal(action, layout if loop == "modes" else None)
                                   or watch_spot_refusal(action, context['facts'], client)
                                   or room_refusal(action, rooms.items,
                                                   room_state(context['facts'].buildings, context['facts'].report)[1],
                                                   layout if loop == "modes" else None,
                                                   len(context['colonists'])))
                        if refusal:
                            raise ValueError(refusal)
                        result = execute(client, action, blueprints, zones, context['operations'],
                                         learned_incapable, chops, mines, combat, layout)
                        if action.action == "build_room" and not result.startswith("unchanged"):
                            rooms.add(action.rect(), action.purpose, action.door_side)
                        hint = success_hint(action, result, client.submitted_blueprints, context['facts'])
                        if not result.startswith("unchanged") and loop == "modes":
                            hint = " ".join(h for h in (hint, layout_hint(action, layout)) if h)
                        if hint:
                            result += f" HINT: {hint}"
                        retries.succeeded(action)  # done, or already queued ("unchanged")
                    except (RimApiError, httpx.HTTPError, ValueError) as exc:
                        error = str(exc)
                        hint, keep = failure_hint(action, error, context['facts'])
                        if hint:
                            error += f" HINT: {hint}"
                        retries.failed(action, error)
                        result = f"FAILED {action.model_dump(exclude={'reason'})}: {error}"
                        # Nothing to fix (e.g. its own blueprint was in the way): don't keep it
                        # in the prompt as "unresolved", which read as a problem to solve.
                        if keep:
                            failures.remember(action, error, context['fingerprint'](action.model_dump()))
                    status = ("FAILED" if error else
                              "UNCHANGED" if result.startswith("unchanged") or ": unchanged: " in result else "OK")
                    counts[status] += 1
                    results.append(f"#{index + 1} {status}: "
                                   + (result.removeprefix("unchanged: ") if status == "UNCHANGED" else
                                      result.removeprefix("FAILED ")))
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
                    "choice_schema": schema if index == 0 else None,
                    "prompt_kind": ("single" if loop == "single" else "threat" if danger else "focus")
                    if index == 0 else None,
                    # Focus replies: how the plan stands, toolkits asked for, and the lesson saved.
                    "plan_status": getattr(turn, 'plan', None) if index == 0 else None,
                    "request_toolkits": getattr(turn, 'request_toolkits', None) or None if index == 0 else None,
                    "lesson": (turn.lesson.model_dump() if isinstance(getattr(turn, 'lesson', None), BaseModel)
                               else getattr(turn, 'lesson', None)) if index == 0 else None,
                })
                print(f"Step {step}.{index}: {result}")
            last_counts = (f"{len(results)} orders: {counts['OK']} OK, {counts['UNCHANGED']} UNCHANGED, "
                           f"{counts['FAILED']} FAILED.") if results else "no orders."
            last_results = ([last_counts] + results) if results else []
            last_results += retries.end_turn()
            tick = context['state'].game_tick
            if isinstance(turn, Turn):
                # Plan and policies are always kept; a step reported completed is
                # held until a later turn confirms it (see memory.held_until_confirmed).
                memory_note = memory.apply_update(step, tick, turn.memory_update)
            elif isinstance(turn, FocusTurn):
                memory_note = (memory.apply_update(step, tick, MemoryUpdate(
                    observation=turn.lesson.text, toolkit=turn.lesson.toolkit)) if turn.lesson else None)
                plan.turns += 1
                track_steps(plan, turn.step, turn.steps_done)
                added = [t for t in turn.request_toolkits if t not in plan.toolkits]
                if added:
                    plan.toolkits = (plan.toolkits + added)[-MAX_TOOLKITS:]
                    last_results.append(f"Toolkits added for the next turn: {', '.join(added)}.")
                memory.save_plan(plan)
                failed_turns = failed_turns + 1 if turn.actions and counts["FAILED"] == len(turn.actions) else 0
                if turn.plan == "done" and plan.finish:
                    progress, met = finish_progress(plan.finish, context.get('finish_state'))
                    if not met:
                        last_results.append("You said the plan is done, but the game ends it by itself once every "
                                            "finish check is met, and some aren't yet: "
                                            + "; ".join(line[2:] for line in progress if "MET" not in line)
                                            + ". Keep working on them, or say replan if they can't be met.")
                    if plan.turns >= MAX_FOCUS_TURNS:
                        need_overview = f"{MAX_FOCUS_TURNS} turns spent on it"
                elif turn.plan != "continue":
                    need_overview = "you said the plan is done" if turn.plan == "done" else "you asked to replan"
                elif plan.turns >= MAX_FOCUS_TURNS:
                    need_overview = f"{MAX_FOCUS_TURNS} turns spent on it"
                elif failed_turns >= FAILED_TURNS_TO_REPLAN:
                    need_overview = "every order failed on consecutive turns"
            else:
                memory_note = (memory.apply_update(step, tick, MemoryUpdate(observation=turn.lesson,
                                                                            toolkit="combat"))
                               if turn.lesson else None)
            if memory_note:
                last_results.append(memory_note)
            mode = mode_after_decision(mode, turn.time)
            if turn.time == 'hold':
                window = None
            else:
                # Catch threat events that arrived while orders were being processed.
                arrived = events.drain() if events else []
                carried_events += arrived
                if any(is_attack(e) for e in arrived):
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
        if combat.drafted:  # drafted colonists do no work: never leave them so
            try:
                for note in release_drafted(client, combat, client.get_colonists()):
                    print(note)
            except (RimApiError, httpx.HTTPError):
                pass


def release_drafted(client, combat: Combat, colonists) -> list[str]:
    """Once a threat is over, send drafted colonists back to work (drafted ones do nothing)."""
    ids = {c.name: c.id for c in colonists}
    notes = []
    for name in sorted(combat.drafted):
        try:
            if name in ids:
                client.set_drafted(ids[name], False)
            notes.append(f"{name} released back to work: the threat is over.")
        except (RimApiError, httpx.HTTPError) as error:
            notes.append(f"Could not release {name} from the draft: {error}")
            continue
        combat.drafted.discard(name)
    return notes


def track_steps(plan: CurrentPlan, working_on: int, finished: list[int]) -> None:
    """Mark the steps a focus reply finished, and the one it works on now.

    Steps already under way stay [now]: queued work carries on by itself while
    the next step starts, so several can be in progress at once.
    """
    for n in finished:
        if 1 <= n <= len(plan.steps):
            plan.steps[n - 1].status = "completed"
    if 1 <= working_on <= len(plan.steps) and plan.steps[working_on - 1].status != "completed":
        plan.steps[working_on - 1].status = "in_progress"


def overview_turn(model, context, memory, combat, logger, step, why, previous, last_counts,
                  interrupted=None) -> CurrentPlan:
    """Ask the overview prompt for a plan and store it as the current plan."""
    prompt = overview_prompt(context, combat.drafted, memory, why, previous, last_counts, interrupted)
    overview, reply = request_overview(model, prompt, logger, step, context["thing_names"], context["items"])
    plan = CurrentPlan(objective=overview.goal, toolkits=list(overview.toolkits),
                       done_when="; ".join(c.label() for c in overview.finish),
                       uses=overview.uses, finish=list(overview.finish),
                       steps=[PlanStep(description=s, status="pending") for s in overview.steps if s.strip()])
    memory.save_plan(plan)
    logger.log_decision({"step": step, "action": "overview", "reason": why, "prompt": prompt, "reply": reply,
                         "thinking": getattr(model, 'last_thinking', None),
                         "model_stats": getattr(model, 'last_stats', None), "plan": plan.model_dump(),
                         "prompt_kind": "overview"})
    print(f"Step {step}: plan ({why}): {plan.objective} [{', '.join(plan.toolkits)}]")
    return plan


# When the overview can't be read: keep the colony going on the basics.
FALLBACK_OVERVIEW = Overview(goal="Get every colonist a bed under a roof.",
                             uses="wood or steel from the colony stock, colonists with Construction on",
                             toolkits=["construction"],
                             finish=[FinishCheck(kind="room_closed", what="bedroom")])


def request_overview(model, prompt: str, logger: RunLogger, step: int,
                     thing_names: frozenset[str], items=None) -> tuple[Overview, object | None]:
    """Ask for a plan twice if needed; fall back to FALLBACK_OVERVIEW."""
    if getattr(model, 'schema', None) is not None:
        model.schema = Overview.model_json_schema()
    attempt_prompt, reply = prompt, None
    unfinishable = None  # a valid plan its toolkit can't finish, kept in case the retry goes wrong
    for attempt in range(1, 3):
        try:
            reply = model(attempt_prompt)
        except Exception as error:  # noqa: BLE001 - isolate arbitrary model adapters
            logger.log_model_failure({"step": step, "attempt": attempt, "failure_type": "model_call_error",
                                      "response": None, "error": f"{type(error).__name__}: {error}"})
            continue
        try:
            overview = Overview.model_validate_json(reply)
            on_map = stock_by_def(items, False) if items is not None else None
            if on_map is not None:
                for name, n in stock_by_def(items, True).items():
                    on_map[name] = on_map.get(name, 0) + n
            for check in overview.finish:
                if check.kind not in ("have_items", "allowed"):
                    continue
                known = ALL if check.kind == "allowed" and check.what.casefold() == ALL else                     canonical_item(check.what, thing_names)
                if known is None:
                    raise ValueError(f"Unknown item def {check.what!r} in {check.kind}. Use an exact game def name "
                                     "shown in supplies, e.g. WoodLog, not wood; Steel, not steel; or Food for "
                                     "meals and raw food together. Display names cannot be measured.")
                check.what = known
                if on_map is not None:
                    total = (sum(n for name, n in on_map.items() if is_food(name)) if known == FOOD
                             else on_map.get(known, 0))
                    why = unreachable_count(check, list(overview.toolkits), total)
                    if why:
                        raise ValueError(why)
        except (ValidationError, TypeError, ValueError) as error:
            logger.log_model_failure({"step": step, "attempt": attempt, "failure_type": "invalid_overview",
                                      "response": reply, "error": f"{type(error).__name__}: {error}"})
            attempt_prompt = (prompt + "\n\nYour previous response was invalid:\n" + str(reply)[:4000]
                              + "\n\nValidation error:\n" + str(error) + "\nReturn one corrected JSON object only.")
            continue
        # A plan its toolkit can't finish is sent back once, then taken as it is.
        mismatch = plan_mismatch(overview) if unfinishable is None else None
        if mismatch is None:
            return overview, reply
        logger.log_model_failure({"step": step, "attempt": attempt, "failure_type": "overview_mismatch",
                                  "response": reply, "error": mismatch})
        unfinishable = overview
        attempt_prompt = (prompt + "\n\nYour plan:\n" + str(reply)[:4000] + "\n\nIt can't be finished as it "
                          "stands: " + mismatch + "\nReturn one corrected JSON object only.")
    if unfinishable is not None:
        return unfinishable, reply
    logger.log_model_failure({"step": step, "attempt": None, "failure_type": "fallback_overview",
                              "response": None, "error": "overview failed twice; using the fallback plan"})
    return FALLBACK_OVERVIEW, reply
