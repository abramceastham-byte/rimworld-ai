"""The agent loop: observe the game, ask a model what to do, act, and log it."""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rimagent.blueprints import BlueprintTracker, Status, TrackedBlueprint
from rimagent.construction import (
    CROPS,
    MAX_CHOP_TREES,
    MAX_DESIGNATE_CELLS,
    MAX_ZONE_CELLS,
    STUFF_BY_CATEGORY,
    Rect,
    TerrainMap,
    available_buildings,
    map_symbol,
)
from rimagent.events import EventListener, GameEvent
from rimagent.memory import AgentMemory, MemoryUpdate
from rimagent.rimapi import (
    NORMAL,
    Alert,
    Colonist,
    GameDefs,
    GameState,
    MapThing,
    RimApiClient,
    RimApiError,
    Threat,
    WorkTable,
    Zone,
)
from rimagent.runlog import RunLogger


class Decision(BaseModel):
    """What the model is allowed to answer. Anything else is rejected."""

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "pause",
        "resume",
        "wait",
        "enable_work",
        "disable_work",
        "set_bill",
        "create_growing_zone",
        "create_stockpile",
        "place_blueprint",
        "designate",
        "allow_items",
        "chop_trees",
    ]
    reason: str = Field(min_length=1, max_length=300)
    colonist: str | None = None  # a colonist's name, for enable_work / disable_work
    work_type: str | None = None  # e.g. "Cooking", for enable_work / disable_work
    building_id: int | None = Field(default=None, gt=0)
    recipe_def_name: str | None = None
    target_count: int | None = Field(default=None, ge=1, le=500)
    # A rectangle, for create_growing_zone / create_stockpile / designate.
    x1: int | None = Field(default=None, ge=0)
    z1: int | None = Field(default=None, ge=0)
    x2: int | None = Field(default=None, ge=0)
    z2: int | None = Field(default=None, ge=0)
    plant: str | None = None  # for create_growing_zone, e.g. "Plant_Potato"
    designation: Literal["mine", "harvest", "hunt"] | None = None
    # One building, for place_blueprint.
    building_def: str | None = None
    x: int | None = Field(default=None, ge=0)
    z: int | None = Field(default=None, ge=0)
    rotation: int = Field(default=0, ge=0, le=3)
    stuff: str | None = None  # material, e.g. "WoodLog"
    memory_update: MemoryUpdate | None = None

    @model_validator(mode="after")
    def actions_have_their_fields(self) -> "Decision":
        # Runs after the fields are checked. Raising here makes parsing fail,
        # so an incomplete action is rejected instead of half-run.
        missing = [f for f in REQUIRED_FIELDS.get(self.action, ()) if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.action} needs {', '.join(missing)}")
        return self

    def rect(self) -> Rect:
        return Rect.from_corners(self.x1, self.z1, self.x2, self.z2)


RECT = ("x1", "z1", "x2", "z2")
REQUIRED_FIELDS = {
    "enable_work": ("colonist", "work_type"),
    "disable_work": ("colonist", "work_type"),
    "set_bill": ("building_id", "recipe_def_name", "target_count"),
    "create_growing_zone": ("plant", *RECT),
    "create_stockpile": RECT,
    "place_blueprint": ("building_def", "x", "z"),
    "designate": ("designation", *RECT),
    "allow_items": RECT,
    "chop_trees": RECT,
}


INSTRUCTIONS = (
    "You are managing a RimWorld colony.\n"
    "Choose exactly one action: pause, resume, wait, enable_work, disable_work, "
    "set_bill, create_growing_zone, create_stockpile, place_blueprint, designate, "
    "allow_items, chop_trees.\n"
    "Pause and resume control time. The game is always paused while you decide; while "
    "time is running, it runs for a short while after each decision. Choose pause to "
    "stop time: while paused you can give as many orders as you like, and colonists "
    "carry them out once you resume. Pausing is also useful when a threat arrives, to "
    "set up a response step by step. Waiting while paused does nothing: nothing "
    "changes until you resume, so if you are waiting for something to happen, choose "
    "resume instead of wait.\n"
    "enable_work / disable_work switch one kind of work on or off for one colonist.\n"
    "set_bill creates or updates one work-table bill in TargetCount mode (the "
    "in-game 'Do until X' setting). It only works on a built work table listed under "
    "'Work tables and production bills', using that listed id and one of its recipe "
    "def names. Blueprints and buildings under construction are not work tables yet: "
    "never guess an id. Prefer updating an existing useful bill instead of creating "
    "competing food recipes. target_count must be between 1 and 500.\n"
    "Map actions use cell coordinates from the Map section: x grows east, z grows north.\n"
    f"create_growing_zone plants one crop in the rectangle x1,z1 to x2,z2 (at most "
    f"{MAX_ZONE_CELLS} cells; every cell needs fertile soil). create_stockpile makes a "
    f"storage zone for all normal items. designate marks everything of one kind in a "
    f"rectangle (at most {MAX_DESIGNATE_CELLS} cells): mine (rock), harvest (ripe plants) "
    "or hunt (wild animals; dangerous ones can fight back).\n"
    "place_blueprint places one building from the buildable list at x,z, with rotation "
    "0 north, 1 east, 2 south or 3 west, and a stuff material if the building lists "
    "any. Colonists then build it while time runs, if one has Construction enabled "
    "and the materials are allowed and reachable. Check 'Your blueprints' before "
    "placing more, and don't place a second building on a planned spot.\n"
    f"allow_items unforbids every forbidden item in a rectangle (at most "
    f"{MAX_DESIGNATE_CELLS} cells) so colonists can haul and use it; forbidden items "
    "are marked f on the map. chop_trees marks the trees (T on the map) in a rectangle "
    f"for cutting, up to {MAX_CHOP_TREES} per action; colonists with PlantCutting "
    "enabled cut them for wood.\n"
    "Keep new zones and buildings near the colonists and away from each other. If an "
    "area is blocked, the action fails with the reason, so pick another spot.\n"
    "You may include an optional memory_update object. Use observation only for "
    "durable insights worth remembering across the playthrough. Use replace_plan, "
    "status_updates, and policy changes for intentions and strategy.\n"
    "Do not store live facts already supplied every step: pause state, tick, wealth, "
    "storyteller, colonist count, alerts, threats, colonist age/health/mood/hunger/job/"
    "skills/work settings, new events, or available work types.\n"
    "Reply with one JSON object only. The templates below show the shape of each "
    "reply; they are not suggestions. Replace every <...> with a value you read from "
    "the current state above (a real colonist name, a work-table id, a def name from "
    "the lists, coordinates from the Map). Always include a short reason.\n"
    '{"action": "wait", "reason": "<why>"}\n'
    '{"action": "enable_work", "colonist": "<colonist name>", "work_type": "<work type>", '
    '"reason": "<why>"}\n'
    '{"action": "set_bill", "building_id": <work-table id>, "recipe_def_name": '
    '"<recipe def name from that table>", "target_count": <1-500>, "reason": "<why>"}\n'
    '{"action": "create_growing_zone", "plant": "<crop def name>", "x1": <x>, "z1": <z>, '
    '"x2": <x>, "z2": <z>, "reason": "<why>"}\n'
    '{"action": "create_stockpile", "x1": <x>, "z1": <z>, "x2": <x>, "z2": <z>, '
    '"reason": "<why>"}\n'
    '{"action": "place_blueprint", "building_def": "<buildable def name>", "x": <x>, '
    '"z": <z>, "rotation": <0-3>, "stuff": "<material, if the building lists any>", '
    '"reason": "<why>"}\n'
    '{"action": "designate", "designation": "<mine|harvest|hunt>", "x1": <x>, "z1": <z>, '
    '"x2": <x>, "z2": <z>, "reason": "<why>"}\n'
    '{"action": "allow_items", "x1": <x>, "z1": <z>, "x2": <x>, "z2": <z>, '
    '"reason": "<why>"}\n'
    '{"action": "chop_trees", "x1": <x>, "z1": <z>, "x2": <x>, "z2": <z>, '
    '"reason": "<why>"}\n'
    "Any reply may also carry a memory_update, for example:\n"
    '{"action": "<action>", "reason": "<why>", "memory_update": {"replace_plan": '
    '{"objective": "<your objective>", "status": "active", "steps": [{"description": '
    '"<a step>", "status": "pending"}]}, "policies_to_add": ["<a rule you will follow>"]}}\n'
    "Mark a plan step completed only once the current state shows it is done.\n"
)


def describe_colonist(c: Colonist) -> str:
    """One line per colonist: needs, what they're doing, best skills, enabled work."""
    passion = {0: "", 1: " (interested)", 2: " (burning)"}
    best = sorted(
        (s for s in c.skills if not s.totally_disabled), key=lambda s: -s.level
    )[:3]
    skills = ", ".join(f"{s.name} {s.level}{passion.get(s.passion, '')}" for s in best)
    work = ", ".join(w.work_type for w in c.work_priorities) or "nothing"
    return (
        f"- {c.name} (age {c.age}): mood {c.mood:.0%}, health {c.health:.0%}, "
        f"fed {c.hunger:.0%}, doing {c.current_job or 'nothing'}. "
        f"Best skills: {skills}. Works: {work}."
    )


TICKS_PER_HOUR = 2500  # RimWorld: 60,000 ticks per in-game day


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
    """Whether the agent keeps game time running between decisions.

    The game is always paused while the model decides. Running: after each
    decision the loop lets time run for a while. Paused: it doesn't, so the
    model can give several orders in a row; only resume starts time again.
    """

    paused: bool
    reason: str = ""           # why it's paused, for the prompt
    paused_decisions: int = 0  # decisions made since it was paused

    def to_log(self) -> dict:
        return {"paused": self.paused, "reason": self.reason,
                "paused_decisions": self.paused_decisions}


def mode_after_decision(mode: TimeMode, action: str) -> TimeMode:
    """pause and resume switch the mode; any other action keeps it."""
    if action == "resume":
        return TimeMode(paused=False)
    if action == "pause" and not mode.paused:
        return TimeMode(paused=True, reason="you paused it")
    if mode.paused:
        return TimeMode(True, mode.reason, mode.paused_decisions + 1)
    return mode


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
    seconds: float,
    speed: int = NORMAL,
    poll_seconds: float = 1.0,
) -> TimeWindow:
    """Unpause for up to `seconds`, then pause again.

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
    client.resume(speed)
    try:
        while (elapsed := time.monotonic() - start) < seconds:
            time.sleep(min(poll_seconds, seconds - elapsed))
            new = events.drain() if events else []
            collected += new
            threat = next((e for e in new if e.category.startswith("Threat")), None)
            try:
                paused = client.get_state().is_paused
            except (RimApiError, httpx.HTTPError):
                paused = False  # a missed check is fine; try again next poll
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


def describe_time(
    mode: TimeMode, window: TimeWindow | None, run_seconds: float, threat_run_seconds: float
) -> str:
    """Whether time is running or paused, and what happened since the last decision."""
    lines = []
    if window is not None:
        hours = window.ticks / TICKS_PER_HOUR
        since = f"Since your last decision the game ran {window.seconds:g}s ({hours:.1f} in-game hours)"
        lines.append(since + (f" and stopped early: {window.stopped_by}." if window.stopped_by else "."))
    if mode.paused:
        count = (f" (Paused for {mode.paused_decisions} decision"
                 f"{'' if mode.paused_decisions == 1 else 's'}.)" if mode.paused_decisions else "")
        lines.append(
            f"The game is paused ({mode.reason}). This is a good time to plan, or to set up a "
            "response: every order you give now, such as blueprints, zones, work settings, "
            "bills or chop and harvest marks, is queued for the colonists, and they'll start "
            "on it all once you resume. "
            f"Choose resume when you've set up what you want.{count}"
        )
    else:
        lines.append(
            f"Time is running: after this decision the game runs for {run_seconds:g}s "
            f"({threat_run_seconds:g}s while a threat is active), then pauses for your next "
            "decision. If there's a lot to set up, choose pause to stop time and give several "
            "orders in a row."
        )
    return "\n".join(lines)


def require_work_table(building_id: int | None, work_tables: list[WorkTable]) -> None:
    """Refuse bills on anything but a built work table, before asking RIMAPI."""
    if building_id not in {t.id for t in work_tables}:
        listed = ", ".join(f"{t.id} ({t.label})" for t in work_tables) or "none yet"
        raise ValueError(
            f"{building_id} is not a built work table. Work tables now: {listed}. "
            "Blueprints and buildings under construction can't take bills until built."
        )


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

    recipes = ", ".join(
        f"{recipe.def_name} ({recipe.label})" for recipe in table.recipes
    )
    lines.append("  Available recipes: " + (recipes or "none."))
    return lines


MINIMAP_RADIUS = 12
TREE_REFRESH_STEPS = 10


def colony_center(colonists: list[Colonist]) -> tuple[int, int] | None:
    spots = [c.position for c in colonists if c.position]
    if not spots:
        return None
    return (round(sum(p.x for p in spots) / len(spots)), round(sum(p.z for p in spots) / len(spots)))


def describe_map(
    colonists: list[Colonist],
    terrain: TerrainMap,
    defs: GameDefs,
    zones: list[Zone],
    placements: list[str],
    finished_research: set[str],
    items: list[MapThing] | None = None,
    trees: list[MapThing] | None = None,
    blueprints: list[tuple[TrackedBlueprint, Status]] | None = None,
) -> list[str]:
    """Everything the model needs to pick coordinates without inventing them."""
    forbidden = [i for i in items or [] if i.is_forbidden]
    blueprint_marks = {"waiting": "b", "under construction": "F", "built": "B"}
    lines = [f"Map: {terrain.width} x {terrain.height} cells. x grows east, z grows north."]
    for c in colonists:
        if c.position:
            lines.append(f"- {c.name} is at ({c.position.x}, {c.position.z}).")

    center = colony_center(colonists)
    if center:
        cx, cz = center
        r = MINIMAP_RADIUS
        x1, x2 = max(0, cx - r), min(terrain.width - 1, cx + r)
        z1, z2 = max(0, cz - r), min(terrain.height - 1, cz + r)
        # Most important mark wins when several share a cell.
        marks: dict[tuple[int, int], str] = {}
        for t in trees or []:
            marks[(t.position.x, t.position.z)] = "T"
        for i in forbidden:
            marks[(i.position.x, i.position.z)] = "f"
        for bp, status in blueprints or []:
            if status in blueprint_marks:
                for cell in bp.area():
                    marks[cell] = blueprint_marks[status]
        for c in colonists:
            if c.position:
                marks[(c.position.x, c.position.z)] = "@"
        lines += [
            "",
            f"Map around the colony, x {x1}-{x2} left to right, z {z2} (top) down to {z1}:",
            "  R rich soil  . soil (any crop)  , poor soil  _ stone/floor, nothing grows",
            "  ~ water/marsh (no heavy building)  @ colonist  T tree  f forbidden item",
            "  b your blueprint (not started)  F under construction  B just finished",
            "  Rock and allowed items are not shown; blocked spots are rejected with a reason.",
        ]
        for z in range(z2, z1 - 1, -1):
            row = "".join(
                marks.get((x, z)) or map_symbol(defs.terrain.get(terrain.at(x, z)))
                for x in range(x1, x2 + 1)
            )
            lines.append(f"  z={z:3d} {row}")

        view = Rect(x1, z1, x2, z2)
        nearby = [i for i in forbidden if view.contains(i.position.x, i.position.z)]
        if nearby:
            totals: dict[str, int] = {}
            for i in nearby:
                # RimWorld labels already carry the count ("steel x36"); drop it.
                name = re.sub(r" x\d+$", "", i.label or i.def_name)
                totals[name] = totals.get(name, 0) + i.stack_count
            box = Rect.from_corners(
                min(i.position.x for i in nearby), min(i.position.z for i in nearby),
                max(i.position.x for i in nearby), max(i.position.z for i in nearby),
            )
            listed = ", ".join(f"{label} x{n}" for label, n in sorted(totals.items())[:15])
            more = f" and {len(totals) - 15} more kinds" if len(totals) > 15 else ""
            lines.append(
                f"Forbidden items on this map section ({len(nearby)} stacks, all within {box}): "
                f"{listed}{more}."
            )
        near_trees = sum(1 for t in trees or [] if view.contains(t.position.x, t.position.z))
        lines.append(f"Trees on this map section: {near_trees}.")

    lines += ["", "Zones: " + (", ".join(f"{z.label} ({z.type}, {z.cells_count} cells)" for z in zones) or "none") + "."]
    lines.append("Placed by you: " + ("; ".join(placements) or "nothing yet") + ".")
    if blueprints:
        lines.append("Your blueprints, checked in the game just now:")
        notes = {
            "waiting": "waiting to be built (needs a colonist with Construction and the materials)",
            "under construction": "under construction",
            "built": "built",
            "gone": "gone (cancelled or destroyed)",
        }
        lines += [f"- {bp.label()}, covering {bp.area()}: {notes[status]}" for bp, status in blueprints]
    else:
        lines.append("Your blueprints: none.")

    buildable = available_buildings(finished_research)
    lines += ["", "Buildable (def name, size facing north, material types):"]
    for name, spec in buildable.items():
        material = "/".join(spec.stuff) if spec.stuff else "fixed cost"
        lines.append(f"- {name} ({spec.label}), {spec.size[0]}x{spec.size[1]}, {material}")
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
    return lines


def describe_placement(decision: Decision, result: object) -> str:
    """A short record of a successful map action, for later prompts."""
    if decision.action == "create_growing_zone":
        return f"{decision.plant} field {decision.rect()} (zone {result})"
    if decision.action == "create_stockpile":
        return f"stockpile {decision.rect()} (zone {result})"
    if decision.action == "place_blueprint":
        return f"{decision.building_def} blueprint covering {result}"
    if decision.action == "allow_items":
        return f"allowed {result} item stacks in {decision.rect()}"
    if decision.action == "chop_trees":
        return f"marked {result} trees for cutting in {decision.rect()}"
    return f"{decision.designation} designation {decision.rect()}"


def build_prompt(
    state: GameState,
    colonists: list[Colonist],
    alerts: list[Alert],
    threats: list[Threat],
    events: list[GameEvent],
    work_types: list[str],
    work_tables: list[WorkTable],
    time_status: str,
    memory: AgentMemory,
    map_lines: list[str] | None = None,
) -> str:
    """Build a prompt for the LLM based on the current game state."""
    lines = [
        "Current game state:",
        time_status,
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
        *[describe_colonist(c) for c in colonists],
        "",
        "Work types: " + ", ".join(work_types) + ".",
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
    return "\n".join(lines) + INSTRUCTIONS + "\n" + memory.to_prompt()


def parse_decision(reply: str) -> Decision:
    """Turn the model's raw reply into a Decision, falling back to "wait"."""
    try:
        return Decision.model_validate_json(reply)
    except ValidationError:
        return Decision(action="wait", reason="could not parse reply")


def request_decision(
    model: Callable[[str], str],
    prompt: str,
    logger: RunLogger,
    step: int,
    danger_present: bool,
) -> tuple[Decision, object | None]:
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
            return Decision.model_validate_json(reply), reply
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

    action: Literal["pause", "wait"] = "pause" if danger_present else "wait"
    reason = (
        "model failed twice; paused because a threat is present"
        if danger_present
        else "model failed twice; waiting safely"
    )
    return Decision(action=action, reason=reason), last_reply


def colonist_id(colonists: list[Colonist], name: str | None) -> int:
    """Find a colonist's id from the name the model used (case doesn't matter)."""
    for c in colonists:
        if name and c.name.lower() == name.lower():
            return c.id
    raise ValueError(f"no colonist named {name!r}")


def run(
    client: RimApiClient,
    model: Callable[[str], str],
    logger: RunLogger,
    events: EventListener | None = None,
    max_steps: int = 10,
    run_seconds: float = 10.0,
    threat_run_seconds: float = 3.0,
    speed: int = NORMAL,
    memory: AgentMemory | None = None,
) -> None:
    """Run the agent loop for max_steps decisions.

    The game is always paused while the model decides. In running mode the loop
    then lets time run for run_seconds (threat_run_seconds while a threat is
    active), stopping early if a threat arrives or the game pauses itself. The
    model's pause switches to paused mode: it can give several orders in a row
    and time stays stopped until it chooses resume. If the game pauses itself
    (e.g. RimWorld's auto-pause on a major threat), the loop switches to paused
    mode too. See TimeMode, mode_after_decision and mode_after_window.
    """
    work_tables: list[WorkTable] = []

    def set_bill(d: Decision, cols: list[Colonist]) -> None:
        # work_tables is this step's list (read below, before any action runs).
        require_work_table(d.building_id, work_tables)
        client.set_target_bill(d.building_id, d.recipe_def_name, d.target_count)

    # Each action gets the decision and this step's colonists, so work actions
    # can turn the colonist's name into the id RIMAPI needs. pause and resume do
    # nothing here: the loop decides whether time runs after each step.
    actions: dict[str, Callable[[Decision, list[Colonist]], object]] = {
        "wait": lambda d, cols: None,
        "pause": lambda d, cols: None,
        "resume": lambda d, cols: None,
        "enable_work": lambda d, cols: client.enable_work(
            colonist_id(cols, d.colonist), d.work_type
        ),
        "disable_work": lambda d, cols: client.disable_work(
            colonist_id(cols, d.colonist), d.work_type
        ),
        "set_bill": set_bill,
        # Map actions return the zone id or covered cells, for describe_placement.
        "create_growing_zone": lambda d, cols: client.create_growing_zone(d.plant, d.rect()),
        "create_stockpile": lambda d, cols: client.create_stockpile(d.rect()),
        "place_blueprint": lambda d, cols: client.place_blueprint(
            d.building_def, d.x, d.z, d.rotation, d.stuff
        ),
        "designate": lambda d, cols: client.designate(d.designation, d.rect()),
        "allow_items": lambda d, cols: client.allow_items(d.rect()),
        "chop_trees": lambda d, cols: client.chop_trees(d.rect()),
    }
    # Blueprints are tracked separately (see BlueprintTracker), with live status.
    map_actions = (
        "create_growing_zone",
        "create_stockpile",
        "designate",
        "allow_items",
        "chop_trees",
    )
    placements: list[str] = []  # RIMAPI doesn't say where zones are, so remember
    # /map/plants is ~7 MB, so trees are re-read every few steps, not every step.
    trees: list[MapThing] = []
    trees_read_at: int | None = None
    work_types = client.get_work_types()
    if memory is None:
        memory = AgentMemory()
    blueprints = BlueprintTracker(memory.memory_dir / "blueprints.json")
    window: TimeWindow | None = None  # what happened while time last ran
    carried_events: list[GameEvent] = []  # letters/messages that arrived while time ran
    # Start in the game's own state: if it's paused now, stay paused until the
    # model resumes (that's the situation the first gpt-oss run got stuck in).
    if client.get_state().is_paused:
        mode = TimeMode(paused=True, reason="it was already paused when this session started")
    else:
        mode = TimeMode(paused=False)

    try:
        for step in range(max_steps):
            try:
                client.pause()  # the model always decides on a paused game
                state = client.get_state()
                colonists = client.get_colonists()
                alerts = client.get_alerts()
                threats = client.get_threats()
                work_tables = client.get_work_tables()
                if trees_read_at is None or step - trees_read_at >= TREE_REFRESH_STEPS:
                    trees = client.get_trees()
                    trees_read_at = step
                blueprint_report = blueprints.check(client)
                map_lines = describe_map(
                    colonists,
                    client.get_terrain(),
                    client.get_game_defs(),
                    client.get_zones(),
                    placements,
                    client.get_finished_research(),
                    client.get_items(),
                    trees,
                    blueprint_report,
                )
            except (RimApiError, httpx.TimeoutException) as e:
                print(f"Step {step}: could not read game state: {e}")
                time.sleep(2)
                continue
            new_events = carried_events + (events.drain() if events else [])
            carried_events = []

            logger.log_state(state)
            for event in new_events:
                logger.log_event(event)

            time_status = describe_time(mode, window, run_seconds, threat_run_seconds)
            mode_at_decision = mode
            prompt = build_prompt(
                state,
                colonists,
                alerts,
                threats,
                new_events,
                work_types,
                work_tables,
                time_status,
                memory,
                map_lines,
            )

            danger_present = any(threat.active for threat in threats) or any(
                event.category.startswith("Threat") for event in new_events
            )
            decision, reply = request_decision(
                model,
                prompt,
                logger,
                step,
                danger_present,
            )
            error = None
            try:
                result = actions[decision.action](decision, colonists)
            except (RimApiError, ValueError) as e:
                # RimApiError: RIMAPI refused (e.g. work the colonist can't do).
                # ValueError: our own checks refused, e.g. an unknown colonist,
                # a blocked area, a building the colony can't build yet, or a
                # bill on something that isn't a built work table.
                error = str(e)
                print(f"Step {step}: failed to {decision.action}: {e}")

            if error is None and decision.action == "place_blueprint":
                blueprints.add(
                    decision.building_def, decision.stuff, decision.x, decision.z,
                    decision.rotation, result,
                )
            elif error is None and decision.action in map_actions:
                placements.append(describe_placement(decision, result))
                del placements[:-15]  # keep the prompt bounded
            if decision.action == "chop_trees":
                trees_read_at = None  # re-read next step so cut trees update

            memory.remember_decision(
                step,
                decision.action,
                decision.reason,
                error,
                decision.colonist,
                decision.work_type,
            )
            memory.apply_update(step, state.game_tick, decision.memory_update)
            print(f"Step {step}: {decision.action} ({decision.reason})")

            # pause/resume switch the mode; in running mode, let time run now.
            mode = mode_after_decision(mode, decision.action)
            if mode.paused:
                window = None
                print(f"         game stays paused ({mode.reason})")
            else:
                seconds = threat_run_seconds if danger_present else run_seconds
                window = let_time_run(client, events, seconds, speed)
                carried_events = window.events
                mode = mode_after_window(mode, window)
                note = f", stopped early: {window.stopped_by}" if window.stopped_by else ""
                print(f"         game ran {window.seconds:g}s ({window.ticks} ticks){note}")

            logger.log_decision(
                {
                    "step": step,
                    "reply": reply,
                    "action": decision.action,
                    "colonist": decision.colonist,
                    "work_type": decision.work_type,
                    "building_id": decision.building_id,
                    "recipe_def_name": decision.recipe_def_name,
                    "target_count": decision.target_count,
                    "rect": (
                        [decision.x1, decision.z1, decision.x2, decision.z2]
                        if decision.x1 is not None
                        else None
                    ),
                    "plant": decision.plant,
                    "designation": decision.designation,
                    "building_def": decision.building_def,
                    "position": [decision.x, decision.z] if decision.x is not None else None,
                    "rotation": decision.rotation,
                    "stuff": decision.stuff,
                    "reason": decision.reason,
                    "memory_update": (
                        decision.memory_update.model_dump()
                        if decision.memory_update
                        else None
                    ),
                    "time_status": time_status,
                    "time_mode": mode_at_decision.to_log(),  # as the model saw it
                    "time_after": window.to_log() if window else "stayed paused",
                    "error": error,
                    # The model's reasoning, for reading later. It is not put
                    # back into the next prompt. None for models without it.
                    "thinking": getattr(model, "last_thinking", None),
                    "model_stats": getattr(model, "last_stats", None),
                }
            )
    finally:
        # Runs even on Ctrl+C or a crash: don't leave the colony unattended.
        try:
            client.pause()
        except (RimApiError, httpx.HTTPError):
            pass
