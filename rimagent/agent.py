"""The agent loop: observe the game, ask a model what to do, act, and log it."""

import re
import time
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
    "enable_work / disable_work switch one kind of work on or off for one colonist.\n"
    "set_bill creates or updates one work-table bill in TargetCount mode (the "
    "in-game 'Do until X' setting). Use an exact work-table id and recipe def name "
    "from the current state. Prefer updating an existing useful bill instead of "
    "creating competing food recipes. target_count must be between 1 and 500.\n"
    "Map actions use cell coordinates from the Map section: x grows east, z grows north.\n"
    f"create_growing_zone plants one crop in the rectangle x1,z1 to x2,z2 (at most "
    f"{MAX_ZONE_CELLS} cells; every cell needs fertile soil). create_stockpile makes a "
    f"storage zone for all normal items. designate marks everything of one kind in a "
    f"rectangle (at most {MAX_DESIGNATE_CELLS} cells): mine (rock), harvest (ripe plants) "
    "or hunt (wild animals; dangerous ones can fight back).\n"
    "place_blueprint places one building from the buildable list at x,z, with rotation "
    "0 north, 1 east, 2 south or 3 west, and a stuff material if the building lists "
    "any. Colonists then build it if they have Construction enabled and materials.\n"
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


def describe_pause(
    is_paused: bool, paused_by_agent: bool | None, threat_letter: GameEvent | None
) -> str:
    """Explain to the model whether the game is paused, and why."""
    if not is_paused:
        return "Game is running."
    if paused_by_agent is None:
        return "Game is paused (it was already paused when you started)."
    if paused_by_agent:
        return "Game is paused: you paused it."
    if threat_letter:
        return f"Game is paused: the game paused itself because of a major threat ({threat_letter.text})."
    return "Game is paused by the player or an open menu."


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
) -> list[str]:
    """Everything the model needs to pick coordinates without inventing them."""
    forbidden = [i for i in items or [] if i.is_forbidden]
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
        for c in colonists:
            if c.position:
                marks[(c.position.x, c.position.z)] = "@"
        lines += [
            "",
            f"Map around the colony, x {x1}-{x2} left to right, z {z2} (top) down to {z1}:",
            "  R rich soil  . soil (any crop)  , poor soil  _ stone/floor, nothing grows",
            "  ~ water/marsh (no heavy building)  @ colonist  T tree  f forbidden item",
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
    pause_status: str,
    memory: AgentMemory,
    map_lines: list[str] | None = None,
) -> str:
    """Build a prompt for the LLM based on the current game state."""
    lines = [
        "Current game state:",
        pause_status,
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
    step_seconds: float = 5.0,
    memory: AgentMemory | None = None,
) -> None:
    """Run the agent loop for max_steps steps."""

    # Each action gets the decision and this step's colonists, so work actions
    # can turn the colonist's name into the id RIMAPI needs.
    actions: dict[str, Callable[[Decision, list[Colonist]], object]] = {
        "wait": lambda d, cols: None,
        "pause": lambda d, cols: client.pause(),
        "resume": lambda d, cols: client.resume(),
        "enable_work": lambda d, cols: client.enable_work(
            colonist_id(cols, d.colonist), d.work_type
        ),
        "disable_work": lambda d, cols: client.disable_work(
            colonist_id(cols, d.colonist), d.work_type
        ),
        "set_bill": lambda d, cols: client.set_target_bill(
            d.building_id, d.recipe_def_name, d.target_count
        ),
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
    map_actions = (
        "create_growing_zone",
        "create_stockpile",
        "place_blueprint",
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
    paused_by_agent: bool | None = None  # unknown until the agent acts
    last_threat_letter: GameEvent | None = None

    try:
        for step in range(max_steps):
            try:
                state = client.get_state()
                colonists = client.get_colonists()
                alerts = client.get_alerts()
                threats = client.get_threats()
                work_tables = client.get_work_tables()
                if trees_read_at is None or step - trees_read_at >= TREE_REFRESH_STEPS:
                    trees = client.get_trees()
                    trees_read_at = step
                map_lines = describe_map(
                    colonists,
                    client.get_terrain(),
                    client.get_game_defs(),
                    client.get_zones(),
                    placements,
                    client.get_finished_research(),
                    client.get_items(),
                    trees,
                )
            except (RimApiError, httpx.TimeoutException) as e:
                print(f"Step {step}: could not read game state: {e}")
                time.sleep(step_seconds)
                continue
            new_events = events.drain() if events else []

            for event in new_events:
                if event.category == "ThreatBig":
                    last_threat_letter = event

            if not state.is_paused:
                # The game is running, so any earlier pause is over, whoever caused it.
                paused_by_agent = False
                last_threat_letter = None

            logger.log_state(state)
            for event in new_events:
                logger.log_event(event)

            pause_status = describe_pause(
                state.is_paused,
                paused_by_agent,
                last_threat_letter,
            )
            prompt = build_prompt(
                state,
                colonists,
                alerts,
                threats,
                new_events,
                work_types,
                work_tables,
                pause_status,
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
                # a blocked area, or a building the colony can't build yet.
                error = str(e)
                print(f"Step {step}: failed to {decision.action}: {e}")

            if error is None and decision.action in map_actions:
                placements.append(describe_placement(decision, result))
                del placements[:-15]  # keep the prompt bounded
            if decision.action == "chop_trees":
                trees_read_at = None  # re-read next step so cut trees update

            if error is None:
                if decision.action == "pause":
                    paused_by_agent = True
                elif decision.action == "resume":
                    paused_by_agent = False

            memory.remember_decision(
                step,
                decision.action,
                decision.reason,
                error,
                decision.colonist,
                decision.work_type,
            )
            memory.apply_update(step, state.game_tick, decision.memory_update)
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
                    "pause_status": pause_status,
                    "error": error,
                    # The model's reasoning, for reading later. It is not put
                    # back into the next prompt. None for models without it.
                    "thinking": getattr(model, "last_thinking", None),
                    "model_stats": getattr(model, "last_stats", None),
                }
            )
            print(f"Step {step}: {decision.action} ({decision.reason})")
            time.sleep(step_seconds)
    finally:
        # Runs even on Ctrl+C or a crash: don't leave the colony unattended.
        try:
            client.pause()
        except (RimApiError, httpx.HTTPError):
            pass
