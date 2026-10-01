"""The overview/focus/threat loop: three kinds of prompt instead of one big one.

- Overview: a short picture of the whole colony. The model writes a plan in
  its own words and names the toolkits it needs from a fixed list. No orders.
- Focus: the plan plus only what its toolkits need: their orders, rules,
  state and observations (lessons saved for that toolkit). The model may ask
  for another toolkit, say the plan is done, or ask to replan.
- Threat: only the threat, the colonists and weapons, and combat orders.

Every prompt starts with the same vital-signs header, so nothing urgent is
ever out of sight. This module only assembles text and reply models; the loop
that runs it is agent.run(loop="modes").
"""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rimagent.actions import MAX_ACTIONS, Action
from rimagent.defense import ALERT_NOTES
from rimagent.finish import RAW_FOODS, FinishCheck, finish_mismatch, finish_progress, is_food

MAX_FOCUS_TURNS = 12     # focus turns on one plan before the overview looks again
MAX_STEPS = 8            # a main job and up to two side jobs
MAX_TOOLKITS = 3         # toolkits one plan may carry: one job, with the orders it needs (Sonnet freed
                         # supplies, then idled waiting to switch toolkit while there was work to give)
FAILED_TURNS_TO_REPLAN = 2  # consecutive focus turns where every order failed


@dataclass(frozen=True)
class Toolkit:
    summary: str                     # one line for the overview's menu
    actions: tuple[str, ...]         # orders it offers
    sections: tuple[str, ...] = ()   # prompt blocks it shows (describe_map sections and agent blocks)
    tags: tuple[str, ...] = ()       # observation tags it shows (Operations.tagged, saved lessons)
    designations: tuple[str, ...] = ()  # which kinds of designate it offers


# Every focus: nothing gets done without work settings, and allowing forbidden
# supplies is one instant order. Without it a construction focus was told to
# "include allow_items" for its forbidden wood, couldn't, and placed beds instead.
CORE_ACTIONS = ("enable_work", "disable_work", "allow_items")
PLACE = ("map", "site", "open_ground", "buildings", "blueprints", "catalog", "anchors", "supplies")

TOOLKITS: dict[str, Toolkit] = {
    "construction": Toolkit(
        "rooms, walls, floors, furniture and other buildings",
        ("place_blueprint", "build_room", "build_wall", "build_floor", "plan_area"),
        PLACE + ("floors", "room_cost", "room_slots"), ("shelter", "zones")),
    "storage": Toolkit(
        "stockpiles, shelves, storage rooms, and allowing forbidden items",
        ("create_stockpile", "build_room", "place_blueprint", "allow_items", "plan_area"),
        PLACE + ("forbidden", "room_cost", "room_slots"), ("storage", "zones")),
    "work": Toolkit(
        "who does which work; fixing idle colonists",
        (), ("work_pending",)),
    "production": Toolkit(
        "work tables and bills: cooking, stonecutting, crafting",
        ("set_bill", "place_blueprint", "plan_area"), PLACE + ("tables",)),
    "farming": Toolkit(
        "growing zones and crops; harvesting ripe wild plants",
        ("create_growing_zone", "designate", "plan_area"), ("map", "site", "open_ground", "crops"),
        ("farming", "zones"), ("harvest",)),
    "research": Toolkit(
        "the research project and research benches",
        ("set_research", "place_blueprint", "plan_area"), PLACE + ("research",)),
    "wood_mining": Toolkit(
        "chop_trees directly orders wood gathering; no preliminary designation is needed. "
        "designate mine is only for rock or ore; allow_items unlocks forbidden supplies",
        ("chop_trees", "designate", "allow_items"), ("map", "trees_nearby", "trees", "supplies", "forbidden"),
        ("wood_mining",), ("mine",)),
    "hunting": Toolkit(
        "hunting wild animals for meat and leather; arming hunters",
        ("designate", "equip"), ("map", "weapons"), ("hunting",), ("hunt",)),
    "power": Toolkit(
        "generators, batteries, and connecting them to what needs power",
        ("connect_power", "place_blueprint", "plan_area"), PLACE, ("power",)),
    "defense": Toolkit(
        "defenses against raids: barricades, spike traps, turrets; arming colonists",
        ("place_blueprint", "equip", "plan_area"), PLACE + ("defense", "weapons"), ("defense",)),
    "medical": Toolkit(
        "tending the injured, bed rest, carrying the downed to bed, hospital beds",
        ("tend", "bed_rest", "rescue", "place_blueprint"), ("injured", "site", "buildings", "catalog",
                                                           "anchors"), ("medical", "shelter")),
}
# Threat mode only: not offered by the overview.
COMBAT = Toolkit("drafting and fighting", ("draft", "move", "attack", "equip", "tend", "bed_rest", "rescue"),
                 ("map",), ("combat", "medical"))

# The work types each toolkit's orders depend on: enable_work and disable_work
# offer only these, and not at all when nobody could take one up or drop one.
# With the order always offered, a model filled spare slots with it: Art,
# Warden and Tailoring switched on "to utilize underused skills". The work
# toolkit covers every work type.
TOOLKIT_WORK: dict[str, tuple[str, ...]] = {
    "construction": ("Construction",),
    "storage": ("Hauling", "Construction"),
    "production": ("Cooking", "Crafting", "Smithing", "Tailoring", "Art", "Construction"),
    "farming": ("Growing", "PlantCutting"),
    "research": ("Research", "Construction"),
    "wood_mining": ("PlantCutting", "Mining"),
    "hunting": ("Hunting",),
    "power": ("Construction",),
    "medical": ("Doctor",),
    "defense": ("Construction",),
}

ToolkitName = Literal[tuple(TOOLKITS)]  # type: ignore[valid-type]

# Per-order rules, shown only in prompts that offer the order.
ACTION_RULES = {
    "enable_work": "enable_work / disable_work: colonist, work_type.",
    "set_bill": "set_bill: building_id, recipe_def_name, target_count (1..500); creates/updates Do until X.",
    "set_research": "set_research: project (available name); no bills on research benches.",
    "create_growing_zone": ("create_growing_zone: plant, rectangle sized by the Recommended growing area line "
                            "(fields over twice the recommendation in all are refused; don't copy a whole open-"
                            "ground spot, use a small part of it), in the open: "
                            "never inside a room or where you mean to build one, since a closed room is roofed "
                            "and crops need sunlight."),
    "create_stockpile": ("create_stockpile: rectangle (<=225); normal priority. While storage is under half full it "
                         "is refused except inside a room built with purpose storage. Check existing capacity and "
                         "pending shelves before adding storage. A shelf can be built over stockpile cells and "
                         "takes over those cells; creating a stockpile over an existing shelf is refused. A "
                         "stockpile losing cells as shelves are built does not need replacing."),
    "place_blueprint": ("place_blueprint: building_def, x,z, rotation (0=N,1=E,2=S,3=W), stuff. x,z is RimWorld's "
                        "anchor, not always a corner; use checked candidates. Fixed-cost buildings: stuff=null."),
    "build_room": ("build_room: OUTER rectangle including the walls (4..15 cells per side; inside is 2 smaller each "
                   "way), stuff, door_side (north/east/south/west), purpose (bedroom, barracks, kitchen, dining, "
                   "workshop, storage, hospital, other). One centred door. Only the perimeter must be clear; "
                   "existing walls, doors or rock on it are kept. No floor included."),
    "build_wall": "build_wall: straight rectangle (<=20 cells), stuff; blocked cells are skipped. It only adds walls.",
    "build_floor": ("build_floor: rectangle (<=225), terrain (listed floor name); existing/pending floors are "
                    "skipped. Floor the inside of finished rooms: bare ground is dirty and ugly (food poisoning in "
                    "kitchens, infection in hospitals, worse mood everywhere)."),
    "designate": ("designate: never for trees (use chop_trees). mine = rock only (<=400 cells); hunt = exactly ONE "
                  "cell (x1=x2, z1=z2) at an observed animal, needs an armed hunter and risks retaliation; "
                  "harvest = ripe wild plants (fields are harvested anyway). Only use observed targets."),
    "allow_items": ("allow_items: what = all, Food (meals and raw food) or one forbidden item def; makes every "
                    "forbidden item of that kind on the whole map usable at once. It takes no game time: if a turn "
                    "only allows items, reply time=hold and carry on."),
    "plan_area": ("plan_area: purpose (bedrooms, kitchen_dining, storage, workshop, research, fields, power, "
                  "recreation, hospital), rectangle (<=30 a side). Builds nothing: it reserves where that kind of "
                  "building goes, so the base stays organised. Planning a purpose again moves its area; areas "
                  "of different purposes can't overlap."),
    "chop_trees": ("chop_trees: THE way to get wood. Any rectangle; orders up to 30 trees nearest its centre cut. "
                   "Trees already ordered are skipped."),
    "connect_power": ("connect_power: from_x,from_z = any cell of the power source (generator, battery, or a "
                      "conduit of its grid); to_x,to_z = any cell of what needs power (lamp, stove, bench). The route "
                      "is worked out for you: plain conduits under walls and buildings, hidden conduits on open "
                      "ground, reusing existing ones. Both ends may still be blueprints. Costs Steel."),
    "draft": ("draft: colonist, drafted true/false. Drafted colonists stop all work and only follow move/attack; "
              "release them (drafted false) once the fight is over."),
    "move": "move: a drafted colonist to x,z (e.g. behind a wall, into a doorway, away from melee).",
    "attack": ("attack: a drafted colonist and an enemy id. With a gun they shoot from where they stand; unarmed or "
               "melee-armed they walk up to it."),
    "equip": "equip: colonist, item_id of a weapon on the map (forbidden ones are allowed first).",
    "tend": "tend: patient, doctor (or null for any doctor). Tending needs a colonist with Doctor work.",
    "bed_rest": "bed_rest: an injured patient goes to bed to heal.",
    "rescue": "rescue: colonist carries a downed patient to a bed.",
}
ACTION_RULES["disable_work"] = ACTION_RULES["enable_work"]

REPLY_RULES = """Orders execute while paused, in order, and every one is tried even if an earlier one
failed. Next turn each is reported by number as OK, UNCHANGED or FAILED; a result may end with HINT:
a suggested next step, only if still needed for your goal. A suggested placement is an option, not a new task.
Standing orders (work, research, bills, blueprints, zones, cut orders) stay
in effect: never repeat one. Send only the orders this turn needs, often 1 to 4; fewer is fine and
there is no reward for filling all 10. A repeat of an order in the same reply is dropped, not done twice.
Use exact names and ids from this prompt; x grows east, z north;
rectangles are inclusive x1,z1,x2,z2. time: advance lets the work happen; hold keeps the game paused."""

LESSON_RULE = """lesson (optional): usually omit it. Save one only when an observed result establishes something
useful that the supplied rules do not already explain. Do not turn a failed placement into a universal rule
or save an assumption as fact. Current observations and supplied rules take precedence over saved lessons.
Not what you ordered this turn, no coordinates or stock numbers."""


class Overview(BaseModel):
    """The overview reply: a plan in the model's words and the toolkits it needs."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=300)
    # What from the prompt the plan relies on: writing it down is the check
    # that the plan can be done with what the colony has.
    uses: str = Field(min_length=1, max_length=300)
    steps: list[str] = Field(default_factory=list, max_length=MAX_STEPS)
    toolkits: list[ToolkitName] = Field(min_length=1, max_length=MAX_TOOLKITS)  # type: ignore[valid-type]
    finish: list[FinishCheck] = Field(min_length=1, max_length=6)


def plan_mismatch(overview: "Overview") -> str | None:
    """Why the plan's finish checks can't be met with its toolkit, or None if they can.

    One plan asked wood_mining for "stone blocks in the stockpile", which only
    a bill at a stonecutter's table (production) makes, and drifted for 8 turns.
    """
    return finish_mismatch(overview.finish, list(overview.toolkits))


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=400)
    toolkit: ToolkitName | Literal["combat", "general"] = "general"  # type: ignore[valid-type]


class FocusTurn(BaseModel):
    """A focus reply: orders, time, and how the plan stands."""

    actions: list[Action] = Field(max_length=MAX_ACTIONS)
    time: Literal["advance", "hold"]
    plan: Literal["continue", "done", "replan"]
    # The plan step these orders work on, and steps now finished (1-based).
    # Without them the model re-read six open steps every turn and jumped between them.
    step: int = Field(default=1, ge=1, le=MAX_STEPS)
    steps_done: list[int] = Field(default_factory=list, max_length=MAX_STEPS)
    request_toolkits: list[ToolkitName] = Field(default_factory=list, max_length=1)  # type: ignore[valid-type]
    lesson: Lesson | None = None

    model_config = ConfigDict(extra="forbid", json_schema_extra=lambda schema: schema.setdefault(
        "required", []).append("step"))


class ThreatTurn(BaseModel):
    """A threat reply: combat and medical orders, and time."""

    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(max_length=MAX_ACTIONS)
    time: Literal["advance", "hold"]
    lesson: str | None = Field(default=None, max_length=400)


def allowed_actions(toolkits: list[str]) -> set[str]:
    return set(CORE_ACTIONS).union(*(TOOLKITS[t].actions for t in toolkits if t in TOOLKITS))


def toolkit_work(toolkits: list[str]) -> set[str] | None:
    """The work types these toolkits cover; None for all of them (the work toolkit)."""
    if "work" in toolkits:
        return None
    return set().union(*(TOOLKIT_WORK.get(t, ()) for t in toolkits))


def designations(toolkits: list[str]) -> set[str]:
    return set().union(*(TOOLKITS[t].designations for t in toolkits if t in TOOLKITS))


def narrowed(choices, allowed: set[str], designate: set[str] | None = None, work: set[str] | None = None):
    """A copy of choices offering only allowed orders (and only these designations,
    and enable_work / disable_work only for these work types)."""
    from copy import deepcopy
    c = deepcopy(choices)
    c.fields = {k: v for k, v in c.fields.items() if k in allowed}
    if work is not None:
        c.can_enable = {n: w & work for n, w in c.can_enable.items()}
        c.can_disable = {n: w & work for n, w in c.can_disable.items()}
        for action, options in (("enable_work", c.can_enable), ("disable_work", c.can_disable)):
            c.fields.pop(action, None)
            if action in allowed and any(options.values()):
                c.fields[action] = {"colonist": [n for n, o in options.items() if o],
                                    "work_type": sorted(set().union(*options.values()))}
    if "designate" in c.fields and designate is not None:
        kinds = [d for d in c.fields["designate"]["designation"] if d in designate]
        if kinds:
            c.fields["designate"] = {**c.fields["designate"], "designation": kinds}
        else:
            del c.fields["designate"]
    def relevant(note: str) -> bool:
        words = note.split(":")[0].split(" ")
        if words[0] == "stuff":
            return bool(allowed & {"place_blueprint", "build_room", "build_wall"})
        if words[0] == "designate" and designate is not None and len(words) > 1:
            return "designate" in allowed and words[1] in designate
        return words[0] in allowed

    c.unavailable = [u for u in c.unavailable if relevant(u)]
    return c


# --- the vital-signs header -------------------------------------------------

MEAL_NUTRITION, RAW_NUTRITION = 0.9, 0.05
NUTRITION_PER_DAY = 1.6  # one colonist, roughly


def food_days(items, colonists: int) -> str:
    """Allowed food on the map, and about how many days it feeds everyone."""
    meals = sum(i.stack_count for i in items if not i.is_forbidden and i.def_name.startswith("Meal"))
    raw = sum(i.stack_count for i in items if not i.is_forbidden
              and (i.def_name in RAW_FOODS or i.def_name.startswith("Meat_")))
    days = (meals * MEAL_NUTRITION + raw * RAW_NUTRITION) / (NUTRITION_PER_DAY * max(1, colonists))
    forbidden = sum(i.stack_count for i in items if i.is_forbidden and is_food(i.def_name))
    return (f"Food: {meals} meals, {raw} raw food, about {days:.1f} days for {colonists} colonists"
            + (f"; {forbidden} more food is forbidden (allow_items to use it)." if forbidden else "."))


STARVING = 0.1  # fed below this with no allowed food: nothing else matters


def food_emergency(ctx: dict) -> str | None:
    """Why feeding the colony must come before the plan in hand, or None.

    One run spent twelve focus turns chopping trees while all three colonists
    went from fed 69% to 0% beside 57 forbidden survival meals.
    """
    if any(a.label == "Starvation" for a in ctx["alerts"]):
        return "colonists are starving (Starvation alert): food comes first"
    hungry = [c.name for c in ctx["colonists"] if c.hunger < STARVING]
    if hungry and not any(not i.is_forbidden and is_food(i.def_name) for i in ctx["items"]):
        return f"{', '.join(hungry)} nearly starving and no food is allowed: food comes first"
    return None


def vital_header(ctx: dict, drafted: set[str], last_counts: str = "") -> list[str]:
    """What every prompt shows first: time, food, each colonist, threats, alerts."""
    state = ctx["state"]
    general = [line for tag, line in ctx["operations"].tagged if tag == "general"
               and line.startswith(("Date:", "Outdoors:"))]
    lines = ["VITAL SIGNS", f"Tick {state.game_tick}. " + " ".join(general),
             food_days(ctx["items"], len(ctx["colonists"]))]
    for c in ctx["colonists"]:
        flags = [f for f, on in (("DRAFTED", c.name in drafted), ("BADLY HURT", c.health < 0.6),
                                 ("HUNGRY", c.hunger < 0.25), ("UNHAPPY", c.mood < 0.3)) if on]
        lines.append(f"- {c.name}: health {c.health:.0%}, mood {c.mood:.0%}, fed {c.hunger:.0%}, "
                     f"doing {c.current_job or 'nothing'}" + (f" [{', '.join(flags)}]" if flags else ""))
    lines.append("Threats: " + ("; ".join(t.summary().rstrip(".") for t in ctx["threats"]) or "none") + ".")
    visitors = ctx.get("visitors") or []
    if visitors:
        # A model never mentioned a visiting group: they showed only as an event line.
        lines.append("Others on the map: " + " ".join(v.summary() for v in visitors))
    lines.append("Alerts: " + (", ".join(a.label + (f" ({ALERT_NOTES[a.label]})" if a.label in ALERT_NOTES else "")
                                        for a in ctx["alerts"]) or "none") + ".")
    events = ctx["blocks"].get("events") or []
    if events:
        lines += ["New since last turn:", *events]
    if last_counts:
        lines.append("Last turn: " + last_counts)
    return lines


# --- prompts ------------------------------------------------------------------

def tagged(ctx: dict, tags) -> list[str]:
    return [line for tag, line in ctx["operations"].tagged if tag in tags]


def feedback_block(ctx: dict) -> list[str]:
    feedback = ctx["blocks"].get("feedback") or []
    return ["What your last orders did, unresolved failures and hints:", *(f"- {r}" for r in feedback), ""] \
        if feedback else []


def overview_prompt(ctx: dict, drafted: set[str], memory, why: str, plan=None, last_counts: str = "",
                    interrupted=None) -> str:
    """The open-ended prompt: the whole colony in brief; reply with a plan."""
    sections, blocks = ctx["sections"], ctx["blocks"]
    # Project names only: the focus prompt for research gives labels and costs.
    research = [re.sub(r" \([^()]*points\)", "", line) for line in blocks.get("research", [])]
    glance = [*sections.get("buildings", []), *research,
              *[line for line in sections.get("supplies", []) if line.startswith(
                  ("Colony stock", "Allowed items", "Also allowed", "Forbidden,", "Forbidden building materials",
                   "Building materials", "Your unfinished builds need"))],
              *tagged(ctx, ("storage", "power", "shelter")),
              *[line for line in tagged(ctx, ("farming",)) if line.startswith(("Growing season", "Recommended",
                                                                                "Farms:"))],
              *sections.get("site", [])]
    lessons = memory.observations_for(set(TOOLKITS) | {"general", "combat", "medical"})[-8:]
    lines = [*vital_header(ctx, drafted, last_counts), ""]
    emergency = food_emergency(ctx)
    if emergency:
        lines += [f"URGENT: {emergency}. Plan to feed the colony now: allow forbidden food (storage toolkit, "
                  "allow_items what=Food), or hunt, or harvest ripe wild plants.", ""]
    lines += [*feedback_block(ctx), "Colony at a glance:", *glance, ""]
    if plan is not None:
        lines += [f"Previous plan: {plan.objective} ({why}).", ""]
    else:
        lines += [f"No plan yet ({why}).", ""]
    if interrupted is not None:
        progress, _ = finish_progress(interrupted.finish, ctx.get("finish_state"))
        lines += [f"Unfinished plan an emergency cut short: {interrupted.objective} "
                  f"[{', '.join(interrupted.toolkits)}]. Its finish checks now:", *progress,
                  "Unless something else matters more now, plan it again so the work already done isn't wasted "
                  "(same goal, toolkit and checks).", ""]
    if lessons:
        lines += ["Lessons you saved:", *(f"- {n}" for n in lessons), ""]
    catalog = [line[2:].split(",")[0].split(" (")[0] for line in sections.get("catalog", []) if line.startswith("- ")]
    lines += ["Toolkits (pick 1 to {MAX_TOOLKITS} for the job; each brings its own orders, map details and rules to "
              "the next prompt, so pick only what the job uses):",
              *(f"- {name}: {kit.summary} (orders: {', '.join(kit.actions) or 'work settings only'})"
                for name, kit in TOOLKITS.items()),
              *([f"Buildings you can place now: {', '.join(catalog)}."] if catalog else []), "",
              "You are the colony's manager. Decide the main job that matters most now, plus up to 2 side jobs "
              "other colonists can do at the same time (e.g. a field or tree cutting while a room is built), and "
              "plan them in your own words. Keep every colonist busy: an idle colonist is lost time. Side jobs "
              "must not take the main job's colonists or materials.",
              "Plan only what you are confident is possible: every step must be an order the chosen toolkits "
              "list above, using buildings, materials and colonists this prompt shows. If something needs a "
              "building, research or material you don't have yet, getting that is the job, or leave it for a "
              "later plan. Don't count on items or recipes this prompt doesn't list.",
              "Check forbidden supplies before gathering more: allow_items makes those items usable, including "
              "food and building materials. Use exact item def names in finish checks, e.g. WoodLog, not wood.",
              "Be specific, so the next prompts can follow it: give the size of each room or zone and the count "
              "of each building, chosen for what it has to hold. Size it to what is recommended and to the "
              "materials left after queued construction, and to about 12 focus turns. Specify intended "
              "relationships; leave exact placement coordinates to the focus prompt's checked candidates. "
              "Prefer completing or furnishing suitable existing rooms before adding new ones.",
              "Reply as JSON: {\"goal\": what to achieve, \"uses\": what from this prompt it relies on "
              "(materials with amounts, which colonists do the work, buildings you can place), \"steps\": up to "
              f"{MAX_STEPS} short steps, each an order the toolkits offer with its size or count, \"toolkits\": 1 to "
              f"{MAX_TOOLKITS} names from the list, \"finish\": 1 to 6 checks, covering every job}}.",
              "finish is how the plan ends: the game measures each check every turn, the plan block shows how "
              "far each has got, and the plan ends by itself once all are met. Each check is {\"kind\", "
              "\"what\", \"count\"}: room_closed (what = bedroom, barracks, kitchen, dining, workshop, storage "
              "or hospital; count = rooms; bedroom and barracks count each other), floored (what = a room "
              "purpose; count = rooms of it floored wall to wall), built (what = a building def from the list; count = how many "
              "finished), beds (count = colonists the built beds sleep), zone_cells (what = stockpile or "
              "growing; count = cells in all), research_done (what = project name), have_items (what = item "
              "def, or Food for meals and raw food together; count = how many usable; only for things the plan "
              "makes, e.g. wood it chops or meals it cooks), allowed (what = item def, Food or all; met once "
              "none of it is left forbidden: the check for freeing supplies). Pick checks your steps produce.",
              "Finish counts are desired colony-wide totals, including existing completed objects. If two "
              "shelves exist and you need one more, use built Shelf count 3. Choose a goal with at least one "
              "unmet finish check. These checks do not verify that buildings occupy a particular room.",
              "The next prompts show only your plan and those toolkits' orders; a focus turn can add another "
              "toolkit when a step needs it."]
    return "\n".join(lines)


STEP_MARKS = {"completed": "done", "in_progress": "now"}


def plan_block(plan, finish_state=None) -> list[str]:
    """The plan with each step's status and how far each finish check has got."""
    lines = [f"YOUR PLAN: {plan.objective}", *([f"Relies on: {plan.uses}"] if plan.uses else [])]
    if plan.steps:
        lines.append("Steps ([done] finished, [now] the one you are on):")
        lines += [f"{n}. [{STEP_MARKS.get(s.status, '  ')}] {s.description}" for n, s in enumerate(plan.steps, 1)]
    if plan.finish:
        progress, _ = finish_progress(plan.finish, finish_state)
        lines += ["Finish (the game checks these every turn; the plan ends by itself once all are met):",
                  *progress]
    else:
        lines.append(f"Done when: {plan.done_when}")
    lines.append(f"Focus turn {plan.turns + 1} of at most {MAX_FOCUS_TURNS}.")
    return lines


def rules_for(allowed: set[str]) -> list[str]:
    seen, lines = set(), []
    for action, text in ACTION_RULES.items():
        if action in allowed and text not in seen:
            seen.add(text)
            lines.append("- " + text)
    return lines


def focus_prompt(ctx: dict, plan, toolkits: list[str], drafted: set[str], memory, choices,
                 game_facts: str, last_counts: str = "") -> str:
    """The plan plus only what its toolkits need."""
    sections, blocks = ctx["sections"], ctx["blocks"]
    wanted: list[str] = []
    for name in toolkits:
        for s in TOOLKITS[name].sections:
            if s not in wanted:
                wanted.append(s)
    tags = {t for name in toolkits for t in TOOLKITS[name].tags}
    state: list[str] = []
    for name in wanted:
        body = sections.get(name) or blocks.get(name) or []
        if body:
            state += [*body, ""]
    observed = tagged(ctx, tags)
    lessons = memory.observations_for(tags | set(toolkits))[-6:]
    allowed = allowed_actions(toolkits)
    others = [n for n in TOOLKITS if n not in toolkits]
    lines = [*vital_header(ctx, drafted, last_counts), "", *plan_block(plan, ctx.get("finish_state")), "", *feedback_block(ctx),
             "Colonists (work on / off):", *blocks.get("colonists", []), "",
             *state, *(["Observed:", *observed, ""] if observed else []),
             *(["Lessons for this work:", *(f"- {n}" for n in lessons), ""] if lessons else []),
             game_facts,
             f"Toolkits in this prompt: {', '.join(toolkits)} (up to {MAX_TOOLKITS}). To add one for the next turn "
             "(the oldest is dropped beyond that), request one of: "
             f"{', '.join(others)}.",
             "Orders and their fields:", *rules_for(allowed & set(choices.fields)),
             *(["Unavailable now: " + "; ".join(choices.unavailable)] if choices.unavailable else []),
             REPLY_RULES,
             "Reply as JSON: {\"actions\": [0 to 10 orders], \"time\": \"advance\" or \"hold\", "
             "\"plan\": \"continue\", \"done\" (the game shows it done) or \"replan\" (it no longer makes sense), "
             "\"step\": the number of the plan step these orders work on, \"steps_done\": numbers of steps now "
             "finished (they stay marked), "
             "\"request_toolkits\": [] or [one toolkit to add next turn], \"lesson\": optional {text, toolkit}}.",
             "Steps run side by side: a step whose work is queued (blueprints, zones, cut orders, bills) goes on "
             "by itself while you start the next one. Give every order a reason tied to its step. "
             "Your plan is revisable: current observations and action rules override its steps. Skip an "
             "invalid or unnecessary prerequisite and use a valid action toward the goal; request replan if "
             "the goal or finish checks need changing. chop_trees needs no preliminary designate order. "
             "If forbidden supplies already cover the need, prefer allow_items to gathering more.",
             "Before ordering anything, check each open step: complete, queued, or blocked. Mark complete steps "
             "in steps_done and fix any blocker. Then, if a colonist is idle or about to run out of work, start "
             "the next step that doesn't wait on queued work. Return no actions and advance only when every "
             "colonist has work queued. Waiting for construction is progress; it does not need another blueprint.",
             LESSON_RULE]
    return "\n".join(lines)


def threat_prompt(ctx: dict, drafted: set[str], memory, last_counts: str = "") -> str:
    """Only the fight: enemies, colonists and their weapons, weapons lying around."""
    blocks, choices = ctx["blocks"], ctx["choices"]
    allowed = set(COMBAT.actions) & set(choices.fields)
    unavailable = [u for u in choices.unavailable if u.split(":")[0] in COMBAT.actions]
    lessons = memory.observations_for({"combat", "medical"})[-6:]
    lines = [*vital_header(ctx, drafted, last_counts), "", "THREAT: deal with it; nothing else matters now.",
             *feedback_block(ctx),
             "Enemies:", *(blocks.get("enemies") or ["none located (they may be asleep or out of sight)"]), "",
             "Your colonists:", *blocks.get("fighters", []), "",
             "Weapons lying on the map:", *(blocks.get("weapons") or ["none"]), "",
             *ctx["sections"].get("map", []), "",
             *(["Lessons about fighting:", *(f"- {n}" for n in lessons), ""] if lessons else []),
             "Orders and their fields:", *rules_for(allowed),
             *(["Unavailable now: " + "; ".join(unavailable)] if unavailable else []),
             REPLY_RULES,
             "Reply as JSON: {\"actions\": [0 to 10 orders], \"time\": \"advance\" or \"hold\", "
             "\"lesson\": optional text}. While enemies are active time runs in short windows.",
             LESSON_RULE]
    return "\n".join(lines)
