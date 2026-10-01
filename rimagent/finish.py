"""Plan finish conditions the game can check, instead of the model saying "done".

A free-text done_when ("stone blocks present in stockpile", "3 colonists 100%
fed") left it to the model to decide when a plan was over, and plans either
never ended or ended on a guess. A plan now names 1 to 4 checks from a short
list; each turn the agent measures them against the game, shows the progress
in the plan block, and ends the plan by itself once every check is met.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rimagent.construction import BUILDINGS

FinishKind = Literal["room_closed", "floored", "built", "beds", "zone_cells", "research_done", "have_items",
                     "allowed"]
ROOM_PURPOSES = ("bedroom", "barracks", "kitchen", "dining", "workshop", "storage", "hospital", "other")
ZONE_KINDS = ("stockpile", "growing")
SLEEPING = ("bedroom", "barracks")

# Which toolkits' orders can make each kind of check come true.
PLACING = ("construction", "storage", "production", "research", "power", "medical", "defense")
KIND_TOOLKITS: dict[str, tuple[str, ...]] = {
    "room_closed": ("construction", "storage"),
    "floored": ("construction",),
    "built": PLACING,
    "beds": PLACING,
    "research_done": ("research",),
}
ZONE_TOOLKITS = {"stockpile": ("storage",), "growing": ("farming",)}

# have_items what = "Food": every allowed meal and raw food item, counted together.
FOOD = "Food"
ALL = "all"  # allowed what = "all": nothing at all left forbidden
RAW_FOODS = ("RawPotatoes", "RawRice", "RawCorn", "RawBerries", "RawAgave", "RawFungus", "Pemmican", "Milk",
             "EggChickenUnfertilized", "Kibble")
# Plain words models write for items, and the def each means. One plan's
# finish check "10 wood usable" read 0 of 10 for twelve turns while 518
# WoodLog lay chopped, since nothing is called "wood".
ITEM_ALIASES = {"wood": "WoodLog", "woodlog": "WoodLog", "logs": "WoodLog", "steel": "Steel",
                "food": FOOD, "meals": FOOD, "meal": FOOD, "components": "ComponentIndustrial",
                "component": "ComponentIndustrial"}


def is_food(def_name: str) -> bool:
    return def_name.startswith(("Meal", "Meat_")) or def_name in RAW_FOODS


def canonical_item(what: str, thing_names) -> str | None:
    """The def a have_items check means, or None if it names nothing measurable:
    an exact def, a def with other capitals, a plain word from ITEM_ALIASES, or Food."""
    if what in thing_names or what == FOOD:
        return what
    folded = what.strip().casefold()
    exact = next((n for n in thing_names if n.casefold() == folded), None)
    if exact:
        return exact
    alias = ITEM_ALIASES.get(folded.replace(" ", "").replace("_", ""))
    return alias if alias == FOOD or alias in thing_names else None


def item_toolkits(def_name: str) -> tuple[str, ...]:
    """Toolkits whose orders make more of an item; empty when several can
    (wood and steel come from chopping and mining, but also from allow_items)."""
    if def_name == FOOD:
        return ()  # allowing, cooking, hunting and farming all make food
    if def_name.startswith(("Blocks", "Meal")) or def_name in ("Kibble", "Pemmican"):
        return ("production",)
    if def_name.startswith(("Meat_", "Leather_")):
        return ("hunting",)
    if def_name.startswith("Raw") or def_name in ("Hay",):
        return ("farming",)
    return ()


def makes_more(def_name: str, toolkit: str) -> bool:
    """Whether a toolkit's orders produce more of an item (not just free what exists)."""
    if def_name == FOOD:
        return toolkit in ("production", "hunting", "farming")
    if def_name in ("WoodLog", "Steel") or def_name.startswith(("Chunk", "Mineable")):
        return toolkit == "wood_mining"
    return toolkit in item_toolkits(def_name)


def unreachable_count(check: "FinishCheck", toolkits: list[str], on_map: int) -> str | None:
    """Why a have_items check can't be met and stay met with this toolkit, or None.

    Two plans asked for "57 MealSurvivalPack usable" to end an allow-the-food
    plan: the colonists ate the meals as soon as they were allowed, the count
    peaked at 47, and each plan ran out its twelve turns.
    """
    if check.kind != "have_items" or any(makes_more(check.what, t) for t in toolkits):
        return None
    toolkit = " and ".join(toolkits)
    name = "food" if check.what == FOOD else check.what
    if check.what == FOOD or is_food(check.what):
        return (f"have_items {name} can't end a {toolkit} plan: colonists eat food, so the count falls as soon "
                f"as it is reached. To free forbidden food use allowed what={check.what}.")
    if check.count > on_map:
        return (f"only {on_map} {name} exist on the map (usable or forbidden) and the {toolkit} toolkit doesn't make "
                f"more. To free forbidden ones use allowed what={check.what}; to get more, plan to make them.")
    return None


class FinishCheck(BaseModel):
    """One thing the game will show once the plan is done."""

    model_config = ConfigDict(extra="forbid")

    kind: FinishKind
    what: str = Field(default="", max_length=60)
    # No low cap: with le=1000, schema-constrained decoding let a model write
    # "117" for the colony's 1170 Steel and stopped it adding the 0.
    count: int = Field(default=1, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def _what_fits_kind(self) -> "FinishCheck":
        if self.kind in ("room_closed", "floored") and self.what not in ROOM_PURPOSES:
            raise ValueError(f"{self.kind} needs what = a room purpose ({', '.join(ROOM_PURPOSES)})")
        if self.kind == "built" and self.what not in BUILDINGS:
            raise ValueError("built needs what = a building def name from the buildable list")
        if self.kind == "zone_cells" and self.what not in ZONE_KINDS:
            raise ValueError("zone_cells needs what = stockpile or growing")
        if self.kind in ("research_done", "have_items", "allowed") and not self.what:
            raise ValueError(f"{self.kind} needs what: the research project or item def name")
        if self.kind == "have_items" and self.what in BUILDINGS:
            raise ValueError(f"{self.what} is a building, not an item: use built what={self.what}")
        return self

    def label(self) -> str:
        return {
            "room_closed": f"{self.count} {self.what} room{'s' if self.count > 1 else ''} closed",
            "floored": f"{self.count} {self.what} room{'s' if self.count > 1 else ''} fully floored",
            "built": f"{self.count} {self.what} built",
            "beds": f"beds built for {self.count} colonists",
            "zone_cells": f"{self.count} cells of {self.what} zone",
            "research_done": f"research {self.what} finished",
            "have_items": (f"{self.count} food (meals and raw food) usable" if self.what == FOOD
                           else f"{self.count} {self.what} usable"),
            "allowed": f"no {'item' if self.what == ALL else self.what} left forbidden",
        }[self.kind]

    def toolkits(self) -> tuple[str, ...]:
        """Toolkits whose orders can meet it; empty means any."""
        if self.kind == "zone_cells":
            return ZONE_TOOLKITS[self.what]
        if self.kind == "have_items":
            return item_toolkits(self.what)
        return KIND_TOOLKITS.get(self.kind, ())  # allowed: allow_items is in every toolkit


@dataclass
class FinishState:
    """What the game shows now, for measuring finish checks. Built in prepare_turn."""

    rooms_closed: dict[str, int] = field(default_factory=dict)  # purpose -> closed rooms
    rooms_floored: dict[str, int] = field(default_factory=dict)  # purpose -> rooms floored wall to wall
    built: dict[str, int] = field(default_factory=dict)         # def name -> finished buildings
    sleepers: int = 0                                           # colonists the built beds sleep
    zone_cells: dict[str, int] = field(default_factory=dict)    # "stockpile"/"growing" -> cells
    research: set[str] = field(default_factory=set)             # finished projects
    items: dict[str, int] = field(default_factory=dict)         # def name -> allowed count
    forbidden: dict[str, int] = field(default_factory=dict)     # def name -> forbidden count

    def measure(self, check: FinishCheck) -> int:
        if check.kind in ("room_closed", "floored"):
            # A bedroom check counts barracks too, and the other way round: one plan
            # asked for "1 bedroom room closed", built it as a barracks, and read
            # 0 of 1 for twelve turns with the room finished.
            rooms = self.rooms_closed if check.kind == "room_closed" else self.rooms_floored
            kinds = SLEEPING if check.what in SLEEPING else (check.what,)
            return sum(rooms.get(kind, 0) for kind in kinds)
        if check.kind == "built":
            return self.built.get(check.what, 0)
        if check.kind == "beds":
            return self.sleepers
        if check.kind == "zone_cells":
            return self.zone_cells.get(check.what, 0)
        if check.kind == "research_done":
            return 1 if check.what in self.research else 0
        if check.kind == "allowed":
            left = sum(n for name, n in self.forbidden.items()
                       if check.what == ALL or name == check.what or (check.what == FOOD and is_food(name)))
            return 0 if left else 1
        if check.what == FOOD:
            return sum(n for name, n in self.items.items() if is_food(name))
        return self.items.get(check.what, 0)


def finish_progress(checks: list[FinishCheck], state: FinishState | None) -> tuple[list[str], bool]:
    """(one line per check with how far it is, whether every check is met)."""
    if state is None or not checks:
        return [f"- {c.label()}" for c in checks], False
    lines, done = [], True
    for check in checks:
        need = 1 if check.kind in ("research_done", "allowed") else check.count
        have = state.measure(check)
        met = have >= need
        done &= met
        progress = "not yet" if check.kind == "allowed" else f"{min(have, need)} of {need}"
        lines.append(f"- {check.label()}: {'MET' if met else progress}")
    return lines, done


def finish_mismatch(checks: list[FinishCheck], toolkits: list[str]) -> str | None:
    """Why none of the chosen toolkits can meet a check, or None."""
    picked = " and ".join(toolkits)
    for check in checks:
        kits = check.toolkits()
        if kits and not set(kits) & set(toolkits):
            return (f"the finish check \"{check.label()}\" needs the {' or '.join(kits)} toolkit, and you picked "
                    f"{picked}. Add {kits[0]}, or choose checks your toolkits' own orders can meet.")
    return None
