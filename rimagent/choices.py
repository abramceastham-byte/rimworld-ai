"""One observation's executable vocabulary, shared by decoding and validation."""

from copy import deepcopy
from dataclasses import dataclass, field

from rimagent.actions import Turn
from rimagent.construction import CROPS, FLOORS, available_buildings, stuff_rank
from rimagent.finish import FOOD, is_food
from rimagent.supplies import ALL_STUFF, how_to_get, unlockable_stuff, usable_stuff

# The skill each work type trains (RimWorld's WorkTypeDef.relevantSkills). A
# totally disabled skill means the colonist can't do that work at all.
SKILL_WORK = {
    "Construction": ("Construction",), "Plants": ("Growing", "PlantCutting"),
    "Mining": ("Mining",), "Cooking": ("Cooking",), "Medicine": ("Doctor",),
    "Shooting": ("Hunting",), "Animals": ("Handling",), "Intellectual": ("Research",),
    "Artistic": ("Art",), "Crafting": ("Crafting", "Smithing", "Tailoring"),
    "Social": ("Warden",),
}


def incapable_work(colonist, learned: dict[str, set[str]] | None = None) -> set[str]:
    """Work a colonist can't do. RIMAPI lists only enabled work, so this is
    inferred from disabled skills plus RIMAPI's own refusals seen this session."""
    inferred = {w for s in colonist.skills if s.totally_disabled for w in SKILL_WORK.get(s.name, ())}
    return inferred | (learned or {}).get(colonist.name, set())


@dataclass
class Choices:
    fields: dict[str, dict[str, list]] = field(default_factory=dict)
    work: dict[str, set[str]] = field(default_factory=dict)        # work each colonist can do
    can_enable: dict[str, set[str]] = field(default_factory=dict)  # ...and doesn't have on yet
    can_disable: dict[str, set[str]] = field(default_factory=dict)  # work they have on
    researching: str | None = None
    recipes: dict[int, set[str]] = field(default_factory=dict)
    materials: dict[str, list[str | None]] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @classmethod
    def observe(cls, colonists, work_types, tables, research, defs, terrain,
                items, trees, target_kinds=(), learned_incapable=None, enemies=(), power_ready=False):
        c = cls(width=terrain.width, height=terrain.height)
        c.work = {p.name: set(work_types) - incapable_work(p, learned_incapable) for p in colonists}
        # Standing orders already in effect aren't offered again: a run asked
        # for the same colonists' Construction five times and the current
        # research project four times.
        enabled = {p.name: {w.work_type for w in p.work_priorities if w.priority > 0} for p in colonists}
        c.can_enable = {name: work - enabled[name] for name, work in c.work.items()}
        c.can_disable = {name: enabled[name] & set(work_types) for name in c.work}
        for action, options in (("enable_work", c.can_enable), ("disable_work", c.can_disable)):
            if any(options.values()):
                c.fields[action] = {"colonist": [n for n, o in options.items() if o],
                                    "work_type": sorted(set().union(*options.values()))}
        c.recipes = {t.id: {r.def_name for r in t.recipes} for t in tables if t.recipes}
        if c.recipes:
            c.fields["set_bill"] = {"building_id": list(c.recipes),
                                    "recipe_def_name": sorted(set.union(*c.recipes.values()))}
        else:
            c.unavailable.append("set_bill: no built table with available recipes")
        c.researching = research.current.name if research.current else None
        projects = [p.name for p in research.available if p.name != c.researching]
        if projects:
            c.fields["set_research"] = {"project": projects}
        elif c.researching:
            c.unavailable.append(f"set_research: already researching {c.researching}, nothing else to switch to")
        else:
            c.unavailable.append("set_research: no currently available project/appropriate bench")
        finished = {p.name for p in research.all_projects if p.is_finished}
        # Only materials the colony can use now: offering every stone block
        # let models plan whole rooms in blocks nobody had.
        # Forbidden materials count: allow_items makes them usable, and it may
        # come in the same reply, since a blueprint only waits for materials.
        usable = {**usable_stuff(items), **unlockable_stuff(items)}
        c.materials = {name: ([m for m in spec.materials() if m in defs.thing_names and m in usable]
                             if spec.materials() else [None])
                       for name, spec in available_buildings(finished).items()
                       if name in defs.thing_names}
        c.materials = {k: v for k, v in c.materials.items() if v}
        missing = [m for m in ALL_STUFF if m in defs.thing_names and m not in usable]
        if missing:
            stone = [m for m in missing if m.startswith("Blocks")]
            other = [m for m in missing if m not in stone]
            c.unavailable += [f"stuff {m}: none usable ({how_to_get(m)})" for m in other]
            if stone:
                c.unavailable.append("stuff stone blocks: none (build in wood)")
        if c.materials:
            mats = sorted({m for ms in c.materials.values() for m in ms if m is not None}, key=stuff_rank)
            c.fields["place_blueprint"] = {"building_def": list(c.materials), "stuff": mats + [None]}
        if "Wall" in c.materials:
            c.fields["build_wall"] = {"stuff": c.materials["Wall"]}
            if "Door" in c.materials:
                c.fields["build_room"] = {"stuff": [m for m in c.materials["Wall"]
                                                       if m in c.materials["Door"]]}
        floors = [n for n, f in FLOORS.items() if n in defs.terrain
                  and (not f.research or f.research in finished) and usable.get(f.cost[0])]
        if floors:
            c.fields["build_floor"] = {"terrain": floors}
        else:
            c.unavailable.append("build_floor: no unlocked floor with usable material")
        crops = [n for n in CROPS if n in defs.crop_fertility_min]
        if crops:
            c.fields["create_growing_zone"] = {"plant": crops}
        c.fields["create_stockpile"] = {}
        c.fields["plan_area"] = {}
        forbidden = sorted({i.def_name for i in items if i.is_forbidden})
        if forbidden:
            food = [FOOD] if any(is_food(name) for name in forbidden) else []
            c.fields["allow_items"] = {"what": ["all", *food, *forbidden]}
        else:
            c.unavailable.append("allow_items: no forbidden items")
        if trees:
            c.fields["chop_trees"] = {}
        else:
            c.unavailable.append("chop_trees: no observed wood-producing trees")
        if target_kinds:
            c.fields["designate"] = {"designation": list(target_kinds)}
        for kind in ("mine", "harvest", "hunt"):
            if kind not in target_kinds:
                c.unavailable.append(f"designate {kind}: no verified targets")
        c.add_pawn_orders(colonists, items, enemies, power_ready)
        return c

    def add_pawn_orders(self, colonists, items, enemies, power_ready: bool) -> None:
        """Draft, move, attack, equip, medical orders and connect_power."""
        names = [p.name for p in colonists]
        if names:
            self.fields["draft"] = {"colonist": names}
            self.fields["move"] = {"colonist": names}
        if names and enemies:
            self.fields["attack"] = {"colonist": names, "target": [e.id for e in enemies]}
        elif names:
            self.unavailable.append("attack: no enemy on the map")
        weapons = [i.thing_id for i in items if i.is_weapon]
        if names and weapons:
            self.fields["equip"] = {"colonist": names, "item_id": weapons}
        elif names:
            self.unavailable.append("equip: no weapons lying on the map")
        injured = [p.name for p in colonists if p.health < 0.99]
        doctors = [n for n in names if "Doctor" in self.work.get(n, set())]
        if injured:
            self.fields["tend"] = {"patient": injured, "doctor": doctors + [None]}
            self.fields["bed_rest"] = {"patient": injured}
        else:
            self.unavailable += ["tend: nobody is injured", "bed_rest: nobody is injured"]
        if len(names) > 1 and any(p.health < 0.6 for p in colonists):
            self.fields["rescue"] = {"colonist": names, "patient": [p.name for p in colonists if p.health < 0.6]}
        else:
            self.unavailable.append("rescue: nobody is badly hurt")
        if power_ready:
            self.fields["connect_power"] = {}
        else:
            self.unavailable.append("connect_power: needs Electricity research and a generator or battery")

    def withhold(self, buildings: set[str], why: str) -> None:
        """Stop offering these buildings to place_blueprint, saying why.

        Taken off the menu rather than refused: a refused order comes back turn
        after turn, an order that isn't offered can't be sent.
        """
        offered = [b for b in buildings if b in self.materials]
        if not offered:
            return
        for name in offered:
            del self.materials[name]
        if "place_blueprint" in self.fields:
            defs = [d for d in self.fields["place_blueprint"]["building_def"] if d not in buildings]
            if defs:
                self.fields["place_blueprint"]["building_def"] = defs
            else:
                del self.fields["place_blueprint"]
        self.unavailable.append(f"place_blueprint {' / '.join(sorted(offered))}: {why}")

    def schema(self, allowed: set[str] | None = None, turn_model=Turn):
        """The JSON schema for this turn's reply: only offered orders (and, when
        given, only those in allowed), with every name narrowed to what exists."""
        schema = deepcopy(turn_model.model_json_schema())
        branches = []
        for spec in schema["$defs"].values():
            props = spec.get("properties", {})
            action = props.get("action", {}).get("const")
            if action not in self.fields or (allowed is not None and action not in allowed):
                continue
            branch = deepcopy(spec)
            for name, values in self.fields[action].items():
                branch["properties"][name] = {"enum": values}
            for name in ("x", "x1", "x2", "z", "z1", "z2"):
                if name in branch["properties"]:
                    branch["properties"][name]["maximum"] = (self.width if name[0] == "x" else self.height) - 1
            # Encode dependencies in decoding, rather than rejecting a legal
            # cross-product of unrelated enums after the model has chosen it.
            pairing = {
                "place_blueprint": ("building_def", "stuff", self.materials),
                "set_bill": ("building_id", "recipe_def_name", self.recipes),
                "enable_work": ("colonist", "work_type", self.can_enable),
                "disable_work": ("colonist", "work_type", self.can_disable),
            }.get(action)
            if pairing:
                selector, dependent, mapping = pairing
                # Share branches when selectors have exactly the same choices.
                groups = {}
                for key, values in mapping.items():
                    if values:
                        order = stuff_rank if dependent == "stuff" else str
                        groups.setdefault(tuple(sorted(values, key=order)), []).append(key)
                for values, keys in groups.items():
                    variant = deepcopy(branch)
                    p = variant["properties"]
                    p[selector] = {"enum": keys}
                    p[dependent] = {"enum": list(values)}
                    variant["properties"] = {k: p[k] for k in
                        ["action", selector, dependent] +
                        [k for k in p if k not in ("action", selector, dependent)]}
                    # Optional defaults must not bypass the narrowed choices.
                    variant["required"] = list(dict.fromkeys(
                        variant.get("required", []) + [selector, dependent]))
                    branches.append(variant)
            else:
                branches.append(branch)
        if branches:
            schema["properties"]["actions"]["items"] = {"anyOf": branches}
        else:  # nothing to order this turn: only time can be chosen
            schema["properties"]["actions"]["items"] = {}
            schema["properties"]["actions"]["maxItems"] = 0
        # Unused action definitions are bulky and can imply unavailable choices.
        schema["$defs"] = {k: v for k, v in schema["$defs"].items()
                           if "action" not in v.get("properties", {})}
        return schema

    def validate(self, turn: Turn, allowed: set[str] | None = None):
        for a in turn.actions:
            if allowed is not None and a.action not in allowed:
                raise ValueError(f"{a.action} isn't one of this turn's orders; request its toolkit first")
            if self.already_in_effect(a):
                # Not offered by the schema; without one, let it through to be
                # reported "unchanged" rather than throw away the whole reply.
                continue
            if a.action not in self.fields:
                raise ValueError(f"{a.action} is unavailable: {'; '.join(self.unavailable)}")
            for name, values in self.fields[a.action].items():
                if getattr(a, name) not in values:
                    raise ValueError(f"{a.action}.{name}: choose one of {values}")
            if a.action in ("enable_work", "disable_work") and a.work_type not in self.work[a.colonist]:
                raise ValueError(f"{a.colonist} is incapable of {a.work_type}")
            if a.action == "set_bill" and a.recipe_def_name not in self.recipes[a.building_id]:
                raise ValueError(f"recipe {a.recipe_def_name} is not offered by table {a.building_id}")
            if a.action == "place_blueprint" and a.stuff not in self.materials[a.building_def]:
                raise ValueError(f"{a.building_def} requires stuff from {self.materials[a.building_def]}")
            if a.action == "designate" and a.designation == "hunt" and a.rect().cells != 1:
                raise ValueError("hunt requires one animal cell: x1=x2 and z1=z2")
            for name in ("x", "x1", "x2", "z", "z1", "z2"):
                value = getattr(a, name, None)
                if value is not None and value >= (self.width if name[0] == "x" else self.height):
                    raise ValueError(f"{name}={value} is outside the map")


    def already_in_effect(self, a) -> str | None:
        """Why an order changes nothing, when it repeats one already in effect."""
        if a.action == "set_research" and a.project == self.researching:
            return f"{a.project} is already being researched"
        if a.action == "enable_work" and a.colonist in self.work and a.work_type in self.work[a.colonist]                 and a.work_type not in self.can_enable.get(a.colonist, set()):
            return f"{a.colonist} already has {a.work_type} on"
        if a.action == "disable_work" and a.colonist in self.work                 and a.work_type not in self.can_disable.get(a.colonist, set()):
            return f"{a.colonist} already has {a.work_type} off"
        return None


def action_target(action):
    return action.model_dump(exclude={"reason"}, exclude_none=True)


class FailureMemory:
    """Bounded, exact-target feedback. A changed relevant snapshot releases it."""

    def __init__(self):
        self.entries = []

    def remember(self, action, error, fingerprint):
        target = action_target(action)
        self.entries = [e for e in self.entries if e[0] != target]
        self.entries.append((target, error, fingerprint))
        self.entries = self.entries[-6:]

    def reconcile(self, fingerprint_for):
        self.entries = [e for e in self.entries if fingerprint_for(e[0]) == e[2]]

    def lines(self):
        return [f"Unresolved: {target}: {error}" for target, error, _ in self.entries]
