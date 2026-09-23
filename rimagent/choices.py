"""One observation's executable vocabulary, shared by decoding and validation."""

from copy import deepcopy
from dataclasses import dataclass, field

from rimagent.actions import Turn
from rimagent.construction import CROPS, FLOORS, available_buildings

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
    work: dict[str, set[str]] = field(default_factory=dict)
    recipes: dict[int, set[str]] = field(default_factory=dict)
    materials: dict[str, list[str | None]] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @classmethod
    def observe(cls, colonists, work_types, tables, research, defs, terrain,
                items, trees, target_kinds=(), learned_incapable=None):
        c = cls(width=terrain.width, height=terrain.height)
        c.work = {p.name: set(work_types) - incapable_work(p, learned_incapable) for p in colonists}
        if c.work:
            for action in ("enable_work", "disable_work"):
                c.fields[action] = {"colonist": list(c.work), "work_type": list(work_types)}
        c.recipes = {t.id: {r.def_name for r in t.recipes} for t in tables if t.recipes}
        if c.recipes:
            c.fields["set_bill"] = {"building_id": list(c.recipes),
                                    "recipe_def_name": sorted(set.union(*c.recipes.values()))}
        else:
            c.unavailable.append("set_bill: no built table with available recipes")
        projects = [p.name for p in research.available]
        if projects:
            c.fields["set_research"] = {"project": projects}
        else:
            c.unavailable.append("set_research: no currently available project/appropriate bench")
        finished = {p.name for p in research.all_projects if p.is_finished}
        c.materials = {name: ([m for m in spec.materials() if m in defs.thing_names]
                             if spec.materials() else [None])
                       for name, spec in available_buildings(finished).items()
                       if name in defs.thing_names}
        c.materials = {k: v for k, v in c.materials.items() if v}
        if c.materials:
            mats = sorted({m for ms in c.materials.values() for m in ms if m is not None})
            c.fields["place_blueprint"] = {"building_def": list(c.materials), "stuff": mats + [None]}
        if "Wall" in c.materials:
            c.fields["build_wall"] = {"stuff": c.materials["Wall"]}
            if "Door" in c.materials:
                c.fields["build_room"] = {"stuff": [m for m in c.materials["Wall"]
                                                       if m in c.materials["Door"]]}
        floors = [n for n, f in FLOORS.items() if n in defs.terrain
                  and (not f.research or f.research in finished)]
        if floors:
            c.fields["build_floor"] = {"terrain": floors}
        else:
            c.unavailable.append("build_floor: no unlocked floor definitions")
        crops = [n for n in CROPS if n in defs.crop_fertility_min]
        if crops:
            c.fields["create_growing_zone"] = {"plant": crops}
        c.fields["create_stockpile"] = {}
        if any(i.is_forbidden for i in items):
            c.fields["allow_items"] = {}
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
        return c

    def schema(self):
        schema = deepcopy(Turn.model_json_schema())
        branches = []
        for spec in schema["$defs"].values():
            props = spec.get("properties", {})
            action = props.get("action", {}).get("const")
            if action not in self.fields:
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
                "enable_work": ("colonist", "work_type", self.work),
                "disable_work": ("colonist", "work_type", self.work),
            }.get(action)
            if pairing:
                selector, dependent, mapping = pairing
                # Share branches when selectors have exactly the same choices.
                groups = {}
                for key, values in mapping.items():
                    if values:
                        groups.setdefault(tuple(sorted(values, key=str)), []).append(key)
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

    def validate(self, turn: Turn):
        for a in turn.actions:
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
