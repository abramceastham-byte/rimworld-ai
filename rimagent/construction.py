"""Map geometry and the building/crop catalog for zones, blueprints and designations.

RIMAPI's zone and blueprint endpoints do very little checking of their own
(unknown names are skipped silently, and blueprints skip RimWorld's normal
placement checks), so the client validates requests against the facts here
before sending anything.

Building sizes, materials and research come from RimWorld 1.6's own def files.
"""

from dataclasses import dataclass

# Largest rectangle the agent may turn into one zone, and may designate at once.
MAX_ZONE_CELLS = 15 * 15
MAX_DESIGNATE_CELLS = 20 * 20
MAX_CHOP_TREES = 30       # trees marked by one chop_trees action
MAX_WALL_RUN = 20         # cells in one straight wall run
MAX_FLOOR_CELLS = 15 * 15  # cells floored by one action


def is_tree(def_name: str) -> bool:
    """Trees are the plants that give wood when cut, e.g. Plant_TreeOak."""
    return def_name.startswith("Plant_Tree")

# Designation types RIMAPI offers that are safe to hand to a model. RIMAPI also
# has "deconstruct" (tears down any building, including the colony's own) and
# "remove-all"; neither is exposed.
DESIGNATIONS = ("mine", "harvest", "hunt")

# Concrete materials for each RimWorld stuff category.
STUFF_BY_CATEGORY = {
    "Woody": ["WoodLog"],
    "Metallic": ["Steel"],
    "Stony": ["BlocksGranite", "BlocksSandstone", "BlocksLimestone", "BlocksSlate", "BlocksMarble"],
}


@dataclass(frozen=True)
class BuildSpec:
    label: str
    size: tuple[int, int] = (1, 1)            # (width, depth) when facing north
    stuff: tuple[str, ...] = ()               # stuff categories; empty = fixed cost
    stuff_count: int = 0                      # units of the chosen material
    cost: tuple[tuple[str, int], ...] = ()    # fixed extra cost, e.g. (("Steel", 25),)
    research: str | None = None               # research project that unlocks it

    def materials(self) -> list[str]:
        return [m for category in self.stuff for m in STUFF_BY_CATEGORY.get(category, [])]

    def cost_text(self) -> str:
        """What it takes to build, e.g. "75 of WoodLog/Steel/stone blocks + 25 Steel"."""
        parts = []
        if self.stuff_count:
            kinds = "/".join(
                "stone blocks" if c == "Stony" else STUFF_BY_CATEGORY[c][0] for c in self.stuff
            )
            parts.append(f"{self.stuff_count} of {kinds}")
        parts += [f"{count} {name}" for name, count in self.cost]
        return " + ".join(parts) or "nothing"


# An early-game building set the agent may place as blueprints.
BUILDINGS = {
    "Wall": BuildSpec("wall", stuff=("Metallic", "Woody", "Stony"), stuff_count=5),
    "Door": BuildSpec("door", stuff=("Metallic", "Woody", "Stony"), stuff_count=25),
    # No zero-work "spots" (SleepingSpot, ButcherSpot, CraftingSpot): the game
    # places those instantly, but RIMAPI's blueprint call makes a construction
    # frame for them, which sticks and spams "construction botched" forever.
    "Bed": BuildSpec("bed", size=(1, 2), stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture", stuff_count=45),
    "DoubleBed": BuildSpec("double bed", size=(2, 2), stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture", stuff_count=85),
    "Campfire": BuildSpec("campfire", cost=(("WoodLog", 20),)),
    "TorchLamp": BuildSpec("torch lamp", cost=(("WoodLog", 20),)),
    "TableButcher": BuildSpec("butcher table", size=(3, 1), stuff=("Metallic", "Woody"), stuff_count=75, cost=(("WoodLog", 20),)),
    "FueledStove": BuildSpec("fueled stove", size=(3, 1), cost=(("Steel", 80),)),
    "ElectricStove": BuildSpec("electric stove", size=(3, 1), research="Electricity", cost=(("Steel", 80), ("ComponentIndustrial", 2))),
    "SimpleResearchBench": BuildSpec("simple research bench", size=(3, 2), stuff=("Metallic", "Woody", "Stony"), stuff_count=75, cost=(("Steel", 25),)),
    "HandTailoringBench": BuildSpec("hand tailoring bench", size=(3, 1), stuff=("Metallic", "Woody"), research="ComplexClothing", stuff_count=75),
    "TableStonecutter": BuildSpec("stonecutter's table", size=(3, 1), stuff=("Metallic", "Woody"), research="Stonecutting", stuff_count=75, cost=(("Steel", 30),)),
    "Shelf": BuildSpec("shelf", size=(2, 1), stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture", stuff_count=20),
    "Table2x2c": BuildSpec("table (2x2)", size=(2, 2), stuff=("Metallic", "Woody", "Stony"), stuff_count=50),
    "Stool": BuildSpec("stool", stuff=("Metallic", "Woody", "Stony"), stuff_count=25),
    "DiningChair": BuildSpec("dining chair", stuff=("Metallic", "Woody"), research="ComplexFurniture", stuff_count=45),
    "PassiveCooler": BuildSpec("passive cooler", research="PassiveCooler", cost=(("WoodLog", 50),)),
    "WoodFiredGenerator": BuildSpec("wood-fired generator", size=(2, 2), research="Electricity", cost=(("Steel", 100), ("ComponentIndustrial", 2))),
    "StandingLamp": BuildSpec("standing lamp", research="Electricity", cost=(("Steel", 20),)),
    "Barricade": BuildSpec("barricade", stuff=("Metallic", "Woody", "Stony"), stuff_count=5),
    "HorseshoesPin": BuildSpec("horseshoes pin", stuff=("Metallic", "Woody", "Stony"), stuff_count=10),
    "ChessTable": BuildSpec("chess table", stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture", stuff_count=70),
}

@dataclass(frozen=True)
class FloorSpec:
    label: str
    cost: tuple[str, int]        # material and how much per cell
    research: str | None = None


# Floors the agent may lay. Costs are per cell, from RimWorld 1.6's def files.
FLOORS = {
    "WoodPlankFloor": FloorSpec("wood floor", ("WoodLog", 3)),
    "TileSandstone": FloorSpec("sandstone tile", ("BlocksSandstone", 4), research="Stonecutting"),
    "TileGranite": FloorSpec("granite tile", ("BlocksGranite", 4), research="Stonecutting"),
    "TileLimestone": FloorSpec("limestone tile", ("BlocksLimestone", 4), research="Stonecutting"),
    "TileSlate": FloorSpec("slate tile", ("BlocksSlate", 4), research="Stonecutting"),
    "TileMarble": FloorSpec("marble tile", ("BlocksMarble", 4), research="Stonecutting"),
}


# Food and fibre crops that need no research. Drug and medicine crops are left
# out on purpose.
CROPS = {
    "Plant_Potato": "potatoes",
    "Plant_Rice": "rice",
    "Plant_Corn": "corn",
    "Plant_Strawberry": "strawberries",
    "Plant_Haygrass": "haygrass (animal feed)",
    "Plant_Cotton": "cotton (cloth)",
}


def available_buildings(finished_research: set[str]) -> dict[str, BuildSpec]:
    """The catalog entries this colony has the research to build."""
    return {name: spec for name, spec in BUILDINGS.items()
            if spec.research is None or spec.research in finished_research}


@dataclass(frozen=True)
class Rect:
    """An inclusive rectangle of map cells, normalised so x1 <= x2 and z1 <= z2."""

    x1: int
    z1: int
    x2: int
    z2: int

    @classmethod
    def from_corners(cls, xa: int, za: int, xb: int, zb: int) -> "Rect":
        return cls(min(xa, xb), min(za, zb), max(xa, xb), max(za, zb))

    @property
    def cells(self) -> int:
        return (self.x2 - self.x1 + 1) * (self.z2 - self.z1 + 1)

    def contains(self, x: int, z: int) -> bool:
        return self.x1 <= x <= self.x2 and self.z1 <= z <= self.z2

    def within(self, width: int, height: int) -> bool:
        return 0 <= self.x1 and 0 <= self.z1 and self.x2 < width and self.z2 < height

    def overlaps(self, other: "Rect") -> bool:
        return not (self.x2 < other.x1 or other.x2 < self.x1 or self.z2 < other.z1 or other.z2 < self.z1)

    def __iter__(self):
        for z in range(self.z1, self.z2 + 1):
            for x in range(self.x1, self.x2 + 1):
                yield x, z

    def __str__(self) -> str:
        return f"({self.x1},{self.z1})-({self.x2},{self.z2})"


def footprint(x: int, z: int, size: tuple[int, int], rotation: int) -> Rect:
    """Cells a building covers when placed at (x, z), following RimWorld's
    GenAdj.OccupiedRect. Rotation: 0 north, 1 east, 2 south, 3 west.
    """
    w, d = size
    if (w, d) == (1, 1):
        return Rect(x, z, x, z)
    if rotation in (1, 3):
        w, d = d, w
    if rotation == 1 and d % 2 == 0:
        z -= 1
    elif rotation == 2:
        if w % 2 == 0:
            x -= 1
        if d % 2 == 0:
            z -= 1
    elif rotation == 3 and w % 2 == 0:
        x -= 1
    x1, z1 = x - (w - 1) // 2, z - (d - 1) // 2
    return Rect(x1, z1, x1 + w - 1, z1 + d - 1)


def room_layout(rect: Rect, door_side: str) -> list[tuple[str, int, int]]:
    """Outer walls with exactly one centred door, no duplicate corner cells."""
    if not (4 <= rect.x2 - rect.x1 + 1 <= 15 and 4 <= rect.z2 - rect.z1 + 1 <= 15):
        raise ValueError("room outer dimensions must each be 4..15 cells")
    mx, mz = (rect.x1 + rect.x2) // 2, (rect.z1 + rect.z2) // 2
    doors = {"north": (mx, rect.z2), "south": (mx, rect.z1),
             "east": (rect.x2, mz), "west": (rect.x1, mz)}
    if door_side not in doors:
        raise ValueError("door_side must be north/east/south/west")
    return [("Door" if (x, z) == doors[door_side] else "Wall", x, z)
            for x, z in rect if x in (rect.x1, rect.x2) or z in (rect.z1, rect.z2)]


class TerrainMap:
    """RIMAPI's /map/terrain grid, decoded.

    RIMAPI sends one palette of terrain names plus a run-length encoded grid of
    [count, palette_index, count, palette_index, ...], row by row from z=0,
    with x increasing within each row.
    """

    def __init__(self, width: int, height: int, palette: list[str], grid: list[int]):
        self.width, self.height = width, height
        cells: list[int] = []
        for i in range(0, len(grid) - 1, 2):
            cells.extend([grid[i + 1]] * grid[i])
        if len(cells) != width * height:
            raise ValueError(f"terrain grid has {len(cells)} cells, expected {width * height}")
        self._names = [palette[i] for i in cells]

    def at(self, x: int, z: int) -> str:
        return self._names[z * self.width + x]


@dataclass(frozen=True)
class TerrainInfo:
    fertility: float
    affordances: frozenset[str]  # e.g. {"Light", "Medium", "Heavy", "Walkable"}


def map_symbol(info: TerrainInfo | None) -> str:
    """One character per cell for the prompt's mini-map."""
    if info is None:
        return "?"
    if "Heavy" not in info.affordances:
        return "~"   # water, marsh, mud: poor or no building
    if info.fertility >= 1.1:
        return "R"   # rich soil
    if info.fertility >= 0.7:
        return "."   # soil any crop can grow in
    if info.fertility > 0:
        return ","   # poor soil
    return "_"       # buildable, nothing grows (stone, floors)
