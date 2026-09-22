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
    research: str | None = None               # research project that unlocks it

    def materials(self) -> list[str]:
        return [m for category in self.stuff for m in STUFF_BY_CATEGORY.get(category, [])]


# An early-game building set the agent may place as blueprints.
BUILDINGS = {
    "Wall": BuildSpec("wall", stuff=("Metallic", "Woody", "Stony")),
    "Door": BuildSpec("door", stuff=("Metallic", "Woody", "Stony")),
    "SleepingSpot": BuildSpec("sleeping spot", size=(1, 2)),
    "Bed": BuildSpec("bed", size=(1, 2), stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture"),
    "DoubleBed": BuildSpec("double bed", size=(2, 2), stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture"),
    "Campfire": BuildSpec("campfire"),
    "TorchLamp": BuildSpec("torch lamp"),
    "ButcherSpot": BuildSpec("butcher spot"),
    "CraftingSpot": BuildSpec("crafting spot"),
    "TableButcher": BuildSpec("butcher table", size=(3, 1), stuff=("Metallic", "Woody")),
    "FueledStove": BuildSpec("fueled stove", size=(3, 1)),
    "ElectricStove": BuildSpec("electric stove", size=(3, 1), research="Electricity"),
    "SimpleResearchBench": BuildSpec("simple research bench", size=(3, 2), stuff=("Metallic", "Woody", "Stony")),
    "HandTailoringBench": BuildSpec("hand tailoring bench", size=(3, 1), stuff=("Metallic", "Woody"), research="ComplexClothing"),
    "TableStonecutter": BuildSpec("stonecutter's table", size=(3, 1), stuff=("Metallic", "Woody"), research="Stonecutting"),
    "Shelf": BuildSpec("shelf", size=(2, 1), stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture"),
    "Table2x2c": BuildSpec("table (2x2)", size=(2, 2), stuff=("Metallic", "Woody", "Stony")),
    "Stool": BuildSpec("stool", stuff=("Metallic", "Woody", "Stony")),
    "DiningChair": BuildSpec("dining chair", stuff=("Metallic", "Woody"), research="ComplexFurniture"),
    "PassiveCooler": BuildSpec("passive cooler", research="PassiveCooler"),
    "WoodFiredGenerator": BuildSpec("wood-fired generator", size=(2, 2), research="Electricity"),
    "StandingLamp": BuildSpec("standing lamp", research="Electricity"),
    "Barricade": BuildSpec("barricade", stuff=("Metallic", "Woody", "Stony")),
    "HorseshoesPin": BuildSpec("horseshoes pin", stuff=("Metallic", "Woody", "Stony")),
    "ChessTable": BuildSpec("chess table", stuff=("Metallic", "Woody", "Stony"), research="ComplexFurniture"),
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
