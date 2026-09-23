"""A small client for the RIMAPI mod's REST server.

Every RIMAPI response is wrapped in the same envelope:
    {"success": bool, "data": ..., "errors": [...], "warnings": [...], "timestamp": "..."}
The client unwraps it and raises RimApiError when success is false.
"""

import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from rimagent.config import load_settings
from rimagent.construction import (
    BUILDINGS,
    CROPS,
    DESIGNATIONS,
    FLOORS,
    MAX_FLOOR_CELLS,
    MAX_WALL_RUN,
    MAX_CHOP_TREES,
    MAX_DESIGNATE_CELLS,
    MAX_ZONE_CELLS,
    Rect,
    TerrainInfo,
    TerrainMap,
    footprint,
    is_tree,
)

# Values accepted by POST /api/v1/game/speed
PAUSED, NORMAL, FAST, SUPERFAST = 0, 1, 2, 3


class RimApiError(RuntimeError):
    """RIMAPI answered, but reported a failure."""


class GameState(BaseModel):
    game_tick: int
    colony_wealth: float
    colonist_count: int
    storyteller: str
    is_paused: bool
    program_state: str = "Unknown"  # "Playing" once a colony is loaded
    map_count: int = 0

    # Keep any fields a newer RIMAPI version adds, instead of dropping them.
    model_config = {"extra": "allow"}


class Alert(BaseModel):
    """One of the alerts on the right side of the screen, e.g. "Need colonist beds"."""

    label: str
    explanation: str = ""
    priority: str = ""  # "High", "Medium", ...


class Skill(BaseModel):
    name: str
    level: int
    passion: int = 0  # 0 none, 1 interested, 2 burning
    totally_disabled: bool = False


class WorkPriority(BaseModel):
    work_type: str
    priority: int  # 0 = off, 1 = highest ... 4 = lowest
    is_totally_disabled: bool = False


class MapPosition(BaseModel):
    x: int
    y: int = 0
    z: int


class Colonist(BaseModel):
    id: int
    name: str
    age: int
    health: float  # 0-1
    mood: float    # 0-1
    hunger: float  # 0-1, where 1 = fully fed
    position: MapPosition | None = None
    current_job: str = ""
    skills: list[Skill] = []
    # Only work types that are switched on. A missing work type means
    # priority 0 (off), or work this colonist is incapable of.
    work_priorities: list[WorkPriority] = []


class Weather(BaseModel):
    weather: str
    temperature_c: float  # RIMAPI reports Celsius


class Recipe(BaseModel):
    """A bill recipe available at a particular work table."""

    def_name: str
    label: str
    description: str = ""
    work_amount: float | None = None
    work_skill: str | None = None

    model_config = {"extra": "allow"}


class Bill(BaseModel):
    """One production bill currently queued on a work table."""

    load_id: int
    recipe_def_name: str
    repeat_mode: str = ""
    repeat_count: int | None = None
    target_count: int | None = None
    suspended: bool = False

    model_config = {"extra": "allow"}


class WorkTable(BaseModel):
    """A spawned building that can hold production bills."""

    id: int
    thing_def: str
    label: str
    position: MapPosition
    bills_count: int = 0
    recipes: list[Recipe] = Field(default_factory=list)
    bills: list[Bill] = Field(default_factory=list)


class MapThing(BaseModel):
    """An item or plant on the map, from /map/things or /map/plants."""

    thing_id: int
    def_name: str
    label: str = ""
    position: MapPosition
    stack_count: int = 1
    is_forbidden: bool = False


class Building(BaseModel):
    """A colony building from /map/buildings: finished, or a frame being built."""

    id: int
    def_name: str = Field(alias="def")
    label: str = ""
    position: MapPosition
    rotation: int = 0
    size: MapPosition = MapPosition(x=1, z=1)
    type: str = ""  # "Building", "Frame", "Building_Bed", "Building_WorkTable", ...

    model_config = {"populate_by_name": True}

    @property
    def under_construction(self) -> bool:
        return self.type == "Frame" or self.def_name.startswith("Frame_")

    def area(self) -> Rect:
        return footprint(self.position.x, self.position.z, (self.size.x, self.size.z), self.rotation)


class ResearchProject(BaseModel):
    name: str          # def name, e.g. "Brewing"
    label: str = ""
    research_points: float = 0
    tech_level: str = ""
    can_start_now: bool = False
    is_finished: bool = False
    progress_percent: float = 0
    prerequisites: list[str] = Field(default_factory=list)
    # True when the colony has a bench able to research this project. The
    # /research/progress placeholder for "nothing being researched" always says
    # false, so only the tree's answer can be trusted.
    player_has_any_appropriate_research_bench: bool = False

    model_config = {"extra": "allow"}


class ResearchState(BaseModel):
    current: ResearchProject | None = None   # None when nothing is being researched
    has_bench: bool = False                  # a research bench is built
    available: list[ResearchProject] = Field(default_factory=list)  # can be started now
    all_projects: list[ResearchProject] = Field(default_factory=list)


class Zone(BaseModel):
    """A growing zone or stockpile. RIMAPI reports its size but not its location."""

    id: int
    label: str
    type: str          # e.g. "Zone_Growing", "Zone_Stockpile"
    cells_count: int = 0


class AreaIssue(BaseModel):
    x: int
    z: int
    def_name: str | None = None
    label: str | None = None
    zone_type: str | None = None  # "Growing" or "Stockpile"


class AreaCheck(BaseModel):
    """RIMAPI's report on what is in a rectangle (POST /builder/check-zone)."""

    terrain: list[AreaIssue] = Field(default_factory=list)    # can't hold heavy buildings
    ores: list[AreaIssue] = Field(default_factory=list)       # rock and ore to mine first
    buildings: list[AreaIssue] = Field(default_factory=list)
    zones: list[AreaIssue] = Field(default_factory=list)      # one entry per overlapping zone

    @staticmethod
    def describe_cells(blocked: dict[tuple[int, int], str], limit: int = 3) -> str:
        """Cells that stopped a wall or floor, e.g. "granite at (4,5)"."""
        shown = ", ".join(f"{why} at ({x},{z})" for (x, z), why in list(blocked.items())[:limit])
        more = f" and {len(blocked) - limit} more" if len(blocked) > limit else ""
        return shown + more

    @staticmethod
    def describe(issues: list[AreaIssue], limit: int = 3) -> str:
        shown = ", ".join(
            f"{i.label or i.def_name or i.zone_type} at ({i.x},{i.z})" for i in issues[:limit]
        )
        more = f" and {len(issues) - limit} more" if len(issues) > limit else ""
        return shown + more


@dataclass(frozen=True)
class GameDefs:
    """The parts of RIMAPI's /def/all that zone and blueprint checks need."""

    terrain: dict[str, TerrainInfo]
    crop_fertility_min: dict[str, float]
    thing_names: frozenset[str]


# What a hostile group's RimWorld "lord job" means, in plain words.
# Names checked against RimWorld 1.6's own code. Unknown ones fall back to the raw name.
THREAT_BEHAVIOURS = {
    "LordJob_AssaultColony": "attacking the colony",
    "LordJob_BossgroupAssaultColony": "attacking the colony",
    "LordJob_StageThenAttack": "gathering at the map edge, about to attack",
    "LordJob_Siege": "building a siege camp to bombard the colony",
    "LordJob_SleepThenAssaultColony": "asleep; will attack the colony when woken",
    "LordJob_SleepThenMechanoidsDefend": "asleep; will only fight if disturbed",
    "LordJob_MechanoidsDefend": "guarding their position",
    "LordJob_DefendAndExpandHive": "an insect hive that grows and defends itself",
    "LordJob_Kidnap": "trying to carry off a colonist",
    "LordJob_Steal": "trying to steal items",
    "LordJob_ExitMapBest": "leaving the map",
    "LordJob_ExitMapNear": "leaving the map",
}


class Threat(BaseModel):
    """A group on the map belonging to a hostile faction (raid, mech cluster, ...)."""

    faction_name: str
    faction_type: str
    behaviour: str         # RimWorld's lord job, e.g. LordJob_AssaultColony
    current_step: str      # e.g. LordToil_Sleep for a dormant mech cluster
    pawn_count: int
    active: bool           # False while e.g. a mech cluster is still asleep

    def summary(self) -> str:
        """One line for a prompt, clearly separating live threats from dormant ones."""
        what = THREAT_BEHAVIOURS.get(self.behaviour, self.behaviour)
        group = f"{self.pawn_count} {self.faction_type} ({self.faction_name})"
        if self.active:
            return f"ACTIVE THREAT: {group}, {what}."
        return f"Dormant, not attacking yet: {group}, {what}."


class NewGameOptions(BaseModel):
    """Settings for POST /api/v1/game/start. Defaults match RimWorld's own.

    The scenario is always Crashlanded (RIMAPI does not let you choose it), and
    the starting colonists are the ones the scenario generates.
    World sliders run 0-6, where 3 is Normal (0 = least, 6 = most).
    """

    storyteller_name: Literal["Cassandra", "Phoebe", "Randy"] = "Cassandra"
    # Peaceful, Easy (community builder), Medium (adventure story),
    # Rough (strive to survive), Hard (blood and dust), Extreme (losing is fun)
    difficulty_name: Literal["Peaceful", "Easy", "Medium", "Rough", "Hard", "Extreme"] = "Rough"
    map_size: int = Field(250, gt=0)
    permadeath: bool = False
    planet_coverage: float = Field(0.3, gt=0, le=1)
    world_seed: str | None = None      # None = random. RIMAPI lowercases it.
    starting_tile: int | None = None   # None = random tile
    starting_season: int = Field(1, ge=1, le=6)  # 1 Spring, 2 Summer, 3 Fall, 4 Winter
    overall_rainfall: int = Field(3, ge=0, le=6)
    overall_temperature: int = Field(3, ge=0, le=6)
    overall_population: int = Field(3, ge=0, le=6)
    landmark_density: int = Field(3, ge=0, le=6)

    def to_request(self) -> dict[str, Any]:
        body = self.model_dump()
        # RIMAPI expects the tile and season as strings, and "auto" for a random tile.
        body["starting_tile"] = "auto" if self.starting_tile is None else str(self.starting_tile)
        body["starting_season"] = str(self.starting_season)
        if self.world_seed is None:
            del body["world_seed"]
        return body


def _corners(rect: Rect) -> dict[str, dict[str, int]]:
    """A rectangle in the point_a / point_b form RIMAPI's map endpoints take."""
    return {
        "point_a": {"x": rect.x1, "y": 0, "z": rect.z1},
        "point_b": {"x": rect.x2, "y": 0, "z": rect.z2},
    }


def _check_rect(rect: Rect, terrain: TerrainMap, max_cells: int) -> None:
    if not rect.within(terrain.width, terrain.height):
        raise ValueError(f"{rect} is outside the {terrain.width}x{terrain.height} map")
    if rect.cells > max_cells:
        raise ValueError(f"{rect} is {rect.cells} cells; the limit is {max_cells}")


class RimApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0):
        base_url = base_url or load_settings().rimapi_url
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self._defs: GameDefs | None = None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RimApiClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._http.request(method, path, **kwargs)
        try:
            body = response.json()
        except ValueError:
            # Not a RIMAPI envelope (e.g. a plain HTML error page).
            response.raise_for_status()
            raise RimApiError(f"{method} {path} returned non-JSON: {response.text[:200]}")
        # RIMAPI puts its reason in "errors" even on HTTP 500, so prefer that
        # over a generic status-code error.
        if not body.get("success", False):
            raise RimApiError(
                f"{method} {path} failed (HTTP {response.status_code}): {body.get('errors')}"
            )
        return body.get("data")

    def get_state(self) -> GameState:
        return GameState.model_validate(self._request("GET", "/api/v1/game/state"))

    def set_speed(self, speed: int) -> None:
        self._request("POST", "/api/v1/game/speed", params={"speed": speed})

    def pause(self) -> None:
        self.set_speed(PAUSED)

    def resume(self, speed: int = NORMAL) -> None:
        if speed == PAUSED:
            raise ValueError("resume() needs a running speed (1-3), not 0")
        self.set_speed(speed)

    # --- Reading the colony -------------------------------------------------

    def get_alerts(self) -> list[Alert]:
        """The alerts shown on the right side of the screen."""
        return [Alert.model_validate(a) for a in self._request("GET", "/api/v1/ui/alerts")]

    def get_colonists(self) -> list[Colonist]:
        """Every colonist with their needs, current job, skills and work priorities."""
        colonists = []
        for c in self._request("GET", "/api/v1/colonists/detailed"):
            work = c.get("colonist_work_info") or {}
            colonists.append(Colonist.model_validate({
                **c["colonist"],
                "current_job": work.get("current_job") or "",
                "skills": work.get("skills") or [],
                "work_priorities": work.get("work_priorities") or [],
            }))
        return colonists

    def get_work_types(self) -> list[str]:
        """Names accepted by enable_work/disable_work, e.g. "Cooking", "Construction"."""
        return self._request("GET", "/api/v1/work-list")["work"]

    def get_weather(self, map_id: int = 0) -> Weather:
        data = self._request("GET", "/api/v1/map/weather", params={"map_id": map_id})
        return Weather(weather=data["weather"], temperature_c=data["temperature"])

    def get_datetime(self) -> str:
        """In-game date and time, e.g. "1st of Aprimay, 5500, 6h"."""
        return self._request("GET", "/api/v1/datetime")["datetime"]

    def get_threats(self, map_id: int = 0) -> list[Threat]:
        """Groups on the map whose faction is hostile to the colony."""
        # A faction's name alone is not unique (there are two "Ancients"), so
        # match on name and faction type together.
        hostile = {
            (f["name"], f["def_name"])
            for f in self._request("GET", "/api/v1/factions")
            if f.get("relation") == "Hostile"
        }
        threats = []
        for lord in self._request("GET", "/api/v1/lords", params={"map_id": map_id}):
            if (lord["faction_name"], lord["faction_def_name"]) not in hostile:
                continue
            threats.append(Threat(
                faction_name=lord["faction_name"],
                faction_type=lord["faction_def_name"],
                behaviour=lord["lord_job_type"],
                current_step=lord["current_toil_name"],
                pawn_count=len(lord.get("owned_pawn_ids") or []),
                active=lord.get("any_active_pawn", True),
            ))
        return threats

    def get_work_tables(self, map_id: int = 0) -> list[WorkTable]:
        """Work tables, their available recipes, and their current bills."""
        tables = []
        for raw_table in self._request(
            "GET", "/api/v1/map/work-tables", params={"map_id": map_id}
        ):
            table_id = raw_table["id"]
            recipes = self._request(
                "GET",
                "/api/v1/buildings/recipes",
                params={"building_id": table_id},
            )
            bills = self._request(
                "GET",
                "/api/v1/buildings/bills",
                params={"building_id": table_id},
            )
            tables.append(
                WorkTable.model_validate(
                    {**raw_table, "recipes": recipes or [], "bills": bills or []}
                )
            )
        return tables

    # --- Acting on the colony ------------------------------------------------

    # Work is simply on or off: without "Manual priorities" ticked in the game's
    # Work tab, RimWorld treats every enabled job the same. RIMAPI refuses work
    # a colonist is incapable of (raises RimApiError).

    def enable_work(self, colonist_id: int, work_type: str) -> None:
        """Let a colonist do this kind of work."""
        self._set_work(colonist_id, work_type, 3)

    def disable_work(self, colonist_id: int, work_type: str) -> None:
        """Stop a colonist doing this kind of work."""
        self._set_work(colonist_id, work_type, 0)

    def _set_work(self, colonist_id: int, work_type: str, priority: int) -> None:
        self._request(
            "POST",
            "/api/v1/colonist/work-priority",
            json={"id": colonist_id, "work": work_type, "priority": priority},
        )

    def set_target_bill(
        self, building_id: int, recipe_def_name: str, target_count: int
    ) -> None:
        """Create or update a safe "do until X" production bill.

        The recipe is checked against the table before anything is changed. If a
        bill for the recipe already exists, update it instead of creating a
        duplicate. This deliberately does not expose bill deletion to the model.
        """
        if target_count < 1:
            raise ValueError("target_count must be at least 1")

        recipes = [
            Recipe.model_validate(recipe)
            for recipe in self._request(
                "GET",
                "/api/v1/buildings/recipes",
                params={"building_id": building_id},
            )
        ]
        if recipe_def_name not in {recipe.def_name for recipe in recipes}:
            raise ValueError(
                f"recipe {recipe_def_name!r} is not available at work table "
                f"{building_id}"
            )

        bills = [
            Bill.model_validate(bill)
            for bill in self._request(
                "GET",
                "/api/v1/buildings/bills",
                params={"building_id": building_id},
            )
        ]
        existing = next(
            (bill for bill in bills if bill.recipe_def_name == recipe_def_name),
            None,
        )
        body = {"repeat_mode": "TargetCount", "target_count": target_count}
        if existing is None:
            self._request(
                "POST",
                "/api/v1/buildings/bills/add",
                params={"building_id": building_id},
                json={"recipe_def_name": recipe_def_name, **body},
            )
            return

        params = {"building_id": building_id, "bill_id": existing.load_id}
        self._request(
            "PUT",
            "/api/v1/buildings/bill/update",
            params=params,
            json=body,
        )
        if existing.suspended:
            self._request(
                "PUT",
                "/api/v1/buildings/bill/suspend",
                params=params,
                json={"suspended": False},
            )

    # --- The map: zones, blueprints, designations ------------------------------
    #
    # RIMAPI's own checks here are thin: unknown names are skipped while it still
    # reports success, blueprints skip RimWorld's normal placement checks, and a
    # stockpile created without a priority stores nothing. So every method below
    # validates first and raises ValueError (with a reason the model can read)
    # before changing anything.

    def get_game_defs(self) -> GameDefs:
        """Terrain fertility, crop needs and thing names, from the running game.

        /def/all is about 2 MB, so it is fetched once per client and reused.
        """
        if self._defs is None:
            data = self._request("GET", "/api/v1/def/all")
            self._defs = GameDefs(
                terrain={
                    t["def_name"]: TerrainInfo(
                        float(t.get("fertility") or 0.0), frozenset(t.get("affordances") or [])
                    )
                    for t in data["terrain_defs"]
                },
                crop_fertility_min={
                    p["def_name"]: float(p.get("fertility_min") or 0.0) for p in data["plant_defs"]
                },
                thing_names=frozenset(t["def_name"] for t in data["things_defs"]),
            )
        return self._defs

    def get_terrain(self, map_id: int = 0) -> TerrainMap:
        data = self._request("GET", "/api/v1/map/terrain", params={"map_id": map_id})
        return TerrainMap(data["width"], data["height"], data["palette"], data["grid"])

    def get_buildings(self, map_id: int = 0) -> list[Building]:
        """The colony's own buildings, including frames under construction."""
        return [Building.model_validate(b) for b in
                self._request("GET", "/api/v1/map/buildings", params={"map_id": map_id})]

    def get_zones(self, map_id: int = 0) -> list[Zone]:
        """Growing zones and stockpiles (not the Home/roof areas)."""
        data = self._request("GET", "/api/v1/map/zones", params={"map_id": map_id})
        return [Zone.model_validate(z) for z in data.get("zones") or []]

    def get_research(self) -> ResearchState:
        """What the colony is researching, and what it could start.

        A research bench takes a project, not bills, so this is separate from
        work tables.
        """
        progress = self._request("GET", "/api/v1/research/progress")
        projects = [ResearchProject.model_validate(p)
                    for p in self._request("GET", "/api/v1/research/tree")["projects"]]
        current = None
        if progress.get("name") and progress["name"] != "none":
            current = ResearchProject.model_validate(progress)
        return ResearchState(
            current=current,
            has_bench=any(p.player_has_any_appropriate_research_bench for p in projects),
            available=sorted((p for p in projects if p.can_start_now and not p.is_finished),
                             key=lambda p: p.research_points),
            all_projects=projects,
        )

    def set_research(self, project: str) -> str:
        """Start researching a project. Returns its label."""
        state = self.get_research()
        match = next((p for p in state.available if p.name.lower() == project.lower()), None)
        if match is None:
            choices = ", ".join(p.name for p in state.available[:10]) or "none"
            known = next((p for p in state.all_projects if p.name.lower() == project.lower()), None)
            if known is None:
                raise ValueError(
                    f"{project!r} is not a research project in this game. "
                    f"Projects you can start now: {choices}"
                )
            if known.is_finished:
                raise ValueError(f"{known.label or known.name} is already researched")
            if not known.player_has_any_appropriate_research_bench:
                raise ValueError(
                    f"{known.name} needs a research bench the colony does not have yet"
                )
            missing = ", ".join(known.prerequisites) or "an earlier project"
            raise ValueError(
                f"{known.name} cannot be started yet: it needs {missing} first. "
                f"Projects you can start now: {choices}"
            )
        self._request("POST", "/api/v1/research/target",
                      params={"name": match.name, "force": False})
        return match.label or match.name

    def get_finished_research(self) -> set[str]:
        return set(self._request("GET", "/api/v1/research/finished")["finished_projects"])

    def check_area(self, rect: Rect, map_id: int = 0) -> AreaCheck:
        """What stands in a rectangle: unbuildable terrain, rock, buildings, zones."""
        data = self._request(
            "POST", "/api/v1/builder/check-zone", json={"map_id": map_id, **_corners(rect)}
        )
        return AreaCheck.model_validate((data or {}).get("issues") or {})

    def create_growing_zone(self, plant: str, rect: Rect, map_id: int = 0) -> int:
        """Create a growing zone for a crop and return its zone id."""
        if plant not in CROPS:
            raise ValueError(f"{plant!r} is not an allowed crop; choose from {', '.join(CROPS)}")
        defs = self.get_game_defs()
        if plant not in defs.crop_fertility_min:
            raise ValueError(f"{plant!r} does not exist in this game")
        terrain = self.get_terrain(map_id)
        _check_rect(rect, terrain, MAX_ZONE_CELLS)

        # RIMAPI doesn't check soil at all, so a zone on stone would never grow.
        need = max(defs.crop_fertility_min[plant], 0.01)
        infertile = [
            (x, z) for x, z in rect
            if defs.terrain.get(terrain.at(x, z), TerrainInfo(0.0, frozenset())).fertility < need
        ]
        if infertile:
            x, z = infertile[0]
            raise ValueError(
                f"{len(infertile)} of {rect.cells} cells in {rect} are too infertile for {plant} "
                f"(needs fertility {need}); e.g. ({x},{z}) is {terrain.at(x, z)}"
            )
        check = self.check_area(rect, map_id)
        blocking = check.ores + check.buildings + check.zones
        if blocking:
            raise ValueError(f"area {rect} is blocked by {AreaCheck.describe(blocking)}")

        data = self._request(
            "POST",
            "/api/v1/map/zone/growing",
            json={"map_id": map_id, "plant_def": plant, **_corners(rect)},
        )
        return data["zone"]["id"]

    def create_stockpile(
        self, rect: Rect, priority: Literal["low", "normal"] = "normal", map_id: int = 0
    ) -> int:
        """Create a stockpile that accepts all normal items and return its zone id."""
        terrain = self.get_terrain(map_id)
        _check_rect(rect, terrain, MAX_ZONE_CELLS)
        check = self.check_area(rect, map_id)
        # Terrain issues here mean water or marsh, where items shouldn't be stored.
        blocking = check.terrain + check.ores + check.buildings + check.zones
        if blocking:
            raise ValueError(f"area {rect} is blocked by {AreaCheck.describe(blocking)}")

        data = self._request(
            "POST",
            "/api/v1/map/zone/stockpile",
            json={
                "map_id": map_id,
                **_corners(rect),
                # RimWorld's StoragePriority: 0 = Unstored, 1 = Low, 2 = Normal.
                # RIMAPI defaults to 0 (a stockpile that stores nothing) and caps
                # the value at 2, so always send it explicitly.
                "priority": 1 if priority == "low" else 2,
            },
        )
        return data["zone_id"]

    def place_blueprint(
        self,
        def_name: str,
        x: int,
        z: int,
        rotation: int = 0,
        stuff: str | None = None,
        map_id: int = 0,
    ) -> Rect:
        """Place one building blueprint for colonists to construct.

        Returns the cells the building will cover.
        rotation: 0 north, 1 east, 2 south, 3 west.
        """
        spec = BUILDINGS.get(def_name)
        if spec is None:
            raise ValueError(f"{def_name!r} is not in the building catalog")
        if spec.research and spec.research not in self.get_finished_research():
            raise ValueError(f"{def_name} needs the {spec.research} research first")
        defs = self.get_game_defs()
        if def_name not in defs.thing_names:
            raise ValueError(f"{def_name!r} does not exist in this game")
        if rotation not in (0, 1, 2, 3):
            raise ValueError("rotation must be 0 (north), 1 (east), 2 (south) or 3 (west)")
        materials = spec.materials()
        if materials and stuff not in materials:
            raise ValueError(f"{def_name} needs a material, one of: {', '.join(materials)}")
        if not materials and stuff:
            raise ValueError(f"{def_name} has a fixed cost and takes no material")
        if stuff and stuff not in defs.thing_names:
            raise ValueError(f"material {stuff!r} does not exist in this game")

        area = footprint(x, z, spec.size, rotation)
        _check_rect(area, self.get_terrain(map_id), spec.size[0] * spec.size[1])

        # check-zone can't see blueprints or frames (RimWorld treats them as
        # non-physical), and the game will happily stack blueprints on top of
        # each other, so check the footprint cell by cell first.
        for cell_x, cell_z in area:
            for thing in self.get_things_at(cell_x, cell_z, map_id):
                if thing.def_name.startswith(("Blueprint_", "Frame_")):
                    planned = thing.def_name.split("_", 1)[1]
                    state = "already being built" if thing.def_name.startswith("Frame_") else "already planned"
                    raise ValueError(
                        f"{def_name} at ({x},{z}) would cover ({cell_x},{cell_z}), where "
                        f"{planned} is {state}"
                    )

        check = self.check_area(area, map_id)
        # Buildings are fine inside stockpiles (shelves belong there), but would
        # eat into a growing zone.
        fields = [i for i in check.zones if i.zone_type != "Stockpile"]
        blocking = check.terrain + check.ores + check.buildings + fields
        if blocking:
            raise ValueError(
                f"{def_name} at ({x},{z}) facing {rotation} covers {area}, which is blocked by "
                f"{AreaCheck.describe(blocking)}"
            )

        self._request(
            "POST",
            "/api/v1/builder/blueprint",
            json={
                "map_id": map_id,
                "position": {"x": x, "y": 0, "z": z},
                "blueprint": {
                    "width": spec.size[0],
                    "height": spec.size[1],
                    "floors": [],
                    "buildings": [
                        {
                            "def_name": def_name,
                            "stuff_def_name": stuff,
                            "rel_x": 0,
                            "rel_z": 0,
                            "rotation": rotation,
                        }
                    ],
                },
                # Ignored by the blueprint endpoint today (only paste uses it),
                # but it defaults to true, so never rely on that.
                "clear_obstacles": False,
            },
        )
        return area

    def _blocked_cells(self, rect: Rect, map_id: int) -> dict[tuple[int, int], str]:
        """Cells in a rectangle that can't be built on, and why.

        Combines RIMAPI's area check (rock, buildings, growing zones, terrain)
        with a per-cell look for blueprints and frames, which the area check
        cannot see.
        """
        check = self.check_area(rect, map_id)
        blocked: dict[tuple[int, int], str] = {}
        for issue in check.terrain + check.ores + check.buildings:
            blocked[(issue.x, issue.z)] = issue.label or issue.def_name or "something"
        for issue in check.zones:
            if issue.zone_type != "Stockpile":  # building inside a stockpile is fine
                blocked[(issue.x, issue.z)] = f"{issue.label or 'a zone'} (zone)"
        for x, z in rect:
            for thing in self.get_things_at(x, z, map_id):
                if thing.def_name.startswith(("Blueprint_", "Frame_")):
                    planned = thing.def_name.split("_", 1)[1]
                    blocked[(x, z)] = f"{planned} already planned here"
        return blocked

    def build_wall(self, rect: Rect, stuff: str, def_name: str = "Wall", map_id: int = 0) -> str:
        """Blueprint a straight run of walls, skipping cells that are taken.

        Placing a wall one action at a time made models collide with their own
        placements; a run is one action. Returns a summary for the next prompt.
        """
        spec = BUILDINGS.get(def_name)
        if spec is None or spec.size != (1, 1):
            raise ValueError(f"{def_name!r} cannot be built as a run; use place_blueprint")
        if rect.x1 != rect.x2 and rect.z1 != rect.z2:
            raise ValueError(f"a wall run must be straight: {rect} bends")
        if rect.cells > MAX_WALL_RUN:
            raise ValueError(f"{rect} is {rect.cells} cells; a run is at most {MAX_WALL_RUN}")
        materials = spec.materials()
        if stuff not in materials:
            raise ValueError(f"{def_name} needs a material, one of: {', '.join(materials)}")
        _check_rect(rect, self.get_terrain(map_id), MAX_WALL_RUN)

        blocked = self._blocked_cells(rect, map_id)
        free = [cell for cell in rect if cell not in blocked]
        if not free:
            raise ValueError(f"every cell in {rect} is taken: {AreaCheck.describe_cells(blocked)}")
        self._request("POST", "/api/v1/builder/blueprint", json={
            "map_id": map_id,
            "position": {"x": rect.x1, "y": 0, "z": rect.z1},
            "blueprint": {
                "width": rect.x2 - rect.x1 + 1, "height": rect.z2 - rect.z1 + 1, "floors": [],
                "buildings": [{"def_name": def_name, "stuff_def_name": stuff,
                               "rel_x": x - rect.x1, "rel_z": z - rect.z1, "rotation": 0}
                              for x, z in free],
            },
            "clear_obstacles": False,
        })
        skipped = (f"; skipped {len(blocked)} cells: {AreaCheck.describe_cells(blocked)}"
                   if blocked else "")
        return f"{len(free)} {def_name} blueprints from ({rect.x1},{rect.z1}) to ({rect.x2},{rect.z2}){skipped}"

    def build_floor(self, rect: Rect, terrain: str, map_id: int = 0) -> str:
        """Blueprint flooring over a rectangle, skipping cells that are taken."""
        floor = FLOORS.get(terrain)
        if floor is None:
            raise ValueError(f"{terrain!r} is not a floor; choose from {', '.join(FLOORS)}")
        if floor.research and floor.research not in self.get_finished_research():
            raise ValueError(f"{terrain} needs the {floor.research} research first")
        if rect.cells > MAX_FLOOR_CELLS:
            raise ValueError(f"{rect} is {rect.cells} cells; floors cover at most {MAX_FLOOR_CELLS}")
        _check_rect(rect, self.get_terrain(map_id), MAX_FLOOR_CELLS)

        # Floors go under buildings, so only rock and unbuildable ground stop them.
        check = self.check_area(rect, map_id)
        blocked = {(i.x, i.z): i.label or i.def_name or "something"
                   for i in check.terrain + check.ores}
        free = [cell for cell in rect if cell not in blocked]
        if not free:
            raise ValueError(f"no cell in {rect} can take a floor: {AreaCheck.describe_cells(blocked)}")
        self._request("POST", "/api/v1/builder/blueprint", json={
            "map_id": map_id,
            "position": {"x": rect.x1, "y": 0, "z": rect.z1},
            "blueprint": {
                "width": rect.x2 - rect.x1 + 1, "height": rect.z2 - rect.z1 + 1, "buildings": [],
                "floors": [{"def_name": terrain, "rel_x": x - rect.x1, "rel_z": z - rect.z1}
                           for x, z in free],
            },
            "clear_obstacles": False,
        })
        return f"{len(free)} cells of {floor.label} planned over {rect}"

    def designate(
        self, kind: Literal["mine", "harvest", "hunt"], rect: Rect, map_id: int = 0
    ) -> None:
        """Mark everything of one kind in a rectangle: rock to mine, ripe plants
        to harvest, or wild animals to hunt."""
        if kind not in DESIGNATIONS:
            raise ValueError(f"designation must be one of: {', '.join(DESIGNATIONS)}")
        _check_rect(rect, self.get_terrain(map_id), MAX_DESIGNATE_CELLS)
        self._request(
            "POST",
            "/api/v1/order/designate/area",
            json={"map_id": map_id, "type": kind, **_corners(rect)},
        )

    def get_items(self, map_id: int = 0) -> list[MapThing]:
        """Every haulable item on the map, with its position and forbidden flag."""
        return [MapThing.model_validate(t) for t in self._request(
            "GET", "/api/v1/map/things", params={"map_id": map_id}
        )]

    def get_trees(self, map_id: int = 0) -> list[MapThing]:
        """Every tree on the map. /map/plants lists all ~25,000 plants (about 7 MB),
        so call this sparingly."""
        return [
            MapThing.model_validate(p)
            for p in self._request("GET", "/api/v1/map/plants", params={"map_id": map_id})
            if is_tree(p.get("def_name", ""))
        ]

    def get_things_at(self, x: int, z: int, map_id: int = 0) -> list[MapThing]:
        """Everything on one cell, including blueprints (Blueprint_<def>) and
        buildings under construction (Frame_<def>), which no list endpoint shows."""
        # RIMAPI reads this GET request's parameters from a JSON body.
        data = self._request(
            "GET",
            "/api/v1/map/things-at",
            json={"map_id": map_id, "position": {"x": x, "y": 0, "z": z}},
        )
        return [MapThing.model_validate(t) for t in data or []]

    def allow_items(self, rect: Rect, map_id: int = 0) -> int:
        """Unforbid every forbidden item in a rectangle so colonists can use it.

        Returns how many items were allowed. Limited to a rectangle because
        forbidden items are scattered across the whole map, some of them near
        dangers the colonists shouldn't walk into.
        """
        _check_rect(rect, self.get_terrain(map_id), MAX_DESIGNATE_CELLS)
        ids = [
            t.thing_id for t in self.get_items(map_id)
            if t.is_forbidden and rect.contains(t.position.x, t.position.z)
        ]
        if not ids:
            raise ValueError(f"there are no forbidden items in {rect}")
        self._request(
            "POST",
            "/api/v1/things/set-forbidden",
            json={"map_id": map_id, "thing_ids": ids, "forbidden": False},
        )
        return len(ids)

    def chop_trees(self, rect: Rect, map_id: int = 0) -> int:
        """Mark the trees in a rectangle for cutting, for wood.

        RIMAPI has no chop order, and its "harvest" designation marks every ripe
        plant in an area, crops included. So this marks each tree's own cell.
        At most MAX_CHOP_TREES, nearest the middle of the rectangle first.
        Trees too young to give wood are skipped by the game.
        Returns how many trees were marked.
        """
        _check_rect(rect, self.get_terrain(map_id), MAX_DESIGNATE_CELLS)
        trees = [t for t in self.get_trees(map_id) if rect.contains(t.position.x, t.position.z)]
        if not trees:
            raise ValueError(f"there are no trees in {rect}")
        cx, cz = (rect.x1 + rect.x2) / 2, (rect.z1 + rect.z2) / 2
        trees.sort(key=lambda t: (t.position.x - cx) ** 2 + (t.position.z - cz) ** 2)
        for tree in trees[:MAX_CHOP_TREES]:
            cell = Rect(tree.position.x, tree.position.z, tree.position.x, tree.position.z)
            self._request(
                "POST",
                "/api/v1/order/designate/area",
                json={"map_id": map_id, "type": "harvest", **_corners(cell)},
            )
        return min(len(trees), MAX_CHOP_TREES)

    def start_game(
        self,
        options: NewGameOptions | None = None,
        wait: bool = True,
        timeout: float = 300.0,
    ) -> GameState | None:
        """Start a new colony. This replaces any game that is currently loaded.

        RIMAPI answers straight away and generates the world and map in the
        background. With wait=True this polls until the colony is playable and
        returns its state; with wait=False it returns None immediately.
        """
        options = options or NewGameOptions()
        self._request("POST", "/api/v1/game/start", json=options.to_request())
        if not wait:
            return None

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                state = self.get_state()
            except httpx.TimeoutException:
                continue  # the game can stall while generating the world
            if state.program_state == "Playing" and state.map_count > 0:
                return state
        raise TimeoutError(f"New game was not ready after {timeout:.0f}s")
