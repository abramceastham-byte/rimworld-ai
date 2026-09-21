"""A small client for the RIMAPI mod's REST server.

Every RIMAPI response is wrapped in the same envelope:
    {"success": bool, "data": ..., "errors": [...], "warnings": [...], "timestamp": "..."}
The client unwraps it and raises RimApiError when success is false.
"""

import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from rimagent.config import load_settings

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


class Colonist(BaseModel):
    id: int
    name: str
    age: int
    health: float  # 0-1
    mood: float    # 0-1
    hunger: float  # 0-1, where 1 = fully fed
    current_job: str = ""
    skills: list[Skill] = []
    # Only work types that are switched on. A missing work type means
    # priority 0 (off), or work this colonist is incapable of.
    work_priorities: list[WorkPriority] = []


class Weather(BaseModel):
    weather: str
    temperature_c: float  # RIMAPI reports Celsius


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


class RimApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0):
        base_url = base_url or load_settings().rimapi_url
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

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
        """Names accepted by set_work_priority, e.g. "Cooking", "Construction"."""
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

    # --- Acting on the colony ------------------------------------------------

    def enable_work(self, colonist_id: int, work_type: str) -> None:
        """Let a colonist do this kind of work."""
        self.set_work_priority(colonist_id, work_type, 3)

    def disable_work(self, colonist_id: int, work_type: str) -> None:
        """Stop a colonist doing this kind of work."""
        self.set_work_priority(colonist_id, work_type, 0)

    def set_work_priority(self, colonist_id: int, work_type: str, priority: int) -> None:
        """Set how much a colonist prioritises a type of work.

        priority: 0 = don't do it, 1 = highest ... 4 = lowest.
        Tick "Manual priorities" in the game's Work tab first: without it,
        RimWorld stores every non-zero value as 3, i.e. just "enabled".
        RIMAPI refuses work the colonist is incapable of (raises RimApiError).
        """
        if not 0 <= priority <= 4:
            raise ValueError("priority must be 0 (off) or 1-4")
        self._request(
            "POST",
            "/api/v1/colonist/work-priority",
            json={"id": colonist_id, "work": work_type, "priority": priority},
        )

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
