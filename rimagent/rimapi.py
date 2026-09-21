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
