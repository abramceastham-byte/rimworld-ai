"""A small client for the RIMAPI mod's REST server.

Every RIMAPI response is wrapped in the same envelope:
    {"success": bool, "data": ..., "errors": [...], "warnings": [...], "timestamp": "..."}
The client unwraps it and raises RimApiError when success is false.
"""

from typing import Any

import httpx
from pydantic import BaseModel

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

    # Keep any fields a newer RIMAPI version adds, instead of dropping them.
    model_config = {"extra": "allow"}


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
