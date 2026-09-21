"""Smoke test: read the game state, pause, then resume.

RimWorld must be running with a colony loaded and the RIMAPI mod enabled.
Run from the repo root:  python -m scripts.smoke_test
"""

import sys
import time

import httpx

from rimagent.rimapi import RimApiClient, RimApiError


def main() -> int:
    try:
        with RimApiClient() as client:
            state = client.get_state()
            print(f"Connected. Tick {state.game_tick}, {state.colonist_count} colonists, "
                  f"storyteller {state.storyteller}, paused={state.is_paused}")
            if state.map_count == 0:
                print("FAIL: RIMAPI is up but no colony is loaded. Load a save or start "
                      "a colony, then retry.")
                return 1

            client.pause()
            time.sleep(0.5)
            if not client.get_state().is_paused:
                print("FAIL: game did not pause")
                return 1
            print("Paused OK")

            client.resume()
            time.sleep(0.5)
            if client.get_state().is_paused:
                print("FAIL: game did not resume")
                return 1
            print("Resumed OK")
    except httpx.ConnectError:
        print("FAIL: could not reach RIMAPI. Is RimWorld running with the mod enabled "
              "and a colony loaded? Check RIMAPI_URL in .env.")
        return 1
    except RimApiError as e:
        print(f"FAIL: {e}")
        return 1
    except httpx.TimeoutException:
        print("FAIL: RIMAPI accepted the connection but did not answer. RimWorld stops "
              "updating when its window is unfocused; turn on Options > General > "
              "'Run in background', or click into the game window and retry.")
        return 1

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
