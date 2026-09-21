# rimworld-api

Building blocks for a Python agent that plays RimWorld through the
[RIMAPI](https://github.com/IlyaChichkov/RIMAPI) mod's REST server.
The agent loop itself lives elsewhere; this repo provides:

| File | What it does |
|------|--------------|
| `rimagent/config.py` | Reads `OLLAMA_HOST` and `RIMAPI_URL` from the environment / `.env` |
| `rimagent/rimapi.py` | RIMAPI client: `get_state()`, `pause()`, `resume()`, `start_game()`, `get_colonists()`, `get_alerts()`, `get_threats()`, `get_weather()`, `get_datetime()`, `get_work_types()`, `set_work_priority()` |
| `rimagent/fake_model.py` | Fake model returning canned answers, for offline testing |
| `rimagent/runlog.py` | Writes state snapshots and decisions as JSON lines in `logs/` |
| `scripts/smoke_test.py` | Reads state, pauses, resumes a live game |

## Setup

Requires Python 3.12 and RimWorld with the RIMAPI mod enabled.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` if your addresses differ from the defaults. It is not committed.

## Smoke test

1. In RimWorld, turn on **Options > General > Run in background**. Without it the
   game stops updating when its window loses focus, and every API request hangs.
2. Start RimWorld with RIMAPI enabled and load a colony. RIMAPI listens on
   `http://localhost:8765` by default (the port is configurable in the mod settings).
3. From the repo root, run:

```powershell
python -m scripts.smoke_test
```

It prints the colony's tick, colonist count and storyteller, then pauses and
resumes the game, checking each step took effect.

## Work priorities

`set_work_priority()` only has an effect beyond on/off if **Manual priorities**
is ticked at the top of the game's Work tab (once per colony). Without it,
RimWorld treats every enabled job as priority 3.

## Starting a new colony from code

`start_game()` starts a fresh game (replacing any loaded one) and waits until
the colony is playable. RIMAPI always uses the Crashlanded scenario and its
generated colonists; you can choose the storyteller, difficulty, world
settings and landing tile:

```python
from rimagent.rimapi import RimApiClient, NewGameOptions

with RimApiClient() as client:
    state = client.start_game(NewGameOptions(
        storyteller_name="Randy",
        difficulty_name="Medium",   # Peaceful, Easy, Medium, Rough, Hard, Extreme
        world_seed="myseed",
        starting_tile=None,         # None = random tile
    ))
```

## Running the model on a lab machine (SSH tunnel)

Ollama can run on a more powerful lab machine while RimWorld and the agent
stay here. An SSH tunnel makes the lab machine's Ollama port appear on your
own machine, so the agent still talks to `localhost` and nothing else changes.

```powershell
ssh -N -L 11434:localhost:11434 you@lab-machine
```

- `-L 11434:localhost:11434` means "forward my local port 11434 to port 11434
  on the lab machine" (as seen from the lab machine itself, hence `localhost`).
- `-N` opens the tunnel without starting a remote shell. Leave that window open.

With the tunnel up, the default `OLLAMA_HOST=http://localhost:11434` already
points at the lab machine. Check it with:

```powershell
curl http://localhost:11434/api/tags
```

If you also run Ollama locally on 11434, pick another local port for the tunnel,
for example `-L 11500:localhost:11434`, and set `OLLAMA_HOST=http://localhost:11500`
in `.env`. That one value is the only thing that changes.

Because Ollama only needs to listen on the lab machine's own localhost, it is
never exposed to the rest of the network.
