# rimworld-api

Building blocks for a Python agent that plays RimWorld through the
[RIMAPI](https://github.com/IlyaChichkov/RIMAPI) mod's REST server.
The agent loop itself lives elsewhere; this repo provides:

| File | What it does |
|------|--------------|
| `rimagent/config.py` | Reads `OLLAMA_HOST` and `RIMAPI_URL` from the environment / `.env` |
| `rimagent/rimapi.py` | RIMAPI client for game state, colonists, threats, work settings, work tables, recipes, bills, zones, blueprints, designations, and game control |
| `rimagent/construction.py` | Map geometry, terrain decoding, and the building/crop catalog the map actions are checked against |
| `rimagent/fake_model.py` | Fake model returning canned answers, for offline testing |
| `rimagent/memory.py` | Persistent Markdown observations, structured JSON plans/policies, and recent decision memory |
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

1. In RimWorld, turn on **Options > General > Run in background**.
2. Start RimWorld with RIMAPI enabled and load a colony. RIMAPI listens on
   `http://localhost:8765` by default (the port is configurable in the mod settings).
3. From the repo root, run:

```powershell
python -m scripts.smoke_test
```

It prints the colony's tick, colonist count and storyteller, then pauses and
resumes the game, checking each step took effect.

## Agent memory

The agent stores durable, model-proposed memory under `memory/`:

- `observations.md` is an append-only playthrough notebook.
- `state.json` contains the current plan, non-live statuses, and policies.

Live facts already read from RIMAPI (game state, colonists, alerts, threats,
events, and work types) stay in the fresh prompt and are not copied into the
structured memory. Memory files survive process restarts. Delete or archive the
`memory/` directory yourself before starting a different colony.

Invalid model responses and model-call errors are recorded as `model_failure`
entries in the run log. The agent retries once with validation feedback. If both
attempts fail, it pauses when a threat is present and otherwise waits.

## Work

The agent switches work on or off per colonist with `enable_work()` and
`disable_work()`. It does not rank jobs: RIMAPI cannot turn on the game's
**Manual priorities** setting, and without it RimWorld treats every enabled
job the same.

## Production bills

Every agent step includes the spawned work tables, their exact recipe definition
names, and their existing bills. The model can use `set_bill` to create or update
one production bill using RimWorld's **Do until X** mode:

```json
{
  "action": "set_bill",
  "building_id": 14502,
  "recipe_def_name": "CookMealSimple",
  "target_count": 20,
  "reason": "Maintain a reserve of 20 simple meals."
}
```

The client checks that the recipe is actually available at that table. If the
same recipe already has a bill, it updates and resumes that bill rather than
creating a duplicate. Bill deletion is intentionally not exposed to the model.

## What the model sees and does each step

A reply is a **turn**: 1 to 5 actions carried out in order, each checked on its
own, with the results reported back in the next prompt. Several actions at once
(with the game paused) let it lay out a room or set up a colonist in one go.

The prompt includes a mini-map around the colony showing terrain, colonists,
trees, forbidden items, **the colony's own buildings, frames and blueprints**,
plus what supplies are **usable now versus forbidden** (so "no usable Steel" is
visible before it plans a steel building), and the live status of every
blueprint it placed.

Plan and policy changes in a reply are **ignored when one of its actions
failed** (`honest_memory_update`), since a failed action changes nothing; the
reply's observation is still kept. Each run log records the full prompt with the
first action of every turn.

## Zones, blueprints and designations

The model can shape the map with four actions. Each step's prompt includes
colonist positions, a terrain mini-map around the colony, existing zones, what
the agent has placed, and the buildings and crops it may use.

| Action | What it does | Limits |
|--------|--------------|--------|
| `create_growing_zone` | Plants one crop in a rectangle | 225 cells; every cell fertile enough for the crop; food/fibre crops only |
| `create_stockpile` | Storage for all normal items | 225 cells; not on water or marsh |
| `place_blueprint` | One building for colonists to construct | Catalog buildings only, research finished, valid material, no overlap with an existing blueprint or frame |
| `designate` | Mine rock, harvest ripe plants, or hunt animals in a rectangle | 400 cells |

RIMAPI does little checking of its own here (unknown names are skipped while it
still reports success, and blueprints skip RimWorld's placement checks), so the
client checks every request first: it asks RIMAPI what is in the area and
refuses anything that overlaps rock, buildings or other zones, giving the model
the reason. Deconstruction, copy/paste and zone deletion are deliberately not
exposed. RIMAPI has no endpoint for removing a growing zone or a blueprint; use
the game's own tools for that.

Two RIMAPI quirks the client works around, worth reporting upstream:

- A stockpile created without a priority gets RimWorld's `Unstored` (0), so it
  stores nothing; RIMAPI also caps the priority at Normal. The client always
  sends Normal (or Low).
- Zones in `GET /map/zones` have a size but no location, so the agent keeps its
  own record of where it placed things during a run.

Tests: `python -m unittest tests/test_map_actions.py -v`

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

## Running a real model

With Ollama reachable at `OLLAMA_HOST` (see the SSH tunnel section below):

```bash
# One decision about the live colony, printed and saved to logs/dry-runs/,
# never acted on. Use it after changing the prompt or to compare models.
python -m scripts.dry_run qwen3:14b --think on

# A real run: the model plays for 20 steps.
python -m scripts.run_agent --model gpt-oss:20b --think medium --steps 20
python -m scripts.run_agent --model qwen3:14b --think on --steps 20
```

**Game time is controlled by the loop**, in two modes. The game is always paused
while the model decides.

- **Running:** after each action the game runs for `--run-seconds` (default 10),
  or `--threat-run-seconds` (default 3) while a threat is active, then pauses for
  the next decision. It stops early if a threat letter arrives or the game pauses
  itself, so the model can respond at once.
- **Paused:** the model's `pause` action stops time until it chooses `resume`, so
  it can give several orders in a row (orders are queued and carried out when
  time runs again). The loop also switches to this mode when the game pauses
  itself, e.g. RimWorld's auto-pause on a major threat, or you pressing pause.

A run starts in whichever mode matches the game: paused if the game is paused.
The prompt says which mode is active, why it's paused, how many decisions have
been made while paused, and what happened while time last ran. `--speed` 1-3 sets
the game speed while running. Ctrl+C stops the run and leaves the game paused.

**Blueprints the agent places are tracked** in `memory/blueprints.json` and
checked in the game every step, so the model sees whether each one is waiting,
under construction, built, or gone (RIMAPI has no endpoint that lists
blueprints, so ones placed by hand aren't tracked).

`--think` is `on`/`off` for qwen3 and `low`/`medium`/`high` for gpt-oss (which
can't switch it off).

Logs are organised by run:

```
logs/runs/2026-09-22_063000_gpt-oss-20b_think-medium/run.jsonl   # one folder per run
logs/dry-runs/2026-09-22_021915_gpt-oss-20b_think-low.json       # one file per dry run
```

`python -m scripts.show_log` turns the newest log into Markdown next to it
(`run.md`, or the dry run's name with `.md`); `--all` converts every log.
Open it in VS Code and press Ctrl+Shift+V.

Each run log starts with a `run` record naming the model
and settings, and every `decision` record includes the model's `thinking` and
timings. The thinking is only logged; it is never fed back into the next prompt.
Leave out `--model` to use the fake model. `--num-ctx` (default 16384) sets the
context window; Ollama's own default is smaller than the prompt and would
silently cut it off.

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
