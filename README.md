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
| `rimagent/agent.py` | The turn loop: observe, build the prompt, ask the model, carry out orders, run game time |
| `rimagent/choices.py` | Each turn's exact valid choices (names, ids, materials), the per-turn JSON schema, and remembered failures |
| `rimagent/execution.py` | Carries out one order and reports what actually changed (unchanged, verified, or only submitted) |
| `rimagent/observation.py` | Extra read-only observations: weather, farms, storage, power, bedrooms, mining and hunting targets |
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

- `observations.md` keeps big discoveries and ideas for the whole playthrough ("colonists are
  happier with a bedroom each"), not a diary of turns; the prompt asks for exactly that.
- `state.json` contains the current plan, non-live statuses, and policies.

Live facts already read from RIMAPI (game state, colonists, alerts, threats,
events, and work types) stay in the fresh prompt and are not copied into the
structured memory. Memory files survive process restarts. Delete or archive the
`memory/` directory yourself before starting a different colony.

Invalid model responses and model-call errors are recorded as `model_failure`
entries in the run log. The agent retries once with validation feedback. If both
attempts fail, the turn gives no orders and holds (time does not run).

## Work

The agent switches work on or off per colonist with `enable_work()` and
`disable_work()`. It does not rank jobs: RIMAPI cannot turn on the game's
**Manual priorities** setting, and without it RimWorld treats every enabled
job the same.

RIMAPI only lists work that is switched on, so it never says what a colonist
*can't* do. The agent infers that from totally disabled skills (no
Intellectual means no Research) and from RIMAPI's refusal when it tries; either
way that work leaves the colonist's choices for the rest of the run. Asking for
work that is already on (or off) reports "unchanged" and sends nothing.

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

## Overview, focus and threat prompts (`--loop modes`, the default)

Instead of one prompt with everything, the loop asks three kinds of prompt:

- **Overview:** the colony at a glance. The model writes a plan in its own
  words (`goal`, up to 6 `steps`, `done_when`) and names 1 to 3 **toolkits**:
  construction, storage, work, production, farming, research, wood_mining,
  hunting, power, medical. It gives no orders.
- **Focus:** the plan plus only what its toolkits need: their orders (the reply
  schema offers nothing else), the rules for those orders, the map sections and
  observations they use, and the lessons saved for them. Enabling and disabling
  work is always offered. The reply is a turn plus `plan` (`continue`, `done`
  or `replan`), `request_toolkits` (added from the next turn) and an optional
  `lesson` tagged with a toolkit, so it only shows up in prompts that use that toolkit.
- **Threat:** while enemies are active: only the enemies (id, kind, position,
  health), the colonists with their weapons, weapons lying on the map, and
  combat orders: `draft`, `move`, `attack`, `equip`, `tend`, `bed_rest`, `rescue`.
  Once the threat is over, drafted colonists are released back to work and the
  overview is asked again.

The overview comes back when the model says the plan is done or asks to
replan, after 12 focus turns on one plan, after two turns in a row where every
order failed, and after a threat. Every prompt starts with the same **vital
signs**: date, temperature, food in days, each colonist's health, mood, food
and job (DRAFTED, BADLY HURT, HUNGRY flags), threats, alerts and last turn's
counts. `--loop single` is the old one-prompt loop, with the orders it had before.

`--steps` counts turns (focus and threat turns; an overview runs within the turn it starts). A run doesn't stop mid-plan: past `--steps` it keeps going until the current plan is done, replanned or out of turns, or the fight is over, but never starts a new plan, and never takes more than 18 extra turns. Drafted colonists are always released when a run ends.

**Base layout** (`rimagent/layout.py`). With `plan_area` (in every building
toolkit) the model reserves an area for each kind of building: bedrooms,
kitchen_dining, storage, workshop, research, fields, power, recreation,
hospital. It is saved in `memory/layout.json` and listed, with what stands in
each area, in the overview and building prompts. Placing something outside
its area, or inside another kind's area, gets a HINT. Nothing is refused, and
planning an area again moves it. Construction and storage prompts also list
**room slots**: rooms right beside ones already built with `build_room`,
sharing a wall, checked clear, with the exact `build_room` rectangle and door side.

**connect_power** takes a cell of the power source (generator, battery or one
of its conduits) and a cell of what needs power. The route is worked out in
code (`rimagent/power.py`): generators, batteries and conduits carry power to
whatever touches them, and a building that uses power links to any of them
within about 6 cells. So a route starts touching the source and ends near the
target. It uses a plain `PowerConduit` (1 steel) under walls and other buildings and a
`HiddenConduit` (2 steel) on open ground, and reuses existing conduits. It is
offered once Electricity is researched and a generator or battery exists (built
or planned).

## What the model sees and does each step

A reply is a **turn**: 0 to 10 orders, then a time choice:

```json
{"actions": [{"action": "build_room", "x1": 111, "z1": 116, "x2": 116, "z2": 120,
              "stuff": "WoodLog", "door_side": "north", "purpose": "bedroom",
              "reason": "A bedroom."}],
 "time": "advance"}
```

The orders run in order while the game is paused, each against freshly read
state. Every order is tried, even after one fails: the game refuses one that
relied on a failed order, and the next prompt reports each by number as OK,
UNCHANGED or FAILED, with a count. Every result says what really happened:
`unchanged` when the game already matched, `verified` when a re-read confirmed
it, or `submitted` when RIMAPI can't confirm it (designations, tree chopping).
Failures stay in the prompt, up to six, until the state they depended on
changes.
Repeated construction sends no write when the same building or floor is already
built or queued. Buildings must match the anchor and orientation; existing
materials are retained when RIMAPI omits material information. Rooms reuse
existing walls and rock. A conflicting building or floor is still an obstruction.

**Only valid choices are offered.** Each turn the JSON schema handed to the
model is narrowed to what exists right now: colonist names, work types, table
ids and their recipes, startable research, unlocked buildings and floors, and
materials. Actions with nothing to act on (no forbidden items, no trees, no
work tables) are left out, and the prompt says why. The reply is checked against
the same choices after decoding. Schema branches enforce table/recipe,
colonist/work and building/material pairs; recipe reads request only unlocked
recipes. Runtime checks also enforce geometry, including single-cell hunting.

**Every result carries a hint** (`rimagent/hints.py`), drawn from what the turn
observed. A failure says what to do instead: a free spot that fits, the nearest
targets, who can do the work. A failure caused by the colony's own blueprint
says it is the model's own and nothing needs fixing, and isn't kept in the
prompt as "unresolved". Without that, a model read its own blueprint as a
conflict, "repositioned" it by placing another, and planned twenty
stonecutter's tables. A construction order that works says which copy it is,
what it costs against what the other unfinished builds already claim, and who
builds it. Work tables, stoves, the research bench and generators can't be
placed again while another of the same kind is still unbuilt.

**Standing hints** join the results at the top of the prompt: when colonists
have wandered or idled for two turns in a row (pending work against who has it
on, and what the idle ones could take), and when the stockpiles are 90% full or
more (make another stockpile, or build shelves).

**Open areas avoid zones RIMAPI won't locate.** Its area check names one cell
per zone, so a reported zone rules out the whole square it could cover (from
its cell count in `/map/zones`), and an area is offered only once a check finds
nothing in it.

**Rooms have a purpose.** `build_room` takes `purpose` (bedroom, barracks,
kitchen, dining, workshop, storage, hospital, other), recorded in
`memory/rooms.json`. Each turn the prompt lists the rooms with how many wall
cells stand, what is inside, and what the purpose still needs (a bedroom
without a bed). A room that would enclose a growing zone is refused: the game
roofs a closed room, and crops under a roof get no sunlight.

**Orders are standing orders.** The prompt tells the model it manages the
colony, and colonists carry out what is already in effect by themselves while
time advances. Orders already in effect aren't offered again: the current
research project drops out of `set_research`, and `enable_work` /
`disable_work` only offer work that is off / on for that colonist. The research
line says the project carries on by itself and who has Research on.

The prompt includes a mini-map around the colony showing terrain, colonists,
trees, forbidden items, the colony's own buildings, frames and blueprints, and
the live status of every blueprint it placed, grouped so a room reads as one
line. Supplies are split the way a player sees them: the **colony stock** in
stockpiles and shelves (RimWorld's resource readout), allowed items lying
elsewhere (still usable), and forbidden ones. **Building materials** are listed
on their own. Stone chunks are called out as *not* a material, with the
stonecutter's table and bill that turn them into blocks, and only materials the
colony actually has are offered as `stuff`. A construction order that the
colony can't pay for succeeds with a warning saying what is missing and how to
get it.

Plan and policy changes in a reply are always kept, whatever its orders did. A
plan step (or the whole plan) newly marked **completed is held as in_progress
until a later turn confirms it** (`memory.held_until_confirmed`): models mark a
step done in the same reply that sends its orders, before the game has done
anything. If the next reply still says completed, having seen fresh game state,
it stands; the prompt marks a held step "you reported it done: mark it completed
again once the game shows it". Each run log records the full prompt with the
first action of every turn.

## Zones, blueprints and designations

Each step's prompt includes colonist positions, a terrain mini-map around the
colony, up to five open areas near the base (the biggest clear rectangles, up to
15x15, checked against the game; the model may use any part of one), existing
zones and buildings, and
the buildings, floors and crops it may use.

| Action | What it does | Limits |
|--------|--------------|--------|
| `create_growing_zone` | Plants one crop in a rectangle | 225 cells; every cell fertile enough for the crop; food/fibre crops only |
| `create_stockpile` | Storage for all normal items | 225 cells; not on water or marsh |
| `place_blueprint` | One building for colonists to construct | Catalog buildings only, research finished, valid material, no overlap with an existing blueprint or frame |
| `build_room` | A closed ring of walls with one centred door, in one request | 4-15 cells a side, outer (a 4x4 room is 2x2 inside); only the perimeter must be clear; walls, doors or rock already on it are kept; anything else refuses the whole room |
| `build_wall` | A straight run of walls; taken cells are skipped and named | 20 cells |
| `build_floor` | Flooring over a rectangle; floored or planned cells are skipped | 225 cells; stone tiles need Stonecutting |
| `designate` | Mine rock or harvest ripe plants in a rectangle; hunt at one animal cell | Mine: 400 cells (a bigger box would send colonists into the whole mountain). Harvest: any size. Hunt: one animal cell, x1=x2 and z1=z2. Only offered with observed targets |
| `allow_items` | Unforbid every forbidden item of a kind (`what` = all, Food or an item def) | Whole map; offered in every toolkit |
| `chop_trees` | Order the trees in a rectangle cut | Any size; the 30 trees nearest its centre per order (one request each). Trees already ordered are skipped, so repeating an order reports `unchanged` |

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
- `POST /builder/check-zone` reports only the first cell of each zone, and a
  building only at its anchor. When a field or building touches an area, the
  client checks it cell by cell, so a wall run can't cut through a field.
- `GET /map/animals` puts the z coordinate in `y`; hunting targets take their
  position from `GET /map/pawns` instead.
- `GET /map/plants?def_name=` still scans every plant, so one unfiltered read
  (about 0.6 s) is cheaper than one per tree species (about 8 s for 26).
- `GET /map/farm/summary` sends each crop's `growth_progress_average` as the
  sum of its plants' growth, so the client divides by the plant count; its
  `days_until_harvest` is always 0 and is ignored.
- `GET /resources/stored` also returns plants growing in stockpile cells and
  material already delivered into frames (at its old position), so the client
  only counts stored things that are also loose items in `GET /map/things`.
- `progress_percent` in `GET /research/progress` is a 0-1 fraction, not a
  percentage, so research looked stuck at "0%" until the client fixed it.
- `GET /map/zone/growing` reads `map_id` as the zone id, so it can't look up a
  zone; per-crop figures come from the farm summary instead.

**Tree cutting is tracked** in `memory/chops.json`. RIMAPI's harvest order
silently skips trees too young for wood and reports success without a count,
so the agent remembers which trees it ordered cut and checks them each turn:
the prompt says how many fell, how many still stand, which have stood 2+
in-game hours (most likely too young), and who has PlantCutting on.

Tests: `python -m unittest discover -s tests`

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
python -m scripts.dry_run qwen3:14b --think on            # overview, then a focus turn on its plan
python -m scripts.dry_run qwen3:14b --toolkits power,construction   # skip the overview
python -m scripts.dry_run --prompts-only                    # just print the prompts, no model

# A real run: the model plays for 20 steps.
python -m scripts.run_agent --model gpt-oss:20b --think medium --steps 20
python -m scripts.run_agent --model qwen3:14b --think on --steps 20
python -m scripts.run_agent --model qwen3:14b --think on --steps 20 --loop single   # the old loop
```

**Every turn chooses time.** The game is always paused while the model decides
and while its orders run. Then:

- **advance:** the game runs for `--run-seconds` of *game* time (default 62.5,
  measured at normal speed: 3,750 ticks, 1.5 in-game hours, about 10.4 real
  seconds at speed 3), or `--threat-run-seconds` (default 18.75) while a
  threat is active, then pauses for the next turn. The window is counted in game
  ticks, so `--speed` only changes how long you wait: RimWorld runs 60 ticks/s
  at speed 1, 180 at speed 2 and 360 at speed 3, so `--run-seconds 15 --speed 2`
  is a turn roughly every 5 real seconds. It stops early if a threat letter
  arrives or the game pauses itself, and doesn't start if a threat arrived while
  the orders ran, so the model can respond at once.
- **hold:** no time passes, for setting up more in the next turn.

There is no pause mode to forget: nothing carries over. The prompt counts
consecutive held turns and, after four, says that nothing progresses while
held. It also says why the game last paused and what happened while time last
ran. `--speed` 1-3 sets the game speed while running. Ctrl+C stops the run and
leaves the game paused.

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
