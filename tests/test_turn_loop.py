"""Rooms, per-turn choices, order execution, and the turn loop end to end.

Run from the repo root:  python -m unittest tests/test_turn_loop.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from rimagent.actions import Turn
from rimagent.agent import run
from rimagent.choices import Choices, FailureMemory
from rimagent.construction import Rect, TerrainMap, room_layout
from rimagent.execution import execute
from rimagent.memory import AgentMemory
from rimagent.rimapi import (
    AreaCheck,
    Colonist,
    GameState,
    RimApiError,
    ResearchProject,
    ResearchState,
    Skill,
    WorkPriority,
)
from rimagent.runlog import RunLogger
from tests.test_map_actions import DEFS, HEIGHT, PALETTE, WIDTH, FakeRimApi, MapTestCase, terrain_grid

ROOM_DEFS = {
    **DEFS,
    "things_defs": DEFS["things_defs"] + [{"def_name": "Door"}],
    "plant_defs": DEFS["plant_defs"] + [
        {"def_name": "Plant_TreeOak", "harvested_thing_def": "WoodLog"},
        {"def_name": "ChoppedStump", "harvested_thing_def": "WoodLog"},
    ],
}


class GridFake(FakeRimApi):
    """check-zone answered the way RIMAPI 1.10 does: only what is inside the
    asked rectangle, a building at its anchor (if any of its cells was asked
    about), and just the first matching cell of each zone."""

    def __init__(self, rock=(), buildings=(), zones=(), **kwargs):
        super().__init__(**kwargs)
        self.rock = set(rock)             # (x, z) cells of natural rock
        self.buildings = list(buildings)  # (label, Rect it covers)
        self.zones = list(zones)          # (zone_type, [cells])

    def __call__(self, method, path, **kwargs):
        if path == "/api/v1/def/all":
            self.calls.append((method, path, kwargs))
            return ROOM_DEFS
        if path != "/api/v1/builder/check-zone":
            return super().__call__(method, path, **kwargs)
        self.calls.append((method, path, kwargs))
        a, b = kwargs["json"]["point_a"], kwargs["json"]["point_b"]
        rect = Rect.from_corners(a["x"], a["z"], b["x"], b["z"])
        issues = {
            "ores": [{"x": x, "z": z, "label": "granite"} for x, z in rect if (x, z) in self.rock],
            "buildings": [{"x": area.x1, "z": area.z1, "label": label}
                          for label, area in self.buildings if any(area.contains(*c) for c in rect)],
            "zones": [],
        }
        for zone_type, cells in self.zones:
            inside = [c for c in cells if rect.contains(*c)]
            if inside:  # RIMAPI breaks after the first cell
                issues["zones"].append({"x": inside[0][0], "z": inside[0][1],
                                        "label": zone_type.lower(), "zone_type": zone_type})
        return {"issues": issues}

    def single_cell_checks(self) -> int:
        return sum(1 for m, p, k in self.calls if p == "/api/v1/builder/check-zone"
                   and k["json"]["point_a"] == k["json"]["point_b"])


def placed(fake: FakeRimApi) -> list[dict]:
    (_, _, kwargs), = [w for w in fake.writes() if w[1] == "/api/v1/builder/blueprint"]
    return kwargs["json"]["blueprint"]["buildings"]


class RoomLayoutTests(unittest.TestCase):
    def test_a_ring_with_one_centred_door(self) -> None:
        layout = room_layout(Rect(2, 2, 7, 6), "north")      # 6 x 5 outside
        self.assertEqual(len(layout), 18)                     # 2*6 + 2*5 - 4 perimeter cells
        self.assertEqual([(x, z) for name, x, z in layout if name == "Door"], [(4, 6)])
        self.assertEqual(len({(x, z) for _, x, z in layout}), 18)  # corners only once
        self.assertNotIn((4, 4), {(x, z) for _, x, z in layout})  # nothing inside

    def test_each_side_puts_the_door_on_that_side(self) -> None:
        rect = Rect(2, 2, 7, 6)
        doors = {side: next((x, z) for n, x, z in room_layout(rect, side) if n == "Door")
                 for side in ("north", "south", "east", "west")}
        self.assertEqual(doors, {"north": (4, 6), "south": (4, 2), "east": (7, 4), "west": (2, 4)})

    def test_refuses_rooms_too_small_or_too_big(self) -> None:
        with self.assertRaisesRegex(ValueError, "4..15"):
            room_layout(Rect(0, 0, 2, 5), "north")
        with self.assertRaisesRegex(ValueError, "4..15"):
            room_layout(Rect(0, 0, 15, 5), "north")


class BuildRoomTests(MapTestCase):
    def test_one_request_for_the_whole_room(self) -> None:
        fake = GridFake()
        summary = self.client_with(fake).build_room(Rect(2, 2, 7, 6), "WoodLog", "north")
        buildings = placed(fake)
        self.assertEqual(len(buildings), 18)
        self.assertEqual([(b["rel_x"], b["rel_z"]) for b in buildings if b["def_name"] == "Door"], [(2, 4)])
        self.assertEqual({b["stuff_def_name"] for b in buildings}, {"WoodLog"})
        self.assertIn("17 walls and 1 door planned, needing 110 WoodLog", summary)

    def test_what_is_inside_does_not_matter(self) -> None:
        fake = GridFake(buildings=[("bed", Rect(4, 4, 4, 5))])
        self.client_with(fake).build_room(Rect(2, 2, 7, 6), "WoodLog", "north")
        self.assertEqual(len(placed(fake)), 18)
        self.assertEqual(fake.single_cell_checks(), 0)  # no cell-by-cell sweep of the inside

    def test_existing_walls_and_rock_become_part_of_the_ring(self) -> None:
        fake = GridFake(rock=[(2, 3), (2, 5)], things_at={(7, 4): [
            {"thing_id": 1, "def_name": "Blueprint_Wall", "position": {"x": 7, "y": 0, "z": 4}}]})
        summary = self.client_with(fake).build_room(Rect(2, 2, 7, 6), "WoodLog", "north")
        cells = {(b["rel_x"] + 2, b["rel_z"] + 2) for b in placed(fake)}
        self.assertEqual(len(cells), 15)
        self.assertFalse(cells & {(2, 3), (2, 5), (7, 4)})
        self.assertIn("kept 3 existing wall/rock cells", summary)

    def test_anything_else_on_the_perimeter_refuses_the_whole_room(self) -> None:
        fake = GridFake(buildings=[("campfire", Rect(5, 2, 5, 2))])
        with self.assertRaisesRegex(ValueError, "blocked on its perimeter: campfire at \\(5,2\\).*nothing was placed"):
            self.client_with(fake).build_room(Rect(2, 2, 7, 6), "WoodLog", "north")
        self.assertEqual(fake.writes(), [])

    def test_rock_where_the_door_goes_is_refused(self) -> None:
        fake = GridFake(rock=[(4, 6)])
        with self.assertRaisesRegex(ValueError, "granite at \\(4,6\\)"):
            self.client_with(fake).build_room(Rect(2, 2, 7, 6), "WoodLog", "north")
        self.assertEqual(fake.writes(), [])

    def test_refuses_materials_that_cant_make_walls(self) -> None:
        with self.assertRaisesRegex(ValueError, "can't build walls and doors"):
            self.client_with(GridFake()).build_room(Rect(2, 2, 7, 6), "Cloth", "north")


class BlockedCellTests(MapTestCase):
    def test_a_field_is_blocked_everywhere_not_just_where_rimapi_reports_it(self) -> None:
        # check-zone names one cell per zone; the wall must still avoid all three.
        fake = GridFake(zones=[("Growing", [(2, 3), (2, 4), (2, 5)])])
        summary = self.client_with(fake).build_wall(Rect(2, 2, 2, 6), "WoodLog")
        self.assertEqual({(b["rel_x"], b["rel_z"]) for b in placed(fake)}, {(0, 0), (0, 4)})
        self.assertIn("skipped 3 cells", summary)

    def test_a_stockpile_neither_blocks_nor_triggers_a_cell_by_cell_check(self) -> None:
        fake = GridFake(zones=[("Stockpile", [(2, 3), (2, 4)])])
        self.client_with(fake).build_wall(Rect(2, 2, 2, 6), "WoodLog")
        self.assertEqual(len(placed(fake)), 5)
        self.assertEqual(fake.single_cell_checks(), 0)

    def test_a_building_anchored_outside_the_run_still_blocks_its_cells(self) -> None:
        fake = GridFake(buildings=[("table", Rect(1, 3, 3, 3))])  # anchor (1,3) is off the run
        summary = self.client_with(fake).build_wall(Rect(2, 2, 2, 6), "WoodLog")
        self.assertNotIn((0, 1), {(b["rel_x"], b["rel_z"]) for b in placed(fake)})
        self.assertIn("skipped 1 cells", summary)


class TreeTests(MapTestCase):
    def test_one_plant_read_per_snapshot_and_no_stumps(self) -> None:
        plant = lambda name, x: {"thing_id": x, "def_name": name, "position": {"x": x, "y": 0, "z": 1}}
        fake = GridFake(plants=[plant("Plant_TreeOak", 1), plant("ChoppedStump", 2), plant("Plant_Grass", 3)])
        client = self.client_with(fake)
        self.assertEqual([t.def_name for t in client.get_trees()], ["Plant_TreeOak"])
        client.get_trees()
        reads = lambda: sum(1 for _, p, _ in fake.calls if p == "/api/v1/map/plants")
        self.assertEqual(reads(), 1)
        client.begin_snapshot()
        client.get_trees()
        self.assertEqual(reads(), 2)


def colonist(name="Ada", skills=(), work=("Construction",), x=5, z=5) -> Colonist:
    return Colonist(id=1, name=name, age=30, health=1, mood=0.5, hunger=0.5,
                    position={"x": x, "z": z}, skills=list(skills),
                    work_priorities=[WorkPriority(work_type=w, priority=3) for w in work])


WORK_TYPES = ["Construction", "Research", "Growing"]
RESEARCH = ResearchState(has_bench=True, available=[ResearchProject(name="Brewing", label="brewing")],
                         all_projects=[ResearchProject(name="Brewing")])


class ChoicesTests(MapTestCase):
    def choices(self, learned=None, colonists=None) -> Choices:
        defs = self.client_with(GridFake()).get_game_defs()
        terrain = TerrainMap(WIDTH, HEIGHT, PALETTE, terrain_grid())
        pawn = colonist(skills=[Skill(name="Intellectual", level=0, totally_disabled=True)])
        return Choices.observe(colonists or [pawn], WORK_TYPES, [], RESEARCH, defs, terrain,
                               [], [], (), learned)

    def turn(self, *actions) -> Turn:
        return Turn.model_validate({"actions": list(actions), "time": "hold"})

    def test_actions_without_targets_are_left_out_and_explained(self) -> None:
        c = self.choices()
        offered = [b["properties"]["action"]["const"]
                   for b in c.schema()["properties"]["actions"]["items"]["anyOf"]]
        for missing in ("set_bill", "allow_items", "chop_trees", "designate"):
            self.assertNotIn(missing, offered)
        self.assertIn("build_room", offered)
        self.assertTrue(any(u.startswith("set_bill:") for u in c.unavailable))
        self.assertTrue(any("designate hunt" in u for u in c.unavailable))

    def test_every_branch_still_starts_with_action(self) -> None:
        for branch in self.choices().schema()["properties"]["actions"]["items"]["anyOf"]:
            self.assertEqual(next(iter(branch["properties"])), "action")

    def test_names_are_constrained_to_what_exists(self) -> None:
        branches = {b["properties"]["action"]["const"]: b
                    for b in self.choices().schema()["properties"]["actions"]["items"]["anyOf"]}
        self.assertEqual(branches["set_research"]["properties"]["project"], {"enum": ["Brewing"]})
        self.assertTrue(any("Wall" in b["properties"]["building_def"]["enum"]
                            for b in self.choices().schema()["properties"]["actions"]["items"]["anyOf"]
                            if b["properties"]["action"]["const"] == "place_blueprint"))
        self.assertNotIn("Wall (wall)", branches["place_blueprint"]["properties"]["building_def"]["enum"])
        self.assertEqual(branches["build_room"]["properties"]["x2"]["maximum"], WIDTH - 1)

    def test_validate_catches_what_the_schema_cant(self) -> None:
        c = self.choices()
        bad = [
            {"action": "place_blueprint", "building_def": "Wall (wall)", "x": 1, "z": 1,
             "stuff": "WoodLog", "reason": "label as a name"},
            {"action": "set_research", "project": "Brewin", "reason": "typo"},
            {"action": "enable_work", "colonist": "Ada", "work_type": "Research",
             "reason": "Intellectual is disabled"},
            {"action": "place_blueprint", "building_def": "Wall", "x": 1, "z": 1,
             "stuff": None, "reason": "a wall needs a material"},
            {"action": "build_room", "x1": 2, "z1": 2, "x2": WIDTH, "z2": 6, "stuff": "WoodLog",
             "door_side": "north", "reason": "off the map"},
        ]
        for action in bad:
            with self.subTest(action=action["reason"]), self.assertRaises(ValueError):
                c.validate(self.turn(action))
        c.validate(self.turn({"action": "build_room", "x1": 2, "z1": 2, "x2": 7, "z2": 6,
                              "stuff": "WoodLog", "door_side": "north", "reason": "fine"}))

    def test_refused_work_is_removed_for_that_colonist(self) -> None:
        c = self.choices(learned={"Ada": {"Growing"}})
        self.assertEqual(c.work["Ada"], {"Construction"})


class FailureMemoryTests(unittest.TestCase):
    def action(self, project="Brewing"):
        return Turn.model_validate({"actions": [{"action": "set_research", "project": project,
                                                 "reason": "x"}], "time": "hold"}).actions[0]

    def test_keeps_one_entry_per_target_and_at_most_six(self) -> None:
        memory = FailureMemory()
        memory.remember(self.action(), "first", "f")
        memory.remember(self.action(), "second", "f")
        self.assertEqual(len(memory.lines()), 1)
        self.assertIn("second", memory.lines()[0])
        for n in range(10):
            memory.remember(self.action(f"P{n}"), "no", "f")
        self.assertEqual(len(memory.lines()), 6)

    def test_forgets_a_failure_once_its_state_changes(self) -> None:
        memory = FailureMemory()
        memory.remember(self.action("A"), "no bench", "before")
        memory.remember(self.action("B"), "no bench", "before")
        memory.reconcile(lambda target: "after" if target["project"] == "A" else "before")
        self.assertEqual(len(memory.lines()), 1)
        self.assertIn("'B'", memory.lines()[0])


class WorkClient:
    """Enough of RimApiClient for execute's work orders."""

    def __init__(self, pawn: Colonist, refuse: str | None = None):
        self.pawn, self.refuse, self.writes = pawn, refuse, []
        self.submitted_blueprints = []

    def begin_snapshot(self) -> None:
        pass

    def get_colonists(self) -> list[Colonist]:
        return [self.pawn]

    def enable_work(self, pawn_id: int, work_type: str) -> None:
        self.writes.append(("enable", work_type))
        if self.refuse:
            raise RimApiError(self.refuse)
        self.pawn.work_priorities.append(WorkPriority(work_type=work_type, priority=3))


def work_order(work_type: str):
    return Turn.model_validate({"actions": [{"action": "enable_work", "colonist": "Ada",
                                             "work_type": work_type, "reason": "x"}],
                                "time": "hold"}).actions[0]


class ExecuteWorkTests(unittest.TestCase):
    def test_already_enabled_is_unchanged_and_writes_nothing(self) -> None:
        client = WorkClient(colonist())
        result = execute(client, work_order("Construction"), None, None, None)
        self.assertTrue(result.startswith("unchanged:"))
        self.assertEqual(client.writes, [])

    def test_a_change_is_read_back(self) -> None:
        client = WorkClient(colonist())
        self.assertIn("verified work setting", execute(client, work_order("Growing"), None, None, None))

    def test_a_refusal_teaches_the_incapability(self) -> None:
        client = WorkClient(colonist(), refuse="Cannot set priority for disabled work type Growing on pawn Ada")
        learned: dict[str, set[str]] = {}
        with self.assertRaisesRegex(ValueError, "Ada is incapable of Growing"):
            execute(client, work_order("Growing"), None, None, None, learned)
        self.assertEqual(learned, {"Ada": {"Growing"}})

    def test_other_errors_are_not_mistaken_for_incapability(self) -> None:
        client = WorkClient(colonist(), refuse="Pawn not found")
        learned: dict[str, set[str]] = {}
        with self.assertRaisesRegex(RimApiError, "Pawn not found"):
            execute(client, work_order("Growing"), None, None, None, learned)
        self.assertEqual(learned, {})


class LoopClient:
    """A tiny paused colony: everything the loop reads, and a record of writes."""

    def __init__(self):
        self.pawn = colonist()
        self.tick, self.paused, self.calls = 1000, True, []
        self.submitted_blueprints = []
        self.defs = GridFake()
        self._defs_client = None

    # time
    def pause(self): self.paused = True; self.calls.append("pause")
    def resume(self, speed=1): self.paused = False; self.calls.append("resume")

    def get_state(self):
        if not self.paused:
            self.tick += 100
        return GameState(game_tick=self.tick, colony_wealth=1000, colonist_count=1,
                         storyteller="Cassandra", is_paused=self.paused)

    # reads
    def begin_snapshot(self): pass
    def get_colonists(self): return [self.pawn]
    def get_alerts(self): return []
    def get_threats(self): return []
    def get_work_types(self): return WORK_TYPES
    def get_work_tables(self): return []
    def get_research(self): return RESEARCH
    def get_terrain(self, map_id=0): return TerrainMap(WIDTH, HEIGHT, PALETTE, terrain_grid())
    def get_items(self, map_id=0): return []
    def get_trees(self, map_id=0): return []
    def get_plants(self, map_id=0): return []
    def get_buildings(self, map_id=0): return []
    def get_zones(self, map_id=0): return []
    def get_things_at(self, x, z, map_id=0): return []
    def check_area(self, rect, map_id=0): return AreaCheck()
    def _request(self, method, path, **kwargs): return None

    def get_game_defs(self):
        from rimagent.rimapi import RimApiClient
        if self._defs_client is None:
            self._defs_client = RimApiClient(base_url="http://example.invalid")
            self._defs_client._request = self.defs
        return self._defs_client.get_game_defs()

    # writes
    def enable_work(self, pawn_id, work_type):
        self.calls.append(f"enable {work_type}")
        self.pawn.work_priorities.append(WorkPriority(work_type=work_type, priority=3))

    def set_research(self, project):
        self.calls.append(f"research {project}")
        raise RimApiError("no research bench can reach it")


class ScriptedModel:
    """Replies in order, remembering each prompt, like OllamaModel with a schema."""

    def __init__(self, replies):
        self.replies, self.prompts, self.schema = list(replies), [], {}

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(self.replies.pop(0))


class RunLoopTests(unittest.TestCase):
    def test_orders_stop_at_a_failure_and_time_follows_each_choice(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        client = LoopClient()
        model = ScriptedModel([
            {"actions": [
                {"action": "enable_work", "colonist": "Ada", "work_type": "Construction", "reason": "a"},
                {"action": "set_research", "project": "Brewing", "reason": "b"},
                {"action": "enable_work", "colonist": "Ada", "work_type": "Growing", "reason": "c"},
            ], "time": "hold"},
            {"actions": [], "time": "advance"},
        ])
        run(client, model, RunLogger(tmp / "logs"), max_steps=2, run_ticks=60, threat_run_ticks=60,
            memory=AgentMemory(tmp / "memory"))

        # Construction was already on (no write); research failed; Growing was skipped.
        self.assertIn("research Brewing", client.calls)
        self.assertNotIn("enable Construction", client.calls)
        self.assertNotIn("enable Growing", client.calls)
        # The hold kept time stopped; the advance ran it once, then paused again.
        self.assertEqual(client.calls.count("resume"), 1)
        self.assertEqual(client.calls[-1], "pause")
        self.assertGreater(client.tick, 1000)

        second = model.prompts[1]
        self.assertIn("unchanged:", second)
        self.assertIn("FAILED {'action': 'set_research', 'project': 'Brewing'}", second)
        self.assertIn("skipped: an earlier order failed", second)
        self.assertIn("Unresolved:", second)
        self.assertIn("Consecutive held turns: 1.", second)
        # The model was handed this turn's schema, not the static one.
        self.assertIn("anyOf", json.dumps(model.schema))

        log = [json.loads(line) for line in next((tmp / "logs" / "runs").iterdir()).joinpath("run.jsonl")
               .read_text(encoding="utf-8").splitlines()]
        ends = [r["data"]["time"] for r in log if r["data"].get("action") == "turn_end"]
        self.assertEqual(ends, ["hold", "advance"])

    def test_two_invalid_replies_hold_with_no_orders(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        client = LoopClient()
        model = ScriptedModel([{"actions": [], "time": "pause"}, {"oops": 1}])
        run(client, model, RunLogger(tmp / "logs"), max_steps=1, run_ticks=60, threat_run_ticks=60,
            memory=AgentMemory(tmp / "memory"))
        self.assertNotIn("resume", client.calls)
        self.assertEqual(client.tick, 1000)


if __name__ == "__main__":
    unittest.main()
