"""Hints on order results, room purposes, and the one-at-a-time guard.

Run from the repo root:  python -m unittest tests/test_hints.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rimagent.actions import Turn
from rimagent.agent import room_state, run
from rimagent.blueprints import TrackedBlueprint
from rimagent.construction import Rect
from rimagent.execution import execute
from rimagent.hints import TurnFacts, failure_hint, success_hint
from rimagent.memory import AgentMemory
from rimagent.rimapi import Building, MapThing
from rimagent.rooms import RoomTracker
from rimagent.runlog import RunLogger
from tests.test_map_actions import MapTestCase
from tests.test_turn_loop import GridFake, LoopClient, ScriptedModel, colonist

SPOTS = [("A", Rect(10, 10, 12, 12), "3x3, fertile, crops can grow here"),
         ("B", Rect(20, 20, 24, 24), "5x5, no crops (poor soil)")]


def order(**fields):
    return Turn.model_validate({"actions": [{"reason": "x", **fields}], "time": "hold"}).actions[0]


def stack(def_name, count, thing_id=1):
    return MapThing(thing_id=thing_id, def_name=def_name, position={"x": 1, "z": 1}, stack_count=count)


def facts(**kwargs) -> TurnFacts:
    base = dict(spots=SPOTS, colonists=[colonist(work=("Construction", "Growing"))], anchor=(15, 15))
    return TurnFacts(**{**base, **kwargs})


BENCH = dict(action="place_blueprint", building_def="TableStonecutter", x=21, z=21, rotation=0, stuff="WoodLog")


class FailureHintTests(unittest.TestCase):
    def test_its_own_blueprint_is_nothing_to_fix(self) -> None:
        # The loop that planned ten stonecutter's tables started with this error.
        hint, keep = failure_hint(order(**BENCH), "TableStonecutter at (21,21) would cover (21,21), "
                                  "where TableStonecutter is already planned", facts())
        self.assertIn("is your own, from an earlier turn", hint)
        self.assertIn("placing another makes a second one, not a move", hint)
        self.assertFalse(keep)  # not left in the prompt as "unresolved"

    def test_something_else_in_the_way_gets_a_worked_spot(self) -> None:
        hint, _ = failure_hint(order(**BENCH), "TableStonecutter at (5,5) facing 0 covers (4,5)-(6,5), "
                               "which is blocked by wall at (4,5)", facts())
        self.assertEqual(hint, "Try TableStonecutter at (11,10) facing north covers (10,10)-(12,10), "
                               "inside free spot A.")

    def test_a_blocked_room_gets_spots_big_enough(self) -> None:
        room = order(action="build_room", x1=1, z1=1, x2=6, z2=6, stuff="WoodLog", door_side="north",
                     purpose="bedroom")
        hint, _ = failure_hint(room, "room (1,1)-(6,6) is blocked on its perimeter: bed at (3,1)", facts())
        self.assertIn("B (20,20)-(24,24)", hint)
        self.assertNotIn("A (10,10)", hint)  # 3x3 is too small for a room

    def test_build_wall_never_removes(self) -> None:
        wall = order(action="build_wall", x1=1, z1=1, x2=1, z2=4, stuff="WoodLog")
        hint, keep = failure_hint(wall, "every cell in (1,1)-(1,4) is taken: wall at (1,1)", facts())
        self.assertIn("it never removes anything", hint)
        self.assertFalse(keep)

    def test_missing_targets_name_the_nearest(self) -> None:
        mine = order(action="designate", designation="mine", x1=0, z1=0, x2=3, z2=3)
        hint, _ = failure_hint(mine, "no observed mine target in (0,0)-(3,3)",
                               facts(targets={"mine": [{"name": "granite", "x": 40, "z": 40},
                                                       {"name": "granite", "x": 16, "z": 15}]}))
        self.assertTrue(hint.startswith("Nearest mine targets: granite at (16,15); granite at (40,40)"))


class SuccessHintTests(unittest.TestCase):
    def test_a_second_bench_says_so_and_what_it_costs(self) -> None:
        first = TrackedBlueprint(def_name="TableStonecutter", stuff="WoodLog", x=11, z=10, cells=(10, 10, 12, 10))
        f = facts(report=[(first, "waiting")], items=[stack("WoodLog", 100), stack("Steel", 50, 2)])
        submitted = [dict(def_name="TableStonecutter", stuff="WoodLog", x=21, z=21)]
        hint = success_hint(order(**BENCH), "verified 1 construction orders", submitted, f)
        self.assertIn("You now have 2 TableStonecutter planned and 0 built: more than one is rarely needed early", hint)
        self.assertIn("It needs 75 WoodLog, 30 Steel", hint)
        self.assertIn("WoodLog short by 50 (chop trees", hint)
        self.assertIn("Steel short by 10", hint)
        self.assertIn("Construction is on for Ada", hint)

    def test_a_room_says_what_to_put_inside_next(self) -> None:
        room = order(action="build_room", x1=20, z1=20, x2=24, z2=24, stuff="WoodLog", door_side="south",
                     purpose="bedroom")
        submitted = [dict(def_name="Wall", stuff="WoodLog", x=20, z=20)]
        hint = success_hint(room, "verified", submitted, facts(items=[stack("WoodLog", 500)]))
        self.assertIn("The inside is (21,21)-(23,23); next for a bedroom: place a Bed inside", hint)

    def test_nothing_changed_gets_no_hint(self) -> None:
        self.assertEqual(success_hint(order(**BENCH), "unchanged: x", [], facts()), "")


def building(def_name, x, z, size=(1, 1), frame=False):
    return Building(id=x * 1000 + z, def_name=("Frame_" if frame else "") + def_name,
                    position={"x": x, "z": z}, size={"x": size[0], "z": size[1]},
                    type="Frame" if frame else "Building")


class RoomTrackerTests(unittest.TestCase):
    RECT = Rect(0, 0, 4, 4)  # 16 perimeter cells, inside (1,1)-(3,3)

    def test_reports_walls_contents_and_what_is_missing(self) -> None:
        rooms = RoomTracker()
        rooms.add(self.RECT, "bedroom", "south")
        walls = [building("Wall", x, z) for x, z in self.RECT if x in (0, 4) or z in (0, 4)][:10]
        line, = rooms.describe(*room_state(walls, []))
        self.assertIn("bedroom (0,0)-(4,4) (inside (1,1)-(3,3)): 10/16 wall cells built", line)
        self.assertIn("next: place a Bed inside", line)

    def test_a_closed_furnished_room(self) -> None:
        rooms = RoomTracker()
        rooms.add(self.RECT, "bedroom", "south")
        ring = [building("Wall", x, z) for x, z in self.RECT if x in (0, 4) or z in (0, 4)]
        bed = TrackedBlueprint(def_name="Bed", stuff="WoodLog", x=2, z=2, cells=(2, 2, 2, 3))
        line, = rooms.describe(*room_state(ring, [(bed, "waiting")]))
        self.assertIn("16/16 wall cells built, closed, so the game roofs it", line)
        self.assertIn("inside: Bed (planned)", line)
        self.assertNotIn("next:", line)

    def test_a_room_whose_walls_are_gone_is_forgotten(self) -> None:
        path = Path(tempfile.mkdtemp()) / "rooms.json"
        rooms = RoomTracker(path)
        rooms.add(self.RECT, "kitchen", "north")
        self.assertEqual(rooms.describe({}, []), [])
        self.assertEqual(RoomTracker(path).items, [])


class RoomFieldTests(MapTestCase):
    def test_a_room_around_a_field_is_refused(self) -> None:
        fake = GridFake(zones=[("Growing", [(4, 4), (5, 4)])])
        with self.assertRaisesRegex(ValueError, "would enclose a growing zone.*no sunlight; nothing was placed"):
            self.client_with(fake).build_room(Rect(2, 2, 7, 6), "WoodLog", "north")
        self.assertEqual(fake.writes(), [])


class OneAtATimeTests(unittest.TestCase):
    def test_a_second_unbuilt_bench_is_refused(self) -> None:
        first = TrackedBlueprint(def_name="TableStonecutter", stuff="WoodLog", x=11, z=10, cells=(10, 10, 12, 10))
        tracker = SimpleNamespace(check=lambda client, forget_finished=True: [(first, "waiting")])
        client = SimpleNamespace(begin_snapshot=lambda: None, submitted_blueprints=[],
                                 construction_at=lambda *a, **k: False)
        with self.assertRaisesRegex(ValueError, r"your TableStonecutter at \(11,10\) is still unbuilt"):
            execute(client, order(**BENCH), tracker, None, None)


class PlacingClient(LoopClient):
    """A colony where the campfire spot is already taken by its own blueprint."""

    def construction_at(self, *args, **kwargs):
        return False  # not an exact repeat: a different spot, still blocked

    def place_blueprint(self, def_name, x, z, rotation=0, stuff=None, map_id=0):
        raise ValueError(f"{def_name} at ({x},{z}) would cover ({x},{z}), where {def_name} is already planned")


class LoopHintTests(unittest.TestCase):
    def test_hints_reach_the_next_prompt_and_nothing_to_fix_is_not_unresolved(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        model = ScriptedModel([
            {"actions": [{"action": "place_blueprint", "building_def": "Campfire", "x": 3, "z": 3,
                          "rotation": 0, "stuff": None, "reason": "cook"}], "time": "hold"},
            {"actions": [], "time": "hold"},
        ])
        run(PlacingClient(), model, RunLogger(tmp / "logs"), max_steps=2, memory=AgentMemory(tmp / "memory"))
        second = model.prompts[1]
        self.assertIn("HINT: That Campfire is your own, from an earlier turn", second)
        self.assertNotIn("Unresolved:", second)
        log = [json.loads(line) for line in next((tmp / "logs" / "runs").iterdir())
               .joinpath("run.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any("HINT:" in (r["data"].get("error") or "") for r in log if r["kind"] == "decision"))


if __name__ == "__main__":
    unittest.main()


def job(name, current_job, work=("Construction",)):
    pawn = colonist(work=work)
    return pawn.model_copy(update={"name": name, "current_job": current_job})


class IdleTests(unittest.TestCase):
    """Colonists wandering while work sat unassigned (GotoWander and Wait_Wander
    were the two most common jobs across the logged runs)."""

    def test_counts_consecutive_idle_turns_and_resets_on_work(self) -> None:
        from rimagent.hints import IdleTracker
        idle = IdleTracker()
        idle.update([job("Ada", "GotoWander"), job("Bo", "SocialRelax")])
        turns = idle.update([job("Ada", "Wait_Wander"), job("Bo", "")])
        self.assertEqual(turns, {"Ada": 2, "Bo": 1})  # recreation isn't idle
        self.assertEqual(idle.update([job("Ada", "CutPlant"), job("Bo", "Wait")]), {"Ada": 0, "Bo": 2})

    def test_one_idle_turn_is_not_reported(self) -> None:
        from rimagent.hints import idle_hint
        self.assertIsNone(idle_hint({"Ada": 1}, facts(), [("Construction", "3 blueprints", False)], {}))

    def test_compares_pending_work_with_who_has_it_on(self) -> None:
        from rimagent.hints import idle_hint
        f = facts(colonists=[job("Ada", "GotoWander", ("Construction",)),
                             job("Bo", "CutPlant", ("PlantCutting", "Construction"))])
        pending = [("Construction", "4 blueprints or frames", False),
                   ("PlantCutting", "12 trees ordered cut", False),
                   ("Research", "Batteries", False)]
        hint = idle_hint({"Ada": 3, "Bo": 0}, f, pending,
                         {"Ada": {"Construction", "PlantCutting"}, "Bo": {"PlantCutting", "Construction"}})
        self.assertTrue(hint.startswith("HINT: Ada (3 turns) wandering or idle."))
        self.assertIn("Construction: 4 blueprints or frames, on for Ada, Bo;", hint)
        self.assertIn("PlantCutting: 12 trees ordered cut, on for Bo;", hint)
        self.assertIn("Research: Batteries, on for nobody;", hint)
        self.assertIn("Ada could take PlantCutting (enable_work)", hint)  # not Research: incapable

    def test_blocked_work_explains_the_idling(self) -> None:
        from rimagent.hints import idle_hint
        hint = idle_hint({"Ada": 2}, facts(colonists=[job("Ada", "Wait_Wander")]),
                         [("Construction", "5 blueprints or frames, short of Steel", True)], {"Ada": {"Construction"}})
        self.assertIn("(blocked for now)", hint)
        self.assertIn("All pending work is blocked: unblock it (materials) or give new work.", hint)

    def test_nothing_pending_asks_for_work(self) -> None:
        from rimagent.hints import idle_hint
        hint = idle_hint({"Ada": 2}, facts(), [], {})
        self.assertIn("nothing is pending. Give them work", hint)

    def test_pending_work_is_read_from_the_colony(self) -> None:
        from rimagent.hints import pending_work
        from rimagent.rimapi import ResearchProject, ResearchState, WorkTable, Zone
        wall = TrackedBlueprint(def_name="Wall", stuff="Steel", x=1, z=1, cells=(1, 1, 1, 1))
        stove = WorkTable(id=7, thing_def="FueledStove", label="fueled stove", position={"x": 1, "z": 1},
                          bills=[{"load_id": 1, "recipe_def_name": "CookMealSimple"}])
        pending = pending_work(facts(report=[(wall, "waiting")]),
                               ResearchState(has_bench=True, current=ResearchProject(name="Batteries")),
                               [stove], [Zone(id=1, label="field", type="Zone_Growing", cells_count=9)],
                               chop_orders=6, short={"Steel"})
        self.assertEqual(pending, [
            ("Construction", "1 blueprints or frames, short of Steel", True),
            ("PlantCutting", "6 trees ordered cut", False),
            ("Growing", "1 growing zones", False),
            ("Research", "Batteries", False),
            ("Cooking", "bills at fueled stove 7", False),
        ])


class LoopIdleTests(unittest.TestCase):
    def test_the_hint_appears_after_two_idle_turns(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        model = ScriptedModel([{"actions": [], "time": "hold"}] * 3)
        run(LoopClient(), model, RunLogger(tmp / "logs"), max_steps=3, memory=AgentMemory(tmp / "memory"))
        self.assertNotIn("wandering or idle", model.prompts[0])
        self.assertIn("HINT: Ada (2 turns) wandering or idle", model.prompts[1])
        self.assertIn("HINT: Ada (3 turns) wandering or idle", model.prompts[2])


class IncapableHintTests(unittest.TestCase):
    """A hint that read "Colonists without Handling on who may do it: Four Eyes"
    named neither who failed nor whether Four Eyes could do it."""

    def order(self):
        return order(action="enable_work", colonist="Ada", work_type="Handling")

    def test_names_who_cant_and_who_can_with_the_order(self) -> None:
        f = facts(colonists=[job("Ada", "Wait_Wander"), job("Bo", "Sow", ("Growing",))])
        hint, keep = failure_hint(self.order(), "Ada is incapable of Handling", f)
        self.assertEqual(hint, "Ada can never do Handling, so don't ask again. Bo can do it and has it off: "
                               "e.g. enable_work colonist=Bo work_type=Handling.")
        self.assertFalse(keep)

    def test_someone_who_already_does_it_is_named_instead(self) -> None:
        f = facts(colonists=[job("Ada", ""), job("Bo", "Sow", ("Handling",))])
        hint, _ = failure_hint(self.order(), "Ada is incapable of Handling", f)
        self.assertIn("Handling is already on for Bo.", hint)
        self.assertNotIn("enable_work", hint)

    def test_colonists_who_cant_either_are_not_suggested(self) -> None:
        from rimagent.rimapi import Skill
        no_animals = job("Bo", "Sow", ()).model_copy(update={"skills": [
            Skill(name="Animals", level=0, totally_disabled=True)]})
        refused = job("Cy", "Sow", ())
        f = facts(colonists=[job("Ada", ""), no_animals, refused], learned={"Cy": {"Handling"}})
        hint, _ = failure_hint(self.order(), "Ada is incapable of Handling", f)
        self.assertEqual(hint, "Ada can never do Handling, so don't ask again. Nobody else can do Handling either.")


class StorageHintTests(unittest.TestCase):
    def test_full_storage_suggests_a_stockpile_and_shelves(self) -> None:
        from rimagent.hints import storage_hint
        hint = storage_hint({"total_stockpiles": 2, "total_cells": 47, "used_cells": 47}, shelf_buildable=True)
        self.assertTrue(hint.startswith("HINT: Storage is full (47/47 cells used in 2 stockpiles)"))
        self.assertIn("create_stockpile on open ground", hint)
        self.assertIn("place_blueprint Shelf (2x1)", hint)

    def test_nearly_full_counts_and_shelves_may_need_research(self) -> None:
        from rimagent.hints import storage_hint
        hint = storage_hint({"total_stockpiles": 1, "total_cells": 20, "used_cells": 18}, shelf_buildable=False)
        self.assertIn("Storage is nearly full (18/20", hint)
        self.assertIn("once ComplexFurniture is researched", hint)

    def test_room_to_spare_or_no_stockpiles_is_no_hint(self) -> None:
        from rimagent.hints import storage_hint
        self.assertIsNone(storage_hint({"total_cells": 47, "used_cells": 7}, True))
        self.assertIsNone(storage_hint({"total_cells": 0, "used_cells": 0}, True))
        self.assertIsNone(storage_hint(None, True))
