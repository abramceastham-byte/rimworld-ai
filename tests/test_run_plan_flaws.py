"""Fixes from the 2026-09-30 runs (qwen3:14b for 35 steps, qwen3:235b for 3): plans
whose finish checks could never stay met, nine beds for three colonists, a bedroom
too big for the wood, and a half-built room dropped for good after an emergency.

Run from the repo root:  python -m unittest tests/test_run_plan_flaws.py -v
"""

import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from rimagent.actions import Turn
from rimagent.construction import Rect
from rimagent.finish import FOOD, FinishCheck, FinishState, finish_progress, unreachable_count
from rimagent.layout import room_refusal
from rimagent.modes import CORE_ACTIONS, allowed_actions
from rimagent.rimapi import MapThing


def overview(*checks, toolkit="storage"):
    return json.dumps({"goal": "Free the supplies", "uses": "forbidden meals", "toolkits": [toolkit],
                       "finish": list(checks)})


def forbidden(def_name: str, count: int) -> MapThing:
    return MapThing(thing_id=hash(def_name) % 1000, def_name=def_name, position={"x": 0, "z": 0},
                    stack_count=count, is_forbidden=True)


class FinishCheckTests(unittest.TestCase):
    def test_allowed_is_met_once_nothing_of_it_is_forbidden(self) -> None:
        check = FinishCheck(kind="allowed", what=FOOD)
        self.assertEqual(finish_progress([check], FinishState(forbidden={"MealSurvivalPack": 7}))[0],
                         ["- no Food left forbidden: not yet"])
        self.assertTrue(finish_progress([check], FinishState(forbidden={"Steel": 900}))[1])
        everything = FinishCheck(kind="allowed", what="all")
        self.assertFalse(finish_progress([everything], FinishState(forbidden={"Steel": 900}))[1])

    def test_counts_above_a_thousand_are_kept(self) -> None:
        # le=1000 let constrained decoding turn the colony's 1170 Steel into 117.
        self.assertEqual(FinishCheck(kind="have_items", what="Steel", count=1170).count, 1170)

    def test_a_building_is_not_an_item(self) -> None:
        with self.assertRaisesRegex(ValidationError, "use built what=Bed"):
            FinishCheck(kind="have_items", what="Bed", count=3)

    def test_food_cannot_end_a_plan_that_only_frees_it(self) -> None:
        # Two plans waited on "57 MealSurvivalPack usable" while the colonists ate them.
        why = unreachable_count(FinishCheck(kind="have_items", what=FOOD, count=57), ["storage"], 57)
        self.assertIn("colonists eat food", why)
        self.assertIsNone(unreachable_count(FinishCheck(kind="have_items", what=FOOD, count=20), ["production"], 0))

    def test_more_than_the_map_holds_needs_a_toolkit_that_makes_it(self) -> None:
        check = FinishCheck(kind="have_items", what="Steel", count=2000)
        self.assertIn("only 1170 Steel exist", unreachable_count(check, ["storage"], 1170))
        self.assertIsNone(unreachable_count(check, ["storage", "wood_mining"], 1170))  # mining makes more
        self.assertIsNone(unreachable_count(FinishCheck(kind="have_items", what="Steel", count=300), ["storage"], 1170))

    def test_the_overview_is_asked_again_with_the_reason(self) -> None:
        from rimagent.agent import request_overview

        model = Mock(side_effect=[overview({"kind": "have_items", "what": "MealSurvivalPack", "count": 57}),
                                  overview({"kind": "allowed", "what": "food"})])
        result, _ = request_overview(model, "prompt", Mock(), 0, frozenset({"MealSurvivalPack"}),
                                     [forbidden("MealSurvivalPack", 57)])
        self.assertIn("colonists eat food", model.call_args_list[1].args[0])
        self.assertEqual([(c.kind, c.what) for c in result.finish], [("allowed", FOOD)])


class RoomCheckTests(unittest.TestCase):
    def test_bedroom_and_barracks_count_each_other(self) -> None:
        # "1 bedroom room closed" read 0 of 1 for twelve turns: the room was built as a barracks.
        state = FinishState(rooms_closed={"barracks": 1}, rooms_floored={"bedroom": 1})
        self.assertTrue(finish_progress([FinishCheck(kind="room_closed", what="bedroom")], state)[1])
        self.assertTrue(finish_progress([FinishCheck(kind="floored", what="barracks")], state)[1])
        self.assertFalse(finish_progress([FinishCheck(kind="room_closed", what="kitchen")], state)[1])

    def test_a_floor_plan_can_end(self) -> None:
        from rimagent.agent import measure_finish
        from rimagent.construction import TerrainMap
        from rimagent.rooms import RoomTracker

        rooms = RoomTracker()
        rooms.items = []
        rooms.add(Rect(0, 0, 3, 3), "barracks", "north")
        bare = TerrainMap(4, 4, ["Soil"], [16, 0])
        floored = TerrainMap(4, 4, ["Soil", "WoodPlankFloor"], [5, 0, 2, 1, 2, 0, 2, 1, 5, 0])
        check = FinishCheck(kind="floored", what="bedroom")
        self.assertFalse(finish_progress([check], measure_finish([], [], rooms, [], [], [], bare))[1])
        self.assertTrue(finish_progress([check], measure_finish([], [], rooms, [], [], [], floored))[1])


class AllowEverywhereTests(unittest.TestCase):
    def test_every_toolkit_can_allow(self) -> None:
        # A construction focus was told to allow its forbidden wood and couldn't.
        self.assertIn("allow_items", CORE_ACTIONS)
        self.assertIn("allow_items", allowed_actions(["construction"]))


class BedTests(unittest.TestCase):
    def test_beds_stop_being_offered_once_everyone_has_one(self) -> None:
        from rimagent.choices import Choices

        choices = Choices(materials={"Bed": ["WoodLog"], "DoubleBed": ["WoodLog"], "Wall": ["WoodLog"]},
                          fields={"place_blueprint": {"building_def": ["Bed", "DoubleBed", "Wall"],
                                                      "stuff": ["WoodLog", None]}})
        choices.withhold({"Bed", "DoubleBed"}, "beds for 3 are built or planned")
        self.assertEqual(choices.fields["place_blueprint"]["building_def"], ["Wall"])
        self.assertNotIn("Bed", choices.materials)
        self.assertIn("place_blueprint Bed / DoubleBed: beds for 3 are built or planned", choices.unavailable)
        schema = json.dumps(choices.schema({"place_blueprint"}, Turn))
        self.assertNotIn('"Bed"', schema)


class BedroomSizeTests(unittest.TestCase):
    def room(self, x2: int, z2: int):
        return Turn.model_validate({"actions": [{"action": "build_room", "reason": "x", "x1": 0, "z1": 0, "x2": x2,
                                                 "z2": z2, "stuff": "WoodLog", "door_side": "north",
                                                 "purpose": "bedroom"}], "time": "hold"}).actions[0]

    def test_a_bedroom_far_too_big_is_refused_with_a_size(self) -> None:
        # 12x12 outer for 3 colonists: 100 cells inside against 36 recommended.
        why = room_refusal(self.room(11, 11), [], [], None, colonists=3)
        self.assertIn("100 cells inside, more than twice the 36 recommended", why)
        self.assertIn("about 8x8 outer (6x6 inside)", why)
        self.assertIsNone(room_refusal(self.room(7, 7), [], [], None, colonists=3))
        self.assertIsNone(room_refusal(self.room(11, 11), [], [], None))  # colonists unknown: no limit


class InterruptedPlanTests(unittest.TestCase):
    def test_the_next_calm_overview_offers_the_cut_short_plan(self) -> None:
        from rimagent.rimapi import Alert
        from tests.test_modes import OVERVIEW, ModesLoopTests
        from tests.test_turn_loop import LoopClient

        client = LoopClient()
        alerts = iter([[], [Alert(label="Starvation")], []])
        client.get_alerts = lambda: next(alerts, [])
        focus = {"actions": [], "time": "hold", "plan": "continue"}
        done = {"actions": [], "time": "hold", "plan": "replan"}
        model, _ = ModesLoopTests.run_loop(self, client, [
            {**OVERVIEW, "goal": "Build the bedroom"}, focus,   # step 0
            {**OVERVIEW, "goal": "Feed Ada"}, done,             # step 1: the emergency cuts it short
            {**OVERVIEW, "goal": "Back to the bedroom"}, focus,  # step 2: calm again
        ], steps=3)
        emergency, calm = model.prompts[2], model.prompts[4]
        self.assertNotIn("Unfinished plan an emergency cut short", emergency)
        self.assertIn("Unfinished plan an emergency cut short: Build the bedroom [work]", calm)


if __name__ == "__main__":
    unittest.main()
