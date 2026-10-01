"""The layout plan (plan_area and its hints) and room slots beside existing rooms.

Run from the repo root:  python -m unittest tests/test_layout.py -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rimagent.actions import Turn
from rimagent.construction import Rect
from rimagent.execution import execute
from rimagent.layout import LayoutTracker, layout_hint, room_slots
from rimagent.modes import allowed_actions


def order(**fields):
    return Turn.model_validate({"actions": [dict(reason="test", **fields)], "time": "hold"}).actions[0]


def tracker() -> LayoutTracker:
    return LayoutTracker(Path(tempfile.mkdtemp()) / "layout.json")


class LayoutPlanTests(unittest.TestCase):
    def test_reserve_move_and_survive_a_restart(self) -> None:
        layout = tracker()
        self.assertIn("workshop area reserved", layout.plan("workshop", Rect(10, 10, 20, 18)))
        self.assertIn("unchanged", layout.plan("workshop", Rect(10, 10, 20, 18)))
        self.assertIn("moved", layout.plan("workshop", Rect(30, 10, 40, 18)))
        again = LayoutTracker(layout.path)
        self.assertEqual(again.get("workshop").area(), Rect(30, 10, 40, 18))

    def test_areas_of_different_purposes_cannot_overlap(self) -> None:
        layout = tracker()
        layout.plan("fields", Rect(0, 0, 14, 14))
        with self.assertRaisesRegex(ValueError, "overlaps your fields area"):
            layout.plan("bedrooms", Rect(10, 10, 20, 20))
        with self.assertRaisesRegex(ValueError, "at most 30"):
            layout.plan("storage", Rect(40, 40, 80, 50))

    def test_described_with_what_stands_in_each_area(self) -> None:
        layout = tracker()
        self.assertIn("none yet", tracker().describe([])[0])
        layout.plan("workshop", Rect(10, 10, 20, 18))
        lines = layout.describe([("TableButcher", Rect(12, 12, 14, 12), "planned"),
                                 ("Bed", Rect(50, 50, 50, 51), "built")])
        self.assertIn("workshop (10,10)-(20,18): TableButcher (planned)", lines[1])

    def test_execute_plans_an_area(self) -> None:
        layout = tracker()
        result = execute(Mock(), order(action="plan_area", purpose="storage", x1=5, z1=5, x2=12, z2=10),
                         None, None, None, layout=layout)
        self.assertIn("storage area reserved", result)


class LayoutHintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = tracker()
        self.layout.plan("kitchen_dining", Rect(10, 10, 20, 20))
        self.layout.plan("fields", Rect(30, 10, 44, 24))

    def test_outside_its_area(self) -> None:
        hint = layout_hint(order(action="place_blueprint", building_def="TableButcher", x=50, z=5, stuff="WoodLog"),
                           self.layout)
        self.assertIn("outside your kitchen_dining area", hint)

    def test_inside_its_area_says_nothing(self) -> None:
        self.assertEqual(layout_hint(order(action="place_blueprint", building_def="FueledStove", x=15, z=15),
                                     self.layout), "")

    def test_in_someone_elses_area(self) -> None:
        self.layout.plan("bedrooms", Rect(50, 50, 60, 60))
        hint = layout_hint(order(action="build_room", x1=52, z1=52, x2=57, z2=57, stuff="WoodLog",
                                 door_side="north", purpose="storage"), self.layout)
        self.assertIn("is in your bedrooms area", hint)
        self.layout.plan("storage", Rect(0, 30, 8, 40))
        hint = layout_hint(order(action="create_growing_zone", plant="Plant_Potato", x1=12, z1=12, x2=14, z2=14),
                           self.layout)
        self.assertIn("outside your fields area", hint)

    def test_no_plan_no_hint_and_walls_belong_anywhere(self) -> None:
        self.assertEqual(layout_hint(order(action="place_blueprint", building_def="TableButcher", x=50, z=5,
                                           stuff="WoodLog"), tracker()), "")
        self.assertEqual(layout_hint(order(action="build_wall", x1=0, z1=0, x2=5, z2=0, stuff="WoodLog"),
                                     self.layout), "")


class RoomSlotTests(unittest.TestCase):
    room = Rect(10, 10, 14, 14)  # outer 5x5

    def test_every_open_side_is_offered_sharing_the_wall(self) -> None:
        slots = room_slots([("storage", self.room)], lambda x, z: True, 50, 50)
        self.assertEqual(len(slots), 4)
        north = next(s for s in slots if s.startswith("- north"))
        self.assertIn("its north wall (10,14)-(14,14) can be shared (5 cells, 25 material saved)", north)
        self.assertIn("Clear ground reaches 15 cells out, walls included, as far as (10,14)-(14,28)", north)
        self.assertIn("choose the size within that", north)

    def test_a_blocked_side_offers_as_far_as_is_clear_or_none(self) -> None:
        # Something at z=20 north: clear ground reaches 6 deep (z 14..19).
        slots = room_slots([("storage", self.room)], lambda x, z: z != 20, 50, 50)
        self.assertTrue(any("reaches 6 cells out" in s and "(10,14)-(14,19)" in s for s in slots))
        # The game reports the east side taken: not offered.
        slots = room_slots([("storage", self.room)], lambda x, z: True, 50, 50, is_clear=lambda r: r.x1 <= 14)
        self.assertFalse(any(s.startswith("- east") for s in slots))

    def test_off_the_map_is_skipped(self) -> None:
        slots = room_slots([("bedroom", Rect(0, 0, 4, 4))], lambda x, z: True, 50, 50)
        self.assertEqual({s.split()[1] for s in slots}, {"north", "east"})


class ToolkitTests(unittest.TestCase):
    def test_placing_toolkits_offer_plan_area(self) -> None:
        for kit in ("construction", "storage", "farming", "production", "research", "power"):
            self.assertIn("plan_area", allowed_actions([kit]))
        self.assertNotIn("plan_area", allowed_actions(["hunting"]))


if __name__ == "__main__":
    unittest.main()


class RunFeedbackTests(unittest.TestCase):
    """Fixes from the first modes run: fields, floors, ancient danger, equip."""

    def test_a_field_far_over_the_recommendation_is_called_out(self) -> None:
        from rimagent.hints import TurnFacts, success_hint
        from tests.test_turn_loop import colonist
        facts = TurnFacts(colonists=[colonist(name=n) for n in ("A", "B", "C")], field_cells=210)
        hint = success_hint(order(action="create_growing_zone", plant="Plant_Potato", x1=0, z1=0, x2=10, z2=12),
                            "verified zone 2", [], facts)
        self.assertIn("You now have 353 field cells, 12 times the 30 recommended", hint)
        small = TurnFacts(colonists=facts.colonists)
        self.assertNotIn("times the", success_hint(order(action="create_growing_zone", plant="Plant_Potato",
                                                         x1=0, z1=0, x2=5, z2=4), "verified zone 1", [], small))

    def test_a_closed_room_without_a_floor_says_why_it_needs_one(self) -> None:
        from rimagent.rooms import RoomTracker, perimeter
        rooms = RoomTracker()
        rooms.items = []
        rooms.add(Rect(0, 0, 5, 5), "kitchen", "north")
        walls = {c: "built" for c in perimeter(Rect(0, 0, 5, 5))}
        line = rooms.describe(walls, [], lambda inside: (0, 0, 16))[0]
        self.assertIn("no floor yet: build_floor over (1,1)-(4,4)", line)
        self.assertIn("food poisoning", line)
        self.assertIn("floor laid (16/16 cells): done", rooms.describe(walls, [], lambda inside: (16, 0, 16))[0])
        self.assertIn("floor: 10 of 16 cells laid, 6 more planned",
                      rooms.describe(walls, [], lambda inside: (10, 6, 16))[0])
        half = {c: "built" for c in list(perimeter(Rect(0, 0, 5, 5)))[:5]}
        self.assertNotIn("no floor", rooms.describe(half, [], lambda inside: (0, 0, 16))[0])  # not closed yet

    def test_ancient_danger_is_a_warning_not_an_attack(self) -> None:
        from rimagent.events import GameEvent, describe_event, is_attack
        warning = GameEvent(kind="letter", text="Ancient danger", category="ThreatBig", tick=1)
        raid = GameEvent(kind="letter", text="Raid", category="ThreatBig", tick=1)
        self.assertFalse(is_attack(warning))
        self.assertTrue(is_attack(raid))
        self.assertIn("It is not an attack", describe_event(warning))

    def test_weapons_are_equipped_through_the_job_endpoint(self) -> None:
        from rimagent.rimapi import RimApiClient
        client = RimApiClient(base_url="http://example.invalid")
        client._request = Mock()
        client.equip(1, 44)
        method, path = client._request.call_args.args[:2]
        self.assertEqual((method, path), ("POST", "/api/v1/pawn/job"))
        self.assertEqual(client._request.call_args.kwargs["json"]["job_def"], "Equip")
        client.close()


class RetryListTests(unittest.TestCase):
    def chair(self, x):
        return order(action="place_blueprint", building_def="DiningChair", x=x, z=5, stuff="WoodLog")

    def test_a_near_miss_is_asked_again_until_fixed(self) -> None:
        from rimagent.hints import RetryList
        retries = RetryList()
        retries.failed(self.chair(4), "DiningChair at (4,5) would cover (4,5), where Table2x2c is already planned "
                                      "HINT: Try DiningChair at (6,5) facing north")
        self.assertEqual(retries.end_turn(), [])
        lines = retries.lines()
        self.assertIn("RETRY IF STILL NEEDED", lines[0])
        self.assertIn("place_blueprint DiningChair, unfixed for 1 of 3 turns", lines[1])
        self.assertIn("Try DiningChair at (6,5)", lines[1])
        retries.succeeded(self.chair(6))  # another spot counts as the fix
        self.assertEqual(retries.lines(), [])

    def test_dropped_after_three_turns_unfixed(self) -> None:
        from rimagent.hints import RetryList
        retries = RetryList()
        retries.failed(self.chair(4), "blocked HINT: Try DiningChair at (6,5)")
        retries.end_turn()                                            # turn 1: the failure itself
        retries.failed(self.chair(5), "blocked again HINT: Try DiningChair at (6,5)")
        self.assertEqual(retries.end_turn(), [])                      # turn 2: tried, failed again
        notes = retries.end_turn()                                    # turn 3: not even tried
        self.assertIn("Dropped place_blueprint DiningChair", notes[0])
        self.assertEqual(retries.items, {})

    def test_nothing_to_fix_is_not_kept(self) -> None:
        from rimagent.hints import RetryList
        retries = RetryList()
        retries.failed(self.chair(4), "already planned HINT: That DiningChair is your own ... Nothing to fix: leave it")
        retries.failed(order(action="enable_work", colonist="Ada", work_type="Growing"), "incapable")
        self.assertEqual(retries.items, {})


class RetryLoopTests(unittest.TestCase):
    def test_the_next_prompt_asks_to_fix_a_blocked_placement(self) -> None:
        from rimagent.agent import run
        from rimagent.memory import AgentMemory
        from rimagent.runlog import RunLogger
        from tests.test_turn_loop import LoopClient, ScriptedModel
        tmp = Path(tempfile.mkdtemp())
        client = LoopClient()
        client.construction_at = Mock(return_value=False)
        client.place_blueprint = Mock(side_effect=ValueError("Campfire at (3,3) would cover (3,3), where rock is"))
        campfire = {"action": "place_blueprint", "building_def": "Campfire", "x": 3, "z": 3, "reason": "heat"}
        model = ScriptedModel([{"actions": [campfire], "time": "hold"}] + [{"actions": [], "time": "hold"}] * 3)
        run(client, model, RunLogger(tmp / "logs"), max_steps=4, memory=AgentMemory(tmp / "memory"))
        second, third, fourth = model.prompts[1:]
        self.assertIn("RETRY IF STILL NEEDED", second)
        self.assertIn("place_blueprint Campfire, unfixed for 1 of 3", second)
        self.assertIn("unfixed for 2 of 3", third)
        self.assertIn("Dropped place_blueprint Campfire", fourth)
        self.assertNotIn("FIX FIRST", fourth)


class FieldLimitTests(unittest.TestCase):
    def facts(self, field_cells=0):
        from rimagent.hints import TurnFacts
        from tests.test_turn_loop import colonist
        return TurnFacts(colonists=[colonist(name=n) for n in ("A", "B", "C")], field_cells=field_cells)

    def field(self, x1, z1, x2, z2):
        return order(action="create_growing_zone", plant="Plant_Potato", x1=x1, z1=z1, x2=x2, z2=z2)

    def test_a_whole_open_ground_spot_is_refused_with_a_fitting_rectangle(self) -> None:
        from rimagent.hints import field_refusal
        # The run's first order: all of spot A, 225 cells, for 3 colonists.
        text = field_refusal(self.field(112, 131, 126, 145), self.facts())
        self.assertIn("about 30 more is enough", text)
        self.assertIn("Use (112,131)-(116,136) (30 cells) instead", text)

    def test_within_the_limit_is_allowed(self) -> None:
        from rimagent.hints import field_refusal
        self.assertIsNone(field_refusal(self.field(0, 0, 5, 5), self.facts()))          # 36 cells
        self.assertIsNone(field_refusal(self.field(0, 0, 4, 4), self.facts(10)))        # 25 more, 35 in all

    def test_a_second_field_within_the_limit_is_allowed(self) -> None:
        from rimagent.hints import field_refusal
        self.assertIsNone(field_refusal(self.field(0, 0, 4, 5), self.facts(30)))        # 30 more, 60 in all

    def test_over_the_limit_with_enough_offers_no_replacement(self) -> None:
        from rimagent.hints import field_refusal
        # The run: 30 cells, then 90 asked; the old refusal offered 30 more and it was taken.
        text = field_refusal(self.field(108, 145, 122, 150), self.facts(30))
        self.assertIn("you already have 30, which is enough", text)
        self.assertNotIn("Use", text)

    def test_enough_fields_refuses_any_more(self) -> None:
        from rimagent.hints import field_refusal
        self.assertIn("which is enough", field_refusal(self.field(0, 0, 3, 3), self.facts(60)))


class SuggestedFieldTests(unittest.TestCase):
    def test_sized_to_the_need_fertile_and_near_the_base(self) -> None:
        from rimagent.construction import TerrainInfo, TerrainMap
        from rimagent.mapview import suggested_field
        # 20x20 map: soil everywhere except a band of sand at x 0..4.
        grid = []
        for z in range(20):
            grid += [5, 1, 15, 0]  # 5 sand, 15 soil per row
        terrain = TerrainMap(20, 20, ["Soil", "Sand"], grid)
        defs = {"Soil": TerrainInfo(1.0, frozenset({"Heavy"})), "Sand": TerrainInfo(0.1, frozenset({"Heavy"}))}
        spot = ("A", Rect(0, 0, 14, 14), "15x15 open")
        field = suggested_field([spot], terrain, defs, (2, 2), 30, 0.7)
        self.assertEqual(field.cells, 30)             # the need, not the whole 225-cell spot
        self.assertGreaterEqual(field.x1, 5)          # all on soil
        self.assertEqual((field.x1, field.z1), (5, 0))  # the fertile window closest to the base
        self.assertIsNone(suggested_field([spot], terrain, defs, (2, 2), 2, 0.7))  # nothing needed
