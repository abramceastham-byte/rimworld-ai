"""Turns (several actions per reply), honest memory, supplies, and overlap checks.

Run from the repo root:  python -m unittest tests/test_turns.py -v
"""

import unittest

from pydantic import ValidationError

from rimagent.actions import MAX_ACTIONS, Turn
from rimagent.agent import describe_supplies, honest_memory_update, stock_by_def
from rimagent.memory import MemoryUpdate
from rimagent.rimapi import Building, MapThing


def item(def_name: str, count: int, forbidden: bool) -> MapThing:
    return MapThing.model_validate({
        "thing_id": abs(hash((def_name, count, forbidden))) % 10000, "def_name": def_name,
        "label": def_name.lower(), "position": {"x": 1, "y": 0, "z": 1},
        "stack_count": count, "is_forbidden": forbidden,
    })


class TurnTests(unittest.TestCase):
    def test_several_actions_in_order(self) -> None:
        turn = Turn.model_validate_json(
            '{"actions": ['
            '{"action": "place_blueprint", "building_def": "Wall", "x": 5, "z": 5, '
            '"stuff": "WoodLog", "reason": "west wall"},'
            '{"action": "place_blueprint", "building_def": "Door", "x": 6, "z": 5, '
            '"stuff": "WoodLog", "reason": "doorway"},'
            '{"action": "resume", "reason": "let them build"}]}'
        )
        self.assertEqual([a.action for a in turn.actions],
                         ["place_blueprint", "place_blueprint", "resume"])
        self.assertEqual(turn.actions[1].building_def, "Door")

    def test_limits(self) -> None:
        one = '{"action": "wait", "reason": "x"}'
        with self.assertRaises(ValidationError):  # empty
            Turn.model_validate_json('{"actions": []}')
        with self.assertRaises(ValidationError):  # too many
            Turn.model_validate_json('{"actions": [' + ",".join([one] * (MAX_ACTIONS + 1)) + "]}")
        Turn.model_validate_json('{"actions": [' + ",".join([one] * MAX_ACTIONS) + "]}")

    def test_memory_update_sits_beside_the_actions(self) -> None:
        turn = Turn.model_validate_json(
            '{"actions": [{"action": "wait", "reason": "planning"}], '
            '"memory_update": {"observation": "The steel is forbidden."}}'
        )
        self.assertEqual(turn.memory_update.observation, "The steel is forbidden.")

    def test_a_bad_action_rejects_the_whole_turn(self) -> None:
        with self.assertRaises(ValidationError):
            Turn.model_validate_json(
                '{"actions": [{"action": "wait", "reason": "ok"}, '
                '{"action": "set_bill", "reason": "no ids"}]}'
            )


    def test_an_action_cannot_carry_another_actions_fields(self) -> None:
        # The failure this schema exists to prevent: a zone sent with a
        # blueprint's x/z instead of the rectangle x1/z1/x2/z2.
        with self.assertRaises(ValidationError):
            Turn.model_validate_json(
                '{"actions": [{"action": "create_growing_zone", "plant": "Plant_Potato", '
                '"x": 110, "z": 120, "rotation": 0, "stuff": null, "reason": "potatoes"}]}'
            )
        Turn.model_validate_json(
            '{"actions": [{"action": "create_growing_zone", "plant": "Plant_Potato", '
            '"x1": 110, "z1": 120, "x2": 113, "z2": 123, "reason": "potatoes"}]}'
        )

    def test_action_is_the_first_field_of_every_action(self) -> None:
        # Constrained decoding fills properties in schema order and picks a
        # branch of the union at the first one. With "reason" first, the branch
        # was chosen before the action was known and the model's choice was
        # overwritten: a whole run came out as place_blueprint with reasons
        # describing chopping trees and making zones.
        for name, spec in Turn.model_json_schema()["$defs"].items():
            properties = list(spec.get("properties", {}))
            if "action" in properties:
                self.assertEqual(properties[0], "action", f"{name} must declare action first")

    def test_the_schema_tells_the_model_each_action_shape(self) -> None:
        schema = Turn.model_json_schema()
        zone = schema["$defs"]["CreateGrowingZone"]
        self.assertEqual(set(zone["required"]), {"action", "reason", "plant", "x1", "z1", "x2", "z2"})
        self.assertNotIn("x", zone["properties"])  # no blueprint fields to reach for
        self.assertNotIn("stuff", zone["properties"])


class HonestMemoryTests(unittest.TestCase):
    def test_plan_changes_are_dropped_when_an_action_failed(self) -> None:
        update = MemoryUpdate(
            observation="Bench still not built.",
            replace_plan={"objective": "Tailoring", "status": "active",
                          "steps": [{"description": "Set bill", "status": "completed"}]},
            policies_to_add=["Always tailor"],
        )
        kept = honest_memory_update(update, any_failed=True)
        self.assertIsNone(kept.replace_plan)
        self.assertEqual(kept.policies_to_add, [])
        self.assertEqual(kept.observation, "Bench still not built.")  # observations survive

    def test_nothing_is_dropped_when_everything_worked(self) -> None:
        update = MemoryUpdate(policies_to_add=["Keep 20 meals"])
        self.assertIs(honest_memory_update(update, any_failed=False), update)

    def test_an_update_with_only_a_plan_becomes_nothing(self) -> None:
        update = MemoryUpdate(replace_plan={"objective": "x", "status": "active", "steps": []})
        self.assertIsNone(honest_memory_update(update, any_failed=True))


class SuppliesTests(unittest.TestCase):
    def test_splits_usable_from_forbidden(self) -> None:
        items = [item("WoodLog", 96, False), item("WoodLog", 220, True),
                 item("Steel", 450, True), item("Steel", 30, True)]
        self.assertEqual(stock_by_def(items, forbidden=False), {"WoodLog": 96})
        self.assertEqual(stock_by_def(items, forbidden=True), {"WoodLog": 220, "Steel": 480})

        usable, locked = describe_supplies(items)
        self.assertIn("you can use now: WoodLog 96.", usable)
        self.assertIn("Steel 480, WoodLog 220", locked)  # biggest first

    def test_says_nothing_when_there_is_nothing(self) -> None:
        usable, locked = describe_supplies([])
        self.assertIn("use now: nothing.", usable)
        self.assertIn("allow_items: nothing.", locked)


class BuildingTests(unittest.TestCase):
    def make(self, **kwargs) -> Building:
        base = {"id": 1, "def": "Wall", "label": "wooden wall",
                "position": {"x": 10, "y": 0, "z": 10}, "rotation": 0,
                "size": {"x": 1, "y": 0, "z": 1}, "type": "Building"}
        return Building.model_validate({**base, **kwargs})

    def test_reads_the_games_fields(self) -> None:
        wall = self.make()
        self.assertEqual(wall.def_name, "Wall")
        self.assertFalse(wall.under_construction)
        self.assertEqual(wall.area().cells, 1)

    def test_frames_count_as_under_construction(self) -> None:
        frame = self.make(id=2, **{"def": "Frame_Bed"}, type="Frame",
                          size={"x": 1, "y": 0, "z": 2})
        self.assertTrue(frame.under_construction)
        self.assertEqual(frame.area().cells, 2)  # a bed covers two cells


if __name__ == "__main__":
    unittest.main()
