"""Turns (several actions per reply), honest memory, supplies, and overlap checks.

Run from the repo root:  python -m unittest tests/test_turns.py -v
"""

import tempfile
import unittest

from pydantic import ValidationError

from rimagent.actions import MAX_ACTIONS, Turn
from rimagent.agent import describe_supplies, stock_by_def
from rimagent.memory import AgentMemory, MemoryUpdate
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
            '"stuff": "WoodLog", "reason": "doorway"}], "time": "advance"}'
        )
        self.assertEqual([a.action for a in turn.actions], ["place_blueprint", "place_blueprint"])
        self.assertEqual(turn.actions[1].building_def, "Door")
        self.assertEqual(turn.time, "advance")

    def test_limits(self) -> None:
        one = '{"action": "set_research", "project": "Brewing", "reason": "x"}'
        Turn.model_validate_json('{"actions": [], "time": "advance"}')  # waiting is allowed
        with self.assertRaises(ValidationError):  # too many
            Turn.model_validate_json(
                '{"actions": [' + ",".join([one] * (MAX_ACTIONS + 1)) + '], "time": "hold"}')
        Turn.model_validate_json('{"actions": [' + ",".join([one] * MAX_ACTIONS) + '], "time": "hold"}')

    def test_time_must_be_chosen_every_turn(self) -> None:
        with self.assertRaises(ValidationError):
            Turn.model_validate_json('{"actions": []}')
        with self.assertRaises(ValidationError):
            Turn.model_validate_json('{"actions": [], "time": "pause"}')
        with self.assertRaises(ValidationError):  # the old time actions are gone
            Turn.model_validate_json(
                '{"actions": [{"action": "resume", "reason": "go"}], "time": "hold"}')

    def test_time_comes_after_the_orders(self) -> None:
        # Constrained decoding writes fields in schema order: the model should
        # pick advance/hold knowing what it has just ordered.
        self.assertEqual(list(Turn.model_json_schema()["properties"])[:2], ["actions", "time"])

    def test_memory_update_sits_beside_the_actions(self) -> None:
        turn = Turn.model_validate_json(
            '{"actions": [], "time": "hold", '
            '"memory_update": {"observation": "The steel is forbidden."}}'
        )
        self.assertEqual(turn.memory_update.observation, "The steel is forbidden.")

    def test_a_bad_action_rejects_the_whole_turn(self) -> None:
        with self.assertRaises(ValidationError):
            Turn.model_validate_json(
                '{"actions": [{"action": "set_research", "project": "Brewing", "reason": "ok"}, '
                '{"action": "set_bill", "reason": "no ids"}], "time": "hold"}'
            )


    def test_an_action_cannot_carry_another_actions_fields(self) -> None:
        # The failure this schema exists to prevent: a zone sent with a
        # blueprint's x/z instead of the rectangle x1/z1/x2/z2.
        with self.assertRaises(ValidationError):
            Turn.model_validate_json(
                '{"actions": [{"action": "create_growing_zone", "plant": "Plant_Potato", '
                '"x": 110, "z": 120, "rotation": 0, "stuff": null, "reason": "potatoes"}], '
                '"time": "hold"}'
            )
        Turn.model_validate_json(
            '{"actions": [{"action": "create_growing_zone", "plant": "Plant_Potato", '
            '"x1": 110, "z1": 120, "x2": 113, "z2": 123, "reason": "potatoes"}], "time": "hold"}'
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


class PlanConfirmationTests(unittest.TestCase):
    """A step reported completed is held until a later turn confirms it; the rest
    of the plan is always kept, whatever happened to the turn's orders."""

    def memory(self) -> AgentMemory:
        return AgentMemory(tempfile.mkdtemp())

    def plan(self, *steps, status="active") -> MemoryUpdate:
        return MemoryUpdate(replace_plan={"objective": "Shelter", "status": status, "steps": [
            {"description": d, "status": s} for d, s in steps]})

    def status_of(self, memory: AgentMemory, description: str) -> str:
        return next(s.status for s in memory.state.current_plan.steps if s.description == description)

    def test_a_new_completion_waits_for_a_later_turn(self) -> None:
        memory = self.memory()
        memory.apply_update(1, 100, self.plan(("Build bedroom", "pending")))
        memory.apply_update(2, 200, self.plan(("Build bedroom", "completed")))  # same reply as its orders
        self.assertEqual(self.status_of(memory, "Build bedroom"), "in_progress")
        self.assertIn("in_progress: Build bedroom (you reported it done: mark it completed again once "
                      "the game shows it)", memory.to_prompt())
        memory.apply_update(3, 300, self.plan(("Build bedroom", "completed")))  # still done a turn later
        self.assertEqual(self.status_of(memory, "Build bedroom"), "completed")
        self.assertEqual(memory.state.unconfirmed, [])

    def test_a_claim_dropped_next_turn_is_forgotten(self) -> None:
        memory = self.memory()
        memory.apply_update(1, 100, self.plan(("Build bedroom", "completed")))
        memory.apply_update(2, 200, self.plan(("Build bedroom", "in_progress")))  # it wasn't done after all
        self.assertEqual(self.status_of(memory, "Build bedroom"), "in_progress")
        self.assertEqual(memory.state.unconfirmed, [])
        memory.apply_update(3, 300, self.plan(("Build bedroom", "completed")))  # a fresh claim waits again
        self.assertEqual(self.status_of(memory, "Build bedroom"), "in_progress")

    def test_steps_already_completed_stay_completed(self) -> None:
        memory = self.memory()
        memory.apply_update(1, 100, self.plan(("Chop wood", "completed")))
        memory.apply_update(2, 200, self.plan(("Chop wood", "completed")))
        memory.apply_update(3, 300, self.plan(("Chop wood", "completed"), ("Build bedroom", "pending")))
        self.assertEqual(self.status_of(memory, "Chop wood"), "completed")

    def test_the_whole_plan_is_held_the_same_way(self) -> None:
        memory = self.memory()
        memory.apply_update(1, 100, self.plan(("Build bedroom", "pending")))
        memory.apply_update(2, 200, self.plan(("Build bedroom", "pending"), status="completed"))
        self.assertEqual(memory.state.current_plan.status, "active")
        memory.apply_update(3, 300, self.plan(("Build bedroom", "pending"), status="completed"))
        self.assertEqual(memory.state.current_plan.status, "completed")

    def test_plan_and_policies_are_kept_whatever_the_orders_did(self) -> None:
        # There's no longer a failure flag: nothing the orders did drops a plan.
        memory = self.memory()
        memory.apply_update(1, 100, MemoryUpdate(
            observation="Bench still not built.", policies_to_add=["Build in wood"],
            replace_plan={"objective": "Tailoring", "status": "active",
                          "steps": [{"description": "Set bill", "status": "pending"}]}))
        self.assertEqual(memory.state.current_plan.objective, "Tailoring")
        self.assertEqual(memory.state.policies, ["Build in wood"])


class SuppliesTests(unittest.TestCase):
    def test_splits_usable_from_forbidden(self) -> None:
        items = [item("WoodLog", 96, False), item("WoodLog", 220, True),
                 item("Steel", 450, True), item("Steel", 30, True)]
        self.assertEqual(stock_by_def(items, forbidden=False), {"WoodLog": 96})
        self.assertEqual(stock_by_def(items, forbidden=True), {"WoodLog": 220, "Steel": 480})

        usable, locked, stuff, unlock = describe_supplies(items)
        self.assertIn("Allowed items on the map (usable): WoodLog 96.", usable)
        self.assertIn("Steel 480, WoodLog 220", locked)  # biggest first
        self.assertIn("usable now: WoodLog 96; stone blocks: none.", stuff)
        # Wood is usable already, so only the Steel waits on allow_items.
        self.assertIn("Forbidden building materials, offered in the orders all the same: Steel 480.", unlock)

    def test_says_nothing_when_there_is_nothing(self) -> None:
        usable, locked, stuff = describe_supplies([])
        self.assertIn("(usable): nothing.", usable)
        self.assertIn("allow_items: nothing.", locked)
        self.assertIn("no wood or steel; stone blocks: none", stuff)


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
