"""Fixes from the first modes run (2026-09-24, qwen3:14b): colonists starved
beside 57 forbidden survival meals while a plan chopped trees for 12 turns.

Run from the repo root:  python -m unittest tests/test_run_starving.py -v
"""

import tempfile
import unittest
from types import SimpleNamespace

from rimagent.finish import FOOD, FinishCheck, FinishState, canonical_item
from rimagent.memory import AgentMemory, MemoryUpdate, same_lesson
from rimagent.modes import food_emergency
from rimagent.rimapi import MapThing


def item(def_name: str, count: int = 1, forbidden: bool = False) -> MapThing:
    return MapThing(thing_id=1, def_name=def_name, position={"x": 0, "z": 0}, stack_count=count,
                    is_forbidden=forbidden)


def ctx(hunger: float, items, alerts=()):
    return {"alerts": [SimpleNamespace(label=a) for a in alerts],
            "colonists": [SimpleNamespace(name="Squint", hunger=hunger)], "items": items}


class CanonicalItemTests(unittest.TestCase):
    names = frozenset({"WoodLog", "Steel", "ComponentIndustrial"})

    def test_plain_words_and_other_capitals(self) -> None:
        self.assertEqual(canonical_item("wood", self.names), "WoodLog")
        self.assertEqual(canonical_item("woodlog", self.names), "WoodLog")
        self.assertEqual(canonical_item("steel", self.names), "Steel")
        self.assertEqual(canonical_item("Meals", self.names), FOOD)
        self.assertEqual(canonical_item("Steel", self.names), "Steel")

    def test_unknown_stays_unknown(self) -> None:
        self.assertIsNone(canonical_item("planks", self.names))
        self.assertIsNone(canonical_item("StorageRoom", self.names))

    def test_food_counts_meals_and_raw_food(self) -> None:
        state = FinishState(items={"MealSurvivalPack": 40, "RawRice": 10, "Meat_Muffalo": 7, "WoodLog": 99})
        self.assertEqual(state.measure(FinishCheck(kind="have_items", what=FOOD, count=57)), 57)


class FoodEmergencyTests(unittest.TestCase):
    def test_starvation_alert(self) -> None:
        self.assertIn("Starvation", food_emergency(ctx(0.5, [item("MealSimple")], alerts=["Starvation"])))

    def test_nearly_starving_with_only_forbidden_food(self) -> None:
        reason = food_emergency(ctx(0.04, [item("MealSurvivalPack", 57, forbidden=True)]))
        self.assertIn("Squint nearly starving", reason)

    def test_hungry_with_food_allowed_is_no_emergency(self) -> None:
        self.assertIsNone(food_emergency(ctx(0.04, [item("MealSurvivalPack", 5)])))
        self.assertIsNone(food_emergency(ctx(0.5, [item("MealSurvivalPack", 57, forbidden=True)])))


class LessonTests(unittest.TestCase):
    def test_reworded_lesson_is_the_same_lesson(self) -> None:
        a = "Young trees do not yield wood; selecting a new area with mature trees is necessary."
        b = "Young trees do not yield wood; mature trees must be selected for chopping."
        self.assertTrue(same_lesson(a, b))
        self.assertFalse(same_lesson(a, "Colonists build blueprints by themselves once Construction is on."))

    def test_a_lesson_is_saved_once(self) -> None:
        memory = AgentMemory(tempfile.mkdtemp())
        first = "Young trees do not yield wood; mature trees must be selected for chopping."
        again = "Young trees do not yield wood; selecting a new area with mature trees is necessary."
        self.assertIsNone(memory.apply_update(1, 10, MemoryUpdate(observation=first, toolkit="wood_mining")))
        note = memory.apply_update(2, 20, MemoryUpdate(observation=again, toolkit="wood_mining"))
        self.assertIn("already saved", note)
        self.assertEqual(memory.observations_for({"wood_mining"}), [first])

    def test_duplicates_already_on_disk_are_shown_once(self) -> None:
        memory = AgentMemory(tempfile.mkdtemp())
        a = "Young trees do not yield wood; mature trees must be selected for chopping."
        b = "Young trees do not yield wood; selecting a new area with mature trees is necessary."
        with memory.observations_path.open("a", encoding="utf-8") as f:
            for n, text in enumerate((a, b, a, b)):
                f.write(f"\n## Step {n} (game tick {n}) [wood_mining]\n\n{text}\n")
        self.assertEqual(memory.observations_for({"wood_mining"}), [b])


class AllowItemsTests(unittest.TestCase):
    """allow_items covers the whole map: one 235B turn per scattered pile was four turns with nobody working."""

    def allow(self, what: str, items):
        from unittest.mock import Mock

        from rimagent.actions import Turn
        from rimagent.execution import execute

        state = {i.thing_id: i for i in items}
        client = Mock()
        client.get_items = lambda: list(state.values())

        def allow_things(ids):
            for i in ids:
                state[i] = state[i].model_copy(update={"is_forbidden": False})
            return len(ids)

        client.allow_things = allow_things
        action = Turn.model_validate({"actions": [{"action": "allow_items", "reason": "x", "what": what}],
                                      "time": "hold"}).actions[0]
        return execute(client, action, None, None, None), state

    def items(self):
        return [MapThing(thing_id=1, def_name="WoodLog", position={"x": 0, "z": 0}, stack_count=236, is_forbidden=True),
                MapThing(thing_id=2, def_name="WoodLog", position={"x": 90, "z": 90}, stack_count=64, is_forbidden=True),
                MapThing(thing_id=3, def_name="MealSurvivalPack", position={"x": 59, "z": 170}, stack_count=4,
                         is_forbidden=True),
                MapThing(thing_id=4, def_name="Steel", position={"x": 5, "z": 5}, stack_count=233, is_forbidden=True)]

    def test_one_item_everywhere(self) -> None:
        result, state = self.allow("WoodLog", self.items())
        self.assertIn("verified 2 stacks allowed, now usable: WoodLog 300", result)
        self.assertEqual([i for i, t in state.items() if t.is_forbidden], [3, 4])

    def test_food_and_all(self) -> None:
        result, state = self.allow("Food", self.items())
        self.assertIn("MealSurvivalPack 4", result)
        result, state = self.allow("all", self.items())
        self.assertFalse(any(t.is_forbidden for t in state.values()))

    def test_nothing_left_is_unchanged(self) -> None:
        result, _ = self.allow("Steel", self.items()[:1])
        self.assertTrue(result.startswith("unchanged"))


if __name__ == "__main__":
    unittest.main()
