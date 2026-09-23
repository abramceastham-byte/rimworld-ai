"""Growing season, fields found from plants, stockpile detail, and the stockpile guard.

Run from the repo root:  python -m unittest tests/test_fields_storage.py -v
"""

import unittest

from rimagent.actions import Turn
from rimagent.construction import Rect
from rimagent.hints import TurnFacts, failure_hint, stockpile_refusal
from rimagent.observation import describe_storage, fields_from_plants, growing_season
from rimagent.rooms import TrackedRoom


def plant(def_name, x, z):
    return {"def_name": def_name, "position": {"x": x, "y": 0, "z": z}}


def stockpile(x1, z1, x2, z2):
    return Turn.model_validate({"actions": [{"action": "create_stockpile", "reason": "x",
                                             "x1": x1, "z1": z1, "x2": x2, "z2": z2}], "time": "hold"}).actions[0]


class SeasonTests(unittest.TestCase):
    def test_full_speed_between_10_and_42(self) -> None:
        self.assertEqual(growing_season(22), "Growing season: yes, crops grow at full speed at 22 C (10 to 42 C).")

    def test_slow_when_cool_or_hot(self) -> None:
        self.assertIn("about 50% speed at 5 C", growing_season(5))
        self.assertIn("about 50% speed at 50 C", growing_season(50))

    def test_no_growth_at_freezing(self) -> None:
        self.assertIn("Growing season: NO, at -3 C outdoor crops don't grow at all", growing_season(-3))


class FieldTests(unittest.TestCase):
    def test_each_crop_s_plants_form_fields(self) -> None:
        plants = ([plant("Plant_Potato", x, z) for x in range(10, 16) for z in range(10, 13)]   # 18
                  + [plant("Plant_Potato", x, 40) for x in range(40, 44)]                     # 4, far away
                  + [plant("Plant_Rice", 10, 20), plant("Plant_Rice", 12, 20)]                # gap of 1: one field
                  + [plant("Plant_Grass", 11, 11)])                                           # not a crop
        fields = fields_from_plants(plants, ["Plant_Potato", "Plant_Rice"])
        self.assertEqual(fields, [("Plant_Potato", Rect(10, 10, 15, 12), 18),
                                  ("Plant_Potato", Rect(40, 40, 43, 40), 4),
                                  ("Plant_Rice", Rect(10, 20, 12, 20), 2)])


class FieldAdviceTests(unittest.TestCase):
    def test_short_of_the_recommendation_says_how_much_more(self) -> None:
        from rimagent.observation import field_advice
        self.assertEqual(field_advice(12, 3), "Recommended growing area: about 10 cells per colonist, 30 for 3; "
                                              "you have 12, so a field of about 18 more cells would help.")

    def test_enough_is_only_advice(self) -> None:
        from rimagent.observation import field_advice
        self.assertIn("you have 302, which is enough; more fields are optional.", field_advice(302, 3))


class StorageTests(unittest.TestCase):
    PILES = [{"label": "Stockpile 1", "cells_count": 223, "type": "Zone_Stockpile"},
             {"label": "Stockpile 2", "cells_count": 180, "type": "Zone_Stockpile"}]

    def test_plenty_of_room_says_not_to_add_more(self) -> None:
        line = describe_storage({"used_cells": 7, "total_cells": 403}, self.PILES)
        self.assertIn("Storage: 7/403 cells used across 2 stockpiles (Stockpile 1 223 cells, Stockpile 2 180 cells)", line)
        self.assertIn("396 cells free: plenty of room, so don't create another stockpile", line)

    def test_filling_up_is_just_reported(self) -> None:
        self.assertTrue(describe_storage({"used_cells": 300, "total_cells": 403}, self.PILES).endswith("180 cells)."))

    def test_no_stockpiles_points_to_a_storage_room(self) -> None:
        self.assertIn("no stockpiles yet", describe_storage({"used_cells": 0, "total_cells": 0}, []))


class StockpileGuardTests(unittest.TestCase):
    ROOM = TrackedRoom(purpose="storage", door_side="south", cells=(10, 10, 16, 16))  # inside (11,11)-(15,15)

    def facts(self, used, total=400):
        return TurnFacts(storage={"used_cells": used, "total_cells": total}, rooms=[self.ROOM])

    def test_refused_outdoors_while_storage_has_room(self) -> None:
        refusal = stockpile_refusal(stockpile(30, 30, 34, 34), self.facts(used=10))
        self.assertIn("plenty of room (10/400 cells used)", refusal)
        hint, keep = failure_hint(stockpile(30, 30, 34, 34), refusal, self.facts(used=10))
        self.assertIn("build_room with purpose storage, place Shelf blueprints inside", hint)
        self.assertFalse(keep)

    def test_always_allowed_inside_a_storage_room(self) -> None:
        self.assertIsNone(stockpile_refusal(stockpile(11, 11, 15, 15), self.facts(used=10)))

    def test_allowed_once_storage_fills_up(self) -> None:
        self.assertIsNone(stockpile_refusal(stockpile(30, 30, 34, 34), self.facts(used=250)))

    def test_allowed_when_there_is_no_storage_yet(self) -> None:
        self.assertIsNone(stockpile_refusal(stockpile(30, 30, 34, 34), self.facts(used=0, total=0)))


if __name__ == "__main__":
    unittest.main()
