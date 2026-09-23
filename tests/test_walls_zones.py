"""Wall runs, floors, zone tracking, and the nudge to resume.

Run from the repo root:  python -m unittest tests/test_walls_zones.py -v
"""

import tempfile
import unittest
from pathlib import Path

from rimagent.agent import PAUSED_NUDGE_AFTER, TimeMode, describe_time
from rimagent.construction import MAX_WALL_RUN, Rect
from rimagent.zones import ZoneTracker
from tests.test_map_actions import FakeRimApi, MapTestCase


class WallTests(MapTestCase):
    def test_one_call_places_the_whole_run(self) -> None:
        fake = FakeRimApi()
        summary = self.client_with(fake).build_wall(Rect(2, 2, 2, 6), "WoodLog")

        (method, path, kwargs), = fake.writes()
        self.assertEqual((method, path), ("POST", "/api/v1/builder/blueprint"))
        buildings = kwargs["json"]["blueprint"]["buildings"]
        self.assertEqual(len(buildings), 5)                       # one call, five walls
        self.assertEqual({b["def_name"] for b in buildings}, {"Wall"})
        self.assertEqual({b["stuff_def_name"] for b in buildings}, {"WoodLog"})
        self.assertIn("5 Wall blueprints from (2,2) to (2,6)", summary)

    def test_taken_cells_are_skipped_and_reported(self) -> None:
        fake = FakeRimApi(
            issues={"ores": [{"x": 2, "z": 4, "def_name": "Granite", "label": "granite"}]},
            things_at={(2, 5): [{"thing_id": 1, "def_name": "Blueprint_Wall",
                                 "position": {"x": 2, "y": 0, "z": 5}}]},
        )
        summary = self.client_with(fake).build_wall(Rect(2, 2, 2, 6), "WoodLog")

        placed = fake.writes()[0][2]["json"]["blueprint"]["buildings"]
        self.assertEqual(len(placed), 3)  # 5 cells minus rock minus an existing blueprint
        self.assertIn("skipped 2 cells", summary)
        self.assertIn("granite at (2,4)", summary)

    def test_refuses_bent_and_over_long_runs(self) -> None:
        client = self.client_with(FakeRimApi())
        with self.assertRaisesRegex(ValueError, "must be straight"):
            client.build_wall(Rect(2, 2, 5, 6), "WoodLog")
        with self.assertRaisesRegex(ValueError, f"at most {MAX_WALL_RUN}"):
            client.build_wall(Rect(0, 0, 0, MAX_WALL_RUN), "WoodLog")

    def test_refuses_a_run_with_nowhere_to_build(self) -> None:
        fake = FakeRimApi(issues={"buildings": [{"x": x, "z": 2, "label": "wall"} for x in range(2, 5)]})
        with self.assertRaisesRegex(ValueError, "every cell in"):
            self.client_with(fake).build_wall(Rect(2, 2, 4, 2), "WoodLog")
        self.assertEqual(fake.writes(), [])


class FloorTests(MapTestCase):
    def test_lays_a_rectangle_of_floor(self) -> None:
        fake = FakeRimApi()
        summary = self.client_with(fake).build_floor(Rect(1, 1, 3, 2), "WoodPlankFloor")

        floors = fake.writes()[0][2]["json"]["blueprint"]["floors"]
        self.assertEqual(len(floors), 6)
        self.assertEqual({f["def_name"] for f in floors}, {"WoodPlankFloor"})
        self.assertIn("6 cells of wood floor", summary)

    def test_stone_tiles_need_the_research(self) -> None:
        fake = FakeRimApi(research=[])  # no Stonecutting
        with self.assertRaisesRegex(ValueError, "Stonecutting"):
            self.client_with(fake).build_floor(Rect(1, 1, 2, 2), "TileSandstone")
        self.assertEqual(fake.writes(), [])

    def test_refuses_unknown_floors(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not a floor"):
            self.client_with(FakeRimApi()).build_floor(Rect(1, 1, 2, 2), "GoldFloor")

    def test_floors_go_under_buildings_but_not_through_rock(self) -> None:
        # A building in the way doesn't stop flooring...
        fake = FakeRimApi(issues={"buildings": [{"x": 1, "z": 1, "label": "wall"}]})
        self.client_with(fake).build_floor(Rect(1, 1, 2, 2), "WoodPlankFloor")
        self.assertEqual(len(fake.writes()[0][2]["json"]["blueprint"]["floors"]), 4)

        # ...but rock does, for those cells.
        fake = FakeRimApi(issues={"ores": [{"x": 1, "z": 1, "label": "granite"}]})
        self.client_with(fake).build_floor(Rect(1, 1, 2, 2), "WoodPlankFloor")
        self.assertEqual(len(fake.writes()[0][2]["json"]["blueprint"]["floors"]), 3)


class ZoneTrackerTests(unittest.TestCase):
    def test_remembers_where_zones_are(self) -> None:
        path = Path(tempfile.mkdtemp()) / "zones.json"
        tracker = ZoneTracker(path)
        tracker.add("growing", 0, "Plant_Potato field", Rect(117, 121, 119, 123))
        tracker.add("stockpile", 1, "stockpile", Rect(121, 122, 123, 124))

        reloaded = ZoneTracker(path)  # survives a restart, unlike RIMAPI's own listing
        self.assertEqual([z.describe() for z in reloaded.items],
                         ["Plant_Potato field (117,121)-(119,123)", "stockpile (121,122)-(123,124)"])
        self.assertIn((118, 122), reloaded.cells())
        self.assertNotIn((130, 130), reloaded.cells())


class NudgeTests(unittest.TestCase):
    def test_says_plainly_when_nothing_is_happening(self) -> None:
        quiet = describe_time(TimeMode(True, "you paused it", PAUSED_NUDGE_AFTER - 1), None, 600, 180)
        self.assertNotIn("NOTHING HAS HAPPENED", quiet)

        loud = describe_time(TimeMode(True, "you paused it", PAUSED_NUDGE_AFTER), None, 600, 180)
        self.assertTrue(loud.startswith("NOTHING HAS HAPPENED FOR 4 DECISIONS"))
        self.assertIn("Choose resume", loud)


if __name__ == "__main__":
    unittest.main()
