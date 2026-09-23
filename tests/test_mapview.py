"""The two map views and the pre-checked free spots.

Run from the repo root:  python -m unittest tests/test_mapview.py -v
"""

import unittest

from rimagent.construction import Rect, TerrainInfo, TerrainMap
from rimagent.mapview import (
    MAX_DETAIL_SPAN,
    block_mark,
    centre_of,
    detail_bounds,
    detail_rows,
    free_spots,
    overview_rows,
    ruler,
    terrain_mark,
)

HEAVY = frozenset({"Light", "Medium", "Heavy"})
DEFS = {
    "SoilRich": TerrainInfo(1.4, HEAVY),
    "Soil": TerrainInfo(1.0, HEAVY),
    "Gravel": TerrainInfo(0.4, HEAVY),
    "Granite_Rough": TerrainInfo(0.0, HEAVY),
    "WaterShallow": TerrainInfo(0.0, frozenset({"Light"})),
}
PALETTE = list(DEFS)


def terrain(width: int = 60, height: int = 60, index: int = 1) -> TerrainMap:
    """A map of one terrain type (1 = Soil), run-length encoded as RIMAPI does."""
    return TerrainMap(width, height, PALETTE, [width * height, index])


class MarkTests(unittest.TestCase):
    def test_marks_never_reuse_a_misleading_letter(self) -> None:
        self.assertEqual(terrain_mark(DEFS["SoilRich"]), "*")  # not "R", which reads as rock
        self.assertEqual(terrain_mark(DEFS["Soil"]), ".")
        self.assertEqual(terrain_mark(DEFS["Gravel"]), ",")
        self.assertEqual(terrain_mark(DEFS["Granite_Rough"]), "_")
        self.assertEqual(terrain_mark(DEFS["WaterShallow"]), "~")

    def test_block_mark_shows_the_most_important_thing(self) -> None:
        block = Rect(0, 0, 4, 4)
        grid = terrain()
        self.assertEqual(block_mark(block, grid, DEFS, set(), {(2, 2)}), "n")   # base wins
        self.assertEqual(block_mark(block, grid, DEFS, {(0, 0), (1, 1), (2, 2)}, set()), "T")
        self.assertEqual(block_mark(block, grid, DEFS, set(), set()), ".")
        water = terrain(index=4)
        self.assertEqual(block_mark(block, water, DEFS, set(), set()), "~")


class ViewTests(unittest.TestCase):
    def test_the_view_follows_the_base_and_is_capped(self) -> None:
        wide = [(10, 10), (40, 40)]  # a base too big for one detail view
        view = detail_bounds(centre_of(wide), wide, 60, 60)
        self.assertLessEqual(view.x2 - view.x1 + 1, MAX_DETAIL_SPAN)
        self.assertLessEqual(view.z2 - view.z1 + 1, MAX_DETAIL_SPAN)

        small = [(30, 30), (32, 31)]
        view = detail_bounds(centre_of(small), small, 60, 60)
        for cell in small:
            self.assertTrue(view.contains(*cell))  # the whole base stays visible

    def test_it_stays_inside_the_map(self) -> None:
        view = detail_bounds((1, 1), [(1, 1)], 60, 60)
        self.assertGreaterEqual(view.x1, 0)
        self.assertGreaterEqual(view.z1, 0)

    def test_the_ruler_gives_whole_coordinates(self) -> None:
        label_width = len("  z=123 ")
        line = ruler(Rect(109, 0, 125, 0), label_width)
        # Whole coordinates (110, not "10"), each starting above its own column.
        self.assertEqual(line[label_width + (110 - 109):][:3], "110")
        self.assertEqual(line[label_width + (120 - 109):][:3], "120")

    def test_rows_are_north_up_and_labelled(self) -> None:
        rows = detail_rows(Rect(0, 0, 3, 2), terrain(), DEFS, {(1, 1): "@"})
        grid = [r for r in rows if r.startswith("  z=")]
        self.assertTrue(grid[0].startswith("  z=  2"))   # north at the top
        self.assertTrue(grid[-1].startswith("  z=  0"))
        self.assertIn("@", grid[1])

    def test_the_overview_covers_more_ground(self) -> None:
        rows = overview_rows((30, 30), terrain(), DEFS, set(), {(30, 30)})
        grid = [r for r in rows if r.startswith("  z=")]
        self.assertEqual(len(grid), 12)
        self.assertIn("n", "".join(grid))  # the base shows up


class FreeSpotTests(unittest.TestCase):
    def test_offers_clear_rectangles_near_the_base(self) -> None:
        spots = free_spots((30, 30), terrain(), DEFS, occupied=set())
        self.assertEqual([letter for letter, _, _ in spots], ["A", "B", "C"])
        for _, rect, note in spots:
            self.assertTrue(rect.contains(rect.x1, rect.z1))
            self.assertIn("crops", note)

    def test_skips_occupied_cells_and_water(self) -> None:
        occupied = {(x, z) for x in range(25, 36) for z in range(25, 36)}
        spots = free_spots((30, 30), terrain(), DEFS, occupied)
        for _, rect, _ in spots:
            self.assertFalse(any(cell in occupied for cell in rect))

        spots = free_spots((30, 30), terrain(index=4), DEFS, set())  # all water
        self.assertEqual(spots, [])

    def test_spots_do_not_overlap_each_other(self) -> None:
        spots = free_spots((30, 30), terrain(), DEFS, set())
        rects = [rect for _, rect, _ in spots]
        for i, rect in enumerate(rects):
            for other in rects[i + 1:]:
                self.assertFalse(rect.overlaps(other))


if __name__ == "__main__":
    unittest.main()
