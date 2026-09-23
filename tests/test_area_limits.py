"""Wide area orders and crop readiness.

Run from the repo root:  python -m unittest tests/test_area_limits.py -v
"""

import unittest

from rimagent.construction import MAX_MINE_CELLS, Rect
from rimagent.observation import describe_crop
from tests.test_map_actions import FakeRimApi, MapTestCase, thing

BIG = 40  # a map big enough for a 21x21 box


class BigMapFake(FakeRimApi):
    """FakeRimApi on a 40x40 soil map."""

    def __call__(self, method, path, **kwargs):
        if path == "/api/v1/map/terrain":
            self.calls.append((method, path, kwargs))
            return {"width": BIG, "height": BIG, "palette": ["Soil"], "grid": [BIG * BIG, 0]}
        return super().__call__(method, path, **kwargs)


WIDE = Rect(10, 10, 30, 30)  # 21 x 21 = 441 cells, the box a run was refused for


class WideAreaTests(MapTestCase):
    def test_allow_items_takes_a_wide_box(self) -> None:
        fake = BigMapFake(items=[thing(1, "Steel", 11, 12, forbidden=True),
                                 thing(2, "Steel", 29, 28, forbidden=True)])
        self.assertEqual(self.client_with(fake).allow_items(WIDE), 2)
        (_, path, kwargs), = fake.writes()
        self.assertEqual(path, "/api/v1/things/set-forbidden")
        self.assertCountEqual(kwargs["json"]["thing_ids"], [1, 2])

    def test_harvest_takes_a_wide_box(self) -> None:
        fake = BigMapFake()
        self.client_with(fake).designate("harvest", WIDE)
        self.assertEqual(fake.writes()[0][2]["json"]["type"], "harvest")

    def test_mining_stays_bounded(self) -> None:
        self.assertLess(MAX_MINE_CELLS, WIDE.cells)
        fake = BigMapFake()
        with self.assertRaisesRegex(ValueError, f"limit is {MAX_MINE_CELLS}"):
            self.client_with(fake).designate("mine", WIDE)
        self.assertEqual(fake.writes(), [])

    def test_chop_trees_takes_a_wide_box(self) -> None:
        fake = BigMapFake(plants=[thing(1, "Plant_TreeOak", 10, 10), thing(2, "Plant_TreeOak", 30, 30)])
        done, total = self.client_with(fake).chop_trees(WIDE)
        self.assertEqual((len(done), total), (2, 2))

    def test_still_refuses_boxes_off_the_map(self) -> None:
        fake = BigMapFake(items=[thing(1, "Steel", 11, 12, forbidden=True)])
        with self.assertRaisesRegex(ValueError, "outside the 40x40 map"):
            self.client_with(fake).allow_items(Rect(10, 10, BIG, 20))
        self.assertEqual(fake.writes(), [])


def crop(total: int, growth_sum: float, ripe: int = 0) -> dict:
    """A crop entry as RIMAPI 1.10 sends it, growth summed over the plants."""
    return {"plant_def_name": "Plant_Potato", "total_plants": total, "harvestable_plants": ripe,
            "growth_progress_average": growth_sum, "days_until_harvest": 0.0}


class CropTests(unittest.TestCase):
    def test_growth_is_the_average_not_the_sum(self) -> None:
        # Live: 26 potato plants summing to 0.758 were shown as "76%"; really ~3%.
        self.assertIn("average growth 3%", describe_crop(crop(26, 0.758), {}))
        self.assertIn("average growth 50%", describe_crop(crop(8, 4.0), {}))

    def test_says_there_is_nothing_to_harvest_yet(self) -> None:
        text = describe_crop(crop(8, 0.127), {"grow_days": 5.8})
        self.assertIn("full growth takes 5.8 days in good conditions", text)
        self.assertIn("none ripe yet, so there is nothing to harvest: let time pass", text)

    def test_ripe_crops_are_harvested_by_growers(self) -> None:
        text = describe_crop(crop(8, 8.0, ripe=8), {"grow_days": 5.8})
        self.assertIn("average growth 100%", text)
        self.assertNotIn("full growth takes", text)
        self.assertIn("8 ripe, harvested by colonists with Growing work", text)

    def test_an_empty_field_does_not_divide_by_zero(self) -> None:
        self.assertIn("0 plants, average growth 0%", describe_crop(crop(0, 0.0), {}))


if __name__ == "__main__":
    unittest.main()
