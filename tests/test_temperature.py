"""The Architect's Temperature tab: heater, cooler and vent.

Run from the repo root:  python -m unittest tests/test_temperature.py -v
"""

import copy
import unittest

from rimagent.construction import BUILDING_USE, BUILDINGS, available_buildings
from tests import test_map_actions
from tests.test_map_actions import FakeRimApi, MapTestCase

DEFS = copy.deepcopy(test_map_actions.DEFS)
DEFS["things_defs"] += [{"def_name": n} for n in ("Heater", "Cooler", "Vent", "ComponentIndustrial")]


class TemperatureFake(FakeRimApi):
    """FakeRimApi in a colony with Electricity and Air conditioning done."""

    def __init__(self, **kwargs):
        super().__init__(research=["Electricity", "AirConditioning"], **kwargs)

    def __call__(self, method, path, **kwargs):
        if path == "/api/v1/def/all":
            self.calls.append((method, path, kwargs))
            return DEFS
        return super().__call__(method, path, **kwargs)


WALL = {"buildings": [{"x": 4, "z": 4, "def_name": "Wall", "label": "wall"}]}


class TemperatureBuildingTests(MapTestCase):
    def test_the_tab_is_in_the_catalog_with_its_research(self) -> None:
        self.assertEqual(BUILDINGS["Heater"].research, "Electricity")
        self.assertEqual(BUILDINGS["Cooler"].research, "AirConditioning")
        self.assertEqual(BUILDINGS["Cooler"].cost, (("Steel", 90), ("ComponentIndustrial", 3)))
        offered = available_buildings({"Electricity", "AirConditioning"})
        self.assertTrue({"Heater", "Cooler", "Vent"} <= set(offered))
        self.assertNotIn("Cooler", available_buildings({"Electricity"}))
        self.assertIn("freezer", BUILDING_USE["Cooler"])

    def test_a_cooler_or_vent_goes_over_a_built_wall(self) -> None:
        # The cooler replaces the wall it is built over (checked in game).
        for name in ("Cooler", "Vent"):
            fake = TemperatureFake(issues=WALL)
            self.client_with(fake).place_blueprint(name, 4, 4, rotation=2)
            self.assertEqual(len(fake.writes()), 1, name)

    def test_other_buildings_still_cant(self) -> None:
        fake = TemperatureFake(issues=WALL)
        with self.assertRaisesRegex(ValueError, "wall at \\(4,4\\)"):
            self.client_with(fake).place_blueprint("Heater", 4, 4)
        bed = {"buildings": [{"x": 4, "z": 4, "def_name": "Bed", "label": "bed"}]}
        with self.assertRaisesRegex(ValueError, "bed at \\(4,4\\)"):
            self.client_with(TemperatureFake(issues=bed)).place_blueprint("Cooler", 4, 4)

    def test_a_cooler_needs_air_conditioning(self) -> None:
        fake = FakeRimApi(research=["Electricity"])
        with self.assertRaisesRegex(ValueError, "AirConditioning research"):
            self.client_with(fake).place_blueprint("Cooler", 4, 4)


if __name__ == "__main__":
    unittest.main()
