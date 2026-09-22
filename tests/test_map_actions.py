"""Zones, blueprints and designations, tested against a fake RIMAPI.

Run from the repo root:  python -m unittest tests/test_map_actions.py -v
"""

import unittest

from pydantic import ValidationError

from rimagent.agent import Decision
from rimagent.construction import Rect, TerrainMap, footprint
from rimagent.rimapi import RimApiClient

# A 10x10 map: soil everywhere, except gravel (fertility 0.7) in row z=0 and
# smooth granite (fertility 0) at x=9.
WIDTH = HEIGHT = 10
PALETTE = ["Soil", "Gravel", "Granite_Smooth"]


def terrain_grid() -> list[int]:
    cells = []
    for z in range(HEIGHT):
        for x in range(WIDTH):
            cells.append(2 if x == 9 else 1 if z == 0 else 0)
    grid: list[int] = []  # run-length encode, as RIMAPI does
    for value in cells:
        if grid and grid[-1] == value:
            grid[-2] += 1
        else:
            grid += [1, value]
    return grid


DEFS = {
    "terrain_defs": [
        {"def_name": "Soil", "fertility": 1.0, "affordances": ["Light", "Medium", "Heavy"]},
        {"def_name": "Gravel", "fertility": 0.7, "affordances": ["Light", "Medium", "Heavy"]},
        {"def_name": "Granite_Smooth", "fertility": 0.0, "affordances": ["Light", "Medium", "Heavy"]},
    ],
    "plant_defs": [{"def_name": "Plant_Potato", "fertility_min": 0.7}],
    "things_defs": [{"def_name": n} for n in ("Wall", "Bed", "Campfire", "WoodLog", "Steel")],
}


class FakeRimApi:
    """Answers the endpoints the map methods use, and records every call."""

    def __init__(
        self,
        issues: dict | None = None,
        research: list[str] | None = None,
        items: list[dict] | None = None,
        plants: list[dict] | None = None,
        things_at: dict | None = None,
    ):
        self.calls: list[tuple[str, str, dict]] = []
        self.issues = issues or {}
        self.research = research if research is not None else ["ComplexFurniture"]
        self.items = items or []
        self.plants = plants or []
        self.things_at = things_at or {}

    def __call__(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == "/api/v1/def/all":
            return DEFS
        if path == "/api/v1/map/terrain":
            return {"width": WIDTH, "height": HEIGHT, "palette": PALETTE, "grid": terrain_grid()}
        if path == "/api/v1/research/finished":
            return {"finished_projects": self.research}
        if path == "/api/v1/builder/check-zone":
            return {"can_build": not self.issues, "issues": self.issues}
        if path == "/api/v1/map/zone/growing":
            return {"zone": {"id": 7}}
        if path == "/api/v1/map/zone/stockpile":
            return {"zone_id": 8}
        if path == "/api/v1/map/things":
            return self.items
        if path == "/api/v1/map/plants":
            return self.plants
        if path == "/api/v1/map/things-at":
            cell = kwargs["json"]["position"]
            return self.things_at.get((cell["x"], cell["z"]), [])
        return None

    def writes(self) -> list[tuple[str, str, dict]]:
        """Calls that change the game (everything except reads and area checks)."""
        return [c for c in self.calls if c[0] != "GET" and c[1] != "/api/v1/builder/check-zone"]


class MapTestCase(unittest.TestCase):
    def client_with(self, fake: FakeRimApi) -> RimApiClient:
        client = RimApiClient(base_url="http://example.invalid")
        client._request = fake
        self.addCleanup(client.close)
        return client


class GeometryTests(unittest.TestCase):
    def test_terrain_grid_decodes_row_by_row(self) -> None:
        terrain = TerrainMap(WIDTH, HEIGHT, PALETTE, terrain_grid())
        self.assertEqual(terrain.at(3, 0), "Gravel")
        self.assertEqual(terrain.at(3, 5), "Soil")
        self.assertEqual(terrain.at(9, 5), "Granite_Smooth")

    def test_terrain_grid_rejects_wrong_size(self) -> None:
        with self.assertRaises(ValueError):
            TerrainMap(WIDTH, HEIGHT, PALETTE, [5, 0])

    def test_rect_normalises_corners(self) -> None:
        rect = Rect.from_corners(5, 8, 2, 3)
        self.assertEqual((rect.x1, rect.z1, rect.x2, rect.z2), (2, 3, 5, 8))
        self.assertEqual(rect.cells, 4 * 6)

    def test_footprint_follows_rotation(self) -> None:
        self.assertEqual(footprint(5, 5, (1, 1), 2), Rect(5, 5, 5, 5))
        self.assertEqual(footprint(5, 5, (1, 2), 0), Rect(5, 5, 5, 6))   # bed north
        self.assertEqual(footprint(5, 5, (1, 2), 2), Rect(5, 4, 5, 5))   # bed south
        self.assertEqual(footprint(5, 5, (3, 1), 1), Rect(5, 4, 5, 6))   # table east


class GrowingZoneTests(MapTestCase):
    def test_creates_zone_on_fertile_soil(self) -> None:
        fake = FakeRimApi()
        zone_id = self.client_with(fake).create_growing_zone("Plant_Potato", Rect(1, 0, 3, 2))
        self.assertEqual(zone_id, 7)
        self.assertEqual(fake.writes(), [(
            "POST", "/api/v1/map/zone/growing",
            {"json": {"map_id": 0, "plant_def": "Plant_Potato",
                      "point_a": {"x": 1, "y": 0, "z": 0}, "point_b": {"x": 3, "y": 0, "z": 2}}},
        )])

    def test_refuses_infertile_cells(self) -> None:
        fake = FakeRimApi()
        with self.assertRaisesRegex(ValueError, "too infertile"):
            self.client_with(fake).create_growing_zone("Plant_Potato", Rect(7, 2, 9, 4))
        self.assertEqual(fake.writes(), [])

    def test_refuses_crops_outside_the_list(self) -> None:
        fake = FakeRimApi()
        with self.assertRaisesRegex(ValueError, "not an allowed crop"):
            self.client_with(fake).create_growing_zone("Plant_Smokeleaf", Rect(1, 1, 2, 2))
        self.assertEqual(fake.writes(), [])

    def test_refuses_areas_over_the_limit(self) -> None:
        fake = FakeRimApi()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.client_with(fake).create_growing_zone("Plant_Potato", Rect(0, 0, 12, 3))
        self.assertEqual(fake.writes(), [])

    def test_refuses_blocked_areas(self) -> None:
        fake = FakeRimApi(issues={"ores": [{"x": 2, "z": 2, "def_name": "Granite", "label": "granite"}]})
        with self.assertRaisesRegex(ValueError, "granite at \\(2,2\\)"):
            self.client_with(fake).create_growing_zone("Plant_Potato", Rect(1, 1, 3, 3))
        self.assertEqual(fake.writes(), [])


class StockpileTests(MapTestCase):
    def test_always_sends_a_priority_that_stores_items(self) -> None:
        fake = FakeRimApi()
        self.assertEqual(self.client_with(fake).create_stockpile(Rect(1, 1, 3, 3)), 8)
        (_, _, kwargs), = fake.writes()
        self.assertEqual(kwargs["json"]["priority"], 2)  # 0 would be "Unstored"

    def test_low_priority(self) -> None:
        fake = FakeRimApi()
        self.client_with(fake).create_stockpile(Rect(1, 1, 3, 3), priority="low")
        self.assertEqual(fake.writes()[0][2]["json"]["priority"], 1)

    def test_refuses_to_overlap_another_zone(self) -> None:
        fake = FakeRimApi(issues={"zones": [{"x": 2, "z": 2, "label": "Potato field", "zone_type": "Growing"}]})
        with self.assertRaisesRegex(ValueError, "Potato field"):
            self.client_with(fake).create_stockpile(Rect(1, 1, 3, 3))
        self.assertEqual(fake.writes(), [])


class BlueprintTests(MapTestCase):
    def test_places_one_building_without_clearing_anything(self) -> None:
        fake = FakeRimApi()
        area = self.client_with(fake).place_blueprint("Bed", 4, 4, rotation=0, stuff="WoodLog")
        self.assertEqual(area, Rect(4, 4, 4, 5))
        (method, path, kwargs), = fake.writes()
        self.assertEqual((method, path), ("POST", "/api/v1/builder/blueprint"))
        body = kwargs["json"]
        self.assertIs(body["clear_obstacles"], False)
        self.assertEqual(body["position"], {"x": 4, "y": 0, "z": 4})
        self.assertEqual(body["blueprint"]["buildings"], [
            {"def_name": "Bed", "stuff_def_name": "WoodLog", "rel_x": 0, "rel_z": 0, "rotation": 0}
        ])

    def test_checks_the_buildings_footprint(self) -> None:
        fake = FakeRimApi()
        self.client_with(fake).place_blueprint("Bed", 4, 4, rotation=2, stuff="WoodLog")
        (check,) = [c for c in fake.calls if c[1] == "/api/v1/builder/check-zone"]
        self.assertEqual(check[2]["json"]["point_a"], {"x": 4, "y": 0, "z": 3})
        self.assertEqual(check[2]["json"]["point_b"], {"x": 4, "y": 0, "z": 4})

    def test_refuses_unresearched_buildings(self) -> None:
        fake = FakeRimApi(research=[])
        with self.assertRaisesRegex(ValueError, "ComplexFurniture"):
            self.client_with(fake).place_blueprint("Bed", 4, 4, stuff="WoodLog")
        self.assertEqual(fake.writes(), [])

    def test_refuses_wrong_or_missing_materials(self) -> None:
        client = self.client_with(FakeRimApi())
        with self.assertRaisesRegex(ValueError, "needs a material"):
            client.place_blueprint("Wall", 4, 4)
        with self.assertRaisesRegex(ValueError, "needs a material"):
            client.place_blueprint("Wall", 4, 4, stuff="Plasteel")
        with self.assertRaisesRegex(ValueError, "takes no material"):
            client.place_blueprint("Campfire", 4, 4, stuff="WoodLog")

    def test_refuses_materials_missing_from_the_game(self) -> None:
        fake = FakeRimApi()  # the fake game has WoodLog and Steel, but no stone blocks
        with self.assertRaisesRegex(ValueError, "does not exist in this game"):
            self.client_with(fake).place_blueprint("Wall", 4, 4, stuff="BlocksGranite")
        self.assertEqual(fake.writes(), [])

    def test_refuses_buildings_outside_the_catalog(self) -> None:
        fake = FakeRimApi()
        with self.assertRaisesRegex(ValueError, "not in the building catalog"):
            self.client_with(fake).place_blueprint("ShipReactor", 4, 4)
        self.assertEqual(fake.writes(), [])

    def test_refuses_overlapping_blueprints_and_frames(self) -> None:
        # check-zone can't see blueprints or frames, so place_blueprint checks cells.
        fake = FakeRimApi(things_at={(4, 5): [{"thing_id": 1, "def_name": "Blueprint_Wall",
                                               "position": {"x": 4, "y": 0, "z": 5}}]})
        with self.assertRaisesRegex(ValueError, r"\(4,5\), where Wall is already planned"):
            self.client_with(fake).place_blueprint("Bed", 4, 4, stuff="WoodLog")
        self.assertEqual(fake.writes(), [])

        fake = FakeRimApi(things_at={(4, 4): [{"thing_id": 2, "def_name": "Frame_Bed",
                                               "position": {"x": 4, "y": 0, "z": 4}}]})
        with self.assertRaisesRegex(ValueError, "Bed is already being built"):
            self.client_with(fake).place_blueprint("Wall", 4, 4, stuff="WoodLog")
        self.assertEqual(fake.writes(), [])

    def test_refuses_blocked_footprints(self) -> None:
        fake = FakeRimApi(issues={"buildings": [{"x": 4, "z": 5, "def_name": "Wall", "label": "wall"}]})
        with self.assertRaisesRegex(ValueError, "wall at \\(4,5\\)"):
            self.client_with(fake).place_blueprint("Bed", 4, 4, stuff="WoodLog")
        self.assertEqual(fake.writes(), [])

    def test_building_inside_a_stockpile_is_allowed(self) -> None:
        fake = FakeRimApi(issues={"zones": [{"x": 4, "z": 4, "label": "Stockpile 1", "zone_type": "Stockpile"}]})
        self.client_with(fake).place_blueprint("Wall", 4, 4, stuff="WoodLog")
        self.assertEqual(len(fake.writes()), 1)


class DesignationTests(MapTestCase):
    def test_sends_designation(self) -> None:
        fake = FakeRimApi()
        self.client_with(fake).designate("harvest", Rect(0, 0, 4, 4))
        self.assertEqual(fake.writes(), [(
            "POST", "/api/v1/order/designate/area",
            {"json": {"map_id": 0, "type": "harvest",
                      "point_a": {"x": 0, "y": 0, "z": 0}, "point_b": {"x": 4, "y": 0, "z": 4}}},
        )])

    def test_deconstruct_is_not_allowed(self) -> None:
        fake = FakeRimApi()
        with self.assertRaisesRegex(ValueError, "designation must be one of"):
            self.client_with(fake).designate("deconstruct", Rect(0, 0, 4, 4))
        self.assertEqual(fake.writes(), [])


def thing(thing_id: int, def_name: str, x: int, z: int, forbidden: bool = False) -> dict:
    return {"thing_id": thing_id, "def_name": def_name, "label": def_name.lower(),
            "position": {"x": x, "y": 0, "z": z}, "stack_count": 1, "is_forbidden": forbidden}


class AllowItemsTests(MapTestCase):
    def test_allows_only_forbidden_items_inside_the_rectangle(self) -> None:
        fake = FakeRimApi(items=[
            thing(1, "Steel", 2, 2, forbidden=True),
            thing(2, "MealSurvivalPack", 3, 3, forbidden=True),
            thing(3, "Silver", 3, 3, forbidden=False),      # already allowed
            thing(4, "Steel", 8, 8, forbidden=True),        # outside
        ])
        self.assertEqual(self.client_with(fake).allow_items(Rect(1, 1, 4, 4)), 2)
        self.assertEqual(fake.writes(), [(
            "POST", "/api/v1/things/set-forbidden",
            {"json": {"map_id": 0, "thing_ids": [1, 2], "forbidden": False}},
        )])

    def test_says_so_when_nothing_is_forbidden_there(self) -> None:
        fake = FakeRimApi(items=[thing(4, "Steel", 8, 8, forbidden=True)])
        with self.assertRaisesRegex(ValueError, "no forbidden items"):
            self.client_with(fake).allow_items(Rect(1, 1, 4, 4))
        self.assertEqual(fake.writes(), [])


class ChopTreesTests(MapTestCase):
    def test_marks_each_tree_cell_and_nothing_else(self) -> None:
        fake = FakeRimApi(plants=[
            thing(10, "Plant_TreeOak", 2, 2),
            thing(11, "Plant_Potato", 3, 3),      # a crop: must not be harvested
            thing(12, "Plant_TreePoplar", 4, 4),
            thing(13, "Plant_TreeOak", 8, 8),     # outside
        ])
        self.assertEqual(self.client_with(fake).chop_trees(Rect(1, 1, 5, 5)), 2)
        cells = [c[2]["json"]["point_a"] for c in fake.writes()]
        self.assertCountEqual(cells, [{"x": 2, "y": 0, "z": 2}, {"x": 4, "y": 0, "z": 4}])
        for _, path, kwargs in fake.writes():
            self.assertEqual(path, "/api/v1/order/designate/area")
            self.assertEqual(kwargs["json"]["type"], "harvest")
            self.assertEqual(kwargs["json"]["point_a"], kwargs["json"]["point_b"])

    def test_caps_trees_per_action_nearest_first(self) -> None:
        plants = [thing(100 + i, "Plant_TreeOak", x, z) for i, (x, z) in
                  enumerate((x, z) for x in range(9) for z in range(9))]  # 81 trees
        fake = FakeRimApi(plants=plants)
        self.assertEqual(self.client_with(fake).chop_trees(Rect(0, 0, 8, 8)), 30)
        self.assertEqual(len(fake.writes()), 30)
        first = fake.writes()[0][2]["json"]["point_a"]
        self.assertEqual((first["x"], first["z"]), (4, 4))  # the middle

    def test_says_so_when_there_are_no_trees(self) -> None:
        fake = FakeRimApi(plants=[thing(11, "Plant_Potato", 3, 3)])
        with self.assertRaisesRegex(ValueError, "no trees"):
            self.client_with(fake).chop_trees(Rect(1, 1, 5, 5))
        self.assertEqual(fake.writes(), [])


class DecisionTests(unittest.TestCase):
    def test_map_actions_need_their_fields(self) -> None:
        with self.assertRaises(ValidationError):
            Decision(action="create_growing_zone", x1=1, z1=1, x2=3, z2=3)  # no plant
        with self.assertRaises(ValidationError):
            Decision(action="place_blueprint", building_def="Bed", x=4)  # no z
        with self.assertRaises(ValidationError):
            Decision(action="designate", designation="deconstruct", x1=0, z1=0, x2=1, z2=1)

    def test_complete_map_action_parses(self) -> None:
        decision = Decision.model_validate_json(
            '{"action": "create_stockpile", "x1": 5, "z1": 9, "x2": 2, "z2": 3, '
            '"reason": "Store the supplies."}'
        )
        self.assertEqual(decision.rect(), Rect(2, 3, 5, 9))

    def test_reason_is_required(self) -> None:
        with self.assertRaises(ValidationError):
            Decision.model_validate_json('{"action": "wait"}')
        with self.assertRaises(ValidationError):
            Decision.model_validate_json('{"action": "wait", "reason": ""}')
        self.assertIn("reason", Decision.model_json_schema()["required"])  # so Ollama enforces it


if __name__ == "__main__":
    unittest.main()
