"""Colony stock versus everything else, stone chunks, and grouped blueprints.

Run from the repo root:  python -m unittest tests/test_supplies.py -v
"""

import unittest

from rimagent.agent import describe_blueprints
from rimagent.blueprints import TrackedBlueprint
from rimagent.choices import Choices
from rimagent.construction import BUILDINGS, FLOORS, Rect, TerrainMap, room_layout
from rimagent.rimapi import MapThing, WorkTable
from rimagent.supplies import describe_supplies, how_to_get, material_needs, shortfall
from tests.test_map_actions import MapTestCase
from tests.test_turn_loop import RESEARCH, WORK_TYPES, GridFake, colonist
from tests.test_map_actions import HEIGHT, PALETTE, WIDTH, terrain_grid


def stack(thing_id: int, def_name: str, count: int, forbidden: bool = False, x: int = 1) -> MapThing:
    return MapThing(thing_id=thing_id, def_name=def_name, position={"x": x, "z": 1},
                    stack_count=count, is_forbidden=forbidden)


ITEMS = [
    stack(1, "WoodLog", 120),                  # in a stockpile
    stack(2, "Steel", 75),                     # in a stockpile
    stack(3, "Steel", 45, x=30),               # lying outside storage
    stack(4, "ChunkSandstone", 559, x=40),     # lying around: not a material
    stack(5, "Steel", 720, forbidden=True),
]
STORED = ITEMS[:2]
STONECUTTER = WorkTable(id=9, thing_def="TableStonecutter", label="stonecutter's table",
                        position={"x": 5, "z": 5})


class SupplyTests(unittest.TestCase):
    def test_stock_is_what_the_resource_readout_shows(self) -> None:
        lines = describe_supplies(ITEMS, STORED)
        self.assertIn("Colony stock (allowed, in stockpiles and shelves, as the game's resource "
                      "readout shows): WoodLog 120, Steel 75.", lines)
        self.assertIn("Also allowed but outside storage (usable; colonists fetch it): "
                      "ChunkSandstone 559, Steel 45.", lines)
        self.assertIn("Forbidden, unusable until allow_items: Steel 720.", lines)

    def test_stored_entries_that_are_not_loose_items_are_ignored(self) -> None:
        # Live: steel inside frames and grass growing in a stockpile were
        # listed as stored.
        phantom = [stack(50, "Steel", 45), stack(51, "Plant_Grass", 1)]
        lines = describe_supplies(ITEMS, STORED + phantom)
        self.assertIn("readout shows): WoodLog 120, Steel 75.", lines[0])

    def test_materials_are_listed_apart_and_chunks_are_not_among_them(self) -> None:
        stuff = next(line for line in describe_supplies(ITEMS, STORED) if line.startswith("Building materials"))
        self.assertEqual(stuff, "Building materials (stuff) usable now: WoodLog 120, Steel 120; "
                                "stone blocks: none.")

    def test_chunks_are_mentioned_once_without_pushing_a_stonecutter(self) -> None:
        # Told in four places to build a stonecutter's table, a model planned ten.
        plain = describe_supplies(ITEMS, STORED, (), {"Stonecutting"})[-1]
        self.assertEqual(plain, "Stone chunks (ChunkSandstone 559) are not building material; build in wood.")
        self.assertNotIn("TableStonecutter", " ".join(describe_supplies(ITEMS, STORED, (), {"Stonecutting"})))
        planned = describe_supplies(ITEMS, STORED, (), set(), [dict(def_name="TableStonecutter", stuff="WoodLog")])
        self.assertIn("Your stonecutter's table is planned", planned[-1])
        built = describe_supplies(ITEMS, STORED, [STONECUTTER])[-1]
        self.assertIn("Your stonecutter's table can cut them into blocks (20 per chunk) with a bill such as "
                      "Make_StoneBlocksSandstone, if you want stone.", built)

    def test_without_the_stored_read_everything_allowed_is_one_line(self) -> None:
        self.assertIn("Allowed items on the map (usable): ChunkSandstone 559, WoodLog 120, Steel 120.",
                      describe_supplies(ITEMS, None))

    def test_how_to_get_each_kind_of_material(self) -> None:
        self.assertIn("stonecutter's table (bill Make_StoneBlocksSandstone)", how_to_get("BlocksSandstone"))
        self.assertIn("chop trees", how_to_get("WoodLog"))
        self.assertIn("allow_items on forbidden Steel", how_to_get("Steel"))


class ShortfallTests(unittest.TestCase):
    def test_a_room_needs_its_walls_and_door(self) -> None:
        room = [dict(def_name=name, stuff="WoodLog", x=x, z=z) for name, x, z in
                room_layout(Rect(2, 2, 7, 6), "north")]
        self.assertEqual(material_needs(room, BUILDINGS, FLOORS), {"WoodLog": 17 * 5 + 25})
        self.assertIn("needs 110 WoodLog but only 50 usable now: chop trees",
                      shortfall({"WoodLog": 110}, [stack(1, "WoodLog", 50)]))
        self.assertEqual(shortfall({"WoodLog": 110}, [stack(1, "WoodLog", 200)]), "")

    def test_fixed_costs_and_floors_count_too(self) -> None:
        need = material_needs([dict(def_name="TableStonecutter", stuff="WoodLog", x=1, z=1),
                               dict(def_name="WoodPlankFloor", stuff=None, x=1, z=1, floor=True)],
                              BUILDINGS, FLOORS)
        self.assertEqual(need, {"Steel": 30, "WoodLog": 75 + 3})

    def test_forbidden_material_does_not_count(self) -> None:
        self.assertIn("only 0 usable", shortfall({"Steel": 5}, [stack(1, "Steel", 700, forbidden=True)]))


class ChoiceMaterialTests(MapTestCase):
    def choices(self, items) -> Choices:
        defs = self.client_with(GridFake()).get_game_defs()
        terrain = TerrainMap(WIDTH, HEIGHT, PALETTE, terrain_grid())
        return Choices.observe([colonist()], WORK_TYPES, [], RESEARCH, defs, terrain, items, [], ())

    def test_only_materials_the_colony_has_are_offered(self) -> None:
        c = self.choices([stack(1, "WoodLog", 100), stack(2, "ChunkSandstone", 500),
                          stack(3, "Steel", 300, forbidden=True)])
        self.assertEqual(c.fields["build_room"]["stuff"], ["WoodLog"])
        self.assertEqual(c.materials["Wall"], ["WoodLog"])
        self.assertTrue(any("stuff Steel: none usable" in u for u in c.unavailable))

    def test_the_schema_offers_wood_first(self) -> None:
        c = self.choices([stack(1, "WoodLog", 100), stack(2, "Steel", 300)])
        self.assertEqual(c.fields["build_room"]["stuff"], ["WoodLog", "Steel"])
        branch = next(b for b in c.schema()["properties"]["actions"]["items"]["anyOf"]
                      if b["properties"]["action"]["const"] == "place_blueprint"
                      and "Wall" in b["properties"]["building_def"]["enum"])
        self.assertEqual(branch["properties"]["stuff"]["enum"], ["WoodLog", "Steel"])

    def test_nothing_to_build_with_leaves_out_material_orders(self) -> None:
        c = self.choices([stack(2, "ChunkSandstone", 500)])
        self.assertNotIn("build_room", c.fields)
        self.assertNotIn("build_wall", c.fields)
        self.assertNotIn("Wall", c.materials)


def tracked(def_name: str, x: int, z: int, stuff: str | None = "BlocksSandstone") -> TrackedBlueprint:
    return TrackedBlueprint(def_name=def_name, stuff=stuff, x=x, z=z, cells=(x, z, x, z))


class BlueprintGroupTests(unittest.TestCase):
    def test_a_room_is_one_line(self) -> None:
        room = [(tracked(name, x, z), "waiting") for name, x, z in room_layout(Rect(117, 132, 121, 136), "south")]
        bed = TrackedBlueprint(def_name="Bed", stuff="WoodLog", x=130, z=130, cells=(130, 130, 130, 131))
        lines = describe_blueprints(room + [(bed, "waiting")], {"WoodLog": 300}, [])
        self.assertEqual(len(lines), 2)
        self.assertIn("- 15 Wall + 1 Door (BlocksSandstone) within (117,132)-(121,136): 16 waiting", lines[0])
        self.assertIn("nobody can build with BlocksSandstone, none is usable: cut ChunkSandstone", lines[0])
        self.assertEqual(lines[1], "- Bed (WoodLog) covering (130,130)-(130,131): waiting - "
                                   "nobody has Construction on: enable_work Construction")

    def test_names_the_builders_when_nothing_is_missing(self) -> None:
        bed = TrackedBlueprint(def_name="Bed", stuff="WoodLog", x=1, z=1, cells=(1, 1, 1, 2))
        line, = describe_blueprints([(bed, "waiting")], {"WoodLog": 300}, ["Squint", "Reikguard"])
        self.assertIn("Construction is on for Squint, Reikguard: they build it while time advances", line)

    def test_mixed_progress_is_counted(self) -> None:
        run = [(tracked("Wall", x, 5, "WoodLog"), "under construction" if x < 3 else "waiting")
               for x in range(6)]
        line, = describe_blueprints(run, {"WoodLog": 100})
        self.assertIn("6 Wall (WoodLog) within (0,5)-(5,5): 3 under construction, 3 waiting", line)


class StoredReadTests(MapTestCase):
    def test_things_held_inside_buildings_are_left_out(self) -> None:
        fake = GridFake()
        fake.stored = {"ResourcesRaw": [
            {"thing_id": 1, "def_name": "Steel", "position": {"x": 3, "y": 0, "z": 4}, "stack_count": 75},
            {"thing_id": 2, "def_name": "Steel", "position": {"x": -1000, "y": -1000, "z": -1000},
             "stack_count": 30},
        ]}
        original = fake.__call__
        client = self.client_with(fake)
        client._request = lambda m, p, **k: fake.stored if p == "/api/v1/resources/stored" else original(m, p, **k)
        self.assertEqual([(t.thing_id, t.stack_count) for t in client.get_stored_items()], [(1, 75)])


if __name__ == "__main__":
    unittest.main()


class CommitmentTests(unittest.TestCase):
    """What unfinished builds already claim (a model counted Steel 380 as free
    while a generator and four steel tables still needed it)."""

    TABLE = dict(def_name="TableStonecutter", stuff="Steel")  # 75 + 30 Steel

    def test_claimed_material_is_taken_off_what_is_left(self) -> None:
        from rimagent.supplies import describe_commitments
        line = describe_commitments([stack(1, "Steel", 300)], None, [self.TABLE, self.TABLE])
        self.assertIn("need Steel 210 in all", line)
        self.assertIn("Left for new orders after that: Steel 90", line)

    def test_material_already_in_frames_is_not_counted_twice(self) -> None:
        from rimagent.supplies import describe_commitments
        items = [stack(1, "Steel", 100)]
        in_frame = stack(70, "Steel", 45)  # stored, but not a loose item: inside a frame
        line = describe_commitments(items, items + [in_frame], [self.TABLE])
        self.assertIn("Steel 45 of that is already delivered to their frames", line)
        self.assertIn("still to come from stock: Steel 60", line)
        self.assertIn("Left for new orders after that: Steel 40", line)

    def test_a_shortfall_says_how_to_get_more(self) -> None:
        from rimagent.supplies import describe_commitments, short_materials
        builds = [self.TABLE] * 4
        self.assertIn("Steel short by 344 (allow_items on forbidden Steel, or mine it)",
                      describe_commitments([stack(1, "Steel", 76)], None, builds))
        self.assertEqual(short_materials([stack(1, "Steel", 76)], None, builds), {"Steel"})
        self.assertEqual(short_materials([stack(1, "Steel", 500)], None, builds), set())

    def test_blueprints_short_of_material_say_so(self) -> None:
        bench = TrackedBlueprint(def_name="TableStonecutter", stuff="Steel", x=1, z=1, cells=(1, 1, 3, 1))
        line, = describe_blueprints([(bench, "waiting")], {"Steel": 76}, ["Ada"], {"Steel"})
        self.assertIn("not enough Steel for all your unfinished builds", line)
        self.assertNotIn("materials are there", line)


class MaterialOrderTests(unittest.TestCase):
    def test_wood_and_stone_come_before_steel(self) -> None:
        self.assertEqual(BUILDINGS["Wall"].materials()[0], "WoodLog")
        self.assertEqual(BUILDINGS["Wall"].materials()[-1], "Steel")
        self.assertEqual(BUILDINGS["Wall"].cost_text(), "5 of WoodLog/stone blocks/Steel")
        self.assertEqual(BUILDINGS["TableButcher"].cost_text(), "75 of WoodLog/Steel + 20 WoodLog")
