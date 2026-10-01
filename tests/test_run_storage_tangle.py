"""Fixes from the 2026-09-23 23:22 qwen3 run, where storage failed for ten turns
around a field planted in the storage area.

Run from the repo root:  python -m unittest tests/test_run_storage_tangle.py -v
"""

import tempfile
import unittest
from pathlib import Path

from rimagent.actions import Turn
from rimagent.construction import Rect, TerrainInfo, TerrainMap
from rimagent.execution import execute
from rimagent.hints import RetryList, TurnFacts, failure_hint
from rimagent.layout import LayoutTracker, field_area_refusal
from rimagent.mapview import suggested_field
from rimagent.observation import Operations
from rimagent.rimapi import Zone
from rimagent.zones import ZoneTracker


def order(**fields):
    return Turn.model_validate({"actions": [{"reason": "x", **fields}], "time": "hold"}).actions[0]


class PartZoneClient:
    """The game makes the stockpile but leaves out the cells under a shelf."""

    def __init__(self, made: int):
        self.made, self.zones = made, []

    def begin_snapshot(self): pass

    def get_zones(self, map_id=0): return self.zones

    def create_stockpile(self, rect):
        self.zones = [Zone(id=3, label="Stockpile 3", type="Zone_Stockpile", cells_count=self.made)]
        return 3


class PartZoneTests(unittest.TestCase):
    def test_a_zone_smaller_than_asked_is_made_and_kept(self) -> None:
        # Live: "zone 3 creation incomplete: expected 4 cells; observed 2", reported FAILED
        # though the stockpile existed, then retried for turns.
        zones = ZoneTracker(Path(tempfile.mkdtemp()) / "zones.json")
        result = execute(PartZoneClient(2), order(action="create_stockpile", x1=102, z1=136, x2=103, z2=137),
                         _Blueprints(), zones, Operations())
        self.assertIn("verified zone 3, 2 cells", result)
        self.assertIn("left out 2 of the 4 cells", result)
        self.assertEqual(zones.items[0].cell_count(), 2)   # matches the game, so it stays known
        self.assertEqual(ZoneTracker(zones.path).items[0].describe(), "stockpile (102,136)-(103,137), 2 cells")

    def test_no_cells_at_all_is_a_failure(self) -> None:
        zones = ZoneTracker(Path(tempfile.mkdtemp()) / "zones.json")
        with self.assertRaisesRegex(ValueError, "was not created"):
            execute(PartZoneClient(0), order(action="create_stockpile", x1=1, z1=1, x2=2, z2=2),
                    _Blueprints(), zones, Operations())
        self.assertEqual(zones.items, [])


class _Blueprints:
    submitted = []


class BlockedByZoneHintTests(unittest.TestCase):
    FIELD = {(x, z) for x in range(115, 121) for z in range(136, 141)}

    def test_the_hint_leaves_out_the_field(self) -> None:
        # Live: "The largest clear part of that area is (115,135)-(120,140)", the same blocked rectangle.
        pile = order(action="create_stockpile", x1=115, z1=132, x2=120, z2=140)
        hint, _ = failure_hint(pile, "area is blocked by Growing zone 1 at (117,137)",
                               TurnFacts(zone_cells=self.FIELD))
        self.assertIn("The largest clear part of that area is (115,132)-(120,135)", hint)

    def test_never_offers_the_same_rectangle_back(self) -> None:
        pile = order(action="create_stockpile", x1=110, z1=110, x2=113, z2=113)
        hint, _ = failure_hint(pile, "area is blocked by something at (111,111)", TurnFacts())
        self.assertNotIn("largest clear part", hint)
        self.assertIn("Nothing to fix", hint)


class RetryRepeatTests(unittest.TestCase):
    def test_the_same_order_failing_twice_is_dropped(self) -> None:
        retries = RetryList()
        pile = order(action="create_stockpile", x1=115, z1=135, x2=119, z2=139)
        retries.failed(pile, "blocked by Growing zone 1 HINT: The largest clear part ...")
        retries.end_turn()
        retries.failed(pile, "blocked by Growing zone 1 HINT: The largest clear part ...")
        self.assertEqual(retries.items, {})

    def test_enough_already_is_not_kept(self) -> None:
        retries = RetryList()
        retries.failed(order(action="create_growing_zone", plant="Plant_Potato", x1=0, z1=0, x2=9, z2=9),
                       "which would take your fields past 60; you already have 30, which is enough")
        self.assertEqual(retries.items, {})


class SuggestedFieldPlaceTests(unittest.TestCase):
    TERRAIN = TerrainMap(40, 40, ["Soil"], [1600, 0])
    DEFS = {"Soil": TerrainInfo(1.0, frozenset({"Heavy"}))}
    SPOT = [("A", Rect(0, 0, 39, 39), "40x40 open")]

    def test_keeps_off_the_base_middle(self) -> None:
        field = suggested_field(self.SPOT, self.TERRAIN, self.DEFS, (20, 20), 30, 0.7, gap=5)
        self.assertFalse(field.x1 - 5 <= 20 <= field.x2 + 5 and field.z1 - 5 <= 20 <= field.z2 + 5)

    def test_keeps_out_of_building_areas_and_prefers_the_fields_area(self) -> None:
        storage = Rect(0, 0, 39, 19)
        field = suggested_field(self.SPOT, self.TERRAIN, self.DEFS, (20, 10), 30, 0.7, avoid=[storage])
        self.assertGreaterEqual(field.z1, 20)
        fields = Rect(30, 30, 39, 39)
        field = suggested_field(self.SPOT, self.TERRAIN, self.DEFS, (20, 10), 30, 0.7, prefer=fields)
        self.assertTrue(fields.x1 <= field.x1 and field.x2 <= fields.x2 and fields.z1 <= field.z1)


class FieldAreaRefusalTests(unittest.TestCase):
    def test_a_field_in_the_storage_area_is_refused(self) -> None:
        # Live: plan_area storage (115,135)-(120,140), then a potato field over the same cells.
        layout = LayoutTracker(Path(tempfile.mkdtemp()) / "layout.json")
        layout.plan("storage", Rect(115, 135, 120, 140))
        field = order(action="create_growing_zone", plant="Plant_Potato", x1=115, z1=136, x2=120, z2=140)
        self.assertIn("overlaps your storage area", field_area_refusal(field, layout))
        layout.plan("fields", Rect(100, 100, 110, 110))
        self.assertIn("in your fields area (100,100)-(110,110)", field_area_refusal(field, layout))
        inside = order(action="create_growing_zone", plant="Plant_Potato", x1=100, z1=100, x2=105, z2=104)
        self.assertIsNone(field_area_refusal(inside, layout))


class WorkForToolkitsTests(unittest.TestCase):
    def choices(self):
        from rimagent.choices import Choices
        c = Choices(width=50, height=50)
        c.can_enable = {"Ada": {"Art", "Warden", "Growing"}, "Bo": {"Art"}}
        c.can_disable = {"Ada": {"Hauling"}, "Bo": {"Construction"}}
        c.fields = {"enable_work": {"colonist": ["Ada", "Bo"], "work_type": ["Art", "Growing", "Warden"]},
                    "disable_work": {"colonist": ["Ada", "Bo"], "work_type": ["Construction", "Hauling"]}}
        return c

    def test_only_the_toolkits_work_is_offered(self) -> None:
        from rimagent.modes import allowed_actions, narrowed, toolkit_work
        kits = ["farming"]
        c = narrowed(self.choices(), allowed_actions(kits), None, toolkit_work(kits))
        self.assertEqual(c.fields["enable_work"], {"colonist": ["Ada"], "work_type": ["Growing"]})
        self.assertNotIn("disable_work", c.fields)   # nobody has farming work on

    def test_nobody_able_means_no_work_orders(self) -> None:
        from rimagent.modes import allowed_actions, narrowed, toolkit_work
        kits = ["hunting"]
        c = narrowed(self.choices(), allowed_actions(kits), None, toolkit_work(kits))
        self.assertNotIn("enable_work", c.fields)
        self.assertNotIn("disable_work", c.fields)

    def test_the_work_toolkit_offers_everything(self) -> None:
        from rimagent.modes import allowed_actions, narrowed, toolkit_work
        kits = ["work"]
        c = narrowed(self.choices(), allowed_actions(kits), None, toolkit_work(kits))
        self.assertEqual(c.fields["enable_work"]["work_type"], ["Art", "Growing", "Warden"])


class WatchClient:
    """A 40x40 soil map; check-zone reports only what it is given."""

    def __init__(self, rock=()):
        self.rock = rock

    def get_terrain(self):
        return TerrainMap(40, 40, ["Soil"], [1600, 0])

    def check_area(self, rect):
        from rimagent.rimapi import AreaCheck
        return AreaCheck.model_validate({"ores": [{"x": x, "z": z, "def_name": "Granite"}
                                                  for x, z in self.rock if (x, z) in set(rect)]})


class HorseshoesTests(unittest.TestCase):
    def pin(self, x, z):
        return order(action="place_blueprint", building_def="HorseshoesPin", x=x, z=z, stuff="WoodLog")

    def walls(self, rect):
        from rimagent.blueprints import TrackedBlueprint
        from rimagent.rooms import perimeter
        return [(TrackedBlueprint(def_name="Wall", stuff="WoodLog", x=x, z=z, cells=(x, z, x, z)), "waiting")
                for x, z in perimeter(rect)]

    def test_inside_a_small_room_is_refused(self) -> None:
        # Live: a pin at (102,136) inside the 3x3 storage room (100,134)-(104,138), here moved onto a 40x40 map.
        facts = TurnFacts(report=self.walls(Rect(10, 10, 14, 14)))
        from rimagent.hints import watch_spot_refusal
        text = watch_spot_refusal(self.pin(12, 12), facts, WatchClient())
        self.assertIn("could never be played", text)
        self.assertIn("all 12 of those cells", text)

    def test_open_ground_or_one_side_clear_is_allowed(self) -> None:
        from rimagent.hints import watch_spot_refusal
        self.assertIsNone(watch_spot_refusal(self.pin(20, 20), TurnFacts(), WatchClient()))
        # Rock all around but a corridor east to the east strip.
        open_ = {(x, 20) for x in range(20, 26)} | {(25, 19), (25, 21)}
        rock = [(x, z) for x in range(15, 26) for z in range(15, 26) if (x, z) not in open_]
        self.assertIsNone(watch_spot_refusal(self.pin(20, 20), TurnFacts(), WatchClient(rock)))

    def test_a_wall_between_blocks_the_line(self) -> None:
        from rimagent.hints import watch_spot_refusal
        # A wall ring 3 cells out on every side: the 5-away cells are free but out of sight.
        facts = TurnFacts(report=self.walls(Rect(17, 17, 23, 23)))
        self.assertIsNotNone(watch_spot_refusal(self.pin(20, 20), facts, WatchClient()))

    def test_near_the_map_edge_counts_the_cells_on_the_map(self) -> None:
        from rimagent.hints import watch_spot_refusal
        self.assertIsNone(watch_spot_refusal(self.pin(2, 2), TurnFacts(), WatchClient()))



class RoomRefusalTests(unittest.TestCase):
    def room(self, purpose, x1, z1, x2, z2):
        return order(action="build_room", x1=x1, z1=z1, x2=x2, z2=z2, stuff="WoodLog", door_side="north",
                     purpose=purpose)

    def test_a_room_around_things_that_dont_belong_is_refused(self) -> None:
        # Live (gpt-oss): a storage room walled around its generator and campfire.
        from rimagent.layout import room_refusal
        contents = [("WoodFiredGenerator", Rect(118, 140, 119, 141), "planned"),
                    ("Campfire", Rect(120, 140, 120, 140), "planned")]
        text = room_refusal(self.room("storage", 117, 138, 123, 143), [], contents, None)
        self.assertIn("would shut in WoodFiredGenerator at (118,140)-(119,141), Campfire", text)
        self.assertIsNone(room_refusal(self.room("kitchen", 117, 138, 123, 143), [],
                                       contents[1:], None))  # a campfire belongs in a kitchen

    def test_rooms_may_share_a_wall_but_not_overlap(self) -> None:
        from rimagent.layout import room_refusal
        from rimagent.rooms import TrackedRoom
        rooms = [TrackedRoom(purpose="bedroom", door_side="south", cells=(114, 134, 118, 139))]
        self.assertIsNone(room_refusal(self.room("storage", 118, 134, 122, 139), rooms, [], None))  # shared wall
        self.assertIn("overlaps your bedroom room", room_refusal(self.room("storage", 116, 134, 120, 139),
                                                                 rooms, [], None))
        self.assertIsNone(room_refusal(self.room("bedroom", 114, 134, 118, 139), rooms, [], None))  # the same one

    def test_another_purposes_area_is_refused(self) -> None:
        from rimagent.layout import room_refusal
        layout = LayoutTracker(Path(tempfile.mkdtemp()) / "layout.json")
        layout.plan("fields", Rect(100, 100, 110, 110))
        layout.plan("bedrooms", Rect(120, 100, 130, 110))
        text = room_refusal(self.room("bedroom", 102, 102, 106, 107), [], [], layout)
        self.assertIn("would take over your fields area", text)
        self.assertIn("your bedrooms area (120,100)-(130,110)", text)
        self.assertIsNone(room_refusal(self.room("bedroom", 122, 102, 126, 107), [], [], layout))


class SitePlanTests(unittest.TestCase):
    def test_areas_rooms_and_zones_in_one_block(self) -> None:
        from rimagent.agent import site_plan
        from rimagent.zones import TrackedZone
        layout = LayoutTracker(Path(tempfile.mkdtemp()) / "layout.json")
        layout.plan("storage", Rect(115, 135, 120, 140))
        sections = {"layout": layout.describe([]),
                    "rooms": ["Your rooms (purpose as you tagged it):", "- storage (100,134)-(104,138) ..."]}
        zones = [TrackedZone(kind="growing", zone_id=0, label="Plant_Potato field", cells=(108, 145, 112, 150))]
        lines = site_plan(sections, zones, layout)
        self.assertIn("Nothing new may overlap any of it", lines[0])
        self.assertIn("- [1] storage (115,135)-(120,140): nothing yet", lines)
        self.assertIn("- storage (100,134)-(104,138) ...", lines)
        self.assertIn("- Plant_Potato field (108,145)-(112,150), 30 cells", lines)

    def test_areas_are_numbered_on_the_detail_map(self) -> None:
        from rimagent.mapview import detail_rows
        rows = detail_rows(Rect(0, 0, 4, 2), TerrainMap(5, 3, ["Soil"], [15, 0]),
                           {"Soil": TerrainInfo(1.0, frozenset({"Heavy"}))}, {(1, 1): "1"}, ["storage"])
        self.assertIn("  1 storage area (reserved, see the site plan)", rows)
        self.assertIn("  z=  1 .1...", rows)



class FurnishTests(unittest.TestCase):
    # The 00:12 gpt-oss run's storage room, door east, with the shelves it had built.
    ROOM = Rect(129, 140, 135, 145)

    def test_shelves_fit_inside_clear_of_the_doorway_and_what_stands(self) -> None:
        from rimagent.furnish import doorway, fits
        taken = {(130, 141), (131, 141), (132, 141), (133, 141)}
        spots = fits("Shelf", self.ROOM, "east", taken)
        self.assertEqual(len(spots), 3)
        inside = Rect(130, 141, 134, 144)
        used = set()
        for x, z, rotation, cells in spots:
            self.assertTrue(all(inside.contains(*c) for c in cells))
            self.assertFalse(set(cells) & (taken | used | {doorway(self.ROOM, "east")}))
            used |= set(cells)

    def test_a_work_table_keeps_its_working_spot_free(self) -> None:
        from rimagent.furnish import fits, work_cell
        for x, z, rotation, cells in fits("TableButcher", Rect(123, 145, 127, 150), "south", set()):
            front = work_cell("TableButcher", x, z, rotation)
            self.assertTrue(Rect(124, 146, 126, 149).contains(*front))
            self.assertNotIn(front, set(cells))

    def test_the_room_line_gives_orders_to_copy(self) -> None:
        from rimagent.furnish import furnish_line
        line = furnish_line("storage", self.ROOM, "east", [], {"Shelf": ["WoodLog"]}, stockpile=True)
        self.assertIn("stockpile: it has one already; don't create another.", line)
        self.assertIn("Shelf stuff=WoodLog fits at x=130 z=144 rotation=0 (covers (130,144)-(131,144))", line)
        self.assertIsNone(furnish_line("storage", self.ROOM, "east", [], {}))  # nothing buildable

    def test_a_room_may_hold_several_beds_and_says_when_enough(self) -> None:
        from rimagent.furnish import furnish_line
        bed = [("Bed", Rect(115, 135, 115, 136), "built")]
        line = furnish_line("bedroom", Rect(114, 134, 118, 139), "south", bed, {"Bed": ["WoodLog"]},
                            notes={"Bed": "beds for 3 of 3 colonists, built or planned: enough for everyone"})
        self.assertIn("Bed stuff=WoodLog (beds for 3 of 3 colonists, built or planned: enough for everyone) fits at",
                      line)

    def test_the_stockpile_note_survives_the_building_notes(self) -> None:
        from rimagent.furnish import furnish_line
        line = furnish_line("storage", self.ROOM, "east", [], {"Shelf": ["WoodLog"]}, stockpile=False,
                            notes={"Bed": "x"})
        self.assertTrue(line.startswith("  stockpile: none yet"))

    def test_each_work_table_gets_its_own_place(self) -> None:
        from rimagent.furnish import furnish_line
        line = furnish_line("workshop", Rect(123, 145, 127, 150), "south", [],
                            {"HandTailoringBench": ["WoodLog"], "TableStonecutter": ["WoodLog"]})
        self.assertEqual(line.count("rotation="), 2)
        self.assertNotIn("no room left", line)

    def test_a_failed_placement_in_a_room_is_offered_a_spot_in_that_room(self) -> None:
        # Live: Shelf rotation 1 at (134,141) reached the wall; the hint offered open ground 15 cells away.
        from rimagent.rooms import TrackedRoom
        facts = TurnFacts(rooms=[TrackedRoom(purpose="storage", door_side="east", cells=(129, 140, 135, 145))])
        shelf = order(action="place_blueprint", building_def="Shelf", x=134, z=141, rotation=1, stuff="WoodLog")
        hint, _ = failure_hint(shelf, "Shelf at (134,141) would cover (134,140), where Wall is already planned",
                               facts)
        self.assertIn("In that storage room, Shelf fits at x=130 z=144 rotation=0", hint)
        self.assertNotIn("free spot", hint)


class ShrunkZoneTests(unittest.TestCase):
    def test_a_stockpile_with_shelves_on_it_stays_on_the_site_plan(self) -> None:
        # Live: 20 cells, then 16, 13, 11 as shelves went up; it vanished from the prompt.
        from rimagent.agent import still_known
        from rimagent.zones import TrackedZone
        zone = TrackedZone(kind="stockpile", zone_id=1, label="stockpile", cells=(130, 141, 134, 144))
        field = TrackedZone(kind="growing", zone_id=2, label="Plant_Potato field", cells=(0, 0, 4, 5))
        known = still_known([zone, field], {1: 16, 2: 40})
        self.assertEqual([z.describe() for z in known], ["stockpile (130,141)-(134,144), 16 cells"])
        self.assertEqual(still_known([zone], {}), [])  # gone from the game

if __name__ == "__main__":
    unittest.main()
