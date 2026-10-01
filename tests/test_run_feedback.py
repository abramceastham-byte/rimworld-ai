"""Fixes from the 2026-09-23 run review: tree cutting that went nowhere, rooms
refused over the colony's own door, trees ordered through designate, repeated
enable_work, and longer time windows.

Run from the repo root:  python -m unittest tests/test_run_feedback.py -v
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rimagent.actions import Turn
from rimagent.agent import RUN_TICKS, THREAT_RUN_TICKS, TICKS_PER_SECOND
from rimagent.chopping import STALE_TICKS, ChopTracker
from rimagent.construction import Rect, room_layout
from rimagent.execution import execute
from rimagent.rimapi import MapThing
from tests.test_map_actions import MapTestCase, thing
from tests.test_turn_loop import GridFake, WorkClient, colonist, work_order


def tree(thing_id: int, x: int, z: int) -> MapThing:
    return MapThing(thing_id=thing_id, def_name="Plant_TreeOak", position={"x": x, "z": z})


def order(**fields):
    return Turn.model_validate({"actions": [{"reason": "x", **fields}], "time": "hold"}).actions[0]


class TimeWindowTests(unittest.TestCase):
    def test_a_turn_is_an_hour_and_a_half_of_game_time(self) -> None:
        self.assertEqual(RUN_TICKS, 3750)                          # 1.5 in-game hours
        self.assertAlmostEqual(RUN_TICKS / (6 * TICKS_PER_SECOND), 10.4, places=1)  # speed 3 runs 6x normal
        self.assertEqual(THREAT_RUN_TICKS, 1125)                   # 30% while a threat is active


class ChopTrackerTests(unittest.TestCase):
    def test_felled_trees_are_counted_and_forgotten(self) -> None:
        chops = ChopTracker()
        chops.add([tree(1, 5, 5), tree(2, 6, 5), tree(3, 7, 5)], tick=1000)
        chops.reconcile([tree(3, 7, 5)])  # 1 and 2 are gone: cut
        self.assertEqual(chops.cut_since_last, 2)
        self.assertEqual([o.thing_id for o in chops.orders], [3])
        line = chops.describe(1500, ["Squint"])
        self.assertIn("2 cut since last turn", line)
        self.assertIn("1 still standing", line)
        self.assertIn("PlantCutting is on for: Squint.", line)

    def test_long_standing_trees_are_called_out_as_probably_too_young(self) -> None:
        chops = ChopTracker()
        chops.add([tree(1, 5, 5), tree(2, 9, 8)], tick=1000)
        chops.reconcile([tree(1, 5, 5), tree(2, 9, 8)])
        line = chops.describe(1000 + STALE_TICKS, ["Squint"])
        self.assertIn("2 of them ordered 2+ in-game hours ago, within (5,5)-(9,8), and none has fallen", line)
        self.assertIn("too young", line)

    def test_old_orders_are_not_blamed_on_youth_while_trees_still_fall(self) -> None:
        # One run was told "most likely too young" ten turns running while its wood went from 0 to 518.
        chops = ChopTracker()
        chops.add([tree(1, 5, 5), tree(2, 9, 8), tree(3, 6, 6)], tick=1000)
        chops.reconcile([tree(1, 5, 5), tree(2, 9, 8), tree(3, 6, 6)], tick=1000, wood=0)
        chops.reconcile([tree(1, 5, 5), tree(2, 9, 8)], tick=1000 + STALE_TICKS, wood=25)
        line = chops.describe(1000 + STALE_TICKS, ["Squint"])
        self.assertIn("trees are still falling", line)
        self.assertNotIn("too young", line)
        self.assertIn("Allowed WoodLog on the map: 25 (+25 since last turn)", line)

    def test_says_when_nobody_can_cut(self) -> None:
        chops = ChopTracker()
        chops.add([tree(1, 5, 5)], tick=0)
        self.assertIn("Nobody has PlantCutting on", chops.describe(10, []))

    def test_old_orders_may_be_repeated(self) -> None:
        chops = ChopTracker()
        chops.add([tree(1, 5, 5)], tick=0)
        self.assertEqual(chops.recent_ids(STALE_TICKS - 1), {1})
        self.assertEqual(chops.recent_ids(STALE_TICKS), set())

    def test_survives_a_restart_and_a_dry_run_saves_nothing(self) -> None:
        path = Path(tempfile.mkdtemp()) / "chops.json"
        ChopTracker(path).add([tree(1, 5, 5)], tick=7)
        self.assertEqual([o.thing_id for o in ChopTracker(path).orders], [1])
        dry = ChopTracker(path, read_only=True)
        dry.add([tree(2, 6, 6)], tick=8)
        self.assertEqual([o.thing_id for o in ChopTracker(path).orders], [1])

    def test_nothing_to_report_is_no_line(self) -> None:
        self.assertIsNone(ChopTracker().describe(0, ["Squint"]))


class ChopClient:
    """Enough of RimApiClient for execute's chop_trees and designate."""

    def __init__(self, trees):
        self.trees, self.ordered, self.submitted_blueprints = trees, [], []

    def begin_snapshot(self): pass
    def get_state(self): return SimpleNamespace(game_tick=2000)
    def get_trees(self, map_id=0): return self.trees

    def chop_trees(self, rect, map_id=0, skip=frozenset()):
        inside = [t for t in self.trees if rect.contains(t.position.x, t.position.z)]
        todo = [t for t in inside if t.thing_id not in skip][:30]
        self.ordered += [t.thing_id for t in todo]
        return todo, len(inside)


class ExecuteChopTests(unittest.TestCase):
    def test_repeating_the_same_rectangle_orders_nothing_new(self) -> None:
        client, chops = ChopClient([tree(1, 5, 5), tree(2, 6, 6)]), ChopTracker()
        chop = order(action="chop_trees", x1=0, z1=0, x2=9, z2=9)
        first = execute(client, chop, None, None, None, chops=chops)
        self.assertIn("ordered 2 trees cut", first)
        again = execute(client, chop, None, None, None, chops=chops)
        self.assertTrue(again.startswith("unchanged:"), again)
        self.assertIn("all 2 trees there were already ordered", again)
        self.assertEqual(client.ordered, [1, 2])  # nothing sent the second time

    def test_a_wider_box_only_adds_the_new_trees(self) -> None:
        client, chops = ChopClient([tree(1, 5, 5), tree(2, 6, 6), tree(3, 20, 20)]), ChopTracker()
        execute(client, order(action="chop_trees", x1=0, z1=0, x2=9, z2=9), None, None, None, chops=chops)
        result = execute(client, order(action="chop_trees", x1=0, z1=0, x2=29, z2=29), None, None, None,
                         chops=chops)
        self.assertIn("ordered 1 trees cut (2 there were already ordered)", result)

    def test_designate_over_trees_points_to_chop_trees(self) -> None:
        client = ChopClient([tree(1, 5, 5)])
        ops = SimpleNamespace(targets={"mine": [{"name": "granite", "x": 40, "z": 40}]})
        with self.assertRaisesRegex(ValueError, "holds 1 trees: use chop_trees"):
            execute(client, order(action="designate", designation="mine", x1=0, z1=0, x2=9, z2=9),
                    None, None, ops)


class RepeatWorkTests(unittest.TestCase):
    def test_already_enabled_says_where_to_look_instead(self) -> None:
        result = execute(WorkClient(colonist()), work_order("Construction"), None, None, None)
        self.assertIn("asking again does nothing", result)
        self.assertIn("missing materials", result)


class RoomDoorTests(MapTestCase):
    RECT = Rect(2, 2, 7, 6)  # north door at (4,6)

    def planned(self, def_name: str, x: int, z: int) -> dict:
        return {(x, z): [thing(99, f"Blueprint_{def_name}", x, z)]}

    def test_an_existing_door_on_a_wall_cell_is_kept(self) -> None:
        # Live: a room was refused over the colony's own planned door.
        fake = GridFake(things_at=self.planned("Door", 3, 2))
        summary = self.client_with(fake).build_room(self.RECT, "WoodLog", "north")
        cells = {(b["rel_x"] + 2, b["rel_z"] + 2) for b in
                 fake.writes()[0][2]["json"]["blueprint"]["buildings"]}
        self.assertNotIn((3, 2), cells)
        self.assertEqual(len(cells), 17)
        self.assertIn("kept 1 existing", summary)

    def test_a_wall_where_the_door_goes_is_fine_if_the_ring_has_a_door(self) -> None:
        fake = GridFake(things_at={**self.planned("Wall", 4, 6), **self.planned("Door", 3, 2)})
        self.client_with(fake).build_room(self.RECT, "WoodLog", "north")
        placed = fake.writes()[0][2]["json"]["blueprint"]["buildings"]
        self.assertEqual(len(placed), 16)
        self.assertNotIn("Door", {b["def_name"] for b in placed})

    def test_a_wall_where_the_only_door_goes_is_refused_with_a_hint(self) -> None:
        fake = GridFake(things_at=self.planned("Wall", 4, 6))
        with self.assertRaisesRegex(ValueError, r"\(4,6\) is where the north door goes: choose another door_side"):
            self.client_with(fake).build_room(self.RECT, "WoodLog", "north")
        self.assertEqual(fake.writes(), [])

    def test_the_size_error_explains_outer_versus_inside(self) -> None:
        with self.assertRaisesRegex(ValueError, "walls included; the inside is 2 smaller each way"):
            room_layout(Rect(0, 0, 2, 2), "north")


if __name__ == "__main__":
    unittest.main()


class StandingOrderTests(MapTestCase):
    """Orders already in effect aren't offered again (a run re-selected its
    research project four times and re-enabled Construction five)."""

    def choices(self, research, pawns):
        from rimagent.choices import Choices
        from rimagent.construction import TerrainMap
        from tests.test_map_actions import HEIGHT, PALETTE, WIDTH, terrain_grid
        from tests.test_turn_loop import WORK_TYPES
        defs = self.client_with(GridFake()).get_game_defs()
        return Choices.observe(pawns, WORK_TYPES, [], research, defs,
                               TerrainMap(WIDTH, HEIGHT, PALETTE, terrain_grid()), [], [], ())

    def research(self, current, *available):
        from rimagent.rimapi import ResearchProject, ResearchState
        return ResearchState(has_bench=True, current=ResearchProject(name=current) if current else None,
                             available=[ResearchProject(name=n) for n in available])

    def branches(self, c, action):
        return [b for b in c.schema()["properties"]["actions"]["items"]["anyOf"]
                if b["properties"]["action"]["const"] == action]

    def test_the_current_project_is_not_offered(self) -> None:
        c = self.choices(self.research("Batteries", "Batteries", "Brewing"), [colonist()])
        self.assertEqual(c.fields["set_research"], {"project": ["Brewing"]})
        only = self.choices(self.research("Batteries", "Batteries"), [colonist()])
        self.assertNotIn("set_research", only.fields)
        self.assertIn("set_research: already researching Batteries, nothing else to switch to",
                      only.unavailable)

    def test_enable_offers_only_work_that_is_off(self) -> None:
        # Ada has Construction on; WORK_TYPES are Construction, Research, Growing.
        c = self.choices(self.research(None, "Brewing"), [colonist(work=("Construction",))])
        enable, = self.branches(c, "enable_work")
        self.assertEqual(enable["properties"]["work_type"]["enum"], ["Growing", "Research"])
        disable, = self.branches(c, "disable_work")
        self.assertEqual(disable["properties"]["work_type"]["enum"], ["Construction"])

    def test_a_colonist_with_everything_on_gets_no_enable_branch(self) -> None:
        busy = colonist(work=("Construction", "Research", "Growing"))
        c = self.choices(self.research(None, "Brewing"), [busy])
        self.assertNotIn("enable_work", c.fields)

    def test_a_repeat_without_the_schema_is_reported_not_rejected(self) -> None:
        # Rejecting would throw away the whole reply; execution says "unchanged".
        c = self.choices(self.research("Batteries", "Batteries", "Brewing"), [colonist(work=("Construction",))])
        c.validate(Turn.model_validate({"actions": [
            {"action": "set_research", "project": "Batteries", "reason": "x"},
            {"action": "enable_work", "colonist": "Ada", "work_type": "Construction", "reason": "x"},
        ], "time": "hold"}))
