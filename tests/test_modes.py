"""The overview/focus/threat loop, power routing, and orders to pawns.

Run from the repo root:  python -m unittest tests/test_modes.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rimagent.actions import Turn
from rimagent.agent import LEGACY_ACTIONS, run
from rimagent.construction import Rect
from rimagent.execution import Combat, execute
from rimagent.memory import AgentMemory, MemoryUpdate
from rimagent.modes import FocusTurn, allowed_actions, designations, narrowed
from rimagent.observation import read_operations
from rimagent.power import route
from rimagent.rimapi import Enemy, MapThing, Threat
from rimagent.runlog import RunLogger
from tests.test_turn_loop import ChoicesTests, LoopClient, ScriptedModel, colonist
from tests.test_map_actions import MapTestCase


def order(**fields):
    return Turn.model_validate({"actions": [dict(reason="test", **fields)], "time": "hold"}).actions[0]


def cells(x1, z1, x2, z2):
    return set(Rect(x1, z1, x2, z2))


class RouteTests(unittest.TestCase):
    bounds = Rect(0, 0, 40, 10)

    def test_plain_conduits_under_walls_and_hidden_ones_in_the_open(self) -> None:
        # Generator at the west end, lamp 20 cells east; a wall runs along z=5 between them.
        wall = cells(8, 5, 18, 5)
        path = route(cells(0, 5, 1, 6), {(22, 5)}, self.bounds, set(), wall, lambda x, z: False)
        placed = {(x, z): name for x, z, name in path}
        self.assertTrue(placed)
        self.assertTrue(all(name == "PowerConduit" for cell, name in placed.items() if cell in wall))
        self.assertTrue(all(name == "HiddenConduit" for cell, name in placed.items() if cell not in wall))
        # The cheap cells under the wall are used rather than open ground beside it.
        self.assertGreaterEqual(sum(1 for cell in placed if cell in wall), 8)

    def test_nothing_needed_when_already_in_reach(self) -> None:
        # A generator carries power itself: a lamp 3 cells away links to it directly.
        self.assertEqual(route(cells(0, 0, 1, 1), {(4, 1)}, self.bounds, set(), set(), lambda x, z: False), [])
        # So does a conduit line touching the generator.
        self.assertEqual(route(cells(0, 0, 1, 1), {(30, 1)}, self.bounds, cells(2, 1, 26, 1), set(),
                               lambda x, z: False), [])

    def test_existing_conduits_are_reused_for_free(self) -> None:
        # The old line stops two cells short of the generator: only the gap is new.
        line = cells(4, 2, 26, 2)
        path = route(cells(0, 2, 1, 3), {(30, 2)}, self.bounds, line, set(), lambda x, z: False)
        self.assertEqual(path, [(2, 2, "HiddenConduit"), (3, 2, "HiddenConduit")])

    def test_no_route_through_rock(self) -> None:
        rock = cells(10, 0, 10, 10)
        self.assertIsNone(route(cells(0, 5, 1, 5), {(25, 5)}, self.bounds, set(), set(),
                                lambda x, z: (x, z) in rock))


class ConnectPowerTests(unittest.TestCase):
    def test_posts_one_blueprint_of_conduits_and_says_the_cost(self) -> None:
        from rimagent.construction import TerrainMap
        from rimagent.rimapi import AreaCheck, Building, RimApiClient
        client = RimApiClient(base_url="http://example.invalid")
        generator = Building(id=1, def_name="WoodFiredGenerator", position={"x": 2, "z": 5}, size={"x": 2, "z": 2})
        lamp = Building(id=2, def_name="StandingLamp", position={"x": 25, "z": 5})
        walls = [Building(id=10 + x, def_name="Wall", position={"x": x, "z": 5}) for x in range(8, 16)]
        client.get_buildings = Mock(return_value=[generator, lamp, *walls])
        client.get_terrain = Mock(return_value=TerrainMap(40, 12, ["Soil"], [40 * 12, 0]))
        client.check_area = Mock(return_value=AreaCheck())
        client._request = Mock()
        result = client.connect_power((2, 5), (25, 5))
        body = client._request.call_args.kwargs["json"]["blueprint"]
        defs = {b["def_name"] for b in body["buildings"]}
        self.assertEqual(defs, {"PowerConduit", "HiddenConduit"})
        self.assertIn("under walls or buildings", result)
        self.assertRegex(result, r"needs \d+ Steel")
        client.close()


class PawnOrderTests(unittest.TestCase):
    def client(self, weapon_categories=("WeaponsRanged",)):
        client = Mock()
        client.get_colonists.return_value = [colonist()]
        client.get_equipment.return_value = [MapThing(thing_id=5, def_name="Gun_Revolver", position={"x": 0, "z": 0},
                                                      categories=list(weapon_categories))]
        return client

    def test_attack_needs_a_draft_first(self) -> None:
        client, combat = self.client(), Combat(enemies={7: Enemy(id=7, kind="Mech_Militor", x=9, z=9)})
        with self.assertRaisesRegex(ValueError, "draft them first"):
            execute(client, order(action="attack", colonist="Ada", target=7), None, None, None, combat=combat)
        execute(client, order(action="draft", colonist="Ada", drafted=True), None, None, None, combat=combat)
        client.set_drafted.assert_called_once_with(1, True)
        self.assertIn("Ada", combat.drafted)
        execute(client, order(action="attack", colonist="Ada", target=7), None, None, None, combat=combat)
        client.order_job.assert_called_with(1, "AttackStatic", target_thing_id=7)

    def test_melee_when_there_is_no_gun_and_repeat_draft_is_unchanged(self) -> None:
        client = self.client(weapon_categories=("WeaponsMelee",))
        combat = Combat(enemies={7: Enemy(id=7, kind="Mech_Militor", x=9, z=9)}, drafted={"Ada"})
        result = execute(client, order(action="draft", colonist="Ada", drafted=True), None, None, None,
                         combat=combat)
        self.assertIn("unchanged", result)
        client.set_drafted.assert_not_called()
        execute(client, order(action="attack", colonist="Ada", target=7), None, None, None, combat=combat)
        client.order_job.assert_called_with(1, "AttackMelee", target_thing_id=7)

    def test_equip_allows_a_forbidden_weapon_first(self) -> None:
        client = self.client()
        client.get_items.return_value = [MapThing(thing_id=44, def_name="Gun_Revolver", position={"x": 3, "z": 4},
                                                  is_forbidden=True, categories=["WeaponsRanged"])]
        result = execute(client, order(action="equip", colonist="Ada", item_id=44), None, None, None)
        client.allow_items.assert_called_once_with(Rect(3, 4, 3, 4))
        client.equip.assert_called_once_with(1, 44)
        self.assertIn("now allowed", result)


class NarrowedChoicesTests(MapTestCase):
    def test_only_the_toolkits_orders_and_designations(self) -> None:
        choices = ChoicesTests.choices(self)
        choices.fields["designate"] = {"designation": ["mine", "hunt"]}
        choices.unavailable += ["designate harvest: no verified targets", "chop_trees: no trees"]
        kits = ["hunting"]
        c = narrowed(choices, allowed_actions(kits), designations(kits))
        self.assertEqual(set(c.fields), {"designate", "enable_work", "disable_work"} & set(choices.fields)
                         | {"designate"})
        self.assertEqual(c.fields["designate"]["designation"], ["hunt"])
        self.assertNotIn("chop_trees: no trees", c.unavailable)
        self.assertNotIn("designate harvest: no verified targets", c.unavailable)
        # The original is untouched, and validation names the fix.
        self.assertIn("place_blueprint", choices.fields)
        turn = FocusTurn.model_validate({"actions": [dict(action="set_research", project="Brewing", reason="x")],
                                         "time": "hold", "plan": "continue"})
        with self.assertRaisesRegex(ValueError, "request its toolkit"):
            c.validate(turn, allowed_actions(kits))

    def test_single_loop_offers_only_the_old_orders(self) -> None:
        schema = ChoicesTests.choices(self).schema(LEGACY_ACTIONS)
        offered = {b["properties"]["action"]["const"] for b in schema["properties"]["actions"]["items"]["anyOf"]}
        self.assertTrue(offered <= LEGACY_ACTIONS)
        self.assertNotIn("draft", offered)


class ObservationTagTests(unittest.TestCase):
    def test_lessons_are_shown_only_for_their_toolkit(self) -> None:
        memory = AgentMemory(Path(tempfile.mkdtemp()))
        memory.apply_update(1, 10, MemoryUpdate(observation="Batteries drain at night without generators",
                                                toolkit="power"))
        memory.apply_update(2, 20, MemoryUpdate(observation="Raids come from the east edge"))
        self.assertEqual(memory.observations_for({"power"}), ["Batteries drain at night without generators"])
        self.assertEqual(memory.observations_for({"general"}), ["Raids come from the east edge"])
        self.assertEqual(memory.observations_for({"farming"}), [])


class HarvestTargetTests(unittest.TestCase):
    def test_a_sown_field_does_not_break_the_observation(self) -> None:
        client = LoopClient()
        farm = {"total_growing_zones": 1, "total_plants": 2, "crop_types": [
            {"plant_def_name": "Plant_Potato", "total_plants": 2, "harvestable_plants": 1,
             "growth_progress_average": 1.5}]}
        client._request = lambda method, path, **kw: farm if path.endswith("farm/summary") else None
        client.get_plants = lambda map_id=0: [{"def_name": "Plant_Potato", "position": {"x": 3, "z": 3}},
                                              {"def_name": "Plant_Potato", "position": {"x": 4, "z": 3}}]
        ops = read_operations(client, [colonist()], client.get_game_defs())
        self.assertIn("Field of Plant_Potato", "\n".join(ops.lines))
        self.assertEqual(len(ops.targets["harvest"]), 2)


OVERVIEW = {"goal": "Get Ada growing food", "uses": "Ada, who can grow", "steps": ["enable Growing"],
            "toolkits": ["work"],
            "finish": [{"kind": "have_items", "what": "Beer", "count": 1}]}  # never met in these fakes


class ModesLoopTests(unittest.TestCase):
    def run_loop(self, client, replies, steps):
        from dataclasses import replace

        # One forbidden Beer: "1 Beer usable" is a check the work toolkit can meet in
        # principle (allow it) but never does here, since no focus reply allows it.
        defs = client.get_game_defs()
        client.get_game_defs = lambda: replace(defs, thing_names=defs.thing_names | {"Beer"})
        stocked = client.get_items
        beer = MapThing(thing_id=999, def_name="Beer", position={"x": 9, "z": 0}, is_forbidden=True)
        client.get_items = lambda: [*stocked(), beer]
        tmp = Path(tempfile.mkdtemp())
        model = ScriptedModel(replies)
        memory = AgentMemory(tmp / "memory")
        run(client, model, RunLogger(tmp / "logs"), max_steps=steps, run_ticks=60, threat_run_ticks=60,
            memory=memory, loop="modes")
        return model, memory

    def test_starvation_ends_the_plan_once(self) -> None:
        from rimagent.rimapi import Alert

        # One run chopped trees for 12 focus turns while every colonist starved.
        client = LoopClient()
        alerts = iter([[], [Alert(label="Starvation")], [Alert(label="Starvation")]])
        client.get_alerts = lambda: next(alerts, [Alert(label="Starvation")])
        focus = {"actions": [], "time": "hold", "plan": "continue"}
        model, _ = self.run_loop(client, [OVERVIEW, focus, {**OVERVIEW, "goal": "Feed Ada"}, focus, focus],
                                 steps=3)
        kinds = ["overview" if "Toolkits (pick 1 to" in p else "focus" for p in model.prompts]
        self.assertEqual(kinds[:5], ["overview", "focus", "overview", "focus", "focus"])
        self.assertEqual(kinds.count("overview"), 2)  # not again while the starving goes on
        self.assertIn("URGENT: colonists are starving", model.prompts[2])

    def test_overview_then_focus_then_overview_once_done(self) -> None:
        client = LoopClient()
        model, memory = self.run_loop(client, [
            {**OVERVIEW, "finish": [{"kind": "have_items", "what": "WoodLog", "count": 50}]},  # met: 200 lie about
            {"actions": [{"action": "enable_work", "colonist": "Ada", "work_type": "Growing", "reason": "food"}],
             "time": "hold", "plan": "continue", "request_toolkits": ["farming"],
             "lesson": {"text": "colonists sow fields by themselves once Growing is on", "toolkit": "farming"}},
            {**OVERVIEW, "goal": "Research brewing", "toolkits": ["research"]},
            {"actions": [], "time": "advance", "plan": "replan"},
        ], steps=2)
        overview, focus, overview2, focus2 = model.prompts
        self.assertIn("Toolkits (pick 1 to", overview)
        self.assertIn("VITAL SIGNS", overview)
        self.assertIn("YOUR PLAN: Get Ada growing food", focus)
        self.assertIn("Relies on: Ada, who can grow", focus)
        self.assertIn("give the size of each room or zone", overview)
        self.assertIn("enable_work / disable_work", focus)
        self.assertNotIn("create_growing_zone:", focus)  # farming wasn't asked for yet
        self.assertIn("the game shows your plan finished: 50 WoodLog usable", overview2)
        self.assertIn("#1 OK", overview2)
        self.assertIn("set_research:", focus2)
        self.assertIn("enable Growing", client.calls)
        self.assertEqual(memory.state.current_plan.objective, "Research brewing")
        self.assertEqual(memory.observations_for({"farming"}),
                         ["colonists sow fields by themselves once Growing is on"])

    def test_saying_done_before_the_checks_are_met_keeps_the_plan(self) -> None:
        client = LoopClient()
        model, memory = self.run_loop(client, [
            OVERVIEW,
            {"actions": [], "time": "hold", "plan": "done", "step": 1, "steps_done": [1]},
            {"actions": [], "time": "hold", "plan": "replan"},
        ], steps=2)
        self.assertEqual(len(model.prompts), 3)  # no overview after "done"
        self.assertIn("You said the plan is done, but the game ends it by itself", model.prompts[2])
        self.assertIn("1 Beer usable: 0 of 1", model.prompts[2])
        self.assertIn("1. [done] enable Growing", model.prompts[2])

    def test_a_requested_toolkit_is_added_to_the_current_one(self) -> None:
        client = LoopClient()
        model, memory = self.run_loop(client, [
            OVERVIEW,
            {"actions": [], "time": "hold", "plan": "continue", "request_toolkits": ["farming"]},
            {"actions": [], "time": "hold", "plan": "replan"},
        ], steps=2)
        self.assertEqual(len(model.prompts), 3)  # no second overview
        self.assertIn("create_growing_zone:", model.prompts[2])
        self.assertEqual(memory.state.current_plan.toolkits, ["work", "farming"])  # added, up to 3
        self.assertEqual(memory.state.current_plan.turns, 2)

    def test_threat_prompt_then_release_and_replan(self) -> None:
        client = LoopClient()
        threat = Threat(faction_name="Ancients", faction_type="AncientsHostile", behaviour="LordJob_AssaultColony",
                        current_step="", pawn_count=1, active=True, pawn_ids=[7], pawn_kinds={7: "Mech_Militor"})
        fights = [True]
        client.get_threats = lambda: [threat] if fights[0] else []
        client.get_enemies = lambda threats: [Enemy(id=7, kind="Mech_Militor", x=9, z=9)] if threats else []
        client.get_equipment = lambda pawn_id: []
        client.set_drafted = Mock(side_effect=lambda pid, drafted: fights.__setitem__(0, False) if drafted else None)
        model, memory = self.run_loop(client, [
            {"actions": [{"action": "draft", "colonist": "Ada", "drafted": True, "reason": "fight"}],
             "time": "hold"},
            OVERVIEW,
            {"actions": [], "time": "hold", "plan": "replan"},
        ], steps=2)
        threat_text, overview, focus = model.prompts
        self.assertIn("THREAT:", threat_text)
        self.assertIn("id=7 Mech_Militor at (9,9)", threat_text)
        self.assertIn("weapon: unarmed", threat_text)
        self.assertIn("the threat is over", overview)
        self.assertIn("Ada released back to work", overview)
        self.assertEqual([c.args for c in client.set_drafted.call_args_list], [(1, True), (1, False)])

    def test_a_plan_in_hand_is_finished_past_the_step_limit(self) -> None:
        client = LoopClient()
        model, memory = self.run_loop(client, [
            OVERVIEW,
            {"actions": [], "time": "hold", "plan": "continue"},   # step 0: the only step asked for
            {"actions": [], "time": "hold", "plan": "continue"},   # step 1: still finishing it
            {"actions": [], "time": "hold", "plan": "replan"},     # step 2: it ends, so the run ends
        ], steps=1)
        self.assertEqual(len(model.prompts), 4)  # no new overview after the plan is done
        self.assertEqual(memory.state.current_plan.turns, 3)

    def test_a_fight_past_the_step_limit_ends_the_run_once_over(self) -> None:
        client = LoopClient()
        threat = Threat(faction_name="Red Fangs", faction_type="Pirate", behaviour="LordJob_AssaultColony",
                        current_step="", pawn_count=1, active=True, pawn_ids=[7], pawn_kinds={7: "Pirate"})
        fights = [True]
        client.get_threats = lambda: [threat] if fights[0] else []
        client.get_enemies = lambda threats: [Enemy(id=7, kind="Pirate", x=9, z=9)] if threats else []
        client.get_equipment = lambda pawn_id: []
        client.set_drafted = Mock(side_effect=lambda pid, drafted: fights.__setitem__(0, False) if drafted else None)
        model, memory = self.run_loop(client, [
            {"actions": [{"action": "draft", "colonist": "Ada", "drafted": True, "reason": "fight"}],
             "time": "hold"},
        ], steps=1)
        self.assertEqual(len(model.prompts), 1)  # released, and no new plan started
        self.assertEqual(client.set_drafted.call_args_list[-1].args, (1, False))

    def test_an_endless_fight_is_capped_and_colonists_released(self) -> None:
        from rimagent.agent import OVERRUN_STEPS
        client = LoopClient()
        threat = Threat(faction_name="Red Fangs", faction_type="Pirate", behaviour="LordJob_AssaultColony",
                        current_step="", pawn_count=1, active=True, pawn_ids=[7], pawn_kinds={7: "Pirate"})
        client.get_threats = lambda: [threat]
        client.get_enemies = lambda threats: [Enemy(id=7, kind="Pirate", x=9, z=9)]
        client.get_equipment = lambda pawn_id: []
        client.set_drafted = Mock()
        draft = {"actions": [{"action": "draft", "colonist": "Ada", "drafted": True, "reason": "fight"}],
                 "time": "hold"}
        model, memory = self.run_loop(client, [draft] + [{"actions": [], "time": "hold"}] * 40, steps=1)
        self.assertEqual(len(model.prompts), 1 + OVERRUN_STEPS)
        self.assertEqual(client.set_drafted.call_args_list[-1].args, (1, False))

    def test_a_broken_overview_falls_back_to_a_basic_plan(self) -> None:
        client = LoopClient()
        model, memory = self.run_loop(client, [
            {"nonsense": 1}, {"goal": ""},
            {"actions": [], "time": "hold", "plan": "done"},
        ], steps=1)
        self.assertEqual(memory.state.current_plan.toolkits, ["construction"])
        self.assertIn("place_blueprint:", model.prompts[2])


class ItemFinishValidationTests(unittest.TestCase):
    def test_unknown_item_is_corrected_before_plan_is_accepted(self):
        from rimagent.agent import request_overview
        from rimagent.finish import FinishState, finish_progress

        def reply(name):
            return json.dumps({"goal": "Gather wood", "uses": "trees", "toolkits": ["wood_mining"],
                               "finish": [{"kind": "have_items", "what": name, "count": 10}]})

        model = Mock(side_effect=[reply("planks"), reply("WoodLog")])
        overview, _ = request_overview(model, "prompt", Mock(), 0, frozenset({"WoodLog"}))
        self.assertIn("Unknown item def 'planks'", model.call_args_list[1].args[0])
        self.assertTrue(finish_progress(overview.finish, FinishState(items={"WoodLog": 12}))[1])

    def test_plain_words_are_read_as_the_def_they_mean(self):
        from rimagent.agent import request_overview
        from rimagent.finish import FinishState, finish_progress

        # "10 wood usable" read 0 of 10 for twelve turns while 518 WoodLog lay chopped.
        reply = json.dumps({"goal": "Gather wood", "uses": "trees", "toolkits": ["wood_mining"],
                            "finish": [{"kind": "have_items", "what": "wood", "count": 10},
                                       {"kind": "have_items", "what": "food", "count": 5}]})
        model = Mock(return_value=reply)
        overview, _ = request_overview(model, "prompt", Mock(), 0, frozenset({"WoodLog"}))
        self.assertEqual(model.call_count, 1)
        self.assertEqual([c.what for c in overview.finish], ["WoodLog", "Food"])
        state = FinishState(items={"WoodLog": 518, "MealSurvivalPack": 3, "RawPotatoes": 2, "Steel": 50})
        self.assertTrue(finish_progress(overview.finish, state)[1])

    def test_repeated_unknown_items_never_become_active_finish_checks(self):
        from rimagent.agent import FALLBACK_OVERVIEW, request_overview

        reply = json.dumps({"goal": "Gather wood", "uses": "trees", "toolkits": ["wood_mining"],
                            "finish": [{"kind": "have_items", "what": "planks", "count": 10}]})
        overview, _ = request_overview(Mock(return_value=reply), "prompt", Mock(), 0, frozenset({"WoodLog"}))
        self.assertEqual(overview, FALLBACK_OVERVIEW)

    def test_valid_unstocked_mod_item_is_accepted(self):
        from rimagent.agent import request_overview

        reply = json.dumps({"goal": "Gather crystal", "uses": "ore", "toolkits": ["wood_mining"],
                            "finish": [{"kind": "have_items", "what": "ModCrystal", "count": 10}]})
        model = Mock(return_value=reply)
        overview, _ = request_overview(model, "prompt", Mock(), 0, frozenset({"ModCrystal"}))
        self.assertEqual(overview.finish[0].what, "ModCrystal")
        self.assertEqual(model.call_count, 1)


class FocusSchemaTests(MapTestCase):
    def test_focus_schema_offers_only_its_toolkits(self) -> None:
        choices = ChoicesTests.choices(self)
        allowed = allowed_actions(["research"])
        schema = narrowed(choices, allowed).schema(allowed, FocusTurn)
        offered = {b["properties"]["action"]["const"] for b in schema["properties"]["actions"]["items"]["anyOf"]}
        self.assertTrue(offered <= allowed)
        self.assertIn("plan", schema["properties"])
        json.dumps(schema)  # serialisable for Ollama


if __name__ == "__main__":
    unittest.main()
