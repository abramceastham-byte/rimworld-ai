"""Regression coverage for the six continuation-review fixes."""
import tempfile
from pathlib import Path
from unittest.mock import Mock

from rimagent.actions import Turn
from rimagent.agent import INSTRUCTIONS, TimeMode, prepare_turn
from rimagent.blueprints import BlueprintTracker
from rimagent.zones import ZoneTracker
from rimagent.memory import AgentMemory
from rimagent.observation import read_operations
from rimagent.execution import execute
from rimagent.construction import Rect, room_layout, TerrainMap
from rimagent.rimapi import Building, AreaCheck
from tests import test_turn_loop as fixtures
from tests.test_turn_loop import GridFake, LoopClient, colonist
from tests.test_map_actions import MapTestCase, PALETTE, WIDTH, HEIGHT, terrain_grid


def order(**fields):
    return Turn.model_validate({'actions': [dict(reason='test', **fields)], 'time': 'hold'}).actions[0]


def thing(name, x, z, stuff=None, rotation=0):
    return dict(thing_id=x*100+z, def_name=name, position={'x': x, 'z': z},
                stuff_def_name=stuff, rotation=rotation)


class PairingTests(MapTestCase):
    choices = fixtures.ChoicesTests.choices
    turn = fixtures.ChoicesTests.turn

    def test_schema_allows_only_valid_pairs(self):
        c = self.choices()
        c.work = {'Ada': {'Construction'}, 'Bob': {'Research'}}
        c.fields['enable_work'] = {'colonist': list(c.work), 'work_type': ['Construction', 'Research']}
        c.recipes = {1: {'Meal'}, 2: {'Shirt'}}
        c.fields['set_bill'] = {'building_id': [1, 2], 'recipe_def_name': ['Meal', 'Shirt']}
        branches = c.schema()['properties']['actions']['items']['anyOf']
        for action, selector, dependent, mapping in [
            ('enable_work', 'colonist', 'work_type', c.work),
            ('set_bill', 'building_id', 'recipe_def_name', c.recipes),
            ('place_blueprint', 'building_def', 'stuff', c.materials),
        ]:
            emitted = set()
            for b in branches:
                p = b['properties']
                if p['action']['const'] == action:
                    self.assertEqual(list(p)[:3], ['action', selector, dependent])
                    self.assertIn(dependent, b['required'])
                    emitted.update((a, v) for a in p[selector]['enum'] for v in p[dependent]['enum'])
            self.assertEqual(emitted, {(a, v) for a, vs in mapping.items() for v in vs})

    def test_hunt_single_cell_is_explicit_and_validated(self):
        self.assertIn('x1=x2 and z1=z2', INSTRUCTIONS)
        c = self.choices()
        c.fields['designate'] = {'designation': ['hunt']}
        a = dict(action='designate', designation='hunt', x1=1, z1=1, x2=2, z2=1, reason='test')
        with self.assertRaisesRegex(ValueError, 'one animal cell'):
            c.validate(self.turn(a))
        a['x2'] = 1
        c.validate(self.turn(a))


class ConstructionRepeatTests(MapTestCase):
    def test_wall_and_floor_repeats_write_nothing(self):
        for name, method, material in [('Wall', 'build_wall', 'WoodLog'),
                                        ('WoodPlankFloor', 'build_floor', None)]:
            for prefix in ('Blueprint_', 'Frame_'):
                fake = GridFake(things_at={(2, 2): [thing(prefix+name, 2, 2, material)]})
                c = self.client_with(fake)
                self.assertIn('unchanged:', getattr(c, method)(Rect(2, 2, 2, 2), material or name))
                self.assertEqual(fake.writes(), [])

    def test_different_material_or_floor_is_not_unchanged(self):
        fake = GridFake(things_at={(2, 2): [thing('Blueprint_Wall', 2, 2, 'Steel')]})
        with self.assertRaises(ValueError):
            self.client_with(fake).build_wall(Rect(2, 2, 2, 2), 'WoodLog')
        fake = GridFake(things_at={(2, 2): [thing('Blueprint_TileGranite', 2, 2)]})
        with self.assertRaises(ValueError):
            self.client_with(fake).build_floor(Rect(2, 2, 2, 2), 'WoodPlankFloor')
        self.assertEqual(fake.writes(), [])

    def test_single_building_repeat_matches_rotation_and_stuff(self):
        fake = GridFake(things_at={(2, 2): [thing('Blueprint_Bed', 2, 2, 'WoodLog', 1)]})
        c = self.client_with(fake)
        a = order(action='place_blueprint', building_def='Bed', x=2, z=2, rotation=1, stuff='WoodLog')
        self.assertTrue(execute(c, a, None, None, None).startswith('unchanged:'))
        self.assertFalse(c.construction_at('Bed', 2, 2, 0, stuff='WoodLog'))
        self.assertFalse(c.construction_at('Bed', 2, 2, 1, stuff='Steel'))
        self.assertEqual(fake.writes(), [])

    def test_missing_material_in_rimapi_preserves_existing_construction(self):
        fake = GridFake(things_at={(2, 2): [thing('Blueprint_Wall', 2, 2)]})
        result = self.client_with(fake).build_wall(Rect(2, 2, 2, 2), 'WoodLog')
        self.assertIn('unchanged:', result)
        self.assertIn('existing materials retained', result)
        self.assertEqual(fake.writes(), [])

    def test_complete_room_repeat_writes_nothing(self):
        rect = Rect(2, 2, 6, 6)
        fake = GridFake(things_at={(x, z): [thing('Blueprint_'+n, x, z, 'WoodLog')]
                                  for n, x, z in room_layout(rect, 'north')})
        self.assertIn('unchanged:', self.client_with(fake).build_room(rect, 'WoodLog', 'north'))
        self.assertEqual(fake.writes(), [])

    def test_finished_floor_repeat_writes_nothing(self):
        fake = GridFake()
        c = self.client_with(fake)
        c.get_terrain = Mock(return_value=TerrainMap(1, 1, ['WoodPlankFloor'], [1, 0]))
        self.assertIn('unchanged:', c.build_floor(Rect(0, 0, 0, 0), 'WoodPlankFloor'))
        self.assertEqual(fake.writes(), [])


class ObservationTests(MapTestCase):
    def test_construction_repeat_keeps_later_orders_and_plan(self):
        from rimagent.agent import run
        from rimagent.runlog import RunLogger
        client = LoopClient()
        client.construction_at = Mock(return_value=True)
        model = fixtures.ScriptedModel([{'actions': [
            dict(action='place_blueprint', building_def='Wall', x=2, z=2, stuff='WoodLog', reason='repeat'),
            dict(action='enable_work', colonist='Ada', work_type='Growing', reason='farm'),
        ], 'time': 'hold', 'memory_update': {'replace_plan': {'objective': 'Grow food'}}}])
        with tempfile.TemporaryDirectory() as tmp:
            memory = AgentMemory(Path(tmp)/'memory')
            run(client, model, RunLogger(Path(tmp)/'logs'), max_steps=1, memory=memory)
            self.assertIn('enable Growing', client.calls)
            self.assertEqual(memory.state.current_plan.objective, 'Grow food')

    def test_all_targets_retained_and_each_resource_shown(self):
        client = LoopClient()
        client._request = lambda method, path, **kw: ({'map_width': 100, 'ores': {
            'Granite': {'cells': list(range(30))}, 'Steel': {'cells': [999]}}}
            if path.endswith('/map/ore') else None)
        ops = read_operations(client, [colonist(x=0, z=0)], client.get_game_defs())
        self.assertEqual(len(ops.targets['mine']), 31)
        self.assertIn('Steel at (99,9)', '\n'.join(ops.lines))
        client.designate = Mock()
        execute(client, order(action='designate', designation='mine', x1=29, x2=29, z1=0, z2=0),
                None, None, ops)
        client.designate.assert_called_once()

    def test_failure_fingerprint_tracks_local_terrain_rotation_and_zones(self):
        client = LoopClient()
        with tempfile.TemporaryDirectory() as tmp:
            memory = AgentMemory(Path(tmp), read_only=True)
            bp, zones = BlueprintTracker(Path(tmp)/'bp.json'), ZoneTracker(Path(tmp)/'zones.json')
            target = dict(action='place_blueprint', building_def='Bed', x=1, z=1, stuff='WoodLog')
            def fingerprint():
                return prepare_turn(client, memory, bp, zones, TimeMode(True, 'test'),
                                    read_only=True)['fingerprint'](target)
            before = fingerprint()
            original = client.get_terrain()
            client.get_terrain = lambda: TerrainMap(WIDTH, HEIGHT, ['Soil'], [WIDTH*HEIGHT, 0])
            self.assertNotEqual(before, fingerprint())
            client.get_terrain = lambda: original
            building = Building(id=1, def_name='Bed', position={'x': 2, 'z': 2}, size={'x': 1, 'z': 2})
            client.get_buildings = lambda: [building]
            before = fingerprint()
            building.rotation = 1
            self.assertNotEqual(before, fingerprint())
            fake = GridFake(zones=[('Growing', [(8, 8)])])
            client.check_area = lambda rect: AreaCheck.model_validate(fake('POST', '/api/v1/builder/check-zone',
                json={'point_a': {'x': rect.x1, 'z': rect.z1}, 'point_b': {'x': rect.x2, 'z': rect.z2}})['issues'])
            before = fingerprint()
            fake.zones = [('Growing', [(8, 8), (9, 9)])]
            self.assertEqual(before, fingerprint())
            fake.zones = [('Growing', [(1, 1)])]
            self.assertNotEqual(before, fingerprint())
