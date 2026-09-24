"""Fixes from the 2026-09-23 20:15 gpt-oss run: stockpiles over shelves, what
buildings are for, re-marked mining, repeats within a reply, diary memory.

Run from the repo root:  python -m unittest tests/test_run_fixes.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rimagent.actions import Turn
from rimagent.agent import run
from rimagent.blueprints import TrackedBlueprint
from rimagent.construction import BUILDING_USE
from rimagent.execution import execute
from rimagent.hints import TurnFacts, failure_hint
from rimagent.memory import AgentMemory, MemoryUpdate, diary_like
from rimagent.mining import MineTracker
from rimagent.runlog import RunLogger
from tests.test_turn_loop import LoopClient, ScriptedModel


def order(**fields):
    return Turn.model_validate({"actions": [{"reason": "x", **fields}], "time": "hold"}).actions[0]


class StockpileOverShelfTests(unittest.TestCase):
    def test_offers_the_clear_part_of_a_storage_room(self) -> None:
        # Live: a storage room with a shelf in it, then a stockpile over the whole inside, three times.
        shelf = TrackedBlueprint(def_name="Shelf", stuff="WoodLog", x=116, z=144, cells=(116, 144, 117, 144))
        facts = TurnFacts(report=[(shelf, "waiting")])
        pile = order(action="create_stockpile", x1=116, z1=140, x2=118, z2=144)
        hint, keep = failure_hint(pile, "area (116,140)-(118,144) is blocked by shelf at (116,144)", facts)
        self.assertIn("a shelf stores things by itself", hint)
        self.assertIn("The largest clear part of that area is (116,140)-(118,143): create_stockpile there.", hint)
        self.assertIn("Next time make the stockpile first", hint)
        self.assertFalse(keep)


class BuildingUseTests(unittest.TestCase):
    def test_the_stonecutter_says_it_does_not_mine(self) -> None:
        # Live memory: "A mining table will enable the sole mining-capable colonist to extract..."
        self.assertIn("does not mine", BUILDING_USE["TableStonecutter"])
        self.assertIn("stores items by itself", BUILDING_USE["Shelf"])


STEEL = [{"name": "mineable_steel", "x": 148, "z": 126}, {"name": "mineable_steel", "x": 148, "z": 127}]


class MineTrackerTests(unittest.TestCase):
    def test_mined_cells_are_counted_and_forgotten(self) -> None:
        mines = MineTracker()
        mines.add(STEEL)
        mines.reconcile({(148, 127)})  # (148,126) is no longer rock: mined
        self.assertEqual(mines.mined_since_last, 1)
        line = mines.describe(["Reikguard"])
        self.assertIn("1 cells mined since last turn; still marked: 1 mineable_steel", line)
        self.assertIn("Mining is on for: Reikguard.", line)
        self.assertIn("Nobody has Mining on", mines.describe([]))

    def test_survives_a_restart_and_a_dry_run_saves_nothing(self) -> None:
        path = Path(tempfile.mkdtemp()) / "mining.json"
        MineTracker(path).add(STEEL[:1])
        self.assertEqual(MineTracker(path).cells, {(148, 126): "mineable_steel"})
        MineTracker(path, read_only=True).add(STEEL[1:])
        self.assertEqual(len(MineTracker(path).cells), 1)


class DesignateClient:
    def __init__(self):
        self.marked = []

    def begin_snapshot(self): pass

    def designate(self, kind, rect, map_id=0):
        self.marked.append(rect)


class ExecuteMineTests(unittest.TestCase):
    def mine(self, x1, z1, x2, z2):
        return order(action="designate", designation="mine", x1=x1, z1=z1, x2=x2, z2=z2)

    def test_marking_the_same_rock_again_sends_nothing(self) -> None:
        client, mines = DesignateClient(), MineTracker()
        ops = SimpleNamespace(targets={"mine": STEEL})
        first = execute(client, self.mine(148, 126, 148, 126), None, None, ops, mines=mines)
        self.assertIn("marked 1 rock cells for mining", first)
        again = execute(client, self.mine(148, 126, 148, 126), None, None, ops, mines=mines)
        self.assertTrue(again.startswith("unchanged:"), again)
        wider = execute(client, self.mine(148, 126, 148, 127), None, None, ops, mines=mines)
        self.assertIn("marked 1 rock cells for mining (1 there already were)", wider)
        self.assertEqual(len(client.marked), 2)


class DiaryTests(unittest.TestCase):
    def test_turn_logs_are_diary(self) -> None:
        for text in ("Placed TorchLamp adjacent to research bench to enable research",
                     "Planning to build a WoodFiredGenerator at (131,138)",
                     "Generator needs 200 WoodLog fuel; current stocks: 155 WoodLog, 50 Steel",
                     "Stockpile inside storage room and a 15x15 potato growing zone established at (116,142)"):
            self.assertTrue(diary_like(text), text)

    def test_lessons_are_kept(self) -> None:
        for text in ("Colonists prefer beds inside a room to beds outdoors",
                     "Colonists are comfortable between 16 and 26 C",
                     "Fields need about 10 cells per colonist to keep up with meals",
                     "Raids have come from the east edge"):
            self.assertFalse(diary_like(text), text)

    def test_a_diary_entry_is_turned_away_with_a_note(self) -> None:
        memory = AgentMemory(tempfile.mkdtemp())
        note = memory.apply_update(1, 100, MemoryUpdate(observation="Placed beds in the bedroom."))
        self.assertIn("Memory observation not saved", note)
        self.assertNotIn("Placed beds", memory.observations_path.read_text(encoding="utf-8"))
        self.assertIsNone(memory.apply_update(2, 200, MemoryUpdate(observation="Colonists prefer beds inside.")))
        self.assertIn("Colonists prefer beds inside.", memory.observations_path.read_text(encoding="utf-8"))


class LoopTests(unittest.TestCase):
    def test_a_repeat_within_one_reply_is_sent_once_and_diary_is_reported(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        client = LoopClient()
        grow = {"action": "enable_work", "colonist": "Ada", "work_type": "Growing", "reason": "farm"}
        model = ScriptedModel([
            {"actions": [grow, dict(grow, reason="farm again")], "time": "hold",
             "memory_update": {"observation": "Enabled Growing for Ada."}},
            {"actions": [], "time": "hold"},
        ])
        run(client, model, RunLogger(tmp / "logs"), max_steps=2, memory=AgentMemory(tmp / "memory"))
        self.assertEqual(client.calls.count("enable Growing"), 1)
        second = model.prompts[1]
        self.assertIn("#2 UNCHANGED: the same order as #1 in this reply, so it wasn't sent again", second)
        self.assertIn("2 orders: 1 OK, 1 UNCHANGED, 0 FAILED.", second)
        self.assertIn("Memory observation not saved", second)
        log = [json.loads(line) for line in next((tmp / "logs" / "runs").iterdir())
               .joinpath("run.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(1 for r in log if r["data"].get("action") == "enable_work"), 2)  # both logged


if __name__ == "__main__":
    unittest.main()
