"""Loop-controlled game time, blueprint tracking, and the work-table check.

Run from the repo root:  python -m unittest tests/test_time_and_blueprints.py -v
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rimagent.agent import TimeWindow, describe_time, let_time_run, require_work_table
from rimagent.blueprints import BlueprintTracker, status_from_things
from rimagent.construction import Rect
from rimagent.events import GameEvent
from rimagent.rimapi import WorkTable


class FakeGame:
    """Enough of RimApiClient for let_time_run: ticks pass only while running."""

    def __init__(self, pause_itself_after_checks: int | None = None):
        self.paused, self.tick, self.checks = True, 1000, 0
        self.calls: list[str] = []
        self.pause_itself_after_checks = pause_itself_after_checks

    def resume(self, speed: int = 1) -> None:
        self.calls.append(f"resume({speed})")
        self.paused = False

    def pause(self) -> None:
        self.calls.append("pause")
        self.paused = True

    def get_state(self):
        if not self.paused:
            self.tick += 60
            self.checks += 1
            if self.pause_itself_after_checks and self.checks >= self.pause_itself_after_checks:
                self.paused = True
        return SimpleNamespace(game_tick=self.tick, is_paused=self.paused)


class FakeEvents:
    def __init__(self, batches: list[list[GameEvent]]):
        self.batches = batches

    def drain(self) -> list[GameEvent]:
        return self.batches.pop(0) if self.batches else []


def letter(category: str, text: str) -> GameEvent:
    return GameEvent(kind="letter", text=text, category=category, tick=0)


class LetTimeRunTests(unittest.TestCase):
    def test_runs_for_the_full_time_then_pauses(self) -> None:
        game = FakeGame()
        window = let_time_run(game, None, seconds=0.3, speed=2, poll_seconds=0.05)
        self.assertIsNone(window.stopped_by)
        self.assertGreaterEqual(window.seconds, 0.3)
        self.assertGreater(window.ticks, 0)
        self.assertEqual(game.calls[0], "resume(2)")
        self.assertEqual(game.calls[-1], "pause")
        self.assertTrue(game.paused)

    def test_a_threat_letter_stops_it_early(self) -> None:
        game = FakeGame()
        events = FakeEvents([[], [letter("NeutralEvent", "Trader")], [letter("ThreatBig", "Raid")]])
        window = let_time_run(game, events, seconds=5, poll_seconds=0.02)
        self.assertIn("Raid", window.stopped_by)
        self.assertLess(window.seconds, 1)
        self.assertEqual([e.text for e in window.events], ["Trader", "Raid"])  # kept for the prompt
        self.assertTrue(game.paused)

    def test_the_game_pausing_itself_stops_it_early(self) -> None:
        game = FakeGame(pause_itself_after_checks=3)
        window = let_time_run(game, FakeEvents([]), seconds=5, poll_seconds=0.02)
        self.assertIn("paused by the player", window.stopped_by)
        self.assertLess(window.seconds, 1)


class DescribeTimeTests(unittest.TestCase):
    def test_explains_the_rule_every_time(self) -> None:
        text = describe_time(None, False, 10, 3)
        self.assertIn("paused while you decide", text)
        self.assertIn("runs for 10s (3s while a threat is active)", text)
        self.assertIn("first decision", text)

    def test_after_holding_paused(self) -> None:
        self.assertIn("no game time has passed", describe_time(None, True, 10, 3))

    def test_after_time_ran(self) -> None:
        window = TimeWindow(seconds=2.1, ticks=1250, stopped_by="a threat arrived (letter: Raid)")
        text = describe_time(window, False, 10, 3)
        self.assertIn("ran 2.1s (0.5 in-game hours) and stopped early: a threat arrived", text)


class BlueprintTests(unittest.TestCase):
    def test_status_from_cell_contents(self) -> None:
        self.assertEqual(status_from_things("Bed", ["Blueprint_Bed"]), "waiting")
        self.assertEqual(status_from_things("Bed", ["Steel", "Frame_Bed"]), "under construction")
        self.assertEqual(status_from_things("Bed", ["Bed"]), "built")
        self.assertEqual(status_from_things("Bed", ["Blueprint_Wall"]), "gone")

    def test_tracks_saves_and_forgets_finished_ones(self) -> None:
        path = Path(tempfile.mkdtemp()) / "blueprints.json"
        tracker = BlueprintTracker(path)
        tracker.add("Bed", "WoodLog", 5, 5, 0, Rect(5, 5, 5, 6))
        tracker.add("Wall", "Steel", 9, 9, 0, Rect(9, 9, 9, 9))
        self.assertEqual(len(BlueprintTracker(path).items), 2)  # survives a restart

        cells = {(5, 5): ["Frame_Bed"], (9, 9): ["Wall"]}
        client = SimpleNamespace(get_things_at=lambda x, z, map_id=0: [
            SimpleNamespace(def_name=d) for d in cells.get((x, z), [])])

        dry = tracker.check(client, forget_finished=False)
        self.assertEqual([s for _, s in dry], ["under construction", "built"])
        self.assertEqual(len(BlueprintTracker(path).items), 2)  # a dry run changes nothing

        report = tracker.check(client)
        self.assertEqual([s for _, s in report], ["under construction", "built"])
        self.assertEqual([b.def_name for b in BlueprintTracker(path).items], ["Bed"])


class WorkTableCheckTests(unittest.TestCase):
    def test_refuses_ids_that_are_not_built_work_tables(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a built work table. Work tables now: none yet"):
            require_work_table(1, [])
        table = WorkTable(id=77, thing_def="Campfire", label="campfire", position={"x": 1, "z": 1})
        with self.assertRaisesRegex(ValueError, "77 \\(campfire\\)"):
            require_work_table(3, [table])
        require_work_table(77, [table])  # a real table is fine


if __name__ == "__main__":
    unittest.main()
