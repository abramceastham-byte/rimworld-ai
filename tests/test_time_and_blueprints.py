"""Loop-controlled game time and blueprint tracking.

Run from the repo root:  python -m unittest tests/test_time_and_blueprints.py -v
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rimagent.agent import (
    TimeMode,
    TimeWindow,
    describe_time,
    let_time_run,
    mode_after_decision,
    mode_after_window,
)
from rimagent.blueprints import BlueprintTracker, status_from_things
from rimagent.construction import Rect
from rimagent.events import GameEvent


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
        window = let_time_run(game, None, ticks=180, speed=2, poll_seconds=0.02)
        self.assertIsNone(window.stopped_by)
        self.assertGreaterEqual(window.ticks, 180)  # ran the game time asked for
        self.assertEqual(game.calls[0], "resume(2)")
        self.assertEqual(game.calls[-1], "pause")
        self.assertTrue(game.paused)

    def test_a_threat_letter_stops_it_early(self) -> None:
        game = FakeGame()
        events = FakeEvents([[], [letter("NeutralEvent", "Trader")], [letter("ThreatBig", "Raid")]])
        window = let_time_run(game, events, ticks=6000, poll_seconds=0.02)
        self.assertIn("Raid", window.stopped_by)
        self.assertLess(window.seconds, 1)
        self.assertEqual([e.text for e in window.events], ["Trader", "Raid"])  # kept for the prompt
        self.assertFalse(window.game_paused)  # RimWorld didn't pause for it: time keeps running
        self.assertTrue(game.paused)

    def test_a_threat_the_game_paused_for(self) -> None:
        game = FakeGame(pause_itself_after_checks=3)  # auto-pauses as the letter arrives
        events = FakeEvents([[], [], [letter("ThreatBig", "Raid")]])
        window = let_time_run(game, events, ticks=6000, poll_seconds=0.02)
        self.assertTrue(window.game_paused)
        self.assertIn("paused itself for a threat (letter: Raid)", window.stopped_by)

    def test_the_game_pausing_itself_stops_it_early(self) -> None:
        game = FakeGame(pause_itself_after_checks=3)
        window = let_time_run(game, FakeEvents([]), ticks=6000, poll_seconds=0.02)
        self.assertTrue(window.game_paused)
        self.assertIn("by RimWorld or the player", window.stopped_by)
        self.assertLess(window.seconds, 1)


class TimeModeTests(unittest.TestCase):
    def test_hold_counts_consecutive_turns(self) -> None:
        mode = TimeMode(paused=True, reason="session start")
        for _ in range(3):
            mode = mode_after_decision(mode, "hold")
        self.assertEqual((mode.paused, mode.paused_decisions), (True, 3))

    def test_advance_resets_the_count(self) -> None:
        mode = mode_after_decision(TimeMode(True, "you chose hold", 5), "advance")
        self.assertEqual((mode.paused, mode.paused_decisions), (False, 0))
        # Nothing is inherited: the next hold starts counting again.
        self.assertEqual(mode_after_decision(mode, "hold").paused_decisions, 1)

    def test_anything_else_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            mode_after_decision(TimeMode(paused=False), "resume")

    def test_the_game_pausing_itself_switches_to_paused(self) -> None:
        window = TimeWindow(2.0, 100, "the game paused itself for a threat (letter: Raid)",
                            game_paused=True, pause_reason="RimWorld paused it for a threat: Raid")
        mode = mode_after_window(TimeMode(paused=False), window)
        self.assertTrue(mode.paused)
        self.assertIn("Raid", mode.reason)

    def test_a_threat_without_a_game_pause_keeps_running(self) -> None:
        window = TimeWindow(2.0, 100, "a threat arrived (letter: Raid)", game_paused=False)
        self.assertFalse(mode_after_window(TimeMode(paused=False), window).paused)


class DescribeTimeTests(unittest.TestCase):
    def test_offers_both_choices_with_their_length(self) -> None:
        text = describe_time(TimeMode(paused=False), None, 600, 180)
        self.assertIn("Choose advance for 600 game ticks (180 during danger), or hold", text)
        self.assertIn("Consecutive held turns: 0.", text)

    def test_says_why_it_is_paused(self) -> None:
        text = describe_time(TimeMode(True, "RimWorld paused it for a threat: Raid", 0), None, 600, 180)
        self.assertIn("Last pause: RimWorld paused it for a threat: Raid.", text)

    def test_says_what_happened_while_time_ran(self) -> None:
        window = TimeWindow(seconds=2.1, ticks=1250, stopped_by="a threat arrived (letter: Raid)")
        text = describe_time(TimeMode(paused=False), window, 600, 180)
        self.assertIn("Last window: 1250 ticks, 2.1s real time.", text)
        self.assertIn("Interrupted: a threat arrived (letter: Raid).", text)


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


if __name__ == "__main__":
    unittest.main()
