"""The log-to-Markdown converter, on a small run log written with RunLogger.

Run from the repo root:  python -m unittest tests/test_show_log.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from rimagent.runlog import RunLogger
from scripts.show_log import convert


class ShowLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def test_run_log(self) -> None:
        logger = RunLogger(self.dir)
        logger.log_run_info({"model": "gpt-oss:20b", "think": "medium"})
        logger.log_state({"game_tick": 3230, "is_paused": True})
        logger.log_event({"kind": "letter", "category": "ThreatBig", "text": "Raid"})
        logger.log_model_failure({"step": 0, "attempt": 1, "failure_type": "invalid_response",
                                  "response": '{"action": "dance"}', "error": "bad action",
                                  "thinking": "hmm ~~~~ tricky"})
        logger.log_decision({
            "step": 0, "action": "allow_items", "reason": "Wood for beds.",
            "rect": [124, 90, 141, 111], "rotation": 0, "error": None,
            "pause_status": "Game is paused: the game paused itself because of a major threat (Raid).",
            "memory_update": {"replace_plan": {"objective": "Beds", "status": "active",
                                               "steps": [{"description": "Get wood", "status": "pending"}]}},
            "thinking": "We need wood first.", "model_stats": {"seconds": 7.1, "prompt_tokens": 3159,
                                                               "output_tokens": 67},
        })
        logger.log_state({"game_tick": 3400, "is_paused": False})
        logger.log_decision({"step": 1, "action": "place_blueprint", "reason": "Bed.",
                             "building_def": "Bed", "position": [125, 107], "rotation": 1,
                             "stuff": "WoodLog", "error": "blocked by wall at (125,108)"})

        md_path = convert(logger.path)
        md = md_path.read_text(encoding="utf-8")

        self.assertEqual(logger.path.parent.parent, self.dir / "runs")
        self.assertEqual(md_path, logger.folder / "run.md")
        self.assertIn(f"# Run {logger.folder.name}", md)
        self.assertIn("model `gpt-oss:20b`", md)
        self.assertIn("**Decisions:** 2 (allow_items x1, place_blueprint x1)", md)
        self.assertIn("**Failed actions:** 1", md)
        self.assertIn("## Step 0: allow_items (ok)", md)
        self.assertIn("letter (ThreatBig): Raid", md)
        self.assertIn("Area: `(124,90) to (141,111)`", md)
        self.assertIn("[pending] Get wood", md)
        self.assertIn("**Attempt 1 rejected** (invalid_response): `bad action`", md)
        self.assertIn("We need wood first.", md)
        self.assertIn("~~~~~text", md)  # fence longer than the ~~~~ inside the reasoning
        self.assertIn("## Step 1: place_blueprint (FAILED)", md)
        self.assertIn("Position: `(125,107)`, facing east", md)
        self.assertIn("FAILED: blocked by wall at (125,108)", md)

    def test_run_folders_are_named_by_time_and_label(self) -> None:
        first = RunLogger(self.dir, label="gpt-oss:20b_think-medium")
        second = RunLogger(self.dir, label="gpt-oss:20b_think-medium")
        self.assertRegex(first.folder.name, r"^\d{4}-\d{2}-\d{2}_\d{6}_gpt-oss-20b_think-medium$")
        self.assertNotEqual(first.folder, second.folder)  # same second: no clash

    def test_dry_run(self) -> None:
        path = self.dir / "dry-run-qwen3_14b-x.json"
        path.write_text(json.dumps({
            "settings": {"model": "qwen3:14b"}, "stats": {"seconds": 4.1},
            "prompt": "the prompt", "thinking": None, "reply": '{"action": "wait"}',
            "decision": None, "invalid": "reason: field required",
        }), encoding="utf-8")

        md = convert(path).read_text(encoding="utf-8")

        self.assertIn("## Decision: INVALID", md)
        self.assertIn("reason: field required", md)
        self.assertIn("Full prompt sent to the model", md)


if __name__ == "__main__":
    unittest.main()
