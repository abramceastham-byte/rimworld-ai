"""The claude -p adapter, with the CLI faked.

Run from the repo root:  python -m unittest tests/test_claude_cli_model.py -v
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rimagent.claude_cli_model import ClaudeCliModel


def finished(payload: dict, code: int = 0):
    return SimpleNamespace(stdout=json.dumps(payload).encode(), stderr=b"", returncode=code)


class ClaudeCliModelTests(unittest.TestCase):
    def test_structured_reply_schema_and_prompt_on_stdin(self) -> None:
        model = ClaudeCliModel("opus", effort="low", schema={"type": "object"}, cli="claude")
        payload = {"is_error": False, "structured_output": {"time": "hold"}, "result": "ignored",
                   "usage": {"input_tokens": 2, "cache_creation_input_tokens": 1000, "output_tokens": 50}}
        with patch("subprocess.run", return_value=finished(payload)) as run:
            self.assertEqual(json.loads(model("the prompt")), {"time": "hold"})
        command, kwargs = run.call_args.args[0], run.call_args.kwargs
        self.assertEqual(kwargs["input"], b"the prompt")
        self.assertEqual(command[command.index("--json-schema") + 1], '{"type":"object"}')
        self.assertEqual(command[command.index("--tools") + 1], "")  # no tools for the agent
        self.assertIn("--system-prompt", command)
        self.assertEqual(command[command.index("--effort") + 1], "low")
        self.assertEqual(model.last_stats["prompt_tokens"], 1002)

    def test_errors_are_raised_for_the_loop_to_retry(self) -> None:
        model = ClaudeCliModel("opus", cli="claude")
        with patch("subprocess.run", return_value=finished({"is_error": True, "result": "usage limit reached"})):
            with self.assertRaisesRegex(RuntimeError, "usage limit reached"):
                model("x")
        with patch("subprocess.run", return_value=SimpleNamespace(stdout=b"", stderr=b"not logged in",
                                                                    returncode=1)):
            with self.assertRaisesRegex(RuntimeError, "not logged in"):
                model("x")


if __name__ == "__main__":
    unittest.main()
