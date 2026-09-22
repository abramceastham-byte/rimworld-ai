"""The Ollama adapter, tested with a stand-in for Ollama's client.

Run from the repo root:  python -m unittest tests/test_ollama_model.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from rimagent.ollama_model import OllamaModel, parse_think, think_label


def fake_response(content: str, thinking: str | None = None):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking=thinking),
        prompt_eval_count=3000,
        eval_count=120,
    )


class ParseThinkTests(unittest.TestCase):
    def test_values(self) -> None:
        self.assertIsNone(parse_think(None))
        self.assertIs(parse_think("on"), True)
        self.assertIs(parse_think("OFF"), False)
        self.assertEqual(parse_think("low"), "low")
        with self.assertRaises(ValueError):
            parse_think("maybe")

    def test_labels_match_the_command_line(self) -> None:
        self.assertEqual([think_label(t) for t in (None, True, False, "medium")],
                         ["default", "on", "off", "medium"])
        model = OllamaModel("qwen3:14b", think=True, host="http://example.invalid")
        self.assertEqual(model.label(), "qwen3:14b_think-on")


class OllamaModelTests(unittest.TestCase):
    def make(self, **kwargs) -> tuple[OllamaModel, Mock]:
        model = OllamaModel("qwen3:14b", host="http://example.invalid", **kwargs)
        model.client = Mock()
        return model, model.client

    def test_returns_the_reply_and_keeps_the_thinking_separately(self) -> None:
        model, client = self.make(think=True)
        client.chat.return_value = fake_response('{"action": "wait"}', "Let me think...")

        self.assertEqual(model("the prompt"), '{"action": "wait"}')
        self.assertEqual(model.last_thinking, "Let me think...")
        self.assertEqual(model.last_stats["prompt_tokens"], 3000)
        self.assertEqual(model.last_stats["output_tokens"], 120)

    def test_sends_prompt_schema_context_size_and_think(self) -> None:
        schema = {"type": "object"}
        model, client = self.make(schema=schema, think="low", num_ctx=8192)
        client.chat.return_value = fake_response("{}")

        model("the prompt")

        kwargs = client.chat.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen3:14b")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertEqual(kwargs["format"], schema)
        self.assertEqual(kwargs["options"], {"num_ctx": 8192})
        self.assertEqual(kwargs["think"], "low")

    def test_previous_thinking_is_cleared_each_call(self) -> None:
        model, client = self.make()
        client.chat.return_value = fake_response("{}", "first")
        model("a")
        client.chat.return_value = fake_response("{}", None)
        model("b")
        self.assertIsNone(model.last_thinking)
        # Only the new prompt is sent: nothing from earlier calls carries over.
        self.assertEqual(client.chat.call_args.kwargs["messages"],
                         [{"role": "user", "content": "b"}])


if __name__ == "__main__":
    unittest.main()
