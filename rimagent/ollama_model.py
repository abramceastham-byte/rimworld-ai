"""A real model for the agent loop, served by Ollama (locally or through an SSH tunnel).

OllamaModel works anywhere the loop expects model(prompt) -> str, like the fake
model. It also keeps the last reply's reasoning ("thinking") and timings, which
the loop writes into the run log. The reasoning is never sent back to the model.
"""

import time
from typing import Any, Literal

from ollama import Client

from rimagent.config import load_settings

Think = bool | Literal["low", "medium", "high"] | None


def parse_think(value: str | None) -> Think:
    """Turn a command-line value into Ollama's think setting.

    on/off for models like qwen3; low/medium/high for gpt-oss (which can't
    switch thinking off); None leaves it to the model's default.
    """
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("on", "true"):
        return True
    if lowered in ("off", "false"):
        return False
    if lowered in ("low", "medium", "high"):
        return lowered
    raise ValueError(f"think must be on, off, low, medium or high, not {value!r}")


class OllamaModel:
    def __init__(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
        think: Think = None,
        num_ctx: int = 16384,
        host: str | None = None,
        timeout: float = 600.0,
    ):
        """
        name:    an Ollama model, e.g. "qwen3:14b" or "gpt-oss:20b".
        schema:  a JSON schema the reply must follow (e.g. Decision.model_json_schema()).
                 If a model returns empty or odd replies with it, try None.
        num_ctx: context window in tokens. Ollama's default is smaller than the
                 agent's prompt and silently cuts the start off, so set it.
        timeout: seconds; the first call also loads the model into GPU memory.
        """
        self.name = name
        self.schema = schema
        self.think = think
        self.num_ctx = num_ctx
        self.client = Client(host=host or load_settings().ollama_host, timeout=timeout)
        self.last_thinking: str | None = None
        self.last_stats: dict[str, Any] = {}

    def __call__(self, prompt: str) -> str:
        self.last_thinking, self.last_stats = None, {}
        start = time.monotonic()
        response = self.client.chat(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            format=self.schema,
            options={"num_ctx": self.num_ctx},
            think=self.think,
        )
        self.last_thinking = response.message.thinking
        self.last_stats = {
            "seconds": round(time.monotonic() - start, 1),
            "prompt_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
        }
        return response.message.content

    def describe(self) -> dict[str, Any]:
        """Settings to record at the start of a run log."""
        return {
            "model": self.name,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "schema": self.schema is not None,
        }
