"""Claude as the agent's model, through Claude Code's print mode (`claude -p`).

Runs on a Claude subscription's allowance instead of an API key: each call
starts `claude -p` with the prompt on stdin, Claude Code's own system prompt
replaced, every tool switched off and the reply schema enforced. Works anywhere
the loop expects model(prompt) -> str, like OllamaModel.

The call runs from an empty folder so this project's CLAUDE.md and Claude Code's
notes about it never reach the agent's prompt. Each call costs ~10 s of start-up.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = ("You manage a RimWorld colony through the orders each message describes. Reply with exactly the "
                 "JSON the message asks for.")
MAX_SCHEMA_CHARS = 28_000  # Windows caps a command line at 32,767 characters


def find_claude() -> str:
    """The claude executable: $CLAUDE_CLI, the PATH, or the copy inside the VS Code extension."""
    if os.environ.get("CLAUDE_CLI"):
        return os.environ["CLAUDE_CLI"]
    found = shutil.which("claude")
    if found:
        return found
    bundled = sorted(Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude*"))
    if bundled:
        return str(bundled[-1])
    raise FileNotFoundError("can't find the claude CLI: install Claude Code or set CLAUDE_CLI to its path")


class ClaudeCliModel:
    def __init__(self, model: str = "opus", effort: str | None = None, schema: dict[str, Any] | None = None,
                 timeout: float = 600.0, cli: str | None = None):
        """
        model:  a Claude Code model alias ("opus", "sonnet", "haiku") or full model name.
        effort: low, medium, high, xhigh or max; None leaves the model's default.
        schema: the JSON schema the reply must follow; the loop sets it before each call.
        """
        self.name = model
        self.effort = effort
        self.schema = schema
        self.timeout = timeout
        self.cli = cli or find_claude()
        self.workdir = tempfile.mkdtemp(prefix="rimagent-claude-")
        self.last_thinking: str | None = None
        self.last_stats: dict[str, Any] = {}

    def __call__(self, prompt: str) -> str:
        self.last_thinking, self.last_stats = None, {}
        command = [self.cli, "-p", "--model", self.name, "--output-format", "json", "--tools", "",
                   "--strict-mcp-config", "--no-session-persistence", "--system-prompt", SYSTEM_PROMPT]
        if self.effort:
            command += ["--effort", self.effort]
        if self.schema is not None:
            schema = json.dumps(self.schema, separators=(",", ":"))
            if len(schema) <= MAX_SCHEMA_CHARS:
                command += ["--json-schema", schema]
        start = time.monotonic()
        done = subprocess.run(command, input=prompt.encode("utf-8"), capture_output=True, cwd=self.workdir,
                              timeout=self.timeout)
        output = done.stdout.decode("utf-8", errors="replace")
        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude -p exited {done.returncode}: "
                               f"{(output or done.stderr.decode('utf-8', errors='replace'))[:500]}") from None
        if result.get("is_error"):
            raise RuntimeError(f"claude -p failed: {result.get('result') or result.get('subtype')}")
        usage = result.get("usage") or {}
        self.last_stats = {
            "seconds": round(time.monotonic() - start, 1),
            "prompt_tokens": sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                               "cache_read_input_tokens")),
            "output_tokens": usage.get("output_tokens"),
            "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
            "cost_usd_at_list_price": result.get("total_cost_usd"),
        }
        if result.get("structured_output") is not None:
            return json.dumps(result["structured_output"])
        return result.get("result") or ""

    def label(self) -> str:
        """Short name for log files, e.g. "claude-opus_effort-default"."""
        return f"claude-{self.name}_effort-{self.effort or 'default'}"

    def describe(self) -> dict[str, Any]:
        return {"model": f"claude -p --model {self.name}", "effort": self.effort, "schema": self.schema is not None}
