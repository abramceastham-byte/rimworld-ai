"""Turn agent logs into readable Markdown.

Works on both kinds of log in logs/:
  runs/<name>/run.jsonl   a real run: one section per step, with the decision,
                          its reason, what happened, retries, and the model's
                          reasoning
  dry-runs/<name>.json    a single dry-run decision, with the full prompt

The Markdown is written next to the log (run.md, or <name>.md). Open it in VS
Code and press Ctrl+Shift+V for the preview.

    python -m scripts.show_log                  # the newest log
    python -m scripts.show_log logs/runs/2026-09-22_0630_gpt-oss-20b_think-medium/run.jsonl
    python -m scripts.show_log --all            # every log in logs/
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rimagent.runlog import read_log

LOG_DIR = Path("logs")

# Decision fields that describe *what* an action targets, in display order.
TARGET_FIELDS = [
    ("colonist", "Colonist"),
    ("work_type", "Work type"),
    ("building_id", "Work table"),
    ("recipe_def_name", "Recipe"),
    ("target_count", "Target count"),
    ("plant", "Crop"),
    ("designation", "Designation"),
    ("building_def", "Building"),
    ("position", "Position"),
    ("stuff", "Material"),
    ("rect", "Area"),
]
MAP_ACTIONS = {"place_blueprint"}  # rotation only matters for these


def block(text: str | None, title: str) -> list[str]:
    """Long text (reasoning, prompts) folded away, shown exactly as written."""
    if not text:
        return []
    # A fence the text itself can't close, since model output may contain ``` or ~~~.
    longest = max((len(run) for run in re.findall(r"~+", text)), default=0)
    fence = "~" * max(4, longest + 1)
    return ["<details>", f"<summary>{title}</summary>", "", f"{fence}text", text.strip(), fence, "",
            "</details>", ""]


def describe_targets(d: dict[str, Any]) -> list[str]:
    lines = []
    for key, label in TARGET_FIELDS:
        value = d.get(key)
        if value is None:
            continue
        note = ""
        if key == "rect":
            x1, z1, x2, z2 = value
            value = f"({x1},{z1}) to ({x2},{z2})"
        elif key == "position":
            value = f"({value[0]},{value[1]})"
            if d.get("action") in MAP_ACTIONS:
                note = f", facing {['north', 'east', 'south', 'west'][d.get('rotation') or 0]}"
        lines.append(f"- {label}: `{value}`{note}")
    return lines


def describe_memory(update: dict[str, Any] | None) -> list[str]:
    if not update:
        return []
    lines = ["**Memory update:**"]
    if update.get("observation"):
        lines.append(f"- Observation: {update['observation']}")
    plan = update.get("replace_plan")
    if plan:
        lines.append(f"- New plan: {plan['objective']} ({plan.get('status', 'active')})")
        lines += [f"  - [{s['status']}] {s['description']}" for s in plan.get("steps", [])]
    if update.get("clear_plan"):
        lines.append("- Plan cleared")
    for key, value in (update.get("status_updates") or {}).items():
        lines.append(f"- Status {key}: {value}")
    lines += [f"- Policy added: {p}" for p in update.get("policies_to_add") or []]
    lines += [f"- Policy removed: {p}" for p in update.get("policies_to_remove") or []]
    return lines + [""]


def run_to_markdown(path: Path) -> str:
    records = read_log(path)
    info = next((r["data"] for r in records if r["kind"] == "run"), {})
    decisions = [r["data"] for r in records
                 if r["kind"] == "decision" and r["data"].get("action") != "turn_end"]
    failures = [r["data"] for r in records if r["kind"] == "model_failure"]

    # The folder name says which run this is (run.jsonl is the same everywhere).
    out = [f"# Run {path.parent.name}", ""]
    if info:
        settings = ", ".join(f"{k} `{v}`" for k, v in info.items())
        out.append(f"**Settings:** {settings}")
    if records:
        out.append(f"**Time:** {records[0]['time'][:19]} to {records[-1]['time'][:19]} (UTC)")
    counts = Counter(d["action"] for d in decisions)
    out += [
        f"**Decisions:** {len(decisions)} "
        f"({', '.join(f'{a} x{n}' for a, n in counts.most_common())})",
        f"**Failed actions:** {sum(1 for d in decisions if d.get('error'))} · "
        f"**Invalid or failed model replies:** {len(failures)} · "
        f"**Fallbacks:** {sum(1 for d in decisions if str(d.get('reason', '')).startswith('model failed'))}",
        "",
    ]
    seconds = [d["model_stats"]["seconds"] for d in decisions if d.get("model_stats")]
    if seconds:
        out += [f"**Model time per decision:** average {sum(seconds) / len(seconds):.1f}s, "
                f"longest {max(seconds):.1f}s", ""]

    # Records arrive in order: state, events, any failed attempts, then the decision.
    state, events, step_failures = None, [], []
    for r in records:
        kind, data = r["kind"], r["data"]
        if kind == "state":
            state, events = data, []
        elif kind == "event":
            events.append(data)
        elif kind == "model_failure":
            step_failures.append(data)
        elif kind == "decision" and data.get("action") == "turn_end":
            out += time_after_markdown(data.get("time_after"))
        elif kind == "decision":
            out += step_to_markdown(data, state, events, step_failures)
            events, step_failures = [], []
    return "\n".join(out)


def time_after_markdown(after: object) -> list[str]:
    """The stretch of game time the loop allowed after a turn's actions."""
    if isinstance(after, dict):
        ran = f"**Then:** the game ran {after.get('seconds')}s ({after.get('ticks')} ticks)"
        return [ran + (f", stopped early: {after['stopped_by']}" if after.get("stopped_by") else ""), ""]
    return [f"**Then:** {after}, no time passed", ""] if after else []


def step_to_markdown(d: dict, state: dict | None, events: list, failures: list) -> list[str]:
    error = d.get("error")
    outcome = "FAILED" if error else "ok"
    index, total = d.get("action_index"), d.get("actions_in_turn") or 1
    number = f"{d.get('step', '?')}" if total == 1 or index is None else f"{d.get('step', '?')}.{index}"
    out = [f"## Step {number}: {d['action']} ({outcome})", ""]
    if state and (index in (None, 0)):
        mode = d.get("time_mode")
        if mode:
            if mode.get("paused"):
                n = mode.get("paused_decisions") or 0
                game = f"Time paused ({mode.get('reason')}), {n} decision{'' if n == 1 else 's'} so far"
            else:
                game = "Time running"
            since = [line for line in d.get("time_status", "").splitlines() if line.startswith("Since")]
            game = " ".join([*since, game + "."])
        elif d.get("time_status"):  # loop-controlled time, before pause/resume modes
            game = d["time_status"].splitlines()[-1]
        else:  # logs from before the loop controlled time
            game = d.get("pause_status") or ("paused" if state.get("is_paused") else "running")
        out.append(f"*Tick {state.get('game_tick')}. {game}*")
        out.append("")
    if events:
        out.append("**New since last step:**")
        out += [f"- {e.get('kind')} ({e.get('category')}): {e.get('text')}" for e in events]
        out.append("")
    out += [f"**Reason:** {d.get('reason') or '(none given)'}", ""]
    if d.get("time"):
        out += [f"**Time choice:** {d['time']}", ""]
    targets = describe_targets(d)
    if targets:
        out += targets + [""]
    out += [f"**Result:** {'FAILED: ' + error if error else d.get('result', 'done')}", ""]
    out += describe_memory(d.get("memory_update"))
    for f in failures:
        out.append(f"**Attempt {f.get('attempt')} rejected** ({f.get('failure_type')}): `{f.get('error')}`")
        if f.get("response"):
            out.append(f"  - Reply was: `{str(f['response'])[:300]}`")
    if failures:
        out.append("")
    stats = d.get("model_stats")
    if stats:
        out += [f"*Model: {stats.get('seconds')}s, {stats.get('prompt_tokens')} prompt tokens, "
                f"{stats.get('output_tokens')} output tokens*", ""]
    out += block(d.get("thinking"), "Model reasoning")
    out += block(d.get("prompt"), "Prompt sent to the model")
    for f in failures:
        out += block(f.get("thinking"), f"Reasoning behind rejected attempt {f.get('attempt')}")
    return out


def dry_run_to_markdown(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    settings, stats = data.get("settings", {}), data.get("stats", {})
    out = [f"# Dry run {path.stem}", "",
           "*Dry run: one decision about the live colony, not acted on.*", "",
           "**Settings:** " + ", ".join(f"{k} `{v}`" for k, v in settings.items()),
           f"**Model:** {stats.get('seconds')}s, {stats.get('prompt_tokens')} prompt tokens, "
           f"{stats.get('output_tokens')} output tokens", ""]
    decision = data.get("decision")
    if decision:
        if "actions" in decision:
            out += [f"## Turn: {decision.get('time', 'legacy')} (valid)", ""]
            for action in decision["actions"]:
                out += [f"**{action['action']}**: {action.get('reason', '')}", ""]
                out += describe_targets(action) + [""]
        else:
            out += [f"## Decision: {decision['action']} (valid)", "",
                    f"**Reason:** {decision.get('reason') or '(none given)'}", ""]
            out += describe_targets(decision) + [""]
        out += describe_memory(decision.get("memory_update"))
    else:
        out += ["## Decision: INVALID", "", f"`{data.get('invalid')}`", "",
                "**Reply was:**", "", f"`{data.get('reply')}`", ""]
    out += block(data.get("thinking"), "Model reasoning")
    out += block(data.get("prompt"), "Full prompt sent to the model")
    return "\n".join(out)


def convert(path: Path) -> Path:
    text = dry_run_to_markdown(path) if path.suffix == ".json" else run_to_markdown(path)
    out = path.with_suffix(".md")
    out.write_text(text + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn agent logs into readable Markdown.")
    parser.add_argument("paths", nargs="*", type=Path, help="Log files (default: the newest).")
    parser.add_argument("--all", action="store_true", help="Convert every log in logs/.")
    args = parser.parse_args()

    logs = sorted([*LOG_DIR.glob("runs/*/run.jsonl"), *LOG_DIR.glob("dry-runs/*.json")],
                  key=lambda p: p.stat().st_mtime)
    paths = args.paths or (logs if args.all else logs[-1:])
    if not paths:
        raise SystemExit("No logs found in logs/.")
    for path in paths:
        print(f"{path} -> {convert(path)}")


if __name__ == "__main__":
    main()
