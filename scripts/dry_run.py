"""Ask a model for one decision about the live colony, without acting on it.

Useful after changing the prompt, or to compare models. It builds the same
prompt a real step would (reading memory/ but never writing to it), sends it to
the model, prints the reasoning, reply and whether it parses, and saves all of
it to logs/dry-runs/<date>_<time>_<model>_think-<setting>.json. Nothing is
sent to the game.

    python -m scripts.dry_run qwen3:14b --think off
    python -m scripts.dry_run gpt-oss:20b --think low
"""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from rimagent.agent import TimeMode, Turn, prepare_turn
from rimagent.blueprints import BlueprintTracker
from rimagent.zones import ZoneTracker
from rimagent.memory import AgentMemory
from rimagent.ollama_model import OllamaModel, parse_think
from rimagent.rimapi import RimApiClient
from rimagent.runlog import stamped_name


def live_context(client: RimApiClient) -> dict:
    """The prompt and choices a real run's first turn would get (see agent.run)."""
    memory = AgentMemory(read_only=True)
    return prepare_turn(client, memory,
        BlueprintTracker(memory.memory_dir / "blueprints.json"),
        ZoneTracker(memory.memory_dir / "zones.json"),
        TimeMode(True, "session start"), read_only=True)


def live_prompt(client: RimApiClient) -> str:
    return live_context(client)["prompt"]


def main() -> None:
    parser = argparse.ArgumentParser(description="One model decision about the live colony, not acted on.")
    parser.add_argument("model", help="Ollama model name, e.g. qwen3:14b")
    parser.add_argument("--think", help="on/off (qwen3) or low/medium/high (gpt-oss)")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument("--show-prompt", action="store_true", help="Also print the full prompt.")
    args = parser.parse_args()

    model = OllamaModel(
        args.model,
        schema=None if args.no_schema else Turn.model_json_schema(),
        think=parse_think(args.think),
        num_ctx=args.num_ctx,
    )
    with RimApiClient() as client:
        context = live_context(client)
        prompt = context["prompt"]
        if model.schema is not None:
            model.schema = context["choices"].schema()
    if args.show_prompt:
        print(prompt, "\n")

    reply = model(prompt)
    try:
        turn = Turn.model_validate_json(reply)
        context["choices"].validate(turn)
        decision = turn.model_dump(exclude_none=True)
        problem = None
    except (ValidationError, ValueError) as e:
        decision, problem = None, str(e)

    stats = model.last_stats
    print(f"{args.model} (think={model.think}): {stats['seconds']}s, "
          f"prompt {stats['prompt_tokens']} tokens, output {stats['output_tokens']} tokens")
    print("\n--- thinking ---\n" + (model.last_thinking or "(none)"))
    print("\n--- reply ---\n" + reply)
    print("\n--- " + ("valid decision ---\n" + json.dumps(decision, indent=2) if decision
                     else f"INVALID ---\n{problem}"))

    out = Path("logs") / "dry-runs" / f"{stamped_name(model.label())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "settings": model.describe(), "stats": stats, "prompt": prompt,
        "thinking": model.last_thinking, "reply": reply, "choice_schema": context["choices"].schema(),
        "decision": decision, "invalid": problem,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
