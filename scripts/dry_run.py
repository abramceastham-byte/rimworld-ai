"""Ask a model for one decision about the live colony, without acting on it.

Useful after changing the prompt, or to compare models. It builds the same
prompt a real step would (reading memory/ but never writing to it), sends it to
the model, prints the reasoning, reply and whether it parses, and saves all of
it to logs/dry-run-<model>-<time>.json. Nothing is sent to the game.

    python -m scripts.dry_run qwen3:14b --think off
    python -m scripts.dry_run gpt-oss:20b --think low
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from rimagent.agent import Decision, build_prompt, describe_map, describe_pause
from rimagent.memory import AgentMemory
from rimagent.ollama_model import OllamaModel, parse_think
from rimagent.rimapi import RimApiClient


def live_prompt(client: RimApiClient) -> str:
    """The prompt a real step would build right now (as in agent.run's first step)."""
    state, colonists = client.get_state(), client.get_colonists()
    map_lines = describe_map(
        colonists, client.get_terrain(), client.get_game_defs(), client.get_zones(), [],
        client.get_finished_research(), client.get_items(), client.get_trees(),
    )
    return build_prompt(
        state, colonists, client.get_alerts(), client.get_threats(), [], client.get_work_types(),
        client.get_work_tables(), describe_pause(state.is_paused, None, None), AgentMemory(),
        map_lines,
    )


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
        schema=None if args.no_schema else Decision.model_json_schema(),
        think=parse_think(args.think),
        num_ctx=args.num_ctx,
    )
    with RimApiClient() as client:
        prompt = live_prompt(client)
    if args.show_prompt:
        print(prompt, "\n")

    reply = model(prompt)
    try:
        decision = Decision.model_validate_json(reply).model_dump(exclude_none=True)
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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path("logs") / f"dry-run-{args.model.replace(':', '_')}-{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "settings": model.describe(), "stats": stats, "prompt": prompt,
        "thinking": model.last_thinking, "reply": reply,
        "decision": decision, "invalid": problem,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
