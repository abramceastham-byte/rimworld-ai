"""Ask a model for one decision about the live colony, without acting on it.

Useful after changing the prompt, or to compare models. It builds the same
prompts a real step would (reading memory/ but never writing to it), sends them
to the model, prints the reasoning, reply and whether it parses, and saves all
of it to logs/dry-runs/<date>_<time>_<model>_think-<setting>.json. Nothing is
sent to the game.

With --loop modes (the default) that is an overview (the plan) and then a focus
turn on that plan, or the threat prompt if enemies are active; --loop single
asks the old one-prompt turn.

    python -m scripts.dry_run qwen3:14b --think off
    python -m scripts.dry_run gpt-oss:20b --think low --loop single
    python -m scripts.dry_run --prompts-only            # print the prompts, no model
"""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from rimagent.agent import GAME_FACTS, LEGACY_ACTIONS, TimeMode, Turn, prepare_turn
from rimagent.blueprints import BlueprintTracker
from rimagent.chopping import ChopTracker
from rimagent.memory import AgentMemory, CurrentPlan, PlanStep
from rimagent.mining import MineTracker
from rimagent.modes import (COMBAT, MAX_TOOLKITS, TOOLKITS, FocusTurn, Overview, ThreatTurn, allowed_actions,
                            designations, focus_prompt, narrowed, overview_prompt, threat_prompt, toolkit_work)
from rimagent.ollama_model import OllamaModel, parse_think
from rimagent.rimapi import RimApiClient
from rimagent.layout import LayoutTracker
from rimagent.rooms import RoomTracker
from rimagent.runlog import stamped_name
from rimagent.zones import ZoneTracker


def live_context(client: RimApiClient, memory: AgentMemory | None = None) -> dict:
    """The prompt and choices a real run's first turn would get (see agent.run)."""
    memory = memory or AgentMemory(read_only=True)
    return prepare_turn(client, memory,
        BlueprintTracker(memory.memory_dir / "blueprints.json"),
        ZoneTracker(memory.memory_dir / "zones.json"),
        TimeMode(True, "session start"), read_only=True,
        chops=ChopTracker(memory.memory_dir / "chops.json", read_only=True),
        rooms=RoomTracker(memory.memory_dir / "rooms.json", read_only=True),
        mines=MineTracker(memory.memory_dir / "mining.json", read_only=True),
        layout=LayoutTracker(memory.memory_dir / "layout.json", read_only=True))


def live_prompt(client: RimApiClient) -> str:
    return live_context(client)["prompt"]


def focus_for(context: dict, plan: CurrentPlan, memory: AgentMemory) -> tuple[str, object, set[str]]:
    toolkits = plan.toolkits[:MAX_TOOLKITS]
    allowed = allowed_actions(toolkits)
    choices = narrowed(context["choices"], allowed, designations(toolkits), toolkit_work(toolkits))
    return focus_prompt(context, plan, toolkits, set(), memory, choices, GAME_FACTS), choices, allowed


def ask(model, prompt: str, schema, parse) -> dict:
    """One model call: its stats, thinking, reply and parsed decision (or why not)."""
    if model.schema is not None:
        model.schema = schema
    reply = model(prompt)
    try:
        decision, problem = parse(reply), None
    except (ValidationError, ValueError) as e:
        decision, problem = None, str(e)
    stats = model.last_stats
    print(f"prompt {stats['prompt_tokens']} tokens, output {stats['output_tokens']} tokens, {stats['seconds']}s")
    print("\n--- thinking ---\n" + (model.last_thinking or "(none)"))
    print("\n--- reply ---\n" + reply)
    print("\n--- " + ("valid ---\n" + json.dumps(decision.model_dump(exclude_none=True), indent=2) if decision
                     else f"INVALID ---\n{problem}"))
    return {"prompt": prompt, "stats": stats, "thinking": model.last_thinking, "reply": reply, "schema": schema,
            "decision": decision.model_dump(exclude_none=True) if decision else None, "invalid": problem,
            "parsed": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description="One model decision about the live colony, not acted on.")
    parser.add_argument("model", nargs="?", help="Ollama model name, e.g. qwen3:14b")
    parser.add_argument("--think", help="on/off (qwen3) or low/medium/high (gpt-oss)")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument("--loop", choices=("modes", "single"), default="modes")
    parser.add_argument("--toolkits", help="modes: skip the overview and focus on these, e.g. power,construction")
    parser.add_argument("--show-prompt", action="store_true", help="Also print the full prompts.")
    parser.add_argument("--prompts-only", action="store_true", help="Print the prompts; don't call a model.")
    args = parser.parse_args()
    if not args.model and not args.prompts_only:
        parser.error("give a model name, or --prompts-only")

    memory = AgentMemory(read_only=True)
    with RimApiClient() as client:
        context = live_context(client, memory)
    danger = any(t.active for t in context["threats"])
    plan = memory.state.current_plan
    if args.toolkits:
        names = [t.strip() for t in args.toolkits.split(",") if t.strip()]
        unknown = [t for t in names if t not in TOOLKITS]
        if unknown:
            parser.error(f"unknown toolkits {unknown}; choose from {', '.join(TOOLKITS)}")
        plan = CurrentPlan(objective="(chosen with --toolkits)", toolkits=names, done_when="(not set)")

    if args.prompts_only:
        if args.loop == "single":
            print(context["prompt"])
        elif danger:
            print(threat_prompt(context, set(), memory))
        else:
            if not args.toolkits:
                print(overview_prompt(context, set(), memory, "dry run", plan))
                print("\n" + "=" * 80 + "\n")
            shown = plan if plan and plan.toolkits else CurrentPlan(
                objective="(example) every toolkit", toolkits=list(TOOLKITS)[:MAX_TOOLKITS], done_when="-")
            print(focus_for(context, shown, memory)[0])
        return

    model = OllamaModel(args.model, schema=None if args.no_schema else {}, think=parse_think(args.think),
                        num_ctx=args.num_ctx)
    record: dict = {"settings": model.describe(), "loop": args.loop}
    if args.loop == "single":
        prompt = context["prompt"]
        choices = context["choices"]

        def parse(reply):
            turn = Turn.model_validate_json(reply)
            choices.validate(turn, LEGACY_ACTIONS)
            return turn
        if args.show_prompt:
            print(prompt, "\n")
        record["turn"] = ask(model, prompt, choices.schema(LEGACY_ACTIONS), parse)
    elif danger:
        prompt, allowed, choices = threat_prompt(context, set(), memory), set(COMBAT.actions), context["choices"]

        def parse(reply):
            turn = ThreatTurn.model_validate_json(reply)
            choices.validate(turn, allowed)
            return turn
        if args.show_prompt:
            print(prompt, "\n")
        record["threat"] = ask(model, prompt, choices.schema(allowed, ThreatTurn), parse)
    else:
        if not args.toolkits:
            prompt = overview_prompt(context, set(), memory, "dry run", plan)
            if args.show_prompt:
                print(prompt, "\n")
            print("=== overview ===")
            record["overview"] = ask(model, prompt, Overview.model_json_schema(), Overview.model_validate_json)
            overview = record["overview"].pop("parsed")
            if overview is None:
                print("\nNo valid plan, so no focus turn.")
                plan = None
            else:
                plan = CurrentPlan(objective=overview.goal, toolkits=list(overview.toolkits),
                                   done_when=overview.done_when,
                                   steps=[PlanStep(description=s, status="pending") for s in overview.steps])
        if plan is not None:
            prompt, choices, allowed = focus_for(context, plan, memory)

            def parse(reply):
                turn = FocusTurn.model_validate_json(reply)
                choices.validate(turn, allowed)
                return turn
            if args.show_prompt:
                print(prompt, "\n")
            print("\n=== focus ===")
            record["focus"] = ask(model, prompt, choices.schema(allowed, FocusTurn), parse)
    for part in record.values():
        if isinstance(part, dict):
            part.pop("parsed", None)

    out = Path("logs") / "dry-runs" / f"{stamped_name(model.label())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
