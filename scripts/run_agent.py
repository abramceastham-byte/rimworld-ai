"""Start the agent. Run from the repo root.

RimWorld must be running with a colony loaded. For a real model, Ollama must be
reachable at OLLAMA_HOST (e.g. through the SSH tunnel in the README).

    python -m scripts.run_agent                                   # fake model, 3 steps
    python -m scripts.run_agent --model qwen3:14b --think on --steps 20
    python -m scripts.run_agent --model gpt-oss:20b --think medium --steps 20
    python -m scripts.run_agent --model qwen3:14b --loop single   # the old one-prompt loop
    python -m scripts.run_agent --provider claude-cli --model opus --steps 20   # Claude on your plan

--loop modes (the default) asks an overview prompt for a plan and its toolkits,
then focus prompts showing only those toolkits, and a threat prompt during a fight.

The game is paused while the model decides and its orders run. Each turn then
chooses advance (the game runs --run-seconds, or --threat-run-seconds while a
threat is active) or hold. Press Ctrl+C to stop; the game is left paused.
"""

import argparse

import httpx

from rimagent.agent import RUN_TICKS, THREAT_RUN_TICKS, TICKS_PER_SECOND, Turn, run
from rimagent.claude_cli_model import ClaudeCliModel
from rimagent.events import EventListener
from rimagent.fake_model import make_fake_model, make_fake_modes_model
from rimagent.ollama_model import OllamaModel, parse_think
from rimagent.rimapi import RimApiClient
from rimagent.runlog import RunLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RimWorld agent.")
    parser.add_argument("--provider", choices=("ollama", "claude-cli"), default="ollama",
                        help="ollama (default), or claude-cli: Claude through `claude -p` on your Claude plan.")
    parser.add_argument("--model", help="Ollama model name, e.g. qwen3:14b, or with --provider claude-cli a Claude "
                                        "alias (opus, sonnet, haiku). Omit to use the fake model.")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="claude-cli only: reasoning effort. Default: the model's own.")
    parser.add_argument("--think", help="on/off (qwen3) or low/medium/high (gpt-oss). Default: the model's own.")
    parser.add_argument("--steps", type=int, help="Turns to take (default: 3 fake, 10 real). With --loop modes the run "
                             "finishes the plan or fight in hand before stopping (up to 18 extra turns).")
    parser.add_argument("--run-seconds", type=float, default=RUN_TICKS / TICKS_PER_SECOND,
                        help="Game time per advance, in seconds at normal speed (default 62.5: 1.5 in-game hours, "
                             "about 10.4 real s at speed 3). "
                             "Higher --speed makes the real wait shorter, not the colony time.")
    parser.add_argument("--threat-run-seconds", type=float, default=THREAT_RUN_TICKS / TICKS_PER_SECOND,
                        help="Game time per advance while a threat is active (default 18.75).")
    parser.add_argument("--speed", type=int, choices=(1, 2, 3), default=1,
                        help="Game speed while running: 1 normal, 2 fast, 3 superfast.")
    parser.add_argument("--num-ctx", type=int, default=16384, help="Model context window in tokens.")
    parser.add_argument("--loop", choices=("modes", "single"), default="modes",
                        help="modes: overview, focus and threat prompts (default); single: one big prompt.")
    parser.add_argument("--no-schema", action="store_true",
                        help="Don't force the reply to match each turn's JSON schema.")
    args = parser.parse_args()

    if args.model and args.provider == "claude-cli":
        model = ClaudeCliModel(args.model, effort=args.effort,
                               schema=None if args.no_schema else Turn.model_json_schema())
        info, label = model.describe(), model.label()
        steps = args.steps or 10
    elif args.model:
        model = OllamaModel(
            args.model,
            schema=None if args.no_schema else Turn.model_json_schema(),
            think=parse_think(args.think),
            num_ctx=args.num_ctx,
        )
        info, label = model.describe(), model.label()
        steps = args.steps or 10
    else:
        model = make_fake_modes_model() if args.loop == "modes" else make_fake_model()
        info, label = {"model": "fake"}, "fake"
        steps = args.steps or 3

    with RimApiClient() as client:
        try:
            client.get_state()  # fail fast, before creating an empty log
        except httpx.ConnectError:
            print("Can't reach RIMAPI. Is RimWorld running with the mod enabled and a colony "
                  "loaded? (Check RIMAPI_URL in .env.)")
            return

        logger = RunLogger(label=label)
        logger.log_run_info({
            **info, "steps": steps, "run_seconds": args.run_seconds,
            "threat_run_seconds": args.threat_run_seconds, "speed": args.speed, "loop": args.loop,
        })
        print(f"Running {info['model']} for {steps} steps. Log: {logger.path}")
        try:
            with EventListener() as events:
                run(
                    client, model, logger, events, max_steps=steps,
                    run_ticks=round(args.run_seconds * TICKS_PER_SECOND),
                    threat_run_ticks=round(args.threat_run_seconds * TICKS_PER_SECOND),
                    speed=args.speed, loop=args.loop,
                )
        except KeyboardInterrupt:
            # run() has already paused the game on its way out.
            print(f"\nStopped by you. The game is paused. Log: {logger.path}")
            return
    print(f"Log saved to {logger.path}")


if __name__ == "__main__":
    main()
