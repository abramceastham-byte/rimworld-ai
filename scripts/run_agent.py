"""Start the agent. Run from the repo root.

RimWorld must be running with a colony loaded. For a real model, Ollama must be
reachable at OLLAMA_HOST (e.g. through the SSH tunnel in the README).

    python -m scripts.run_agent                                   # fake model, 3 steps
    python -m scripts.run_agent --model qwen3:14b --think on --steps 20
    python -m scripts.run_agent --model gpt-oss:20b --think medium --steps 20

The loop controls game time: paused while the model decides, then running for
--run-seconds after each action (--threat-run-seconds while a threat is active).
Press Ctrl+C to stop; the game is left paused.
"""

import argparse

import httpx

from rimagent.agent import Turn, run
from rimagent.events import EventListener
from rimagent.fake_model import make_fake_model
from rimagent.ollama_model import OllamaModel, parse_think
from rimagent.rimapi import RimApiClient
from rimagent.runlog import RunLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RimWorld agent.")
    parser.add_argument("--model", help="Ollama model name, e.g. qwen3:14b. Omit to use the fake model.")
    parser.add_argument("--think", help="on/off (qwen3) or low/medium/high (gpt-oss). Default: the model's own.")
    parser.add_argument("--steps", type=int, help="Decisions to make (default: 3 fake, 10 real).")
    parser.add_argument("--run-seconds", type=float, default=10.0,
                        help="How long the game runs after each decision (default 10).")
    parser.add_argument("--threat-run-seconds", type=float, default=3.0,
                        help="How long it runs while a threat is active (default 3).")
    parser.add_argument("--speed", type=int, choices=(1, 2, 3), default=1,
                        help="Game speed while running: 1 normal, 2 fast, 3 superfast.")
    parser.add_argument("--num-ctx", type=int, default=16384, help="Model context window in tokens.")
    parser.add_argument("--no-schema", action="store_true",
                        help="Don't force the reply to match the Turn JSON schema.")
    args = parser.parse_args()

    if args.model:
        model = OllamaModel(
            args.model,
            schema=None if args.no_schema else Turn.model_json_schema(),
            think=parse_think(args.think),
            num_ctx=args.num_ctx,
        )
        info, label = model.describe(), model.label()
        steps = args.steps or 10
    else:
        model = make_fake_model()
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
            "threat_run_seconds": args.threat_run_seconds, "speed": args.speed,
        })
        print(f"Running {info['model']} for {steps} steps. Log: {logger.path}")
        try:
            with EventListener() as events:
                run(
                    client, model, logger, events, max_steps=steps,
                    run_seconds=args.run_seconds, threat_run_seconds=args.threat_run_seconds,
                    speed=args.speed,
                )
        except KeyboardInterrupt:
            # run() has already paused the game on its way out.
            print(f"\nStopped by you. The game is paused. Log: {logger.path}")
            return
    print(f"Log saved to {logger.path}")


if __name__ == "__main__":
    main()
