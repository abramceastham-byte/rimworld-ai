"""Start the agent. Run from the repo root.

RimWorld must be running with a colony loaded. For a real model, Ollama must be
reachable at OLLAMA_HOST (e.g. through the SSH tunnel in the README).

    python -m scripts.run_agent                                   # fake model, 3 steps
    python -m scripts.run_agent --model qwen3:14b --think off --steps 20
    python -m scripts.run_agent --model gpt-oss:20b --think low --steps 20
"""

import argparse

from rimagent.agent import Decision, run
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
    parser.add_argument("--step-seconds", type=float, default=5.0, help="Pause between steps.")
    parser.add_argument("--num-ctx", type=int, default=16384, help="Model context window in tokens.")
    parser.add_argument("--no-schema", action="store_true",
                        help="Don't force the reply to match Decision's JSON schema.")
    args = parser.parse_args()

    if args.model:
        model = OllamaModel(
            args.model,
            schema=None if args.no_schema else Decision.model_json_schema(),
            think=parse_think(args.think),
            num_ctx=args.num_ctx,
        )
        info, label = model.describe(), model.label()
        steps = args.steps or 10
    else:
        model = make_fake_model()
        info, label = {"model": "fake"}, "fake"
        steps = args.steps or 3

    logger = RunLogger(label=label)
    logger.log_run_info({**info, "steps": steps, "step_seconds": args.step_seconds})
    print(f"Running {info['model']} for {steps} steps. Log: {logger.path}")
    with RimApiClient() as client, EventListener() as events:
        run(client, model, logger, events, max_steps=steps, step_seconds=args.step_seconds)
    print(f"Log saved to {logger.path}")


if __name__ == "__main__":
    main()
