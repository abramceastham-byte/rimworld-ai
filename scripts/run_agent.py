"""Start the agent. Run from the repo root:  python -m scripts.run_agent

RimWorld must be running with a colony loaded.
"""

from rimagent.agent import run
from rimagent.fake_model import make_fake_model
from rimagent.rimapi import RimApiClient
from rimagent.runlog import RunLogger


def main() -> None:
    logger = RunLogger()
    with RimApiClient() as client:
        run(client, make_fake_model(), logger, max_steps=3, step_seconds=5)
    print(f"Log saved to {logger.path}")


if __name__ == "__main__":
    main()
