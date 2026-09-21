"""The agent loop: observe the game, ask a model what to do, act, and log it."""

import time
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel

from rimagent.rimapi import GameState, RimApiClient, RimApiError
from rimagent.runlog import RunLogger


class Decision(BaseModel):
    """What the model is allowed to answer. Anything else is rejected."""

    action: Literal["pause", "resume", "wait"]
    reason: str = ""


INSTRUCTIONS = (
    "You are managing a RimWorld colony.\n"
    "Choose exactly one action: pause, resume, wait.\n"
    'Reply only with JSON, like: {"action": "wait", "reason": "short explanation"}\n'
)


def build_prompt(state: GameState) -> str:
    """Build a prompt for the LLM based on the current game state."""
    return (
        "Current game state:\n"
        f"Game is {'paused' if state.is_paused else 'running'}.\n"
        f"Tick {state.game_tick}. {state.colonist_count} colonists.\n"
        f"Wealth {state.colony_wealth:,.0f}. Storyteller {state.storyteller}.\n"
        "\n" + INSTRUCTIONS
    )


def parse_decision(reply: str) -> Decision:
    """Turn the model's raw reply into a Decision, falling back to "wait".

    TODO 1: Replace the line below.
      - Decision.model_validate_json(reply) turns a JSON string into a Decision.
        It raises pydantic.ValidationError if the JSON is broken OR the action
        isn't one of the allowed ones.
      - Wrap it in try/except ValidationError, and in the except branch
        return Decision(action="wait", reason="could not parse reply").
      - Import ValidationError from pydantic at the top of the file.
    Test it: make_fake_model(["not json", '{"action": "dance"}']) should
    give you two "wait" decisions instead of a crash.
    """
    return Decision(action="wait", reason="parse_decision not written yet")


def run(
    client: RimApiClient,
    model: Callable[[str], str],
    logger: RunLogger,
    max_steps: int = 10,
    step_seconds: float = 5.0,
) -> None:
    """Run the agent loop for max_steps steps."""

    # TODO 2: Fill in this table so each action name maps to a function
    # that performs it. "wait" should do nothing: use `lambda: None`.
    # Example entry:  "pause": client.pause,
    actions: dict[str, Callable[[], None]] = {
        "wait": lambda: None,
    }

    try:
        for step in range(max_steps):
            try:
                state = client.get_state()
            except (RimApiError, httpx.TimeoutException) as e:
                print(f"Step {step}: could not read game state: {e}")
                time.sleep(step_seconds)
                continue

            logger.log_state(state)
            prompt = build_prompt(state)

            # TODO 3: Decide and act. Replace the placeholder line below with:
            #   a. reply = model(prompt)                  ask the model
            #   b. decision = parse_decision(reply)       make it safe
            #   c. look up actions[decision.action] and call it, inside
            #      try/except RimApiError so a failed action doesn't end the run
            #   d. logger.log_decision({...}) with the step, the raw reply,
            #      decision.action and decision.reason, so you can debug later
            decision = Decision(action="wait", reason="loop not finished yet")

            print(f"Step {step}: {decision.action} ({decision.reason})")
            time.sleep(step_seconds)
    finally:
        # Runs even on Ctrl+C or a crash: don't leave the colony unattended.
        try:
            client.pause()
        except (RimApiError, httpx.HTTPError):
            pass
