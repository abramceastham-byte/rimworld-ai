"""The agent loop: observe the game, ask a model what to do, act, and log it."""

import time
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

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
    """Turn the model's raw reply into a Decision, falling back to "wait"."""    
    try:
        return Decision.model_validate_json(reply)
    except ValidationError:
        return Decision(action="wait", reason="could not parse reply")


def run(
    client: RimApiClient,
    model: Callable[[str], str],
    logger: RunLogger,
    max_steps: int = 10,
    step_seconds: float = 5.0,
) -> None:
    """Run the agent loop for max_steps steps."""

    actions: dict[str, Callable[[], None]] = {
        "wait": lambda: None,
        "pause": client.pause,
        "resume": client.resume,
        
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

            reply = model(prompt)
            decision = parse_decision(reply)
            try:
                actions[decision.action]()
            except RimApiError:
                print(f"Step {step}: failed to {decision.action}")
            logger.log_decision(
                {
                    "step": step,
                    "reply": reply,
                    "action": decision.action,
                    "reason": decision.reason,
                }
            )
            print(f"Step {step}: {decision.action} ({decision.reason})")
            time.sleep(step_seconds)
    finally:
        # Runs even on Ctrl+C or a crash: don't leave the colony unattended.
        try:
            client.pause()
        except (RimApiError, httpx.HTTPError):
            pass
