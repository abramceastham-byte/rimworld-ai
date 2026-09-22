"""The agent loop: observe the game, ask a model what to do, act, and log it."""

import time
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError, model_validator

from rimagent.events import EventListener, GameEvent
from rimagent.memory import AgentMemory, MemoryUpdate
from rimagent.rimapi import (
    Alert,
    Colonist,
    GameState,
    RimApiClient,
    RimApiError,
    Threat,
)
from rimagent.runlog import RunLogger


class Decision(BaseModel):
    """What the model is allowed to answer. Anything else is rejected."""

    action: Literal["pause", "resume", "wait", "enable_work", "disable_work"]
    reason: str = ""
    colonist: str | None = None  # a colonist's name, for enable_work / disable_work
    work_type: str | None = None  # e.g. "Cooking", for enable_work / disable_work
    memory_update: MemoryUpdate | None = None

    @model_validator(mode="after")
    def work_actions_need_a_target(self) -> "Decision":
        # Runs after the fields are checked. Raising here makes parsing fail,
        # so parse_decision turns an incomplete work action into "wait".
        if self.action in ("enable_work", "disable_work") and not (
            self.colonist and self.work_type
        ):
            raise ValueError(f"{self.action} needs both colonist and work_type")
        return self


INSTRUCTIONS = (
    "You are managing a RimWorld colony.\n"
    "Choose exactly one action: pause, resume, wait, enable_work, disable_work.\n"
    "enable_work / disable_work switch one kind of work on or off for one colonist.\n"
    "You may include an optional memory_update object. Use observation only for "
    "durable insights worth remembering across the playthrough. Use replace_plan, "
    "status_updates, and policy changes for intentions and strategy.\n"
    "Do not store live facts already supplied every step: pause state, tick, wealth, "
    "storyteller, colonist count, alerts, threats, colonist age/health/mood/hunger/job/"
    "skills/work settings, new events, or available work types.\n"
    "Reply only with JSON, like one of these:\n"
    '{"action": "wait", "reason": "short explanation"}\n'
    '{"action": "enable_work", "colonist": "Skye", "work_type": "Cooking", "reason": "..."}\n'
    '{"action": "wait", "reason": "planning", "memory_update": '
    '{"replace_plan": {"objective": "Stabilize food production", "status": "active", '
    '"steps": [{"description": "Assign a capable cook", "status": "pending"}]}, '
    '"policies_to_add": ["Keep at least one capable colonist assigned to cooking"]}}\n'
)


def describe_colonist(c: Colonist) -> str:
    """One line per colonist: needs, what they're doing, best skills, enabled work."""
    passion = {0: "", 1: " (interested)", 2: " (burning)"}
    best = sorted(
        (s for s in c.skills if not s.totally_disabled), key=lambda s: -s.level
    )[:3]
    skills = ", ".join(f"{s.name} {s.level}{passion.get(s.passion, '')}" for s in best)
    work = ", ".join(w.work_type for w in c.work_priorities) or "nothing"
    return (
        f"- {c.name} (age {c.age}): mood {c.mood:.0%}, health {c.health:.0%}, "
        f"fed {c.hunger:.0%}, doing {c.current_job or 'nothing'}. "
        f"Best skills: {skills}. Works: {work}."
    )


def describe_pause(
    is_paused: bool, paused_by_agent: bool | None, threat_letter: GameEvent | None
) -> str:
    """Explain to the model whether the game is paused, and why."""
    if not is_paused:
        return "Game is running."
    if paused_by_agent is None:
        return "Game is paused (it was already paused when you started)."
    if paused_by_agent:
        return "Game is paused: you paused it."
    if threat_letter:
        return f"Game is paused: the game paused itself because of a major threat ({threat_letter.text})."
    return "Game is paused by the player or an open menu."


def build_prompt(
    state: GameState,
    colonists: list[Colonist],
    alerts: list[Alert],
    threats: list[Threat],
    events: list[GameEvent],
    work_types: list[str],
    pause_status: str,
    memory: AgentMemory,
) -> str:
    """Build a prompt for the LLM based on the current game state."""
    lines = [
        "Current game state:",
        pause_status,
        f"Tick {state.game_tick}. {state.colonist_count} colonists.",
        f"Wealth {state.colony_wealth:,.0f}. Storyteller {state.storyteller}.",
        "",
        "Threats:",
        *([t.summary() for t in threats] or ["None."]),
        "",
        "New since last step:",
        *([f"- {e.kind} ({e.category}): {e.text}" for e in events] or ["Nothing."]),
        "",
        "Alerts: " + (", ".join(a.label for a in alerts) or "none") + ".",
        "",
        "Colonists:",
        *[describe_colonist(c) for c in colonists],
        "",
        "Work types: " + ", ".join(work_types) + ".",
        "",
    ]
    return "\n".join(lines) + INSTRUCTIONS + "\n" + memory.to_prompt()


def parse_decision(reply: str) -> Decision:
    """Turn the model's raw reply into a Decision, falling back to "wait"."""
    try:
        return Decision.model_validate_json(reply)
    except ValidationError:
        return Decision(action="wait", reason="could not parse reply")


def request_decision(
    model: Callable[[str], str],
    prompt: str,
    logger: RunLogger,
    step: int,
    danger_present: bool,
) -> tuple[Decision, object | None]:
    """Ask twice if needed, logging failures before choosing a safe fallback."""
    attempt_prompt = prompt
    last_reply: object | None = None

    for attempt in range(1, 3):
        try:
            reply = model(attempt_prompt)
            last_reply = reply
        except Exception as error:  # noqa: BLE001 - isolate arbitrary model adapters
            # Model adapters can fail because of timeouts, connection errors, or
            # server errors. KeyboardInterrupt/SystemExit are intentionally not caught.
            logger.log_model_failure(
                {
                    "step": step,
                    "attempt": attempt,
                    "failure_type": "model_call_error",
                    "response": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        try:
            return Decision.model_validate_json(reply), reply
        except (ValidationError, TypeError, ValueError) as error:
            logger.log_model_failure(
                {
                    "step": step,
                    "attempt": attempt,
                    "failure_type": "invalid_response",
                    "response": reply,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if attempt == 1:
                attempt_prompt = (
                    prompt
                    + "\n\nYour previous response was invalid:\n"
                    + str(reply)[:4000]
                    + "\n\nValidation error:\n"
                    + str(error)
                    + "\nReturn one corrected JSON object only."
                )

    action: Literal["pause", "wait"] = "pause" if danger_present else "wait"
    reason = (
        "model failed twice; paused because a threat is present"
        if danger_present
        else "model failed twice; waiting safely"
    )
    return Decision(action=action, reason=reason), last_reply


def colonist_id(colonists: list[Colonist], name: str | None) -> int:
    """Find a colonist's id from the name the model used (case doesn't matter)."""
    for c in colonists:
        if name and c.name.lower() == name.lower():
            return c.id
    raise ValueError(f"no colonist named {name!r}")


def run(
    client: RimApiClient,
    model: Callable[[str], str],
    logger: RunLogger,
    events: EventListener | None = None,
    max_steps: int = 10,
    step_seconds: float = 5.0,
    memory: AgentMemory | None = None,
) -> None:
    """Run the agent loop for max_steps steps."""

    # Each action gets the decision and this step's colonists, so work actions
    # can turn the colonist's name into the id RIMAPI needs.
    actions: dict[str, Callable[[Decision, list[Colonist]], None]] = {
        "wait": lambda d, cols: None,
        "pause": lambda d, cols: client.pause(),
        "resume": lambda d, cols: client.resume(),
        "enable_work": lambda d, cols: client.enable_work(
            colonist_id(cols, d.colonist), d.work_type
        ),
        "disable_work": lambda d, cols: client.disable_work(
            colonist_id(cols, d.colonist), d.work_type
        ),
    }
    work_types = client.get_work_types()
    if memory is None:
        memory = AgentMemory()
    paused_by_agent: bool | None = None  # unknown until the agent acts
    last_threat_letter: GameEvent | None = None

    try:
        for step in range(max_steps):
            try:
                state = client.get_state()
                colonists = client.get_colonists()
                alerts = client.get_alerts()
                threats = client.get_threats()
            except (RimApiError, httpx.TimeoutException) as e:
                print(f"Step {step}: could not read game state: {e}")
                time.sleep(step_seconds)
                continue
            new_events = events.drain() if events else []

            for event in new_events:
                if event.category == "ThreatBig":
                    last_threat_letter = event

            if not state.is_paused:
                # The game is running, so any earlier pause is over, whoever caused it.
                paused_by_agent = False
                last_threat_letter = None

            logger.log_state(state)
            for event in new_events:
                logger.log_event(event)

            pause_status = describe_pause(
                state.is_paused,
                paused_by_agent,
                last_threat_letter,
            )
            prompt = build_prompt(
                state,
                colonists,
                alerts,
                threats,
                new_events,
                work_types,
                pause_status,
                memory,
            )

            danger_present = any(threat.active for threat in threats) or any(
                event.category.startswith("Threat") for event in new_events
            )
            decision, reply = request_decision(
                model,
                prompt,
                logger,
                step,
                danger_present,
            )
            error = None
            try:
                actions[decision.action](decision, colonists)
            except (RimApiError, ValueError) as e:
                # RimApiError: RIMAPI refused (e.g. work the colonist can't do).
                # ValueError: the model named a colonist who doesn't exist.
                error = str(e)
                print(f"Step {step}: failed to {decision.action}: {e}")

            if error is None:
                if decision.action == "pause":
                    paused_by_agent = True
                elif decision.action == "resume":
                    paused_by_agent = False

            memory.remember_decision(
                step,
                decision.action,
                decision.reason,
                error,
                decision.colonist,
                decision.work_type,
            )
            memory.apply_update(step, state.game_tick, decision.memory_update)
            logger.log_decision(
                {
                    "step": step,
                    "reply": reply,
                    "action": decision.action,
                    "colonist": decision.colonist,
                    "work_type": decision.work_type,
                    "reason": decision.reason,
                    "memory_update": (
                        decision.memory_update.model_dump()
                        if decision.memory_update
                        else None
                    ),
                    "pause_status": pause_status,
                    "error": error,
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
