"""Persistent and working memory for the colony agent."""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanStep(BaseModel):
    """One durable step in the agent's current plan."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=500)
    status: Literal["pending", "in_progress", "completed", "blocked", "abandoned"]


class CurrentPlan(BaseModel):
    """The current multi-step objective, separate from live game telemetry."""

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=500)
    status: Literal["active", "completed", "blocked", "abandoned"] = "active"
    steps: list[PlanStep] = Field(default_factory=list, max_length=20)


class MemoryUpdate(BaseModel):
    """Validated memory changes a model may request with a decision."""

    model_config = ConfigDict(extra="forbid")

    observation: str | None = Field(default=None, max_length=2000)
    replace_plan: CurrentPlan | None = None
    clear_plan: bool = False
    status_updates: dict[str, str] = Field(default_factory=dict)
    clear_statuses: list[str] = Field(default_factory=list)
    policies_to_add: list[str] = Field(default_factory=list)
    policies_to_remove: list[str] = Field(default_factory=list)


class PersistentMemoryState(BaseModel):
    """Structured memory that is saved to JSON between agent runs."""

    model_config = ConfigDict(extra="forbid")

    current_plan: CurrentPlan | None = None
    statuses: dict[str, str] = Field(default_factory=dict)
    policies: list[str] = Field(default_factory=list)
    # Plan steps (and PLAN_KEY for the whole plan) reported completed last
    # turn, held as in_progress until a later reply confirms them.
    unconfirmed: list[str] = Field(default_factory=list)


PLAN_KEY = "(the whole plan)"

# Openings that report this turn rather than state a lesson, and details that
# go stale: coordinates and stock counts. A run's memory filled with lines like
# "Placed TorchLamp adjacent to research bench..." and "current stocks: 155 WoodLog".
DIARY_OPENINGS = ("placed", "planned", "planning", "plan to", "planning to", "created", "established",
                  "built", "building", "designated", "ordered", "scheduled", "set up", "added", "assigned",
                  "enabled", "started", "queued", "marked", "allowed", "will ")
DIARY_DETAIL = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)|\b\d+\s*(WoodLog|Steel|Blocks\w+|Component\w*|Medicine\w*|ticks)\b"
                          r"|\bcurrent stocks?\b", re.IGNORECASE)


def diary_like(observation: str) -> bool:
    """True when an observation reports this turn instead of stating a lasting lesson."""
    text = observation.strip().lower()
    return text.startswith(DIARY_OPENINGS) or bool(DIARY_DETAIL.search(observation))


def held_until_confirmed(plan: CurrentPlan, previous: CurrentPlan | None,
                         unconfirmed: list[str]) -> tuple[CurrentPlan, list[str]]:
    """Accept a step's "completed" only on a later turn than the one it is claimed in.

    Models mark a step completed in the same reply that sends its orders,
    before the game has done anything. A newly completed step is therefore
    kept as in_progress and remembered; if the next reply still says
    completed, the model has seen fresh game state since, and it stands.
    The rest of the plan is always kept, whatever happened to the orders.
    Returns the plan to store and the claims still awaiting confirmation.
    """
    done_before = {s.description.casefold() for s in (previous.steps if previous else [])
                   if s.status == "completed"}
    claimed = {c.casefold() for c in unconfirmed}
    waiting: list[str] = []
    steps = []
    for step in plan.steps:
        key = step.description.casefold()
        if step.status == "completed" and key not in done_before and key not in claimed:
            waiting.append(step.description)
            step = step.model_copy(update={"status": "in_progress"})
        steps.append(step)
    status = plan.status
    plan_done_before = previous is not None and previous.status == "completed"
    if status == "completed" and not plan_done_before and PLAN_KEY.casefold() not in claimed:
        waiting.append(PLAN_KEY)
        status = "active"
    return plan.model_copy(update={"steps": steps, "status": status}), waiting


class DecisionMemoryEntry(BaseModel):
    """One recent decision retained for the current process."""

    step: int
    action: str
    reason: str
    result: str
    colonist: str | None = None
    work_type: str | None = None


class AgentMemory:
    """Manage recent decisions plus durable Markdown and JSON memory."""

    max_recent_decisions = 8
    max_observation_prompt_chars = 1600

    def __init__(self, memory_dir: str | Path = "memory", read_only: bool = False) -> None:
        self.memory_dir = Path(memory_dir)
        self.read_only = read_only
        if not read_only:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.memory_dir / "state.json"
        self.observations_path = self.memory_dir / "observations.md"
        self.recent_decisions: list[DecisionMemoryEntry] = []
        self.state = self._load_state()
        if not read_only:
            self._ensure_observations_file()

    def _load_state(self) -> PersistentMemoryState:
        if not self.state_path.exists():
            state = PersistentMemoryState()
            if not self.read_only:
                self._save_state(state)
            return state
        return PersistentMemoryState.model_validate_json(
            self.state_path.read_text(encoding="utf-8")
        )

    def _save_state(self, state: PersistentMemoryState | None = None) -> None:
        if state is None:
            state = self.state
        temporary_path = self.state_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            state.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(self.state_path)

    def _ensure_observations_file(self) -> None:
        if not self.observations_path.exists():
            self.observations_path.write_text(
                "# Discoveries and ideas\n\n"
                "Big lessons from this playthrough about the colony and how RimWorld works. "
                "Live telemetry and turn-by-turn events are intentionally not kept here.\n",
                encoding="utf-8",
            )

    def remember_decision(
        self,
        step: int,
        action: str,
        reason: str,
        error: str | None,
        colonist: str | None = None,
        work_type: str | None = None,
    ) -> None:
        """Keep a bounded history of decisions for subsequent model calls."""
        self.recent_decisions.append(
            DecisionMemoryEntry(
                step=step,
                action=action,
                reason=reason or "No reason given.",
                result="succeeded" if error is None else f"failed: {error}",
                colonist=colonist,
                work_type=work_type,
            )
        )
        self.recent_decisions = self.recent_decisions[-self.max_recent_decisions :]

    def apply_update(
        self, step: int, game_tick: int, update: MemoryUpdate | None
    ) -> str | None:
        """Apply a model-proposed update to the Markdown and JSON stores.

        Returns a note for the next prompt when the observation was turned
        away as a diary entry (see diary_like), else None.
        """
        if update is None:
            return None

        observation = (update.observation or "").strip()
        note = None
        if observation and diary_like(observation):
            note = ("Memory observation not saved: \"" + observation[:80] + "\" reads like a log of this turn "
                    "(what you placed or planned, stock numbers, coordinates). Save lessons that stay true, "
                    "e.g. \"colonists prefer beds inside a room to beds outdoors\".")
            observation = ""
        if observation:
            with self.observations_path.open("a", encoding="utf-8") as observations:
                observations.write(
                    f"\n## Step {step} (game tick {game_tick})\n\n{observation}\n"
                )

        if update.clear_plan:
            self.state.current_plan = None
            self.state.unconfirmed = []
        elif update.replace_plan is not None:
            self.state.current_plan, self.state.unconfirmed = held_until_confirmed(
                update.replace_plan, self.state.current_plan, self.state.unconfirmed)

        for key in update.clear_statuses:
            self.state.statuses.pop(key.strip(), None)
        for key, value in update.status_updates.items():
            clean_key = key.strip()
            clean_value = value.strip()
            if clean_key and clean_value:
                self.state.statuses[clean_key] = clean_value

        removals = {policy.strip().casefold() for policy in update.policies_to_remove}
        self.state.policies = [
            policy
            for policy in self.state.policies
            if policy.casefold() not in removals
        ]
        known_policies = {policy.casefold() for policy in self.state.policies}
        for policy in update.policies_to_add:
            clean_policy = policy.strip()
            if clean_policy and clean_policy.casefold() not in known_policies:
                self.state.policies.append(clean_policy)
                known_policies.add(clean_policy.casefold())

        self._save_state()
        return note

    def _observation_prompt(self) -> str:
        observations = self.observations_path.read_text(encoding="utf-8") if self.observations_path.exists() else ""
        if len(observations) <= self.max_observation_prompt_chars:
            return observations.strip()
        return (
            "[Earlier observations remain on disk but are omitted from this prompt.]\n"
            + observations[-self.max_observation_prompt_chars :].lstrip()
        )

    def _recent_decisions_prompt(self) -> str:
        if not self.recent_decisions:
            return "- None yet."
        lines = []
        for entry in self.recent_decisions:
            target = ""
            if entry.colonist:
                target += f"; colonist={entry.colonist}"
            if entry.work_type:
                target += f"; work_type={entry.work_type}"
            lines.append(
                f"- Step {entry.step}: {entry.action}{target}; {entry.result}. "
                f"Reason: {entry.reason}"
            )
        return "\n".join(lines)

    def to_prompt(self) -> str:
        """Compact intentions; live outcomes are supplied once by the loop."""
        plan = self.state.current_plan
        lines = ["Agent memory (intentions, not proof of outcomes):"]
        if plan:
            waiting = {c.casefold() for c in self.state.unconfirmed}
            note = " (you reported it done: mark it completed again once the game shows it)"
            lines.append(f"Objective: {plan.objective} [{plan.status}]"
                         + (note if PLAN_KEY.casefold() in waiting else ""))
            lines.extend(f"- {step.status}: {step.description}"
                         + (note if step.description.casefold() in waiting else "")
                         for step in plan.steps if step.status not in ("completed", "abandoned"))
        if self.state.policies:
            lines.append("Policies: " + "; ".join(self.state.policies[-6:]))
        if self.state.statuses:
            lines.append("Notes: " + str(dict(list(self.state.statuses.items())[-6:])))
        lines.append(self._observation_prompt())
        return "\n".join(lines)
