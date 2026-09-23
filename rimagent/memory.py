"""Persistent and working memory for the colony agent."""

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
                "# Agent observations\n\n"
                "Durable observations from this playthrough. Live RIMAPI telemetry "
                "is intentionally not copied here.\n",
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
    ) -> None:
        """Apply a model-proposed update to the Markdown and JSON stores."""
        if update is None:
            return

        observation = (update.observation or "").strip()
        if observation:
            with self.observations_path.open("a", encoding="utf-8") as observations:
                observations.write(
                    f"\n## Step {step} (game tick {game_tick})\n\n{observation}\n"
                )

        if update.clear_plan:
            self.state.current_plan = None
        elif update.replace_plan is not None:
            self.state.current_plan = update.replace_plan

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
            lines.append(f"Objective: {plan.objective} [{plan.status}]")
            lines.extend(f"- {step.status}: {step.description}" for step in plan.steps
                         if step.status not in ("completed", "abandoned"))
        if self.state.policies:
            lines.append("Policies: " + "; ".join(self.state.policies[-6:]))
        if self.state.statuses:
            lines.append("Notes: " + str(dict(list(self.state.statuses.items())[-6:])))
        lines.append(self._observation_prompt())
        return "\n".join(lines)
