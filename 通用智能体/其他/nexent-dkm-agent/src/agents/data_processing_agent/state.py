"""Execution state tracking for the task-1 agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class StepRecord:
    """State for one scheduled agent step."""

    name: str
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    message: str = ""
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskStateTracker:
    """Track task status, step timing, and failures in a serializable form."""

    def __init__(self) -> None:
        self.status = "pending"
        self.steps: list[StepRecord] = []
        self.started_at: str | None = None
        self.finished_at: str | None = None

    def start_task(self) -> None:
        self.status = "running"
        self.started_at = _now()

    def complete_task(self, status: str = "completed") -> None:
        self.status = status
        self.finished_at = _now()

    def start_step(self, name: str) -> StepRecord:
        step = StepRecord(name=name, status="running", started_at=_now())
        self.steps.append(step)
        return step

    def complete_step(
        self,
        step: StepRecord,
        message: str = "",
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        step.status = "completed"
        step.finished_at = _now()
        step.message = message
        step.artifacts = artifacts or {}

    def fail_step(self, step: StepRecord, error: Exception) -> None:
        step.status = "failed"
        step.finished_at = _now()
        step.error = str(error)
        self.status = "failed"
        self.finished_at = step.finished_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [step.to_dict() for step in self.steps],
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
