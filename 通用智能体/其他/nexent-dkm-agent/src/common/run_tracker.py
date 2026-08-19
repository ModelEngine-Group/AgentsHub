"""Shared lightweight run-state tracker for task agents."""

from __future__ import annotations

from typing import Any, Callable


def summarize_artifact(value: Any) -> dict[str, Any]:
    """Normalize step artifacts for serializable run_state output."""

    if isinstance(value, str):
        return {"length": len(value)}
    if isinstance(value, list):
        return {"count": len(value)}
    if not isinstance(value, dict):
        return {}

    summary: dict[str, Any] = {}
    for key in (
        "status",
        "record_count",
        "valid_count",
        "invalid_count",
        "output_path",
        "node_count",
        "edge_count",
    ):
        if key in value:
            summary[key] = value[key]
    if "statistics" in value and isinstance(value["statistics"], dict):
        summary.update(value["statistics"])
    if "charts" in value and isinstance(value["charts"], dict):
        summary["chart_count"] = len(value["charts"])
    if "rows" in value and isinstance(value["rows"], list):
        summary["row_count"] = len(value["rows"])
    return summary


class AgentRunTracker:
    """Track agent step execution with compact summaries."""

    def __init__(self, summarize: Callable[[Any], dict[str, Any]] | None = None) -> None:
        self.status = "pending"
        self.steps: list[dict[str, Any]] = []
        self._summarize = summarize or summarize_artifact

    def start(self) -> None:
        self.status = "running"

    def complete(self, status: str) -> None:
        self.status = status

    def fail(self) -> None:
        self.status = "failed"

    def run_step(self, name: str, operation: Callable[[], Any], message: str) -> Any:
        step = {"name": name, "status": "running", "message": ""}
        self.steps.append(step)
        try:
            result = operation()
        except Exception as exc:
            step["status"] = "failed"
            step["error"] = str(exc)
            self.status = "failed"
            raise
        step["status"] = "completed"
        step["message"] = message
        step["artifacts"] = self._summarize(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "steps": self.steps}
