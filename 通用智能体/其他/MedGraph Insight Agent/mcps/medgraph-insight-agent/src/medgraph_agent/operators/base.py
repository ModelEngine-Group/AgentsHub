from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from medgraph_agent.core.models import OperatorResult, utc_now


class Operator(ABC):
    """Small DataMate-style operator contract used by CLI, API, and MCP."""

    name: str

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> OperatorResult:
        raise NotImplementedError


def ok_result(
    name: str,
    started_at: str,
    *,
    records_processed: int = 0,
    output: dict[str, Any] | None = None,
) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="succeeded",
        started_at=started_at,
        finished_at=utc_now(),
        records_processed=records_processed,
        output=output or {},
    )


def fail_result(name: str, started_at: str, error: Exception) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="failed",
        started_at=started_at,
        finished_at=utc_now(),
        error=f"{type(error).__name__}: {error}",
    )
