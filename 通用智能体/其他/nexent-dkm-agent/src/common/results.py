"""Shared result objects used across task agents and pipelines."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineResult:
    """Minimal pipeline result used by demo entrypoints and smoke tests."""

    task: str
    status: str
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)
