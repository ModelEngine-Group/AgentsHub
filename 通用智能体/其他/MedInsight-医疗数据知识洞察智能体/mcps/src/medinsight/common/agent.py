from __future__ import annotations

from typing import Any, Dict, List

from medinsight.common.schemas import Event


class AgentRun:
    def __init__(self) -> None:
        self.events: List[Event] = []

    def emit(self, stage: str, status: str, message: str, **detail: Any) -> None:
        self.events.append(Event(stage=stage, status=status, message=message, detail=detail))

    def event_dicts(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.events]
