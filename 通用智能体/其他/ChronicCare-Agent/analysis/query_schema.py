from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimeWindow:
    value: int
    unit: str = "day"
    direction: str = "future"
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    intent: str
    time_window: Optional[TimeWindow] = None
    disease_filters: List[str] = field(default_factory=list)
    risk_filters: List[str] = field(default_factory=list)
    chart_types: List[str] = field(default_factory=list)
    output_preference: List[str] = field(default_factory=list)
    tool_plan: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    canonical_question: Optional[str] = None
    route: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.time_window is not None:
            payload["time_window"] = self.time_window.to_dict()
        return payload
