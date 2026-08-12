"""任务三分析计划与执行结果的数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AnalysisQuery:
    """一项可独立执行并生成证据的只读分析。"""

    title: str
    purpose: str
    sql: str
    params: tuple[Any, ...] = ()
    chart_type: str = "auto"
    source: str = "semantic_layer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisPlan:
    """由用户问题解析得到的分析计划。"""

    question: str
    subject: str | None = None
    queries: list[AnalysisQuery] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    planner: str = "semantic_layer"
    planner_status: str = "ready"
    planner_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "subject": self.subject,
            "queries": [query.to_dict() for query in self.queries],
            "unsupported": list(self.unsupported),
            "planner": self.planner,
            "planner_status": self.planner_status,
            "planner_note": self.planner_note,
        }
