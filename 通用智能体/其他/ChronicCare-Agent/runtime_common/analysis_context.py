from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from runtime_common.common import PROJECT_ROOT

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_METRIC_DEFINITION_VERSION = "1.0.0"


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_current_metrics() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs/current_metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_date(value: str | date | None, timezone: str) -> date:
    if isinstance(value, date):
        return value
    if value:
        return date.fromisoformat(str(value).strip())
    demo_enabled = os.getenv("CHRONICCARE_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes"}
    configured = os.getenv("CHRONICCARE_AS_OF_DATE")
    if demo_enabled:
        configured = os.getenv("CHRONICCARE_DEMO_AS_OF_DATE") or configured
    if configured:
        return date.fromisoformat(configured)
    return datetime.now(ZoneInfo(timezone)).date()


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    as_of_date: str
    timezone: str
    data_version: str
    sqlite_version: str
    graph_version: str
    cohort_id: str | None = None
    cohort_definition: Mapping[str, Any] | None = None
    window_start: str | None = None
    window_end: str | None = None
    window_inclusive: bool = True
    latest_record_policy: str = "latest_by_observed_at_then_visit_id"
    metric_definition_version: str = DEFAULT_METRIC_DEFINITION_VERSION
    demo_fixed_date: bool = False

    @classmethod
    def current(
        cls,
        *,
        as_of_date: str | date | None = None,
        timezone: str = DEFAULT_TIMEZONE,
        cohort_id: str | None = None,
        cohort_definition: Mapping[str, Any] | None = None,
    ) -> "AnalysisContext":
        metrics = _read_current_metrics()
        parsed = _parse_date(as_of_date, timezone)
        demo_fixed = bool(
            os.getenv("CHRONICCARE_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes"}
            and os.getenv("CHRONICCARE_DEMO_AS_OF_DATE")
        )
        sqlite_hash = _file_hash(PROJECT_ROOT / "data/sqlite/chroniccare.db")
        graph_hash = _file_hash(PROJECT_ROOT / "data/graph/graph.json")
        return cls(
            as_of_date=parsed.isoformat(),
            timezone=timezone,
            data_version=str(metrics.get("data_version") or "unknown"),
            sqlite_version=f"sha256:{sqlite_hash}" if sqlite_hash else "unavailable",
            graph_version=f"sha256:{graph_hash}" if graph_hash else "unavailable",
            cohort_id=cohort_id,
            cohort_definition=dict(cohort_definition or {}),
            metric_definition_version=str(metrics.get("metric_definition_version") or DEFAULT_METRIC_DEFINITION_VERSION),
            demo_fixed_date=demo_fixed,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AnalysisContext":
        value = value or {}
        base = cls.current(
            as_of_date=value.get("as_of_date"),
            timezone=str(value.get("timezone") or DEFAULT_TIMEZONE),
            cohort_id=value.get("cohort_id"),
            cohort_definition=value.get("cohort_definition") or {},
        )
        allowed = {field for field in cls.__dataclass_fields__}
        overrides = {key: item for key, item in value.items() if key in allowed and item is not None}
        return replace(base, **overrides)

    def with_window(self, days: int) -> "AnalysisContext":
        exact_days = max(1, min(int(days), 366))
        start = date.fromisoformat(self.as_of_date)
        end = start + timedelta(days=exact_days - 1)
        return replace(self, window_start=start.isoformat(), window_end=end.isoformat(), window_inclusive=True)

    def with_cohort(self, cohort_id: str, definition: Mapping[str, Any]) -> "AnalysisContext":
        return replace(self, cohort_id=cohort_id, cohort_definition=dict(definition))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def attach_analysis_context(payload: dict[str, Any], context: AnalysisContext) -> dict[str, Any]:
    payload["analysis_context"] = context.to_dict()
    payload["as_of_date"] = context.as_of_date
    payload["data_version"] = context.data_version
    if context.demo_fixed_date:
        payload["demo_fixed_date_notice"] = f"演示固定日期：{context.as_of_date}"
    return payload
