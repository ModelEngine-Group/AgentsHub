"""根据分析意图和查询结果选择可视化形式。"""

from __future__ import annotations

import re
from typing import Any

from .contracts import AnalysisQuery


SUPPORTED_CHART_TYPES = {"auto", "bar", "column", "donut", "line", "table", "metric"}
_DATE_LIKE_RE = re.compile(
    r"^(?:\d{4}(?:[-年/.]\d{1,2})?(?:[-月/.]\d{1,2})?日?|\d{1,2}月|\d{1,2}日)$"
)


def _explicit_chart_type(question: str) -> str | None:
    text = str(question or "").lower()
    if any(word in text for word in ("饼图", "环形图", "圆环图")):
        return "donut"
    if any(word in text for word in ("折线图", "趋势图", "趋势变化", "时间趋势")):
        return "line"
    if "柱状图" in text:
        return "column"
    if any(word in text for word in ("条形图", "排行", "排名", "top")):
        return "bar"
    return None


def _is_part_to_whole(question: str, query: AnalysisQuery) -> bool:
    text = " ".join(
        (str(question or ""), str(query.title or ""), str(query.purpose or ""))
    ).lower()
    if any(word in text for word in ("占比", "构成", "比例")):
        return True
    return "分布" in text and any(
        word in text for word in ("实体类型", "关系类型", "实体分布", "关系分布")
    )


def _numeric_key(rows: list[dict[str, Any]], columns: list[str]) -> str | None:
    for key in columns:
        values = [row.get(key) for row in rows[:20] if row.get(key) is not None]
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            if key not in {"置信度", "confidence"}:
                return key
    return None


def _looks_temporal(rows: list[dict[str, Any]], label_key: str) -> bool:
    labels = [str(row.get(label_key) or "").strip() for row in rows[:12]]
    return bool(labels) and sum(bool(_DATE_LIKE_RE.match(label)) for label in labels) >= max(
        2, len(labels) // 2
    )


def _prepare_donut_rows(
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: float(row.get(value_key) or 0),
        reverse=True,
    )
    if len(ranked) <= 9:
        return ranked
    visible = [dict(row) for row in ranked[:8]]
    remainder = sum(float(row.get(value_key) or 0) for row in ranked[8:])
    if remainder:
        visible.append({label_key: "其他", value_key: remainder})
    return visible


def build_chart(
    question: str,
    query: AnalysisQuery,
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """返回统一图表契约；不适合可视化时返回 ``None``。"""

    if not rows:
        return None
    requested = str(query.chart_type or "auto").strip().lower()
    if requested not in SUPPORTED_CHART_TYPES:
        requested = "auto"
    explicit = _explicit_chart_type(question)
    chart_type = explicit or requested
    if chart_type in {"table", "metric"}:
        return None

    columns = list(rows[0])
    value_key = _numeric_key(rows, columns)
    label_key = next((key for key in columns if key != value_key), None)
    if not value_key or not label_key:
        return None

    if chart_type == "line" and not _looks_temporal(rows, label_key):
        chart_type = "column"
    if chart_type == "auto":
        if _looks_temporal(rows, label_key):
            chart_type = "line"
        elif _is_part_to_whole(question, query) and len(rows) <= 30:
            chart_type = "donut"
        elif len(rows) <= 7 and all(len(str(row.get(label_key) or "")) <= 10 for row in rows):
            chart_type = "column"
        else:
            chart_type = "bar"

    limits = {"bar": 20, "column": 12, "line": 16}
    if chart_type == "donut":
        data = _prepare_donut_rows(rows, label_key, value_key)
    else:
        data = rows[: limits.get(chart_type, 20)]

    return {
        "type": chart_type,
        "title": query.title,
        "subtitle": query.purpose,
        "label_key": label_key,
        "value_key": value_key,
        "data": data,
    }
