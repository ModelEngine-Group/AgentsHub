from __future__ import annotations

from typing import Any, Dict, List

SAFETY_NOTE = "本结果仅用于慢性病随访数据分析，不构成临床诊断或治疗建议。"


def _metric_name(columns: List[str]) -> str:
    return columns[0] if columns else "value"


def build_insight(question: str, chart_type: str, rows: List[Dict[str, Any]]) -> str:
    if chart_type == "metric_card":
        if rows:
            value = next(iter(rows[0].values()))
            return f"围绕“{question}”的核心统计结果为 {value}。{SAFETY_NOTE}"
        return f"该指标未查询到有效结果。{SAFETY_NOTE}"
    if chart_type == "bar":
        return f"该结果展示了“{question}”对应的分组对比，可用于观察不同类别之间的差异。{SAFETY_NOTE}"
    if chart_type == "line":
        return f"该结果展示了“{question}”对应的时间趋势，可用于后续生成趋势图。{SAFETY_NOTE}"
    if chart_type == "table":
        return f"该结果展示了“{question}”对应的 Top 列表，便于快速定位重点对象。{SAFETY_NOTE}"
    return f"该结果可用于后续分析展示。{SAFETY_NOTE}"


def build_indicator_item(sql_result: Dict[str, Any]) -> Dict[str, Any]:
    rows = sql_result.get("rows", [])
    columns = sql_result.get("columns", [])
    chart_type = sql_result.get("expected_chart_type", "table")
    indicator = {
        "id": sql_result["id"],
        "question": sql_result["question"],
        "intent": sql_result["intent"],
        "status": sql_result["status"],
        "chart_type": chart_type,
        "table": {"columns": columns, "rows": rows},
        "insight": build_insight(sql_result["question"], chart_type, rows),
    }
    if chart_type == "metric_card":
        metric_name = _metric_name(columns)
        metric_value = rows[0].get(metric_name) if rows else None
        indicator["metric"] = {"name": metric_name, "value": metric_value, "unit": "项"}
    elif chart_type in {"line", "bar"} and len(columns) >= 2:
        indicator["x_field"] = columns[0]
        indicator["y_field"] = columns[1]
    return indicator
