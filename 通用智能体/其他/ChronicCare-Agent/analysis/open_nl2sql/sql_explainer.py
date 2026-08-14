from __future__ import annotations

from typing import Any, Dict, List


def build_metric_definition(question: str, filters: Dict[str, Any]) -> str:
    parts: List[str] = [f"查询对象：{question}"]
    time_window = filters.get("time_window") or {}
    if time_window:
        parts.append(f"时间范围：{time_window.get('label') or time_window}")
    diseases = filters.get("disease_filters") or []
    if diseases:
        parts.append(f"疾病过滤：{', '.join(diseases)}")
    risks = filters.get("risk_filters") or []
    if risks:
        parts.append(f"风险过滤：{', '.join(risks)}")
    return "；".join(parts)


def summarize_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "查询已执行，但没有返回结果。"
    if len(rows) == 1:
        row = rows[0]
        pieces = [f"{key}={value}" for key, value in row.items()]
        return f"查询返回 1 行结果：{'；'.join(pieces)}。"
    return f"查询返回 {len(rows)} 行结果，已输出结构化表格。"


def build_sql_response(
    *,
    question: str,
    sql: str,
    result: List[Dict[str, Any]],
    filters: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "sql": sql,
        "is_safe": True,
        "result": result,
        "summary": summarize_rows(result),
        "metric_definition": build_metric_definition(question, filters),
        "filters": filters,
        "warnings": warnings,
    }
