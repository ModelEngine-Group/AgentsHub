from __future__ import annotations

from typing import Any, Dict

from analysis.open_nl2sql.schema_linker import build_schema_links
from analysis.open_nl2sql.sql_candidate_builder import build_sql_candidate
from analysis.open_nl2sql.sql_guard import validate_sql
from orchestration.intent_router import route_intent
from tool_server.utils import fetch_rows


def execute_open_query(question: str, last_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    route = route_intent({"query": question, "last_context": last_context or {}})
    schema_links = build_schema_links(question)
    candidate = build_sql_candidate(question, {"route": "analysis", "intent": route["intent"]}, schema_links)
    sql = candidate.get("sql")
    if not sql:
        return {
            "status": "failed",
            "intent": route["intent"],
            "tool": route["tool"],
            "reason": route["reason"],
            "errors": ["No SQL candidate generated."],
        }
    is_safe, errors, safe_sql, warnings = validate_sql(sql)
    if not is_safe or not safe_sql:
        return {
            "status": "failed",
            "intent": route["intent"],
            "tool": route["tool"],
            "reason": route["reason"],
            "sql": sql,
            "warnings": warnings,
            "errors": errors,
        }
    rows = fetch_rows(safe_sql)
    return {
        "status": "success",
        "intent": route["intent"],
        "tool": route["tool"],
        "reason": route["reason"],
        "sql": safe_sql,
        "row_count": len(rows),
        "rows": rows[:50],
    }

