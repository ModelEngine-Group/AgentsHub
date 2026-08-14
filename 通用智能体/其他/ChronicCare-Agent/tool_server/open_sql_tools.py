from __future__ import annotations

from typing import Any, Dict

from analysis.open_sql import (
    get_open_sql_examples,
    get_open_sql_schema,
    open_sql_eval,
    open_sql_query,
    recent_open_sql_traces,
)


def open_sql_query_tool(
    question: str,
    prefer_llm: bool = True,
    allow_chart: bool = True,
    force_llm: bool = False,
    as_of_date: str | None = None,
) -> Dict[str, Any]:
    return open_sql_query(
        question=question,
        prefer_llm=prefer_llm,
        allow_chart=allow_chart,
        force_llm=force_llm,
        analysis_context={"as_of_date": as_of_date} if as_of_date else None,
    )


def open_sql_schema_tool() -> Dict[str, Any]:
    return get_open_sql_schema()


def open_sql_eval_tool() -> Dict[str, Any]:
    return open_sql_eval()


def open_sql_examples_tool() -> Dict[str, Any]:
    return get_open_sql_examples()


def open_sql_recent_traces_tool(limit: int = 10) -> Dict[str, Any]:
    return recent_open_sql_traces(limit=limit)
