from __future__ import annotations

from typing import Any, Dict

from orchestration.answer_formatter import format_answer
from orchestration.query_executor import execute_query_plan
from orchestration.question_classifier import classify_question
from orchestration.question_parser import build_query_plan


def build_question_pipeline(query: str, last_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    classified = classify_question({"query": query, "last_context": last_context or {}})
    plan = build_query_plan(classified, query)
    return {
        "classification": classified,
        "plan": plan,
    }


def run_question_pipeline(query: str, last_context: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    pipeline = build_question_pipeline(query, last_context=last_context)
    payload = execute_query_plan(pipeline["plan"])
    if payload is None:
        return None
    payload = format_answer(pipeline["plan"], payload)
    payload["rule_pipeline"] = pipeline
    return payload
