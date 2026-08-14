from __future__ import annotations

from typing import Any, Dict

from orchestration.question_classifier import classify_question
from orchestration.question_parser import build_query_plan


def route_intent(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    classification = classify_question({"query": query, "last_context": payload.get("last_context") or {}})
    plan = build_query_plan(classification, query)
    return {
        "intent": plan["intent"],
        "tool": plan["tool"],
        "normalized_entities": plan["normalized_entities"],
        "confidence": classification.get("confidence", 0.75),
        "reason": classification.get("reason", ""),
        "executor": plan.get("executor"),
    }
