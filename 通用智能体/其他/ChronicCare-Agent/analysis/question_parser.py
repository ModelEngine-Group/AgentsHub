from __future__ import annotations

from typing import Any, Dict, List, Tuple

from analysis.demo_questions import default_question_struct, supported_intents


def parse_question_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    intent = entry.get("intent")
    parsed = {
        "id": entry.get("id"),
        "question": entry.get("question"),
        "intent": intent,
        "expected_chart_type": entry.get("expected_chart_type"),
        "sql_template": entry.get("sql_template"),
    }
    defaults = default_question_struct(intent or "")
    parsed.update(defaults)
    if entry.get("sql_template"):
        parsed["status"] = "success"
    elif intent not in supported_intents():
        parsed["status"] = "failed"
        errors.append(f"Unsupported intent: {intent}")
    else:
        parsed["status"] = "success"
    return parsed, errors


def parse_questions(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    parsed_items: List[Dict[str, Any]] = []
    errors: List[str] = []
    for entry in entries:
        parsed, item_errors = parse_question_entry(entry)
        parsed_items.append(parsed)
        errors.extend(item_errors)
    return parsed_items, errors
