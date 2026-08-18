from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_sql_candidates(parsed_items: List[Dict[str, Any]], template_map: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    items: List[Dict[str, Any]] = []
    errors: List[str] = []
    for item in parsed_items:
        candidate = {
            "id": item["id"],
            "question": item["question"],
            "intent": item["intent"],
            "expected_chart_type": item.get("expected_chart_type"),
        }
        sql = item.get("sql_template") or template_map.get(item["intent"])
        if item.get("status") != "success":
            candidate.update({"sql": None, "status": "failed", "error": "question_parse_failed"})
            errors.append(f"{item['id']}: question parse failed")
        elif not sql:
            candidate.update({"sql": None, "status": "failed", "error": "template_not_found"})
            errors.append(f"{item['id']}: SQL template not found for intent {item['intent']}")
        else:
            candidate.update({"sql": sql.strip(), "status": "success"})
        items.append(candidate)
    return items, errors
