from __future__ import annotations

import json
from typing import Any, Dict, List

from analysis.open_sql.open_sql_service import open_sql_query
from runtime_common.common import resolve_path
from tool_server.utils import load_server_config, safety_note

EVAL_CONFIG = "configs/open_sql_eval_questions.json"
REPORT_JSON = "outputs/evaluation/open_sql_eval_report.json"
REPORT_MD = "outputs/evaluation/open_sql_eval_report.md"


def _load_questions() -> List[Dict[str, Any]]:
    path = resolve_path(EVAL_CONFIG)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    expanded: List[Dict[str, Any]] = []
    for item in raw:
        variants = item.get("variants")
        if not variants:
            expanded.append(item)
            continue
        for idx, question in enumerate(variants, start=1):
            clone = {key: value for key, value in item.items() if key != "variants"}
            clone["id"] = f"{item.get('id')}_{idx:02d}"
            clone["question"] = question
            expanded.append(clone)
    return expanded


def run_open_sql_eval() -> Dict[str, Any]:
    cfg = load_server_config()
    questions = _load_questions()
    rows = []
    for item in questions:
        payload = open_sql_query(str(item.get("question") or ""), prefer_llm=True, allow_chart=False)
        expected_intent = item.get("expected_intent")
        expected_stage = item.get("expected_stage")
        ok_intent = expected_intent in {payload.get("intent"), payload.get("template_id")}
        ok_stage = expected_stage in {payload.get("stage"), "any"}
        executable = payload.get("status") == "success" and payload.get("sql_safe") is True and (payload.get("result") or {}).get("status") == "success"
        result_success = executable and (payload.get("result") or {}).get("row_count", 0) > 0
        rows.append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "expected_intent": expected_intent,
                "actual_intent": payload.get("intent"),
                "stage": payload.get("stage"),
                "intent_ok": ok_intent,
                "stage_ok": ok_stage,
                "sql_generated": bool(payload.get("sql")),
                "guard_pass": payload.get("sql_safe") is True,
                "executable": executable,
                "result_success": result_success,
                "answer_format_ok": bool(payload.get("answer_markdown")),
                "status": payload.get("status"),
            }
        )

    total = len(rows) or 1
    def rate(key: str) -> float:
        return round(sum(1 for row in rows if row.get(key)) / total, 4)

    template_rows = [row for row in rows if row.get("stage") == "template"]
    llm_rows = [row for row in rows if row.get("stage") == "llm_candidate"]
    fallback_rows = [row for row in rows if row.get("stage") == "fallback"]
    unsupported_rows = [row for row in rows if row.get("stage") == "unsupported"]
    report = {
        "status": "success",
        "total_questions": len(rows),
        "intent_accuracy": rate("intent_ok"),
        "schema_link_success_rate": round(sum(1 for row in rows if row.get("status") != "unsupported") / total, 4),
        "sql_generation_success_rate": rate("sql_generated"),
        "sql_guard_pass_rate": rate("guard_pass"),
        "sql_executable_rate": rate("executable"),
        "result_success_rate": rate("result_success"),
        "answer_format_pass_rate": rate("answer_format_ok"),
        "template_stage_success_rate": round(sum(1 for row in template_rows if row.get("result_success")) / (len(template_rows) or 1), 4),
        "llm_candidate_stage_success_rate": round(sum(1 for row in llm_rows if row.get("result_success")) / (len(llm_rows) or 1), 4),
        "fallback_count": len(fallback_rows),
        "unsupported_count": len(unsupported_rows),
        "llm_status": "llm_unavailable" if not llm_rows else "llm_used",
        "rows": rows,
        "safety_note": safety_note(cfg),
    }
    write_report(report)
    return report


def write_report(report: Dict[str, Any]) -> None:
    json_path = resolve_path(REPORT_JSON)
    md_path = resolve_path(REPORT_MD)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Open SQL Eval Report",
        "",
        f"- total_questions: `{report.get('total_questions')}`",
        f"- intent_accuracy: `{report.get('intent_accuracy')}`",
        f"- sql_executable_rate: `{report.get('sql_executable_rate')}`",
        f"- result_success_rate: `{report.get('result_success_rate')}`",
        f"- template_stage_success_rate: `{report.get('template_stage_success_rate')}`",
        f"- llm_candidate_stage_success_rate: `{report.get('llm_candidate_stage_success_rate')}`",
        f"- fallback_count: `{report.get('fallback_count')}`",
        f"- unsupported_count: `{report.get('unsupported_count')}`",
        f"- llm_status: `{report.get('llm_status')}`",
        "",
        report.get("safety_note", ""),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
