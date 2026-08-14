from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from analysis.open_sql.llm_sql_candidate import generate_llm_sql_candidate, llm_available
from analysis.open_sql.nl_security import classify_nl_security
from analysis.open_sql.question_rewriter import rewrite_question
from analysis.open_sql.result_formatter import format_result
from analysis.open_sql.schema_catalog import get_schema_catalog
from analysis.open_sql.schema_linker import build_schema_links
from analysis.open_sql.sql_executor import execute_sql
from analysis.open_sql.sql_guard import validate_sql
from analysis.open_sql.sql_template_builder import build_template_sql
from runtime_common.analysis_context import AnalysisContext, attach_analysis_context
from runtime_common.common import resolve_path
from tool_server.utils import load_server_config, safety_note

TRACE_DIR = "outputs/open_sql/traces"

EXAMPLE_QUESTIONS = [
    "高血压患者平均 HbA1c 是多少？",
    "高血压和糖尿病都有的人，糖化平均是多少？",
    "糖尿病患者空腹血糖平均值是多少？",
    "高脂血症患者 LDL-C 异常率是多少？",
    "肥胖患者 BMI 平均值是多少？",
    "高尿酸患者尿酸异常率是多少？",
    "慢性肾病风险患者 eGFR 异常率是多少？",
    "不同风险等级的 HbA1c 平均值是多少？",
    "不同风险等级的血压异常比例是多少？",
    "最近 6 个月 HbA1c 异常趋势如何？",
    "最近 3 个月血压异常趋势如何？",
    "未来 45 天高风险随访人数是多少？",
    "未来 60 天糖尿病患者随访人数是多少？",
    "高盐饮食患者血压异常率是多少？",
    "运动不足患者 BMI 超标比例是多少？",
    "用降压药患者血压控制情况如何？",
    "用降糖药患者 HbA1c 控制情况如何？",
    "高血压合并高脂血症患者 LDL-C 异常比例是多少？",
    "高风险糖尿病患者最近半年 HbA1c 趋势如何？",
    "同时有高血压、糖尿病和高脂血症的人平均 BMI 是多少？",
]

FIXED_TOOL_HINTS = (
    "疾病分布",
    "风险等级分布",
    "图谱",
    "知识图谱",
    "子图",
    "datamate",
    "pipeline",
    "npu",
    "系统状态",
    "图谱规模",
)


def _trace_id(question: str) -> str:
    safe = abs(hash((question, time.time())))
    return f"open_sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe % 1000000:06d}"


def _write_trace(trace: Dict[str, Any]) -> str:
    out_dir = resolve_path(TRACE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{trace['trace_id']}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"{TRACE_DIR}/{path.name}"


def _unsupported(
    question: str,
    reason: str,
    query_spec: Dict[str, Any] | None = None,
    llm_candidate: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = load_server_config()
    context = AnalysisContext.from_mapping((query_spec or {}).get("analysis_context"))
    trace = {
        "trace_id": _trace_id(question),
        "question": question,
        "stage": "unsupported",
        "query_spec": query_spec or {},
        "schema_link": {},
        "sql": None,
        "sql_guard": {"safe": False, "reason": reason},
        "execution": {},
        "answer_summary": reason,
        "chart_url": None,
        "llm_candidate": llm_candidate,
        "error": reason,
    }
    trace_path = _write_trace(trace)
    return attach_analysis_context({
        "status": "unsupported",
        "question": question,
        "intent": (query_spec or {}).get("intent"),
        "stage": "unsupported",
        "sql_safe": False,
        "sql": None,
        "result": {},
        "summary_text": reason,
        "answer_markdown": reason,
        "chart_url": None,
        "trace_id": trace["trace_id"],
        "trace_path": trace_path,
        "llm_status": "available" if llm_available() else "llm_unavailable",
        "llm_candidate": llm_candidate,
        "safety_note": safety_note(cfg),
    }, context)


def should_defer_to_fixed_tool(question: str) -> bool:
    text = str(question or "").lower()
    if "未来" in text and "随访" in text and "高风险" in text and not any(token in text for token in ("糖尿病", "高血压", "合并", "指标", "hba1c", "ldl", "血糖")):
        return True
    return any(token.lower() in text for token in FIXED_TOOL_HINTS)


def open_sql_query(
    question: str,
    prefer_llm: bool = True,
    allow_chart: bool = True,
    force_llm: bool = False,
    last_context: Dict[str, Any] | None = None,
    analysis_context: Dict[str, Any] | AnalysisContext | None = None,
    allow_fixed_tool_overlap: bool = False,
) -> Dict[str, Any]:
    cfg = load_server_config()
    context = analysis_context if isinstance(analysis_context, AnalysisContext) else AnalysisContext.from_mapping(analysis_context)
    question = " ".join(str(question or "").strip().split())
    if not question:
        return _unsupported(question, "问题为空，无法生成 SQL。", {"analysis_context": context.to_dict()})
    nl_security = classify_nl_security(question)
    if not nl_security["safe"]:
        return _unsupported(question, nl_security["reason"], {"analysis_context": context.to_dict(), "nl_security": nl_security})
    if should_defer_to_fixed_tool(question) and not allow_fixed_tool_overlap:
        return _unsupported(question, "该问题属于已有固定工具优先范围，请使用对应专用工具。", {"analysis_context": context.to_dict()})

    catalog = get_schema_catalog()
    query_spec = rewrite_question(question, last_context=last_context)
    if (query_spec.get("time_range") or {}).get("type") == "future_days":
        context = context.with_window(int((query_spec.get("time_range") or {}).get("value") or 1))
    query_spec["analysis_context"] = context.to_dict()
    if query_spec.get("needs_context"):
        return _unsupported(question, "问题包含“他们/这些患者”等上下文指代，但当前没有可用群体上下文。", query_spec)
    if query_spec.get("intent") == "unsupported":
        return _unsupported(question, "未识别到可支持的慢病统计意图。", query_spec)

    schema_link = build_schema_links(query_spec, catalog)
    if schema_link.get("status") != "success":
        return _unsupported(question, "schema linking 失败：" + "；".join(schema_link.get("errors") or []), query_spec)

    template = build_template_sql(query_spec, schema_link)
    stage = "template"
    sql = template.get("sql")
    llm_candidate: Dict[str, Any] | None = None
    force_llm = force_llm or str(os.getenv("OPEN_SQL_FORCE_LLM", "false")).lower() in {"1", "true", "yes"}
    if not sql and prefer_llm:
        llm_candidate = generate_llm_sql_candidate(question, schema_link, catalog)
        if llm_candidate.get("status") == "success":
            stage = "llm_candidate"
            sql = llm_candidate.get("sql")
            template = {"template_id": None, "explanation": "LLM SQL candidate 通过 Guard 后执行。"}
        else:
            stage = "fallback"
    elif prefer_llm and force_llm:
        llm_candidate = generate_llm_sql_candidate(question, schema_link, catalog)
        if llm_candidate.get("status") == "success":
            stage = "llm_candidate"
            sql = llm_candidate.get("sql")
            template = {"template_id": None, "explanation": "Forced LLM SQL candidate; candidate has no execution authority and must pass Guard."}
        else:
            return _unsupported(
                question,
                "强制 LLM Candidate 未成功，未以模板结果冒充模型结果。",
                query_spec,
                llm_candidate,
            )
    if not sql:
        detail = ""
        if llm_candidate:
            detail = f" LLM 状态：{llm_candidate.get('status')}；原因：{llm_candidate.get('reason') or '未返回原因'}。"
        return _unsupported(question, "阶段 1 模板未覆盖，阶段 2 LLM 不可用或未生成安全候选。" + detail, query_spec, llm_candidate)

    guard = validate_sql(sql, catalog)
    if not guard.get("safe"):
        if stage == "llm_candidate":
            fallback_template = build_template_sql(query_spec, schema_link)
            fallback_sql = fallback_template.get("sql")
            fallback_guard = validate_sql(fallback_sql, catalog) if fallback_sql else {"safe": False, "reason": "no fallback template"}
            if fallback_sql and fallback_guard.get("safe"):
                stage = "fallback"
                sql = fallback_sql
                template = fallback_template
                guard = fallback_guard
            else:
                return _unsupported(question, "SQL Guard 未通过：" + str(guard.get("reason")), query_spec)
        else:
            return _unsupported(question, "SQL Guard 未通过：" + str(guard.get("reason")), query_spec)

    normalized_sql = guard["normalized_sql"]
    execution = execute_sql(normalized_sql)
    formatted = format_result(
        question=question,
        query_spec=query_spec,
        template=template,
        execution=execution,
        allow_chart=allow_chart,
    )
    status = "success" if execution.get("status") == "success" else "failed"
    trace = {
        "trace_id": _trace_id(question),
        "question": question,
        "stage": stage,
        "query_spec": query_spec,
        "schema_link": schema_link,
        "sql": normalized_sql,
        "sql_guard": guard,
        "execution": {k: v for k, v in execution.items() if k != "rows"},
        "answer_summary": formatted.get("summary_text"),
        "chart_url": formatted.get("chart_url"),
        "image_url": formatted.get("image_url"),
        "llm_candidate": llm_candidate,
        "error": execution.get("error"),
    }
    trace_path = _write_trace(trace)
    return attach_analysis_context({
        "status": status,
        "question": question,
        "intent": query_spec.get("intent"),
        "stage": stage,
        "sql_safe": bool(guard.get("safe")),
        "sql": normalized_sql,
        "sql_guard": guard,
        "query_spec": query_spec,
        "schema_link": schema_link,
        "result": execution,
        "table": {"rows": (execution.get("rows") or [])[:50], "row_count": execution.get("row_count", 0)},
        "summary_text": formatted.get("summary_text"),
        "answer_markdown": formatted.get("answer_markdown"),
        "chart_url": formatted.get("chart_url"),
        "image_url": formatted.get("image_url"),
        "image_service_url": formatted.get("image_service_url"),
        "charts": formatted.get("charts") or [],
        "trend_rows": formatted.get("trend_rows") or [],
        "trace_id": trace["trace_id"],
        "trace_path": trace_path,
        "template_id": template.get("template_id"),
        "llm_status": "available" if llm_available() else "llm_unavailable",
        "force_llm": force_llm,
        "safety_note": safety_note(cfg),
    }, context)


def get_open_sql_schema() -> Dict[str, Any]:
    cfg = load_server_config()
    catalog = get_schema_catalog()
    catalog["safety_note"] = safety_note(cfg)
    return catalog


def get_open_sql_examples() -> Dict[str, Any]:
    cfg = load_server_config()
    return {
        "status": "success",
        "example_count": len(EXAMPLE_QUESTIONS),
        "examples": EXAMPLE_QUESTIONS,
        "supported_intents": [
            "count",
            "avg",
            "abnormal_rate",
            "distribution",
            "trend",
            "followup_count",
            "cohort_metric",
        ],
        "llm_status": "available" if llm_available() else "llm_unavailable",
        "safety_note": safety_note(cfg),
    }


def recent_open_sql_traces(limit: int = 10) -> Dict[str, Any]:
    cfg = load_server_config()
    trace_dir = resolve_path(TRACE_DIR)
    traces: List[Dict[str, Any]] = []
    if trace_dir.exists():
        for path in sorted(trace_dir.glob("open_sql_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
            try:
                traces.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    return {"status": "success", "trace_count": len(traces), "traces": traces, "safety_note": safety_note(cfg)}


def open_sql_eval() -> Dict[str, Any]:
    from analysis.open_sql.evaluator import run_open_sql_eval

    return run_open_sql_eval()
