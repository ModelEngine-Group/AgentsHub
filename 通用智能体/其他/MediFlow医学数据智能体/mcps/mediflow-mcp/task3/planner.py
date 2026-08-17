"""任务三分析计划生成器。"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from core.llm_client import LLMClient
from core.nl2sql import deterministic_sql

from .contracts import AnalysisPlan, AnalysisQuery
from .semantic_layer import (
    PATIENT_COUNT_RE,
    allowed_schema_prompt,
    detect_fact_specs,
    needs_llm_planner,
    semantic_plan,
)
from .sql_safety import SqlSafetyError, validate_readonly_sql


PLANNER_SYSTEM_PROMPT = f"""
你是医疗数据分析智能体的规划器。把中文问题拆成 1 至 4 项可执行分析，并生成 SQLite 只读 SQL。

{allowed_schema_prompt()}

严格要求：
1. 仅输出 JSON，不输出 Markdown。
2. JSON 结构：
{{
  "subject": "主要疾病或空字符串",
  "queries": [
    {{
      "title": "面向用户的分析标题",
      "purpose": "该查询回答什么",
      "sql": "单条 SELECT/WITH SQL",
      "chart_type": "auto|bar|column|donut|line|table|metric"
    }}
  ],
  "unsupported": ["数据不足时的客观说明"]
}}
3. 不得虚构患者表、病例表、处方表或时间字段。
4. 用户询问患者人数、患病率、发病率时，必须写入 unsupported，不能把疾病知识条目数冒充患者人数。
5. 复合问题必须拆分；每项 SQL 都要直接对应一个子问题。
6. “导出报告”是输出动作，不生成数据库查询。
7. 所有结论必须可以从 SQL 返回结果得到。
8. SQL 只允许使用上面列出的表、视图和字段。
9. 占比或构成问题优先 donut，排名优先 bar，时间趋势优先 line，
   少量分类对比优先 column；明细使用 table，单值使用 metric。
""".strip()


def _parse_llm_plan(question: str, payload: Any) -> AnalysisPlan | None:
    if not isinstance(payload, dict):
        return None
    plan = AnalysisPlan(
        question=question,
        subject=str(payload.get("subject") or "").strip() or None,
        unsupported=[
            str(item).strip()
            for item in payload.get("unsupported", [])
            if str(item).strip()
        ],
        planner="llm_nl2sql",
        planner_status="ready",
        planner_note="复杂问题已由模型生成受限查询计划。",
    )
    for item in payload.get("queries", [])[:4]:
        if not isinstance(item, dict):
            continue
        try:
            sql = validate_readonly_sql(str(item.get("sql") or ""))
        except SqlSafetyError:
            continue
        plan.queries.append(
            AnalysisQuery(
                title=str(item.get("title") or "数据分析").strip(),
                purpose=str(item.get("purpose") or "回答用户问题").strip(),
                sql=sql,
                chart_type=str(item.get("chart_type") or "auto").strip(),
                source="llm_nl2sql",
            )
        )
    return plan


def _query_signature(query: AnalysisQuery) -> str:
    return re.sub(r"\s+", " ", query.sql.strip().lower())


def _source_signature(query: AnalysisQuery) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(
                re.findall(
                    r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
                    query.sql,
                    flags=re.IGNORECASE,
                )
            )
        )
    )


def _merge_plans(
    question: str,
    semantic: AnalysisPlan,
    generated: AnalysisPlan | None,
) -> AnalysisPlan:
    if generated is None:
        return semantic
    merged = AnalysisPlan(
        question=question,
        subject=semantic.subject or generated.subject,
        planner="hybrid_semantic_nl2sql",
        planner_status=generated.planner_status,
        planner_note=generated.planner_note,
    )
    merged.unsupported.extend(semantic.unsupported)
    for item in generated.unsupported:
        if item not in merged.unsupported:
            merged.unsupported.append(item)

    if PATIENT_COUNT_RE.search(question):
        # 当前分析库没有患者粒度数据。复合问题仍由语义层回答可支持的
        # 疾病知识部分，但不执行模型生成的患者统计查询。
        generated.queries = []

    seen: set[str] = set()
    seen_sources: set[tuple[str, ...]] = set()
    for query in generated.queries + semantic.queries:
        signature = _query_signature(query)
        sources = _source_signature(query)
        if signature in seen or (sources and sources in seen_sources):
            continue
        seen.add(signature)
        if sources:
            seen_sources.add(sources)
        merged.queries.append(query)

    expected_fact_count = len(detect_fact_specs(question))
    if expected_fact_count and len(merged.queries) < expected_fact_count:
        for query in semantic.queries:
            signature = _query_signature(query)
            sources = _source_signature(query)
            if signature not in seen and (not sources or sources not in seen_sources):
                seen.add(signature)
                if sources:
                    seen_sources.add(sources)
                merged.queries.append(query)
    return merged


def build_plan(
    conn: sqlite3.Connection,
    question: str,
    llm: LLMClient | None,
) -> AnalysisPlan:
    """生成兼顾稳定性与开放式 NL2SQL 的混合分析计划。"""

    compiled_sql = deterministic_sql(question)
    if compiled_sql:
        return AnalysisPlan(
            question=question,
            planner="deterministic_nl2sql",
            planner_status="ready",
            planner_note="当前问题已由稳定业务语义模板生成只读查询。",
            queries=[
                AnalysisQuery(
                    title="医学数据查询",
                    purpose="回答用户提出的结构化医学数据问题",
                    sql=validate_readonly_sql(compiled_sql),
                    chart_type="auto",
                    source="deterministic_nl2sql",
                )
            ],
        )

    semantic = semantic_plan(conn, question)
    requires_llm = needs_llm_planner(question, semantic)
    if not requires_llm:
        semantic.planner_status = "ready"
        semantic.planner_note = "当前问题已由确定性语义规则生成查询计划。"
        return semantic
    if llm is None:
        semantic.planner_status = "degraded"
        semantic.planner_note = "复杂规划服务当前未配置，已保留可验证的语义层结果。"
        return semantic
    try:
        payload = llm.chat_json(
            f"用户问题：{question}\n请生成分析计划。",
            system=PLANNER_SYSTEM_PROMPT,
        )
        generated = _parse_llm_plan(question, payload)
    except Exception as exc:
        semantic.planner_status = "degraded"
        semantic.planner_note = (
            "复杂规划服务调用失败，已保留可验证的语义层结果；"
            f"故障类型：{type(exc).__name__}。"
        )
        return semantic
    if generated is None:
        semantic.planner_status = "degraded"
        semantic.planner_note = "复杂规划服务未返回有效计划，已保留可验证的语义层结果。"
        return semantic
    return _merge_plans(question, semantic, generated)
