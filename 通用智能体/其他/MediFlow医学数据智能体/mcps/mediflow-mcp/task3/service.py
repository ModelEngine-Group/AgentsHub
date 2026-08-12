"""任务三端到端分析服务。"""

from __future__ import annotations

import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

from core.llm_client import LLMClient

from .chart_policy import build_chart
from .contracts import AnalysisPlan
from .planner import build_plan
from .sql_safety import execute_readonly


def _result_sentence(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"**{title}**：当前分析库未返回匹配记录。"
    if len(rows) == 1 and len(rows[0]) == 1:
        key, value = next(iter(rows[0].items()))
        return f"**{title}**：{key}为 **{value}**。"
    columns = list(rows[0])
    label = columns[0]
    numeric = next(
        (key for key in columns[1:] if isinstance(rows[0].get(key), (int, float))),
        None,
    )
    if numeric:
        items = [f"{row.get(label)}（{row.get(numeric)}）" for row in rows[:8]]
    else:
        items = [str(row.get(label)) for row in rows[:10]]
    return f"**{title}**：" + "、".join(items) + ("。" if items else "")


def _build_answer(
    plan: AnalysisPlan,
    analyses: list[dict[str, Any]],
    analysis_scope: dict[str, Any],
) -> str:
    parts: list[str] = []
    if plan.unsupported:
        parts.append(
            "### 数据口径说明\n"
            + "\n".join(f"- {item}" for item in plan.unsupported)
        )
    if analyses:
        parts.append(
            "### 分析结果\n"
            + "\n\n".join(
                _result_sentence(item["title"], item.get("rows", []))
                if item.get("status") == "ok"
                else f"**{item['title']}**：查询未完成，{item.get('error', '未知错误')}。"
                for item in analyses
            )
        )
    if not parts:
        parts.append(
            "当前问题无法映射到分析库中的可用指标。"
            "请明确疾病、症状、药物、检查、科室、并发症或统计维度。"
        )
    scope_statement = str(analysis_scope.get("statement") or "").strip()
    if scope_statement:
        parts.append(f"### 数据来源范围\n{scope_statement}")
    parts.append(
        "### 证据说明\n"
        "以上内容仅来自本轮只读 SQL 的实际返回结果；"
        "图表、证据表与导出报告使用同一分析记录。"
    )
    return "\n\n".join(parts)


class MedicalAnalysisService:
    """把自然语言问题转换为计划、SQL、证据、图表和可导出结果。"""

    def __init__(
        self,
        db_path: str | Path,
        llm: LLMClient | None = None,
        scope_provider: Callable[[], dict[str, Any]] | None = None,
    ):
        self.db_path = Path(db_path)
        self.llm = llm
        self.scope_provider = scope_provider

    def analyze(self, question: str) -> dict[str, Any]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question is required")
        if not self.db_path.exists():
            raise FileNotFoundError(f"分析库不存在：{self.db_path}")

        started = time.perf_counter()
        with sqlite3.connect(
            f"file:{self.db_path.resolve().as_posix()}?mode=ro",
            uri=True,
        ) as conn:
            plan = build_plan(conn, question, self.llm)

        analyses: list[dict[str, Any]] = []
        for index, query in enumerate(plan.queries[:4], start=1):
            item: dict[str, Any] = {
                "index": index,
                "title": query.title,
                "purpose": query.purpose,
                "sql": query.sql,
                "source": query.source,
                "status": "ok",
            }
            try:
                execution = execute_readonly(
                    self.db_path,
                    query.sql,
                    query.params,
                )
                item.update(execution)
                item["chart"] = build_chart(question, query, execution["rows"])
            except Exception as exc:
                item.update(
                    {
                        "status": "error",
                        "error": "查询执行失败，请检查分析口径或数据结构。",
                        "error_type": type(exc).__name__,
                        "columns": [],
                        "rows": [],
                        "row_count": 0,
                        "chart": None,
                    }
                )
            analyses.append(item)

        executed = [item for item in analyses if item["status"] == "ok"]
        evidence = [item for item in executed if int(item.get("row_count") or 0) > 0]
        errors = [item for item in analyses if item["status"] == "error"]
        first = evidence[0] if evidence else (executed[0] if executed else {})
        charts = [item["chart"] for item in evidence if item.get("chart")]
        total_rows = sum(int(item.get("row_count") or 0) for item in evidence)
        if evidence and (errors or len(evidence) < len(analyses)):
            result_status = "partial"
        elif evidence:
            result_status = "success"
        elif errors and len(errors) == len(analyses):
            result_status = "error"
        else:
            result_status = "no_evidence"
        analysis_scope = (
            self.scope_provider()
            if self.scope_provider is not None
            else {
                "mode": "analysis_snapshot",
                "status": "unavailable",
                "statement": "本轮查询基于当前分析库快照，未提供来源清单。",
            }
        )
        analysis_id = str(uuid.uuid4())
        return {
            "analysis_id": analysis_id,
            "status": result_status,
            "question": question,
            "subject": plan.subject,
            "answer": _build_answer(plan, analyses, analysis_scope),
            "plan": plan.to_dict(),
            "analyses": analyses,
            "unsupported": plan.unsupported,
            "planner": plan.planner,
            "planner_status": plan.planner_status,
            "planner_note": plan.planner_note,
            "analysis_scope": analysis_scope,
            "columns": first.get("columns", []),
            "rows": first.get("rows", []),
            "row_count": total_rows,
            "chart": charts[0] if charts else None,
            "charts": charts,
            "disease": plan.subject,
            "template": f"agentic_analysis+{plan.planner}",
            "steps": [
                {
                    "name": "理解问题并生成计划",
                    "status": "done" if plan.planner_status == "ready" else "warn",
                    "detail": (
                        f"{plan.planner}；{len(plan.queries)} 项只读分析；"
                        f"{plan.planner_note}"
                    ),
                },
                {
                    "name": "生成并校验 NL2SQL",
                    "status": "done" if plan.queries else "warn",
                    "detail": f"{len(plan.queries)} 条 SQL 通过只读校验",
                },
                {
                    "name": "执行查询并绑定证据",
                    "status": "done" if evidence else "warn",
                    "detail": f"{len(executed)}/{len(analyses)} 项执行完成，返回 {total_rows} 行证据",
                },
                {
                    "name": "生成图表与报告数据",
                    "status": "done",
                    "detail": f"{len(charts)} 个图表；报告可导出",
                },
            ],
            "provenance": {
                "analysis_id": analysis_id,
                "database": self.db_path.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "query_count": len(plan.queries),
                "successful_query_count": len(executed),
                "executed_query_count": len(executed),
                "evidence_query_count": len(evidence),
                "failed_query_count": len(errors),
                "row_count": total_rows,
            },
        }
