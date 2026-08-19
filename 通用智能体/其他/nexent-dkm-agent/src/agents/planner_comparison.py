"""Compare rule-based vs enhanced planner outputs across DKM tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.dkm_orchestrator import plan_dkm_workflow
from src.agents.kg_agent.planner import KGHybridPlanner
from src.common.integration import summarize_graph_for_planning
from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner
from src.operators.data_ops.csv_profile import profile_csv
from src.agents.data_processing_agent.planner import HybridPlanner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "samples" / "task1_patients.csv"


def _load_json_document(path: Path) -> dict[str, Any]:
    """Load JSON that may be prefixed with a module-style docstring comment."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith('"""'):
        end = stripped.find('"""', 3)
        if end != -1:
            text = stripped[end + 3 :].lstrip()
    elif stripped.startswith("'''"):
        end = stripped.find("'''", 3)
        if end != -1:
            text = stripped[end + 3 :].lstrip()
    return json.loads(text)

_MINIMAL_GRAPH = {
    "nodes": [
        {"id": "d1", "name": "高血压", "type": "Disease", "mention_count": 2},
        {"id": "s1", "name": "头晕", "type": "Symptom", "mention_count": 1},
        {"id": "s2", "name": "头痛", "type": "Symptom", "mention_count": 1},
        {"id": "dr1", "name": "氨氯地平", "type": "Drug", "mention_count": 1},
    ],
    "edges": [
        {"source": "d1", "target": "s1", "predicate": "has_symptom"},
        {"source": "d1", "target": "s2", "predicate": "has_symptom"},
        {"source": "d1", "target": "dr1", "predicate": "treated_by"},
    ],
}


def _load_graph_summary(graph_file: str | Path | None) -> dict[str, Any]:
    if graph_file:
        path = Path(graph_file)
        if path.is_file():
            graph = json.loads(path.read_text(encoding="utf-8"))
            return summarize_graph_for_planning(graph)
    return summarize_graph_for_planning(_MINIMAL_GRAPH)


def compare_task1_planners(
    task_request: str,
    input_path: str | Path | None = None,
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
) -> dict[str, Any]:
    """Return rule and hybrid task-1 plans for the same request."""

    csv_path = Path(input_path) if input_path else DEFAULT_CSV
    profile = profile_csv(csv_path)
    rule = HybridPlanner().plan(task_request, profile)
    hybrid = HybridPlanner(
        llm_config=llm_config,
        local_model_path=local_model_path,
    ).plan(task_request, profile)
    return {
        "task": "task1",
        "task_request": task_request,
        "input_path": str(csv_path),
        "rule_plan": rule.to_dict(),
        "hybrid_plan": hybrid.to_dict(),
        "operators_match": rule.operators == hybrid.operators,
        "hybrid_mode": hybrid.planner_mode,
    }


def compare_task2_planners(
    task_request: str,
    question: str | None = None,
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
) -> dict[str, Any]:
    """Return rule and hybrid task-2 plans for the same request."""

    rule = KGHybridPlanner().plan(task_request, question=question)
    hybrid = KGHybridPlanner(
        llm_config=llm_config,
        local_model_path=local_model_path,
    ).plan(task_request, question=question)
    return {
        "task": "task2",
        "task_request": task_request,
        "question": question,
        "rule_plan": rule.to_dict(),
        "hybrid_plan": hybrid.to_dict(),
        "operators_match": rule.operators == hybrid.operators,
        "hybrid_mode": hybrid.planner_mode,
    }


def compare_task3_planners(
    task_request: str,
    question: str | None = None,
    graph_file: str | Path | None = None,
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
) -> dict[str, Any]:
    """Return rule and hybrid task-3 plans for the same request."""

    graph_path = Path(graph_file) if graph_file else None
    graph_summary = _load_graph_summary(graph_path)
    rule = AnalysisHybridPlanner().plan(
        task_request,
        question=question,
        graph_summary=graph_summary,
    )
    hybrid = AnalysisHybridPlanner(
        llm_config=llm_config,
        local_model_path=local_model_path,
    ).plan(
        task_request,
        question=question,
        graph_summary=graph_summary,
    )
    return {
        "task": "task3",
        "task_request": task_request,
        "question": question,
        "graph_file": str(graph_path) if graph_path else None,
        "rule_plan": rule,
        "hybrid_plan": hybrid,
        "operators_match": rule["operators"] == hybrid["operators"],
        "hybrid_mode": hybrid.get("planner_mode", "rule"),
    }


def compare_dkm_orchestrator_planners(
    request: str,
    question: str | None = None,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return rule and optional LLM DKM stage plans."""

    rule_plan = plan_dkm_workflow(request, question=question, llm_config=None)
    comparison: dict[str, Any] = {
        "request": request,
        "question": question,
        "rule_plan": rule_plan,
        "enhanced_plan": None,
        "stages_match": None,
        "enhanced_mode": None,
    }
    if llm_config:
        enhanced_plan = plan_dkm_workflow(request, question=question, llm_config=llm_config)
        comparison["enhanced_plan"] = enhanced_plan
        comparison["stages_match"] = [
            stage["task"] for stage in rule_plan["stages"]
        ] == [stage["task"] for stage in enhanced_plan["stages"]]
        comparison["enhanced_mode"] = enhanced_plan.get("planner_mode")
    return comparison


def compare_all_planners(
    *,
    task1_request: str = "请清洗患者CSV，删除重复记录，填补空值，并统一字段类型后导出",
    task2_request: str = "构建医疗知识图谱并回答相关问题",
    task3_request: str = "分析图谱统计、关联与可视化洞察",
    question: str | None = "哪些疾病关联最多症状？",
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
) -> dict[str, Any]:
    """Bundle planner comparisons for all three tasks plus orchestrator."""

    return {
        "task1": compare_task1_planners(
            task1_request,
            llm_config=llm_config,
            local_model_path=local_model_path,
        ),
        "task2": compare_task2_planners(
            task2_request,
            question=question,
            llm_config=llm_config,
            local_model_path=local_model_path,
        ),
        "task3": compare_task3_planners(
            task3_request,
            question=question,
            llm_config=llm_config,
            local_model_path=local_model_path,
        ),
        "orchestrator": compare_dkm_orchestrator_planners(
            "请清洗医疗文本，构建知识图谱并生成分析洞察",
            question=question,
            llm_config=llm_config,
        ),
    }


def _operator_list(task_report: dict[str, Any], *, enhanced: bool) -> list[str]:
    if task_report["task"] == "task3":
        plan = task_report["hybrid_plan" if enhanced else "rule_plan"]
        return list(plan.get("operators", []))
    plan = task_report["hybrid_plan" if enhanced else "rule_plan"]
    return list(plan.get("operators", []))


def summarize_planner_diff(rule_report: dict[str, Any], enhanced_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize per-task operator differences between rule and enhanced planners."""

    summary: list[dict[str, Any]] = []
    for task_key in ("task1", "task2", "task3"):
        rule_task = rule_report[task_key]
        enhanced_task = enhanced_report[task_key]
        rule_ops = _operator_list(rule_task, enhanced=False)
        enhanced_ops = _operator_list(enhanced_task, enhanced=True)
        summary.append({
            "task": task_key,
            "rule_mode": rule_task.get("hybrid_mode") or rule_task["rule_plan"].get("planner_mode", "rule"),
            "enhanced_mode": enhanced_task.get("hybrid_mode")
            or enhanced_task["hybrid_plan"].get("planner_mode", "rule"),
            "rule_operator_count": len(rule_ops),
            "enhanced_operator_count": len(enhanced_ops),
            "operators_match": rule_ops == enhanced_ops,
            "rule_operators": rule_ops,
            "enhanced_operators": enhanced_ops,
        })

    rule_stages = [stage["task"] for stage in rule_report["orchestrator"]["rule_plan"]["stages"]]
    enhanced_stages = []
    enhanced_mode = None
    if enhanced_report["orchestrator"].get("enhanced_plan"):
        enhanced_stages = [
            stage["task"] for stage in enhanced_report["orchestrator"]["enhanced_plan"]["stages"]
        ]
        enhanced_mode = enhanced_report["orchestrator"].get("enhanced_mode")
    summary.append({
        "task": "orchestrator",
        "rule_mode": rule_report["orchestrator"]["rule_plan"].get("planner_mode", "rule"),
        "enhanced_mode": enhanced_mode,
        "rule_stages": rule_stages,
        "enhanced_stages": enhanced_stages,
        "stages_match": rule_stages == enhanced_stages if enhanced_stages else None,
    })
    return summary


def collect_planner_llm_evidence(
    *,
    llm_config: dict[str, Any] | None = None,
    snapshot_path: str | Path | None = None,
    local_model_path: str | None = None,
) -> dict[str, Any]:
    """Collect rule-only and LLM-enhanced planner evidence with optional snapshot fallback."""

    rule_report = compare_all_planners(local_model_path=local_model_path)
    result: dict[str, Any] = {
        "rule_only": rule_report,
        "llm_enhanced": None,
        "diff_summary": None,
        "collection_mode": "rule_only",
        "llm_available": False,
        "snapshot_used": False,
    }

    enhanced_report: dict[str, Any] | None = None
    if llm_config:
        try:
            enhanced_report = compare_all_planners(
                llm_config=llm_config,
                local_model_path=local_model_path,
            )
            result["llm_enhanced"] = enhanced_report
            result["collection_mode"] = "live_llm"
            result["llm_available"] = True
            result["llm_model"] = llm_config.get("model_name")
        except Exception as exc:
            result["llm_error"] = str(exc) or type(exc).__name__

    if enhanced_report is None and snapshot_path:
        path = Path(snapshot_path)
        if path.is_file():
            snapshot = _load_json_document(path)
            enhanced_report = snapshot.get("llm_enhanced") or snapshot.get("comparison")
            if enhanced_report:
                result["llm_enhanced"] = enhanced_report
                result["collection_mode"] = "rule_plus_recorded_snapshot"
                result["snapshot_used"] = True
                result["snapshot_metadata"] = snapshot.get("metadata", {})

    if enhanced_report is not None:
        result["diff_summary"] = summarize_planner_diff(rule_report, enhanced_report)

    return result
