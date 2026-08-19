from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_evolution_experience(report: dict[str, Any]) -> dict[str, Any]:
    """Convert one Task1 run report into an auditable self-evolution experience."""
    plan = report.get("plan") or {}
    profile = report.get("input_profile") or {}
    quality = report.get("quality") or {}
    quality_gate = report.get("quality_gate") or {}
    performance = report.get("performance") or {}
    anomalies = report.get("anomalies") or []
    steps = plan.get("steps") or []
    tools = [str(step.get("tool")) for step in steps if step.get("tool")]
    failed_rules = [
        str(rule.get("id"))
        for rule in quality_gate.get("rules", [])
        if rule.get("status") == "failed"
    ]
    warning_rules = [
        str(rule.get("id"))
        for rule in quality_gate.get("rules", [])
        if rule.get("status") == "warning"
    ]
    suggested_tools = suggest_planner_tools(tools, profile, quality_gate, anomalies)
    experience = {
        "schema_version": "task1-evolution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": report.get("task"),
        "outcome": {
            "quality_gate_status": quality_gate.get("status"),
            "passed": bool(quality_gate.get("passed")),
            "success_rate": quality.get("success_rate"),
            "warning_count": quality.get("warning_count"),
            "error_count": quality.get("error_count"),
            "records_per_second": performance.get("records_per_second"),
        },
        "profile": {
            "detected_mode": profile.get("detected_mode"),
            "record_count": profile.get("record_count"),
            "text_coverage": profile.get("text_coverage"),
            "structured_field_coverage": profile.get("structured_field_coverage"),
            "avg_text_length": profile.get("avg_text_length"),
        },
        "plan": {
            "runtime": plan.get("runtime"),
            "tools": tools,
            "scheduler_strategy": (plan.get("batch_scheduler") or {}).get("strategy"),
        },
        "diagnosis": {
            "failed_rules": failed_rules,
            "warning_rules": warning_rules,
            "anomaly_actions": sorted({str(item.get("action")) for item in anomalies if item.get("action")}),
            "primary_issue": primary_issue(failed_rules, warning_rules, anomalies),
        },
        "suggested_update": {
            "planner_label": suggested_tools,
            "optimization_goal": infer_optimization_goal(report),
            "reason": build_update_reason(failed_rules, warning_rules, anomalies, performance),
        },
        "reuse": {
            "skill_name": build_skill_name(profile, quality_gate),
            "trigger": build_trigger(profile, quality_gate),
        },
    }
    return experience


def build_workflow_skill(experience: dict[str, Any]) -> dict[str, Any]:
    """Create a reusable workflow-skill record from one evolution experience."""
    return {
        "name": experience["reuse"]["skill_name"],
        "trigger": experience["reuse"]["trigger"],
        "workflow": experience["suggested_update"]["planner_label"],
        "quality_gate_status": experience["outcome"]["quality_gate_status"],
        "reason": experience["suggested_update"]["reason"],
        "source_task": experience["task"],
    }


def build_planner_training_sample(experience: dict[str, Any]) -> dict[str, Any]:
    """Convert an experience into the same coarse format used by flow-planner data."""
    profile = experience.get("profile") or {}
    return {
        "task": canonical_task_from_experience(experience),
        "data_mode": profile.get("detected_mode") or "hybrid",
        "optimization_goal": experience["suggested_update"]["optimization_goal"],
        "profile": {
            "record_bucket": record_bucket(profile.get("record_count")),
            "text_coverage_bucket": coverage_bucket(profile.get("text_coverage")),
            "structured_coverage_bucket": coverage_bucket(profile.get("structured_field_coverage")),
            "noise_types": experience["diagnosis"].get("anomaly_actions", []),
            "missing_fields": [],
        },
        "labels": experience["suggested_update"]["planner_label"],
        "source": "self_evolution_experience",
        "source_reason": experience["suggested_update"]["reason"],
    }


def suggest_planner_tools(
    current_tools: list[str],
    profile: dict[str, Any],
    quality_gate: dict[str, Any],
    anomalies: list[dict[str, Any]],
) -> list[str]:
    mode = str(profile.get("detected_mode") or "hybrid")
    tools: list[str]
    if mode == "structured":
        tools = [
            "raw_record_profiler",
            "medical_table_profiler",
            "medical_table_normalizer",
            "medical_table_document_builder",
            "medical_text_cleaner",
            "medical_text_segmenter",
        ]
    else:
        tools = [
            "raw_record_profiler",
            "medical_record_normalizer",
            "medical_text_loader",
            "medical_text_cleaner",
            "medical_text_segmenter",
        ]

    rule_ids = {str(rule.get("id")) for rule in quality_gate.get("rules", []) if rule.get("status") != "passed"}
    anomaly_actions = {str(item.get("action")) for item in anomalies if item.get("action")}
    if rule_ids or anomaly_actions:
        insert_after_profiler(tools, "quality_anomaly_detector")
    if int(profile.get("record_count") or 0) >= 100 and "batch_scheduler" not in tools:
        insert_after_profiler(tools, "batch_scheduler")
    return [tool for tool in tools if tool]


def insert_after_profiler(tools: list[str], tool: str) -> None:
    if tool in tools:
        return
    try:
        index = tools.index("raw_record_profiler") + 1
    except ValueError:
        index = 0
    tools.insert(index, tool)


def primary_issue(failed_rules: list[str], warning_rules: list[str], anomalies: list[dict[str, Any]]) -> str:
    if failed_rules:
        return failed_rules[0]
    if warning_rules:
        return warning_rules[0]
    if anomalies:
        return str(anomalies[0].get("action") or "anomaly")
    return "none"


def infer_optimization_goal(report: dict[str, Any]) -> str:
    quality_gate = report.get("quality_gate") or {}
    scheduler = ((report.get("plan") or {}).get("batch_scheduler") or {})
    if quality_gate.get("status") in {"failed", "warning"}:
        return "quality_first"
    if scheduler.get("strategy") == "throughput_first":
        return "speed_first"
    return "balanced"


def build_update_reason(
    failed_rules: list[str],
    warning_rules: list[str],
    anomalies: list[dict[str, Any]],
    performance: dict[str, Any],
) -> str:
    if failed_rules:
        return f"Quality gate failed on {', '.join(failed_rules)}; prioritize validation and recovery operators."
    if warning_rules:
        return f"Quality gate warned on {', '.join(warning_rules)}; keep recovery-aware planning."
    if anomalies:
        actions = sorted({str(item.get("action")) for item in anomalies if item.get("action")})
        return f"Recovered anomalies: {', '.join(actions)}; keep anomaly-aware normalization."
    return f"Run passed quality gate at {performance.get('records_per_second')} records/s; preserve current workflow as reusable skill."


def build_skill_name(profile: dict[str, Any], quality_gate: dict[str, Any]) -> str:
    mode = str(profile.get("detected_mode") or "unknown")
    status = str(quality_gate.get("status") or "unknown")
    if status == "passed":
        return f"{mode}_stable_etl_workflow"
    return f"{mode}_{status}_recovery_workflow"


def build_trigger(profile: dict[str, Any], quality_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "detected_mode": profile.get("detected_mode"),
        "quality_gate_status": quality_gate.get("status"),
        "text_coverage_lte": 0.85 if float(profile.get("text_coverage") or 0) < 0.85 else None,
        "record_count_gte": 100 if int(profile.get("record_count") or 0) >= 100 else None,
    }


def canonical_task_from_experience(experience: dict[str, Any]) -> str:
    issue = experience.get("diagnosis", {}).get("primary_issue")
    mode = experience.get("profile", {}).get("detected_mode")
    if issue and issue != "none":
        return "缺字段数据修复和质量检查"
    if mode == "structured":
        return "处理结构化表格并映射字段"
    return "公开医疗数据 ETL"


def record_bucket(value: Any) -> str:
    number = int(value or 0)
    if number >= 1000:
        return "large"
    if number >= 100:
        return "medium"
    return "small"


def coverage_bucket(value: Any) -> str:
    number = float(value or 0)
    if number >= 0.8:
        return "high"
    if number >= 0.4:
        return "medium"
    return "low"
