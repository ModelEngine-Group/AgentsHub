from __future__ import annotations

from typing import Any, Dict

INTENT_TOOL_MAP = {
    "system_status": "chroniccare_health_check",
    "data_summary": "chroniccare_data_summary",
    "report_summary": "chroniccare_report_summary",
    "kg_summary": "chroniccare_kg_summary",
    "capability_examples": "chroniccare_open_sql_examples",
    "kg_patient_path_query": "chroniccare_kg_patient_path_query",
    "kg_relation_query": "chroniccare_kg_relation_query",
    "kg_entity_query": "chroniccare_kg_entity_query",
    "kg_subgraph_render": "chroniccare_kg_subgraph_render",
    "future_n_days_high_risk_followup": "chroniccare_followup_high_risk",
    "future_n_days_followup": "chroniccare_followup_high_risk",
    "cohort_disease_distribution": "chroniccare_cohort_disease_distribution",
    "risk_level_distribution": "chroniccare_risk_level_distribution",
    "disease_combination_distribution": "chroniccare_disease_combination_distribution",
    "disease_distribution": "chroniccare_disease_distribution",
    "indicator_analysis": "chroniccare_metric_query",
    "datamate_pipeline_run": "chroniccare_datamate_pipeline_run",
    "datamate_pipeline_run_npu": "chroniccare_datamate_pipeline_run_npu",
    "datamate_pipelines": "chroniccare_datamate_pipelines",
    "datamate_pipeline_status": "chroniccare_datamate_pipeline_status",
    "unsupported_negation_query": "chroniccare_open_analysis_query",
    "performance_query": "chroniccare_report_summary",
    "npu_readiness_query": "chroniccare_npu_readiness",
    "npu_supported_operators": "chroniccare_npu_supported_operators",
    "npu_operator_benchmark": "chroniccare_npu_operator_benchmark",
    "open_sql_analysis": "chroniccare_open_analysis_query",
}


DIRECT_EXECUTOR_INTENTS = {
    "system_status",
    "data_summary",
    "report_summary",
    "kg_summary",
    "capability_examples",
    "kg_patient_path_query",
    "kg_relation_query",
    "kg_entity_query",
    "kg_subgraph_render",
    "datamate_pipeline_run",
    "datamate_pipeline_run_npu",
    "datamate_pipelines",
    "datamate_pipeline_status",
    "npu_readiness_query",
    "npu_supported_operators",
    "npu_operator_benchmark",
    "unsupported_negation_query",
}


def build_query_plan(classification: Dict[str, Any], query: str) -> Dict[str, Any]:
    intent = str(classification.get("intent") or "open_sql_analysis")
    entities = classification.get("normalized_entities") or {}
    tool = INTENT_TOOL_MAP.get(intent, "chroniccare_open_analysis_query")
    if intent == "indicator_analysis" and any(token in query for token in ("趋势", "近 3 个月", "近3个月", "近 6 个月", "近6个月", "最近 3 个月", "最近3个月", "最近 6 个月", "最近6个月", "半年")):
        tool = "chroniccare_trend_query"
    return {
        "intent": intent,
        "tool": tool,
        "query": query,
        "normalized_entities": entities,
        "executor": "direct_tool" if intent in DIRECT_EXECUTOR_INTENTS else "legacy_open_analysis",
        "requires_chart": bool(entities.get("indicators") or any(token in query for token in ("图", "图表", "趋势", "饼图", "折线图"))),
        "reason": classification.get("reason"),
        "confidence": classification.get("confidence"),
    }
