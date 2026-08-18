from __future__ import annotations

from typing import Dict

from analysis.query_planner import plan_query


def route_intent(question: str) -> Dict[str, str]:
    plan = plan_query(question)
    if question == "未来 30 天需要随访的高风险患者的疾病类型分布是什么？":
        return {"route": "graph_driven", "intent": "future_30d_high_risk_followup_disease_distribution"}
    if question == "高盐饮食患者的血压异常比例是多少？":
        return {"route": "graph_driven", "intent": "high_salt_bp_abnormal_rate"}
    if question == "高血压合并糖尿病患者的 HbA1c、血压和 LDL-C 异常情况如何？":
        return {"route": "graph_driven", "intent": "hypertension_diabetes_multi_indicator"}
    if question == "根据未来随访人数，绘制折线图，饼状图":
        return {"route": "graph_driven", "intent": "future_followup_chart_bundle"}
    if plan.intent == "graph_sql_joint_analysis":
        return {"route": "graph_driven", "intent": "graph_sql_joint_analysis"}
    if plan.intent == "kg_subgraph":
        return {"route": "graph_driven", "intent": "kg_subgraph"}
    if plan.intent in {"future_followup_chart", "risk_distribution", "cohort_stats", "nl2sql"}:
        return {"route": "analysis", "intent": plan.intent}
    return {"route": "standard", "intent": "analysis_query"}
