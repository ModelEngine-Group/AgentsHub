from __future__ import annotations

from typing import Any, Dict

from analysis.query_planner import plan_query


def build_sql_candidate(question: str, route: Dict[str, str], schema_links: Dict[str, Any]) -> Dict[str, Any]:
    plan = plan_query(question)
    if route["intent"] == "future_30d_high_risk_followup_disease_distribution":
        return {
            "sql": "SELECT DISTINCT p.patient_id, p.disease_tags FROM patient_profile p JOIN followup_plan f ON p.patient_id=f.patient_id WHERE f.priority='high' AND f.status IN ('pending','scheduled') AND date(f.followup_date) BETWEEN date('now') AND date('now','+30 day');",
            "executable": True,
        }
    if route["intent"] == "high_salt_bp_abnormal_rate":
        return {
            "sql": "SELECT ROUND(SUM(CASE WHEN l.abnormal_flag!='normal' THEN 1 ELSE 0 END)*1.0/COUNT(*), 4) AS abnormal_rate FROM lifestyle_record s JOIN lab_result l ON s.patient_id=l.patient_id AND s.visit_id=l.visit_id WHERE s.salt_intake_level='high' AND l.item_name IN ('systolic_bp','diastolic_bp');",
            "executable": True,
        }
    if route["intent"] == "hypertension_diabetes_multi_indicator":
        return {
            "sql": "SELECT ... FROM patient_profile p JOIN lab_result l ON p.patient_id=l.patient_id WHERE lower(p.disease_tags) LIKE '%hypertension%' AND lower(p.disease_tags) LIKE '%diabetes%';",
            "executable": True,
        }
    if route["intent"] == "future_followup_chart_bundle":
        return {
            "sql": "SELECT followup_date, COUNT(DISTINCT patient_id) AS patient_count FROM followup_plan WHERE status IN ('pending','scheduled') AND date(followup_date) BETWEEN date('now') AND date('now','+30 day') GROUP BY followup_date ORDER BY followup_date;",
            "executable": True,
        }
    if plan.intent == "risk_distribution":
        return {
            "sql": "SELECT prs.risk_level, COUNT(DISTINCT prs.patient_id) AS patient_count FROM patient_risk_score prs GROUP BY prs.risk_level ORDER BY patient_count DESC;",
            "executable": True,
        }
    if plan.intent == "cohort_stats" and "hypertension" in plan.disease_filters and "diabetes" in plan.disease_filters:
        return {
            "sql": "SELECT COUNT(DISTINCT p.patient_id) AS patient_count FROM patient_profile p WHERE lower(p.disease_tags) LIKE '%hypertension%' AND lower(p.disease_tags) LIKE '%diabetes%';",
            "executable": True,
        }
    if plan.intent == "cohort_stats" and "hypertension" in plan.disease_filters:
        return {
            "sql": "SELECT COUNT(DISTINCT p.patient_id) AS patient_count FROM patient_profile p WHERE lower(p.disease_tags) LIKE '%hypertension%';",
            "executable": True,
        }
    if plan.intent == "cohort_stats" and "diabetes" in plan.disease_filters:
        return {
            "sql": "SELECT COUNT(DISTINCT p.patient_id) AS patient_count FROM patient_profile p WHERE lower(p.disease_tags) LIKE '%diabetes%';",
            "executable": True,
        }
    if plan.intent in {"future_followup_chart", "nl2sql"} and plan.time_window is not None and "随访" in question:
        return {
            "sql": f"SELECT date(fp.followup_date) AS followup_date, COUNT(DISTINCT fp.patient_id) AS patient_count FROM followup_plan fp WHERE fp.status IN ('pending','scheduled') AND date(fp.followup_date) BETWEEN date('now') AND date('now','+{plan.time_window.value} day') GROUP BY date(fp.followup_date) ORDER BY followup_date;",
            "executable": True,
        }
    return {
        "sql": None,
        "executable": False,
        "planner_intent": plan.intent,
        "schema_links": schema_links,
        "reason": "delegate_to_standard_analysis_or_fallback",
    }
