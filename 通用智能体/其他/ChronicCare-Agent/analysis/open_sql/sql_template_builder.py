from __future__ import annotations

from typing import Any, Dict, List

from analysis.open_sql.alias_registry import INDICATOR_LABELS
from runtime_common.analysis_context import AnalysisContext

LATEST_RISK_CTE = (
    "WITH latest_risk AS ("
    "SELECT patient_id, risk_level FROM ("
    "SELECT patient_id, risk_level, created_at, visit_id, "
    "ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY date(created_at) DESC, visit_id DESC) AS rn "
    "FROM patient_risk_score"
    ") WHERE rn = 1"
    ") "
)


def _disease_conditions(diseases: List[str], alias: str = "p") -> List[str]:
    return [f"lower({alias}.disease_tags) LIKE '%{disease.lower()}%'" for disease in diseases]


def _indicator_filter(items: List[str], alias: str = "l") -> str:
    quoted = ", ".join(f"'{item}'" for item in items)
    return f"lower({alias}.item_name) IN ({quoted})"


def _abnormal_count(alias: str = "l") -> str:
    return f"COUNT(DISTINCT CASE WHEN lower({alias}.abnormal_flag) != 'normal' THEN {alias}.lab_id END)"


def _time_filter(field: str, time_range: Dict[str, Any] | None, as_of_date: str) -> str:
    if not time_range:
        return ""
    value = int(time_range.get("value") or 0)
    if time_range.get("type") == "past_months":
        # "最近 N 个月"按自然月计数并包含数据中的最新月份。直接减 N
        # 个月会同时覆盖起止月份，实际返回 N+1 个月。
        month_offset = max(value - 1, 0)
        return (
            f" AND date({field}) >= "
            f"date((SELECT max(date(test_date)) FROM lab_result), 'start of month', '-{month_offset} months')"
        )
    if time_range.get("type") == "future_days":
        offset_days = max(value - 1, 0)
        return f" AND date({field}) BETWEEN date('{as_of_date}') AND date('{as_of_date}', '+{offset_days} day')"
    return ""


def _primary_indicator(query_spec: Dict[str, Any], schema_link: Dict[str, Any]) -> tuple[str, List[str]]:
    indicators = query_spec.get("indicators") or []
    indicator = indicators[0] if indicators else "hba1c"
    items = schema_link.get("indicator_items") or [indicator]
    if indicator == "blood_pressure":
        items = ["systolic_bp", "diastolic_bp"]
    return indicator, items


def build_template_sql(query_spec: Dict[str, Any], schema_link: Dict[str, Any]) -> Dict[str, Any]:
    intent = str(query_spec.get("intent") or "")
    confidence = float(query_spec.get("confidence") or 0.0)
    if confidence < 0.7:
        return {"sql": None, "template_id": None, "params": {}, "explanation": "Low confidence template parse; defer to LLM candidate."}
    diseases = query_spec.get("diseases") or []
    risk_level = query_spec.get("risk_level")
    time_range = query_spec.get("time_range")
    context_payload = query_spec.get("analysis_context") or {}
    as_of_date = str(context_payload.get("as_of_date") or AnalysisContext.current().as_of_date)
    medication = query_spec.get("medication_category")
    lifestyle = query_spec.get("lifestyle")
    indicator, indicator_items = _primary_indicator(query_spec, schema_link)
    indicator_label = INDICATOR_LABELS.get(indicator, indicator)
    where = ["1=1", *_disease_conditions(diseases)]

    if intent == "count_by_disease":
        sql = (
            "SELECT COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM patient_profile p WHERE " + " AND ".join(where)
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": "统计疾病筛选患者人数"}
    if intent == "count_by_risk_level":
        risk_filter = f"WHERE lower(r.risk_level) = '{risk_level}'" if risk_level else ""
        sql = (
            LATEST_RISK_CTE
            +
            "SELECT r.risk_level, COUNT(DISTINCT r.patient_id) AS patient_count "
            f"FROM latest_risk r {risk_filter} GROUP BY r.risk_level ORDER BY patient_count DESC"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": "统计风险等级患者人数"}
    if intent == "count_by_disease_and_risk":
        sql = (
            LATEST_RISK_CTE
            +
            "SELECT r.risk_level, COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM patient_profile p JOIN latest_risk r ON p.patient_id = r.patient_id "
            "WHERE "
            + " AND ".join(where)
            + (f" AND lower(r.risk_level) = '{risk_level}'" if risk_level else "")
            + " GROUP BY r.risk_level ORDER BY patient_count DESC"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": "统计疾病与风险等级交叉患者人数"}
    if intent in {"avg_indicator_by_disease", "avg_indicator_by_disease_combo"}:
        if indicator == "bmi":
            sql = (
                "SELECT COUNT(DISTINCT p.patient_id) AS patient_count, "
                "ROUND(AVG(CAST(p.bmi AS REAL)), 4) AS avg_bmi "
                "FROM patient_profile p WHERE "
                + " AND ".join(where)
            )
        else:
            sql = (
                "SELECT COUNT(DISTINCT p.patient_id) AS patient_count, COUNT(DISTINCT l.lab_id) AS lab_count, "
                "ROUND(AVG(CAST(COALESCE(l.value, l.item_value) AS REAL)), 4) AS avg_value "
                "FROM patient_profile p JOIN lab_result l ON p.patient_id = l.patient_id "
                "WHERE "
                + " AND ".join(where)
                + f" AND {_indicator_filter(indicator_items)}"
                + _time_filter("l.test_date", time_range, as_of_date)
            )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"统计{indicator_label}平均值"}
    if intent == "avg_indicator_by_risk_level":
        sql = (
            LATEST_RISK_CTE
            +
            "SELECT r.risk_level, COUNT(DISTINCT r.patient_id) AS patient_count, "
            "COUNT(DISTINCT l.lab_id) AS lab_count, "
            "ROUND(AVG(CAST(COALESCE(l.value, l.item_value) AS REAL)), 4) AS avg_value "
            "FROM latest_risk r JOIN lab_result l ON r.patient_id = l.patient_id "
            f"WHERE {_indicator_filter(indicator_items)} "
            "GROUP BY r.risk_level ORDER BY CASE r.risk_level WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 ELSE 9 END"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"按风险等级统计{indicator_label}平均值"}
    if intent in {"abnormal_rate_by_disease", "abnormal_rate_by_disease_combo"}:
        if indicator == "bmi":
            sql = (
                "SELECT COUNT(DISTINCT p.patient_id) AS denominator, "
                "COUNT(DISTINCT CASE WHEN CAST(p.bmi AS REAL) >= 24 THEN p.patient_id END) AS numerator, "
                "ROUND(1.0 * COUNT(DISTINCT CASE WHEN CAST(p.bmi AS REAL) >= 24 THEN p.patient_id END) / NULLIF(COUNT(DISTINCT p.patient_id), 0), 4) AS abnormal_rate "
                "FROM patient_profile p WHERE "
                + " AND ".join(where)
            )
        else:
            sql = (
                "SELECT COUNT(DISTINCT p.patient_id) AS patient_count, COUNT(DISTINCT l.lab_id) AS denominator, "
                f"{_abnormal_count('l')} AS numerator, "
                f"ROUND(1.0 * {_abnormal_count('l')} / NULLIF(COUNT(DISTINCT l.lab_id), 0), 4) AS abnormal_rate "
                "FROM patient_profile p JOIN lab_result l ON p.patient_id = l.patient_id "
                "WHERE "
                + " AND ".join(where)
                + f" AND {_indicator_filter(indicator_items)}"
                + _time_filter("l.test_date", time_range, as_of_date)
            )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"统计{indicator_label}异常比例"}
    if intent == "abnormal_rate_by_risk_level":
        sql = (
            LATEST_RISK_CTE
            +
            "SELECT r.risk_level, COUNT(DISTINCT l.lab_id) AS denominator, "
            "COUNT(DISTINCT r.patient_id) AS patient_count, "
            f"{_abnormal_count('l')} AS numerator, "
            f"ROUND(1.0 * {_abnormal_count('l')} / NULLIF(COUNT(DISTINCT l.lab_id), 0), 4) AS abnormal_rate "
            "FROM latest_risk r JOIN lab_result l ON r.patient_id = l.patient_id "
            f"WHERE {_indicator_filter(indicator_items)} GROUP BY r.risk_level "
            "ORDER BY CASE r.risk_level WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 ELSE 9 END"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"按风险等级统计{indicator_label}异常比例"}
    if intent == "abnormal_rate_by_lifestyle" and lifestyle:
        field = lifestyle["field"]
        op = lifestyle["op"]
        value = lifestyle["value"]
        condition = f"CAST(ls.{field} AS REAL) {op} {value}" if op in {"<", ">", "<=", ">="} else f"lower(ls.{field}) {op} '{value}'"
        sql = (
            "SELECT COUNT(DISTINCT ls.patient_id) AS patient_count, COUNT(DISTINCT l.lab_id) AS denominator, "
            f"{_abnormal_count('l')} AS numerator, "
            f"ROUND(1.0 * {_abnormal_count('l')} / NULLIF(COUNT(DISTINCT l.lab_id), 0), 4) AS abnormal_rate "
            "FROM lifestyle_record ls JOIN visit_record v ON ls.visit_id = v.visit_id "
            "JOIN lab_result l ON v.visit_id = l.visit_id "
            f"WHERE {condition} AND {_indicator_filter(indicator_items)}"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"统计{lifestyle['label']}人群{indicator_label}异常率"}
    if intent == "abnormal_rate_by_medication" and medication:
        if indicator == "hba1c":
            sql = (
                "SELECT COUNT(DISTINCT l.patient_id) AS patient_count, COUNT(DISTINCT l.lab_id) AS denominator, "
                "COUNT(DISTINCT CASE WHEN CAST(COALESCE(l.value, l.item_value) AS REAL) < 7.0 THEN l.lab_id END) AS numerator, "
                "ROUND(1.0 * COUNT(DISTINCT CASE WHEN CAST(COALESCE(l.value, l.item_value) AS REAL) < 7.0 THEN l.lab_id END) / NULLIF(COUNT(DISTINCT l.lab_id), 0), 4) AS control_rate, "
                "ROUND(AVG(CAST(COALESCE(l.value, l.item_value) AS REAL)), 4) AS avg_value "
                "FROM lab_result l "
                f"WHERE {_indicator_filter(indicator_items)} "
                "AND EXISTS ("
                "SELECT 1 FROM medication_record m "
                "WHERE m.patient_id = l.patient_id "
                f"AND lower(m.drug_category) = '{medication}'"
                ")"
            )
            return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"统计用药人群{indicator_label}控制达标情况"}
        sql = (
            "SELECT COUNT(DISTINCT l.patient_id) AS patient_count, COUNT(DISTINCT l.lab_id) AS denominator, "
            f"{_abnormal_count('l')} AS numerator, "
            f"ROUND(1.0 * {_abnormal_count('l')} / NULLIF(COUNT(DISTINCT l.lab_id), 0), 4) AS abnormal_rate, "
            "ROUND(AVG(CAST(COALESCE(l.value, l.item_value) AS REAL)), 4) AS avg_value "
            "FROM lab_result l "
            f"WHERE {_indicator_filter(indicator_items)} "
            "AND EXISTS ("
            "SELECT 1 FROM medication_record m "
            "WHERE m.patient_id = l.patient_id "
            f"AND lower(m.drug_category) = '{medication}'"
            ")"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"统计用药人群{indicator_label}控制情况"}
    if intent == "distribution_by_risk_level":
        sql = (
            LATEST_RISK_CTE
            +
            "SELECT risk_level, COUNT(DISTINCT patient_id) AS patient_count "
            "FROM latest_risk GROUP BY risk_level ORDER BY patient_count DESC"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": "统计风险等级分布"}
    if intent == "distribution_by_medication_category":
        sql = (
            "SELECT drug_category, COUNT(DISTINCT patient_id) AS patient_count "
            "FROM medication_record GROUP BY drug_category ORDER BY patient_count DESC LIMIT 20"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": "统计用药类别分布"}
    if intent == "distribution_by_disease":
        sql = (
            "SELECT disease_tags, COUNT(DISTINCT patient_id) AS patient_count "
            "FROM patient_profile GROUP BY disease_tags ORDER BY patient_count DESC LIMIT 20"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": "统计疾病标签组合分布"}
    if intent == "trend_indicator_abnormal":
        sql = (
            "SELECT substr(l.test_date, 1, 7) AS month, "
            "COUNT(DISTINCT l.patient_id) AS tested_patient_count, "
            "COUNT(DISTINCT CASE WHEN lower(l.abnormal_flag) != 'normal' THEN l.patient_id END) AS abnormal_patient_count, "
            "ROUND(1.0 * COUNT(DISTINCT CASE WHEN lower(l.abnormal_flag) != 'normal' THEN l.patient_id END) "
            "/ NULLIF(COUNT(DISTINCT l.patient_id), 0), 4) AS abnormal_rate "
            "FROM lab_result l "
            f"WHERE {_indicator_filter(indicator_items)}"
            + _time_filter("l.test_date", time_range, as_of_date)
            + " GROUP BY substr(l.test_date, 1, 7) ORDER BY month"
        )
        return {"sql": sql, "template_id": intent, "params": {}, "explanation": f"统计{indicator_label}异常趋势"}
    if intent == "followup_count":
        follow_where = ["f.status IN ('pending','scheduled')", *_disease_conditions(diseases)]
        if risk_level:
            follow_where.append(f"lower(f.priority) = '{risk_level}'")
        sql = (
            "SELECT COUNT(DISTINCT f.patient_id) AS patient_count, COUNT(f.plan_id) AS plan_count "
            "FROM followup_plan f JOIN patient_profile p ON f.patient_id = p.patient_id "
            "WHERE "
            + " AND ".join(follow_where)
            + _time_filter("f.followup_date", time_range, as_of_date)
        )
        return {"sql": sql, "template_id": "followup_count_by_days_and_disease" if diseases else "followup_count_by_days", "params": {}, "explanation": "统计未来随访人数"}
    return {"sql": None, "template_id": None, "params": {}, "explanation": "No template matched."}
