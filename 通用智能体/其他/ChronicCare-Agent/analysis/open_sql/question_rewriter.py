from __future__ import annotations

from typing import Any, Dict

from analysis.open_sql.alias_registry import (
    detect_aggregation,
    detect_lifestyle,
    detect_medication_category,
    detect_time_range,
    normalize_diseases,
    normalize_indicators,
    normalize_risk_level,
)

PRONOUN_HINTS = ("他们", "这些患者", "该群体", "这类患者", "上述患者")
MEDICATION_GROUP_HINTS = ("药物类别", "用药类别", "用药分类", "药物分类", "drug_category")
UNSUPPORTED_FIELD_HINTS = ("收入", "工资", "薪资", "家庭住址", "身份证号")
UNSUPPORTED_CERTAINTY_HINTS = ("一定会", "必然会", "保证会", "肯定会")


def rewrite_question(question: str, last_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    text = " ".join(str(question or "").strip().split())
    last_context = last_context or {}
    diseases = normalize_diseases(text)
    indicators = normalize_indicators(text)
    risk_level = normalize_risk_level(text)
    aggregation = detect_aggregation(text)
    time_range = detect_time_range(text)
    lifestyle = detect_lifestyle(text)
    medication_category = detect_medication_category(text)
    medication_group_query = any(token.lower() in text.lower() for token in MEDICATION_GROUP_HINTS)
    need_chart = any(token in text for token in ("图", "图表", "可视化", "折线", "柱状", "趋势"))
    cohort_ref = any(token in text for token in PRONOUN_HINTS)
    risk_group_query = any(token in text for token in ("不同风险", "各风险", "风险等级", "风险分组", "风险层级", "risk_level"))
    needs_context = False
    has_explicit_scope = bool(diseases or indicators or risk_level or lifestyle or medication_category)
    if cohort_ref and not last_context and not has_explicit_scope:
        needs_context = True

    explicitly_unsupported = any(
        token in text for token in UNSUPPORTED_FIELD_HINTS + UNSUPPORTED_CERTAINTY_HINTS
    )
    if explicitly_unsupported:
        intent = "unsupported"
    elif ("随访" in text or "followup" in text.lower()) and time_range and time_range.get("type") == "future_days":
        intent = "followup_count"
    elif medication_group_query and aggregation in {"count", "distribution"}:
        intent = "distribution_by_medication_category"
    elif aggregation == "trend":
        intent = "trend_indicator_abnormal"
    elif aggregation == "distribution":
        if indicators and (risk_level or risk_group_query):
            intent = "avg_indicator_by_risk_level"
            aggregation = "avg"
        elif risk_level or risk_group_query:
            intent = "distribution_by_risk_level"
        elif medication_category or "药" in text:
            intent = "distribution_by_medication_category"
        else:
            intent = "distribution_by_disease"
    elif aggregation == "avg" or (aggregation == "count" and indicators):
        if risk_level or risk_group_query:
            intent = "avg_indicator_by_risk_level"
        elif len(diseases) >= 2:
            intent = "avg_indicator_by_disease_combo"
        else:
            intent = "avg_indicator_by_disease"
    elif aggregation == "abnormal_rate":
        if lifestyle:
            intent = "abnormal_rate_by_lifestyle"
        elif medication_category:
            intent = "abnormal_rate_by_medication"
        elif risk_level or risk_group_query:
            intent = "abnormal_rate_by_risk_level"
        elif len(diseases) >= 2:
            intent = "abnormal_rate_by_disease_combo"
        else:
            intent = "abnormal_rate_by_disease"
    elif diseases and risk_level:
        intent = "count_by_disease_and_risk"
    elif risk_level or risk_group_query:
        intent = "count_by_risk_level"
    elif diseases:
        intent = "count_by_disease"
    else:
        intent = "unsupported"

    confidence = 0.9
    if intent == "unsupported":
        confidence = 0.2
    elif not indicators and intent.startswith(("avg", "abnormal_rate", "trend")) and intent not in {"abnormal_rate_by_lifestyle"}:
        confidence = 0.55
    if needs_context and not explicitly_unsupported:
        confidence = 0.0
        intent = "needs_context"

    return {
        "question": text,
        "intent": intent,
        "aggregation": aggregation,
        "diseases": diseases,
        "indicators": indicators,
        "risk_level": risk_level,
        "time_range": time_range,
        "lifestyle": lifestyle,
        "medication_category": medication_category,
        "cohort_ref": cohort_ref,
        "needs_context": needs_context,
        "need_chart": need_chart,
        "confidence": confidence,
    }
