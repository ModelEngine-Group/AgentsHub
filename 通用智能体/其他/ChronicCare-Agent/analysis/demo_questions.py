from __future__ import annotations

from typing import Any, Dict, List

QUESTION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "count_patients_by_disease_recent": {
        "analysis_type": "count",
        "target_entity": "Patient",
        "target_indicator": None,
        "disease_filters": ["hypertension"],
        "drug_filters": [],
        "time_range": "recent_3_months",
    },
    "avg_indicator_by_disease_combo": {
        "analysis_type": "average",
        "target_entity": "Patient",
        "target_indicator": "hba1c",
        "disease_filters": ["hypertension", "diabetes"],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "abnormal_rate_by_disease_group": {
        "analysis_type": "rate",
        "target_entity": "DiseaseGroup",
        "target_indicator": "ldl_c",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "avg_indicator_by_drug": {
        "analysis_type": "average",
        "target_entity": "Patient",
        "target_indicator": "fasting_glucose",
        "disease_filters": [],
        "drug_filters": ["metformin"],
        "time_range": "all_time",
    },
    "abnormal_rate_by_bmi_group": {
        "analysis_type": "rate",
        "target_entity": "Patient",
        "target_indicator": "blood_pressure",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "avg_ldl_by_lipid_drug": {
        "analysis_type": "average",
        "target_entity": "Patient",
        "target_indicator": "ldl_c",
        "disease_filters": [],
        "drug_filters": ["atorvastatin", "rosuvastatin"],
        "time_range": "all_time",
    },
    "top_patients_by_abnormal_latest_visit": {
        "analysis_type": "ranking",
        "target_entity": "Patient",
        "target_indicator": "abnormal_indicator_count",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "latest_visit",
    },
    "hba1c_high_rate_in_diabetes": {
        "analysis_type": "rate",
        "target_entity": "Patient",
        "target_indicator": "hba1c",
        "disease_filters": ["diabetes"],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "systolic_high_count_in_hypertension": {
        "analysis_type": "count",
        "target_entity": "Patient",
        "target_indicator": "systolic_bp",
        "disease_filters": ["hypertension"],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "avg_bmi_three_highs": {
        "analysis_type": "average",
        "target_entity": "Patient",
        "target_indicator": "bmi",
        "disease_filters": ["hypertension", "diabetes", "hyperlipidemia"],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "monthly_fasting_glucose_abnormal_trend": {
        "analysis_type": "trend",
        "target_entity": "Patient",
        "target_indicator": "fasting_glucose",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "monthly",
    },
    "monthly_bp_abnormal_trend": {
        "analysis_type": "trend",
        "target_entity": "Patient",
        "target_indicator": "blood_pressure",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "monthly",
    },
    "patient_count_by_drug_category": {
        "analysis_type": "distribution",
        "target_entity": "DrugCategory",
        "target_indicator": "patient_count",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "graph_entity_type_count": {
        "analysis_type": "distribution",
        "target_entity": "GraphNode",
        "target_indicator": "entity_type",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "all_time",
    },
    "graph_relation_type_count": {
        "analysis_type": "distribution",
        "target_entity": "GraphEdge",
        "target_indicator": "relation_type",
        "disease_filters": [],
        "drug_filters": [],
        "time_range": "all_time",
    },
}


def default_question_struct(intent: str) -> Dict[str, Any]:
    return QUESTION_DEFAULTS.get(
        intent,
        {
            "analysis_type": "unknown",
            "target_entity": None,
            "target_indicator": None,
            "disease_filters": [],
            "drug_filters": [],
            "time_range": None,
        },
    ).copy()


def supported_intents() -> List[str]:
    return sorted(QUESTION_DEFAULTS)
