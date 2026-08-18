from __future__ import annotations

from typing import Dict, List

INDICATOR_ALIASES: Dict[str, str] = {
    "hba1c": "hba1c",
    "糖化血红蛋白": "hba1c",
    "空腹血糖": "fasting_glucose",
    "fpg": "fasting_glucose",
    "fasting_glucose": "fasting_glucose",
    "ldl-c": "ldl_c",
    "ldl": "ldl_c",
    "低密度脂蛋白": "ldl_c",
    "hdl-c": "hdl_c",
    "hdl": "hdl_c",
    "高密度脂蛋白": "hdl_c",
    "总胆固醇": "total_cholesterol",
    "tc": "total_cholesterol",
    "甘油三酯": "triglyceride",
    "tg": "triglyceride",
    "收缩压": "systolic_bp",
    "sbp": "systolic_bp",
    "systolic_bp": "systolic_bp",
    "舒张压": "diastolic_bp",
    "dbp": "diastolic_bp",
    "diastolic_bp": "diastolic_bp",
    "bmi": "bmi",
    "体重指数": "bmi",
    "尿酸": "uric_acid",
    "ua": "uric_acid",
    "肌酐": "creatinine",
    "creatinine": "creatinine",
    "egfr": "egfr",
}


def normalize_indicators(text: str) -> List[str]:
    lowered = str(text or "").lower()
    matched: List[str] = []
    for alias, indicator in INDICATOR_ALIASES.items():
        if alias.lower() in lowered and indicator not in matched:
            matched.append(indicator)
    return matched

