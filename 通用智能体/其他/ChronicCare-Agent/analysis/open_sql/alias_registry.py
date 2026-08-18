from __future__ import annotations

import re
from typing import Any, Dict, List

DISEASE_ALIASES = {
    "高血压": "hypertension",
    "hypertension": "hypertension",
    "糖尿病": "diabetes",
    "高血糖": "diabetes",
    "diabetes": "diabetes",
    "高脂血症": "hyperlipidemia",
    "高脂血": "hyperlipidemia",
    "高血脂": "hyperlipidemia",
    "hyperlipidemia": "hyperlipidemia",
    "肥胖": "obesity",
    "高尿酸": "hyperuricemia",
    "高尿酸血症": "hyperuricemia",
    "冠心病": "coronary_heart_disease",
    "冠心病风险": "coronary_risk",
    "慢性肾病": "chronic_kidney_disease",
    "慢性肾病风险": "ckd_risk",
    "肾病": "chronic_kidney_disease",
    "脂肪肝": "fatty_liver_disease",
    "脂肪肝风险": "fatty_liver_risk",
    "代谢综合征": "metabolic_syndrome",
    "脑卒中": "stroke_post",
    "中风": "stroke_post",
    "骨质疏松": "osteoporosis",
    "慢阻肺": "copd",
    "慢性阻塞性肺疾病": "copd",
    "copd": "copd",
    "哮喘": "asthma",
    "骨关节炎": "osteoarthritis",
    "痛风": "gout",
    "慢性心力衰竭": "chronic_heart_failure",
    "心力衰竭": "chronic_heart_failure",
    "糖尿病肾病": "diabetic_kidney_disease",
    "阻塞性睡眠呼吸暂停": "obstructive_sleep_apnea",
    "睡眠呼吸暂停": "obstructive_sleep_apnea",
    "脑血管病": "cerebrovascular_disease",
    "心房颤动": "atrial_fibrillation",
    "房颤": "atrial_fibrillation",
    "慢性肝炎": "chronic_hepatitis",
    "甲状腺功能减退": "hypothyroidism",
    "甲减": "hypothyroidism",
    "慢病": "chronic_disease",
    "慢病患者": "chronic_disease",
}

DISEASE_LABELS = {
    "hypertension": "高血压",
    "diabetes": "糖尿病",
    "hyperlipidemia": "高脂血症",
    "obesity": "肥胖",
    "hyperuricemia": "高尿酸血症",
    "coronary_risk": "冠心病风险",
    "ckd_risk": "慢性肾病风险",
    "fatty_liver_risk": "脂肪肝风险",
    "metabolic_syndrome": "代谢综合征",
    "stroke_post": "脑卒中后状态",
    "osteoporosis": "骨质疏松",
    "copd": "慢阻肺",
    "chronic_kidney_disease": "慢性肾病",
    "coronary_heart_disease": "冠心病",
    "fatty_liver_disease": "脂肪肝",
    "asthma": "哮喘",
    "osteoarthritis": "骨关节炎",
    "gout": "痛风",
    "chronic_heart_failure": "慢性心力衰竭",
    "diabetic_kidney_disease": "糖尿病肾病",
    "obstructive_sleep_apnea": "阻塞性睡眠呼吸暂停",
    "cerebrovascular_disease": "脑血管病",
    "atrial_fibrillation": "心房颤动",
    "chronic_hepatitis": "慢性肝炎",
    "hypothyroidism": "甲状腺功能减退",
}

INDICATOR_ALIASES = {
    "hba1c": "hba1c",
    "HbA1c": "hba1c",
    "糖化血红蛋白": "hba1c",
    "糖化": "hba1c",
    "空腹血糖": "fasting_glucose",
    "血糖空腹值": "fasting_glucose",
    "空腹血糖值": "fasting_glucose",
    "fpg": "fasting_glucose",
    "FPG": "fasting_glucose",
    "fasting glucose": "fasting_glucose",
    "fasting_glucose": "fasting_glucose",
    "ldl-c": "ldl_c",
    "LDL-C": "ldl_c",
    "ldl": "ldl_c",
    "LDL": "ldl_c",
    "低密度脂蛋白": "ldl_c",
    "hdl-c": "hdl_c",
    "HDL-C": "hdl_c",
    "hdl": "hdl_c",
    "HDL": "hdl_c",
    "高密度脂蛋白": "hdl_c",
    "总胆固醇": "total_cholesterol",
    "TC": "total_cholesterol",
    "甘油三酯": "triglyceride",
    "TG": "triglyceride",
    "血压": "blood_pressure",
    "收缩压": "systolic_bp",
    "SBP": "systolic_bp",
    "舒张压": "diastolic_bp",
    "DBP": "diastolic_bp",
    "blood pressure": "blood_pressure",
    "bmi": "bmi",
    "BMI": "bmi",
    "体重指数": "bmi",
    "尿酸": "uric_acid",
    "UA": "uric_acid",
    "肌酐": "creatinine",
    "creatinine": "creatinine",
    "eGFR": "egfr",
    "egfr": "egfr",
}

INDICATOR_LABELS = {
    "hba1c": "HbA1c",
    "fasting_glucose": "空腹血糖",
    "ldl_c": "LDL-C",
    "hdl_c": "HDL-C",
    "total_cholesterol": "总胆固醇",
    "triglyceride": "甘油三酯",
    "blood_pressure": "血压",
    "systolic_bp": "收缩压",
    "diastolic_bp": "舒张压",
    "bmi": "BMI",
    "uric_acid": "尿酸",
    "creatinine": "肌酐",
    "egfr": "eGFR",
}

INDICATOR_ITEM_NAMES = {
    "hba1c": ["hba1c"],
    "fasting_glucose": ["fasting_glucose"],
    "ldl_c": ["ldl_c"],
    "hdl_c": ["hdl_c"],
    "total_cholesterol": ["total_cholesterol"],
    "triglyceride": ["triglyceride"],
    "blood_pressure": ["systolic_bp", "diastolic_bp"],
    "systolic_bp": ["systolic_bp"],
    "diastolic_bp": ["diastolic_bp"],
    "bmi": ["bmi"],
    "uric_acid": ["uric_acid"],
    "creatinine": ["creatinine"],
    "egfr": ["egfr"],
}

RISK_ALIASES = {"高风险": "high", "高危": "high", "high": "high", "中风险": "medium", "中危": "medium", "medium": "medium", "低风险": "low", "低危": "low", "low": "low"}

AGG_ALIASES = {
    "count": ("多少", "有多少", "人数", "数量", "几人"),
    "avg": ("平均", "均值", "平均值"),
    "abnormal_rate": ("异常比例", "异常率", "超标比例", "超标率", "控制情况", "达标", "控制", "abnormal rate"),
    "distribution": ("分布", "分组", "各类", "不同"),
    "trend": ("趋势", "变化", "最近几个月", "trend"),
    "max": ("最大", "最高", "前 10", "前十"),
    "min": ("最小", "最低"),
}

LIFESTYLE_ALIASES = {
    "高盐饮食": {"field": "salt_intake_level", "op": "=", "value": "high", "label": "高盐饮食"},
    "高盐人群": {"field": "salt_intake_level", "op": "=", "value": "high", "label": "高盐饮食"},
    "高盐摄入": {"field": "salt_intake_level", "op": "=", "value": "high", "label": "高盐饮食"},
    "盐摄入高": {"field": "salt_intake_level", "op": "=", "value": "high", "label": "高盐饮食"},
    "运动不足": {"field": "exercise_minutes_per_week", "op": "<", "value": "150", "label": "运动不足"},
    "睡眠不足": {"field": "sleep_hours", "op": "<", "value": "7", "label": "睡眠不足"},
    "吸烟": {"field": "smoking_status", "op": "=", "value": "yes", "label": "吸烟"},
}

MEDICATION_ALIASES = {
    "降压药": "antihypertensive",
    "降糖药": "glucose_lowering",
    "降脂药": "lipid_lowering",
    "他汀": "lipid_lowering",
    "利尿剂": "antihypertensive",
    "抗血小板": "antiplatelet",
    "降尿酸药": "uric_acid_lowering",
}


def _scan_aliases(question: str, aliases: Dict[str, str]) -> List[str]:
    matches = []
    compact = str(question or "").replace(" ", "")
    lowered = question.lower()
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(r"[A-Za-z]", alias):
            needle = alias.lower()
            matched = bool(needle and re.search(rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])", lowered))
        else:
            needle = alias.replace(" ", "")
            matched = bool(needle and needle in compact)
        if matched and canonical not in matches:
            matches.append(canonical)
    return matches


def normalize_diseases(question: str) -> List[str]:
    diseases = _scan_aliases(question, DISEASE_ALIASES)
    compact = str(question or "").replace(" ", "")
    if "三高" in compact:
        for disease in ("hypertension", "diabetes", "hyperlipidemia"):
            if disease not in diseases:
                diseases.append(disease)
    if "chronic_disease" in diseases:
        diseases.remove("chronic_disease")
    return diseases


def normalize_indicators(question: str) -> List[str]:
    matches = _scan_aliases(question, INDICATOR_ALIASES)
    text = str(question or "")
    lowered = text.lower()
    bp_context = any(token in text for token in ("血压异常", "血压控制", "收缩压", "舒张压", "SBP", "DBP")) or "blood pressure" in lowered
    if "blood_pressure" in matches and "高血压" in text and not bp_context:
        matches = [item for item in matches if item != "blood_pressure"]
    return matches


def normalize_risk_level(question: str) -> str | None:
    matches = _scan_aliases(question, RISK_ALIASES)
    return matches[0] if matches else None


def detect_aggregation(question: str) -> str:
    text = str(question or "")
    if any(token in text for token in AGG_ALIASES["trend"]):
        return "trend"
    if any(token in text for token in AGG_ALIASES["abnormal_rate"]):
        return "abnormal_rate"
    if any(token in text for token in AGG_ALIASES["avg"]):
        return "avg"
    if any(token in text for token in AGG_ALIASES["distribution"]):
        return "distribution"
    if any(token in text for token in AGG_ALIASES["max"]):
        return "max"
    if any(token in text for token in AGG_ALIASES["min"]):
        return "min"
    return "count"


def detect_time_range(question: str) -> Dict[str, Any] | None:
    text = str(question or "")
    compact = text.replace(" ", "")
    named = {
        "最近半年": ("past_months", 6),
        "最近三个月": ("past_months", 3),
        "最近一个月": ("past_months", 1),
        "未来一周": ("future_days", 7),
        "未来一个月": ("future_days", 30),
        "未来两个月": ("future_days", 60),
        "未来三个月": ("future_days", 90),
    }
    for token, value in named.items():
        if token in compact:
            return {"type": value[0], "value": value[1], "label": token}
    match = re.search(r"最近\s*(\d{1,3})\s*个?月", text) or re.search(r"最近(\d{1,3})个?月", compact)
    if match:
        return {"type": "past_months", "value": max(1, min(int(match.group(1)), 60)), "label": f"最近 {match.group(1)} 个月"}
    match = re.search(r"未来\s*(\d{1,3})\s*天", text) or re.search(r"未来(\d{1,3})天", compact)
    if match:
        return {"type": "future_days", "value": max(1, min(int(match.group(1)), 365)), "label": f"未来 {match.group(1)} 天"}
    return None


def detect_lifestyle(question: str) -> Dict[str, Any] | None:
    for alias, spec in LIFESTYLE_ALIASES.items():
        if alias in str(question or ""):
            return dict(spec)
    return None


def detect_medication_category(question: str) -> str | None:
    for alias, canonical in MEDICATION_ALIASES.items():
        if alias in str(question or ""):
            return canonical
    return None
