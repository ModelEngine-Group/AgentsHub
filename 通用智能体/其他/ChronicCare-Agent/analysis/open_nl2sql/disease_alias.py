from __future__ import annotations

from typing import Dict, List

DISEASE_ALIASES: Dict[str, str] = {
    "高血压": "hypertension",
    "hypertension": "hypertension",
    "糖尿病": "diabetes",
    "diabetes": "diabetes",
    "高脂血症": "hyperlipidemia",
    "高脂血": "hyperlipidemia",
    "hyperlipidemia": "hyperlipidemia",
    "肥胖": "obesity",
    "obesity": "obesity",
    "高尿酸": "hyperuricemia",
    "高尿酸血症": "hyperuricemia",
    "冠心病": "coronary_heart_disease",
    "冠心病风险": "coronary_risk",
    "慢性肾病": "chronic_kidney_disease",
    "肾病": "chronic_kidney_disease",
    "脂肪肝": "fatty_liver_disease",
    "代谢综合征": "metabolic_syndrome",
    "脑卒中": "stroke_post",
    "中风": "stroke_post",
    "骨质疏松": "osteoporosis",
    "慢阻肺": "copd",
    "慢性阻塞性肺疾病": "copd",
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


def normalize_diseases(text: str) -> List[str]:
    lowered = str(text or "").lower()
    matched: List[str] = []
    for alias, disease in DISEASE_ALIASES.items():
        if alias.lower() in lowered and disease not in matched:
            matched.append(disease)
    return matched

