"""
知识图谱字段标准化模块。

该模块把抽取标签映射到统一实体类型和关系类型。
"""

from __future__ import annotations

from typing import Any


ENTITY_TYPE_MAP = {
    "dis": "disease",
    "disease": "disease",
    "疾病": "disease",
    "sym": "symptom",
    "symptom": "symptom",
    "症状": "symptom",
    "dru": "drug",
    "drug": "drug",
    "药物": "drug",
    "药品": "drug",
    "ite": "test",
    "test": "test",
    "检查": "test",
    "检验": "test",
    "pro": "procedure",
    "procedure": "procedure",
    "治疗": "procedure",
    "手术": "procedure",
    "dep": "department",
    "department": "department",
    "科室": "department",
    "bod": "body_part",
    "body": "body_part",
    "身体部位": "body_part",
    "mic": "microorganism",
    "microorganism": "microorganism",
    "微生物": "microorganism",
}


RELATION_MAP = {
    "has_symptom": "has_symptom",
    "临床表现": "has_symptom",
    "相关（症状）": "has_symptom",
    "治疗后症状": "has_symptom",
    "treated_by_drug": "treated_by_drug",
    "药物治疗": "treated_by_drug",
    "treated_by_procedure": "treated_by_procedure",
    "辅助治疗": "treated_by_procedure",
    "手术治疗": "treated_by_procedure",
    "放射治疗": "treated_by_procedure",
    "化疗": "treated_by_procedure",
    "requires_test": "requires_test",
    "检查": "requires_test",
    "辅助检查": "requires_test",
    "实验室检查": "requires_test",
    "影像学检查": "requires_test",
    "内窥镜检查": "requires_test",
    "组织学检查": "requires_test",
    "筛查": "requires_test",
    "visit_department": "visit_department",
    "就诊科室": "visit_department",
    "所属科室": "visit_department",
    "has_complication": "has_complication",
    "并发症": "has_complication",
    "has_cause": "has_cause",
    "病因": "has_cause",
    "发病机制": "has_cause",
    "病理生理": "has_cause",
    "遗传因素": "has_cause",
    "has_prevention": "has_prevention",
    "预防": "has_prevention",
    "susceptible_population": "susceptible_population",
    "多发群体": "susceptible_population",
    "发病年龄": "susceptible_population",
    "发病性别倾向": "susceptible_population",
    "transmission_way": "transmission_way",
    "传播途径": "transmission_way",
    "affects_body_part": "affects_body_part",
    "发病部位": "affects_body_part",
    "转移部位": "affects_body_part",
    "belongs_to_category": "belongs_to_category",
    "疾病分类": "belongs_to_category",
    "differential_diagnosis": "differential_diagnosis",
    "鉴别诊断": "differential_diagnosis",
    "alias_of": "alias_of",
    "同义词": "alias_of",
    "related_to": "related_to",
    "病史": "related_to",
    "相关（导致）": "related_to",
    "相关（转化）": "related_to",
    "高危因素": "related_to",
    "风险评估因素": "related_to",
}


RELATION_DISPLAY_NAMES = {
    "has_symptom": "临床表现",
    "treated_by_drug": "药物治疗",
    "treated_by_procedure": "治疗方式",
    "requires_test": "检查",
    "visit_department": "就诊科室",
    "has_complication": "并发症",
    "has_cause": "病因",
    "has_prevention": "预防",
    "susceptible_population": "易感人群",
    "transmission_way": "传播途径",
    "affects_body_part": "发病部位",
    "belongs_to_category": "疾病分类",
    "differential_diagnosis": "鉴别诊断",
    "alias_of": "别名",
    "related_to": "相关",
}


def normalize_kg_entity_type(value: Any) -> str:
    raw = str(value or "").strip()
    return ENTITY_TYPE_MAP.get(raw, ENTITY_TYPE_MAP.get(raw.lower(), "medical_entity"))


def normalize_kg_relation_code(value: Any) -> str:
    raw = str(value or "").strip()
    return RELATION_MAP.get(raw, raw)


def relation_display_name(code: str) -> str:
    normalized = normalize_kg_relation_code(code)
    return RELATION_DISPLAY_NAMES.get(normalized, str(code or normalized))
