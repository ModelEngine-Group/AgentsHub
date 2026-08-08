"""Medical knowledge-graph ontology (entity types, relation types, constraints).

Designed for general clinical text and extended with pathology-specific types so
it works on both the Chinese pathology textbooks and the English
pathologyoutlines-style data in this repo. Extraction is LLM-driven, so the
schema is bilingual-friendly: type names are Chinese (display) with English keys.
"""
from __future__ import annotations

from dataclasses import dataclass

# Entity types: key -> Chinese display name.
ENTITY_TYPES: dict[str, str] = {
    "Disease": "疾病",
    "Symptom": "症状",
    "Drug": "药物",
    "Examination": "检查",
    "Procedure": "手术",
    "Body": "身体部位",
    "Department": "科室",
    # Pathology-specific extensions
    "Tumor": "肿瘤",
    "Biomarker": "标志物",
    "Gene": "基因",
    "Morphology": "形态学特征",
    # Public benchmark / open-schema extensions
    "Epidemiology": "流行病学",
    "SocialFactor": "社会因素",
    "Prognosis": "预后",
    "Other": "其他医学概念",
    "Class": "类别",
    "TestValue": "检查值",
    "Frequency": "频率",
    "Level": "程度",
    "Reason": "原因",
    "Duration": "持续时间",
    "Amount": "用量",
    "Method": "用药方法",
    "Pathogenesis": "发病机制",
}

# Relation types: key -> Chinese display name.
RELATION_TYPES: dict[str, str] = {
    "has_symptom": "有症状",
    "recommend_drug": "推荐药物",
    "need_examination": "需做检查",
    "complication": "并发",
    "contraindication": "禁忌",
    "treated_in_department": "就诊于",
    "positive_marker": "阳性标志物",
    "negative_marker": "阴性标志物",
    "associated_gene": "关联基因",
    "has_morphology": "形态学表现",
    "located_in": "位于",
    "subtype_of": "亚型",
    "treated_by_procedure": "手术/操作治疗",
    "adverse_reaction": "药物不良反应",
}

# Allowed (head_type, tail_type) pairs per relation. Used by triple_validator to
# reject schema-illegal triples (e.g. a "recommend_drug" whose tail is not a Drug).
# Disease and Tumor are treated as interchangeable "lesion" heads for convenience.
_LESION = {"Disease", "Tumor"}
RELATION_CONSTRAINTS: dict[str, tuple[set[str], set[str]]] = {
    "has_symptom": (_LESION, {"Symptom"}),
    "recommend_drug": (_LESION, {"Drug"}),
    "need_examination": (_LESION, {"Examination"}),
    "complication": (_LESION, _LESION),
    "contraindication": ({"Drug"}, _LESION | {"Symptom"}),
    "treated_in_department": (_LESION, {"Department"}),
    "positive_marker": (_LESION, {"Biomarker"}),
    "negative_marker": (_LESION, {"Biomarker"}),
    "associated_gene": (_LESION, {"Gene"}),
    "has_morphology": (_LESION, {"Morphology"}),
    "located_in": (_LESION | {"Symptom"}, {"Body"}),
    "subtype_of": (_LESION, _LESION),
    "treated_by_procedure": (_LESION, {"Procedure"}),
    "adverse_reaction": ({"Drug"}, {"Symptom", "Disease"}),
}


@dataclass(frozen=True)
class Entity:
    name: str
    type: str

    def key(self) -> str:
        """Normalized identity key for dedup (case/space-insensitive)."""
        return f"{self.type}::{self.name.strip().lower()}"


@dataclass(frozen=True)
class Triple:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    confidence: float = 1.0
    source: str = ""

    def key(self) -> str:
        return f"{self.head_type}:{self.head.lower()}|{self.relation}|{self.tail_type}:{self.tail.lower()}"


def is_valid_entity_type(t: str) -> bool:
    return t in ENTITY_TYPES


def is_valid_relation(r: str) -> bool:
    return r in RELATION_TYPES


def check_relation_constraint(relation: str, head_type: str, tail_type: str) -> bool:
    """Return True if (head_type, tail_type) is allowed for `relation`."""
    if relation not in RELATION_CONSTRAINTS:
        return False
    head_ok, tail_ok = RELATION_CONSTRAINTS[relation]
    return head_type in head_ok and tail_type in tail_ok


def ontology_prompt_block() -> str:
    """Human/LLM-readable ontology description injected into extraction prompts."""
    ents = "\n".join(f"  - {k} ({v})" for k, v in ENTITY_TYPES.items())
    rels_lines = []
    for k, v in RELATION_TYPES.items():
        head_ok, tail_ok = RELATION_CONSTRAINTS[k]
        rels_lines.append(
            f"  - {k} ({v}): {'/'.join(sorted(head_ok))} -> {'/'.join(sorted(tail_ok))}"
        )
    rels = "\n".join(rels_lines)
    return f"实体类型 (entity types):\n{ents}\n\n关系类型 (relation types, 含两端类型约束):\n{rels}"
