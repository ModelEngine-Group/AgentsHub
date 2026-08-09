"""医疗知识图谱 Schema：实体与关系类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

ENTITY_TYPES = {
    "Disease": "疾病",
    "Symptom": "症状",
    "Drug": "药物",
    "Treatment": "治疗方式",
    "Examination": "检查",
    "BodyPart": "部位",
    "Population": "人群",
}

RELATION_TYPES = {
    "HAS_SYMPTOM": ("Disease", "Symptom", "疾病-临床表现"),
    "TREATED_BY_DRUG": ("Disease", "Drug", "疾病-药物治疗"),
    "TREATED_BY": ("Disease", "Treatment", "疾病-治疗方式"),
    "DIAGNOSED_BY": ("Disease", "Examination", "疾病-检查诊断"),
    "COMPLICATION_OF": ("Disease", "Disease", "并发症"),
    "AFFECTS": ("Disease", "BodyPart", "疾病-累及部位"),
    "HIGH_RISK_GROUP": ("Disease", "Population", "疾病-高发人群"),
    "CONTRAINDICATED": ("Drug", "Population", "药物-禁忌人群"),
}


@dataclass(frozen=True)
class Entity:
    text: str
    etype: str
    start: int = -1
    end: int = -1
    norm: str = ""          # 归一化后的规范名（实体链接结果）
    link_method: str = ""   # exact|alias|normalized|fuzzy|none

    @property
    def canonical(self) -> str:
        return self.norm or self.text

    def key(self) -> tuple[str, str]:
        return (self.canonical, self.etype)


@dataclass(frozen=True)
class Triple:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    source: str = ""
    confidence: float = 1.0
    evidence: str = ""
    assertion: str = "present"  # present|negated|uncertain（断言状态，否定检测结果）

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


@dataclass
class ValidationReport:
    total: int = 0
    valid: int = 0
    dropped_schema: int = 0
    dropped_self_loop: int = 0
    dropped_duplicate: int = 0
    dropped_negated: int = 0
    issues: list[str] = field(default_factory=list)
