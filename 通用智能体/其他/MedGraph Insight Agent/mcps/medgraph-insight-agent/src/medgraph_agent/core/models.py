from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Literal

EntityType = Literal[
    "disease",
    "symptom",
    "drug",
    "test",
    "treatment",
    "department",
    "risk_factor",
]

RelationType = Literal[
    "has_symptom",
    "treated_by",
    "diagnosed_by",
    "complicates",
    "contraindicated_with",
    "belongs_to_department",
    "has_risk_factor",
]

RunStatus = Literal["pending", "running", "succeeded", "failed"]

ENTITY_LABELS: dict[str, str] = {
    "disease": "疾病",
    "symptom": "症状",
    "drug": "药物",
    "test": "检查",
    "treatment": "治疗",
    "department": "科室",
    "risk_factor": "风险因素",
}

RELATION_LABELS: dict[str, str] = {
    "has_symptom": "症状",
    "treated_by": "治疗",
    "diagnosed_by": "诊断",
    "complicates": "并发",
    "contraindicated_with": "禁忌",
    "belongs_to_department": "科室归属",
    "has_risk_factor": "风险因素",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_id(prefix: str, *parts: str) -> str:
    raw = "||".join(parts)
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class DataRecord:
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: EntityType
    label: str
    confidence: float
    source_record_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Relation:
    id: str
    subject_id: str
    subject_name: str
    predicate: RelationType
    predicate_label: str
    object_id: str
    object_name: str
    evidence: str
    source_record_id: str
    confidence: float


@dataclass(frozen=True)
class OperatorResult:
    name: str
    status: RunStatus
    started_at: str
    finished_at: str
    records_processed: int = 0
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    operator: str
    intent: str
    depends_on: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowPlan:
    id: str
    task: str
    created_at: str
    steps: list[WorkflowStep]


@dataclass
class PipelineRun:
    id: str
    task: str
    source: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    plan: WorkflowPlan | None = None
    operator_results: list[OperatorResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class GraphSnapshot:
    entities: list[Entity]
    relations: list[Relation]
    generated_at: str
    source_record_count: int

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_relation: dict[str, int] = {}
        for entity in self.entities:
            by_type[entity.type] = by_type.get(entity.type, 0) + 1
        for relation in self.relations:
            by_relation[relation.predicate] = by_relation.get(relation.predicate, 0) + 1
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "source_record_count": self.source_record_count,
            "entities_by_type": by_type,
            "relations_by_type": by_relation,
        }


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    confidence: float
    evidence: list[dict[str, Any]]
    plan: list[str]


@dataclass(frozen=True)
class AnalysisResult:
    question: str
    intent: str
    sql: str
    rows: list[dict[str, Any]]
    chart: dict[str, Any]
    narrative: str
    confidence: float


@dataclass(frozen=True)
class BackendResult:
    backend: str
    available: bool
    latency_ms: float | None
    throughput_items_per_second: float | None
    notes: str


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
