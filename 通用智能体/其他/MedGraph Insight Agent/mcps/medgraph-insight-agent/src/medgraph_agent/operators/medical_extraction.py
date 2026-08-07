from __future__ import annotations

import re
from typing import Any

from medgraph_agent.core.knowledge_graph import KnowledgeGraphBuilder
from medgraph_agent.core.models import Entity, EntityType, GraphSnapshot, RelationType, utc_now
from medgraph_agent.operators.base import Operator, fail_result, ok_result

LEXICON: dict[EntityType, list[str]] = {
    "disease": [
        "高血压",
        "2型糖尿病",
        "糖尿病",
        "冠心病",
        "慢性阻塞性肺疾病",
        "慢阻肺",
        "肺炎",
        "哮喘",
        "胃食管反流病",
        "抑郁障碍",
        "脑卒中",
        "慢性肾病",
        "甲状腺功能减退",
        "高脂血症",
        "心力衰竭",
    ],
    "symptom": [
        "头痛",
        "头晕",
        "乏力",
        "多饮",
        "多尿",
        "胸痛",
        "气促",
        "咳嗽",
        "咳痰",
        "喘息",
        "发热",
        "反酸",
        "烧心",
        "情绪低落",
        "失眠",
        "肢体无力",
        "水肿",
        "畏寒",
        "心悸",
    ],
    "drug": [
        "硝苯地平",
        "二甲双胍",
        "阿司匹林",
        "阿托伐他汀",
        "沙丁胺醇",
        "布地奈德",
        "奥美拉唑",
        "左氧氟沙星",
        "左甲状腺素",
        "呋塞米",
    ],
    "test": [
        "血压监测",
        "空腹血糖",
        "糖化血红蛋白",
        "心电图",
        "胸部CT",
        "肺功能检查",
        "胃镜",
        "头颅CT",
        "尿白蛋白",
        "甲状腺功能",
        "血脂",
    ],
    "treatment": [
        "低盐饮食",
        "运动干预",
        "胰岛素治疗",
        "吸氧",
        "抗感染治疗",
        "心理治疗",
        "康复训练",
        "戒烟",
    ],
    "department": [
        "心内科",
        "内分泌科",
        "呼吸科",
        "消化科",
        "神经内科",
        "肾内科",
        "精神心理科",
        "全科医学科",
    ],
    "risk_factor": [
        "肥胖",
        "吸烟",
        "高盐饮食",
        "家族史",
        "年龄增长",
        "长期饮酒",
        "久坐",
        "免疫力低下",
    ],
}

CANONICAL_TERMS: dict[tuple[EntityType, str], str] = {
    ("disease", "慢阻肺"): "慢性阻塞性肺疾病",
}

ALLOWED_RELATIONS: dict[RelationType, tuple[set[EntityType], set[EntityType]]] = {
    "has_symptom": ({"disease"}, {"symptom"}),
    "treated_by": ({"disease"}, {"drug", "treatment"}),
    "diagnosed_by": ({"disease"}, {"test"}),
    "complicates": ({"disease"}, {"disease"}),
    "contraindicated_with": ({"drug", "treatment"}, {"disease", "risk_factor"}),
    "belongs_to_department": ({"disease"}, {"department"}),
    "has_risk_factor": ({"disease"}, {"risk_factor"}),
}


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。；;！？!?]\s*", text) if part.strip()]


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < used_end and end > used_start for used_start, used_end in occupied)


def _canonical_name(entity_type: EntityType, term: str) -> str:
    return CANONICAL_TERMS.get((entity_type, term), term)


def find_entities(text: str, record_id: str) -> list[Entity]:
    builder = KnowledgeGraphBuilder()
    occupied: list[tuple[int, int]] = []
    candidates = [
        (entity_type, term)
        for entity_type, terms in LEXICON.items()
        for term in terms
    ]
    for entity_type, term in sorted(candidates, key=lambda item: len(item[1]), reverse=True):
        for match in re.finditer(re.escape(term), text):
            span = match.span()
            if _overlaps(span, occupied):
                continue
            occupied.append(span)
            confidence = 0.97 if len(term) >= 4 else 0.9
            builder.add_entity(_canonical_name(entity_type, term), entity_type, record_id, confidence)
    return builder.snapshot(source_record_count=1).entities


def _sentence_mentions(sentence: str, entities: list[Entity]) -> list[Entity]:
    return [entity for entity in entities if entity.name in sentence]


def _position(sentence: str, entity: Entity) -> int:
    pos = sentence.find(entity.name)
    return pos if pos >= 0 else 10**9


def _by_type(mentions: list[Entity], entity_type: EntityType) -> list[Entity]:
    return [entity for entity in mentions if entity.type == entity_type]


def _objects(mentions: list[Entity], entity_types: set[EntityType]) -> list[Entity]:
    return [entity for entity in mentions if entity.type in entity_types]


def _primary_disease_subjects(sentence: str, mentions: list[Entity], trigger_tokens: list[str]) -> list[Entity]:
    diseases = sorted(_by_type(mentions, "disease"), key=lambda item: _position(sentence, item))
    if not diseases:
        return []
    trigger_positions = [sentence.find(token) for token in trigger_tokens if token in sentence]
    if not trigger_positions:
        return diseases
    trigger_pos = min(trigger_positions)
    before_trigger = [disease for disease in diseases if _position(sentence, disease) <= trigger_pos]
    return before_trigger[:1] or diseases[:1]


def _add_pairwise(
    builder: KnowledgeGraphBuilder,
    mentions: list[Entity],
    predicate: RelationType,
    evidence: str,
    record_id: str,
    confidence: float,
) -> None:
    subject_types, object_types = ALLOWED_RELATIONS[predicate]
    subjects = [entity for entity in mentions if entity.type in subject_types]
    objects = [entity for entity in mentions if entity.type in object_types]
    for subject in subjects:
        for obj in objects:
            if subject.id == obj.id:
                continue
            builder.add_relation(subject, predicate, obj, evidence, record_id, confidence)


def _add_from_subjects(
    builder: KnowledgeGraphBuilder,
    subjects: list[Entity],
    objects: list[Entity],
    predicate: RelationType,
    evidence: str,
    record_id: str,
    confidence: float,
) -> None:
    for subject in subjects:
        for obj in objects:
            if subject.id != obj.id:
                builder.add_relation(subject, predicate, obj, evidence, record_id, confidence)


def _add_schema_guided_relations(
    builder: KnowledgeGraphBuilder,
    sentence: str,
    mentions: list[Entity],
    record_id: str,
) -> None:
    symptom_subjects = _primary_disease_subjects(sentence, mentions, ["表现", "出现", "常见", "症状"])
    treatment_subjects = _primary_disease_subjects(sentence, mentions, ["治疗", "管理", "采用", "用于", "计划"])
    diagnosis_subjects = _primary_disease_subjects(sentence, mentions, ["诊断", "检查", "筛查", "监测", "发现"])
    department_subjects = _primary_disease_subjects(sentence, mentions, ["归属", "负责", "随访", "管理", "评估"])
    risk_subjects = _primary_disease_subjects(sentence, mentions, ["风险", "相关", "增加", "加重", "由"])

    if any(token in sentence for token in ["症状", "表现", "出现", "常见"]):
        _add_from_subjects(builder, symptom_subjects, _objects(mentions, {"symptom"}), "has_symptom", sentence, record_id, 0.88)
    if any(token in sentence for token in ["治疗", "管理", "采用", "用于", "计划"]):
        _add_from_subjects(
            builder,
            treatment_subjects,
            _objects(mentions, {"drug", "treatment"}),
            "treated_by",
            sentence,
            record_id,
            0.86,
        )
    if any(token in sentence for token in ["诊断", "检查", "筛查", "监测", "发现"]):
        _add_from_subjects(builder, diagnosis_subjects, _objects(mentions, {"test"}), "diagnosed_by", sentence, record_id, 0.88)
    if any(token in sentence for token in ["归属", "负责", "随访", "评估", "调整剂量"]):
        _add_from_subjects(
            builder,
            department_subjects,
            _objects(mentions, {"department"}),
            "belongs_to_department",
            sentence,
            record_id,
            0.9,
        )
    if any(token in sentence for token in ["风险", "相关", "增加", "加重", "由"]):
        _add_from_subjects(
            builder,
            risk_subjects,
            _objects(mentions, {"risk_factor"}),
            "has_risk_factor",
            sentence,
            record_id,
            0.84,
        )
    if any(token in sentence for token in ["并发", "合并", "风险升高"]):
        subject = _primary_disease_subjects(sentence, mentions, ["并发", "合并", "风险升高"])[:1]
        diseases = sorted(_by_type(mentions, "disease"), key=lambda item: _position(sentence, item))
        objects = [disease for disease in diseases if disease.id not in {item.id for item in subject}]
        _add_from_subjects(builder, subject, objects, "complicates", sentence, record_id, 0.82)
    if any(token in sentence for token in ["禁忌", "慎用", "避免"]):
        _add_from_subjects(
            builder,
            _objects(mentions, {"drug", "treatment"}),
            _objects(mentions, {"disease", "risk_factor"}),
            "contraindicated_with",
            sentence,
            record_id,
            0.8,
        )


def build_graph(records: list[Any]) -> GraphSnapshot:
    builder = KnowledgeGraphBuilder()
    for record in records:
        entities = find_entities(record.text, record.id)
        entity_by_key: dict[tuple[str, str], Entity] = {}
        for entity in entities:
            entity_by_key[(entity.type, entity.name)] = builder.add_entity(
                entity.name, entity.type, record.id, entity.confidence
            )
        merged_entities = list(entity_by_key.values())
        for sentence in split_sentences(record.text):
            mentions = _sentence_mentions(sentence, merged_entities)
            if not mentions:
                continue
            _add_schema_guided_relations(builder, sentence, mentions, record.id)
    return builder.snapshot(source_record_count=len(records))


class EntityRecognitionOperator(Operator):
    name = "entity_recognition"

    def execute(self, context: dict[str, Any]):
        started_at = utc_now()
        try:
            entities_by_record: dict[str, list[dict[str, Any]]] = {}
            for record in context.get("records", []):
                entities_by_record[record.id] = [entity.__dict__ for entity in find_entities(record.text, record.id)]
            context["entities_by_record"] = entities_by_record
            return ok_result(
                self.name,
                started_at,
                records_processed=len(context.get("records", [])),
                output={"entities_by_record": entities_by_record},
            )
        except Exception as exc:
            return fail_result(self.name, started_at, exc)


class RelationExtractionOperator(Operator):
    name = "relation_extraction"

    def execute(self, context: dict[str, Any]):
        started_at = utc_now()
        try:
            graph = build_graph(context.get("records", []))
            context["graph"] = graph
            return ok_result(
                self.name,
                started_at,
                records_processed=graph.source_record_count,
                output={"relation_count": len(graph.relations), "entity_count": len(graph.entities)},
            )
        except Exception as exc:
            return fail_result(self.name, started_at, exc)


class TripleValidationOperator(Operator):
    name = "triple_validation"

    def execute(self, context: dict[str, Any]):
        started_at = utc_now()
        try:
            graph: GraphSnapshot = context["graph"]
            valid_relations = []
            entity_by_id = {entity.id: entity for entity in graph.entities}
            for relation in graph.relations:
                subject = entity_by_id.get(relation.subject_id)
                obj = entity_by_id.get(relation.object_id)
                if not subject or not obj:
                    continue
                subject_types, object_types = ALLOWED_RELATIONS[relation.predicate]
                if subject.type in subject_types and obj.type in object_types and relation.confidence >= 0.5:
                    valid_relations.append(relation)
            validated = GraphSnapshot(
                entities=graph.entities,
                relations=valid_relations,
                generated_at=graph.generated_at,
                source_record_count=graph.source_record_count,
            )
            context["graph"] = validated
            return ok_result(
                self.name,
                started_at,
                records_processed=len(valid_relations),
                output={"valid_triples": len(valid_relations), "graph_stats": validated.stats()},
            )
        except Exception as exc:
            return fail_result(self.name, started_at, exc)
