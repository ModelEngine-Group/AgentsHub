from __future__ import annotations

from medgraph_agent.core.models import Answer, GraphSnapshot


def answer_question(question: str, graph: GraphSnapshot) -> Answer:
    question = question.strip()
    plan = ["识别问题中的医疗实体", "按关系类型检索图谱", "汇总证据并生成可解释答案"]
    entities = [entity for entity in graph.entities if entity.name in question]
    evidence_relations = []

    relation_hints = {
        "症状": "has_symptom",
        "表现": "has_symptom",
        "治疗": "treated_by",
        "用药": "treated_by",
        "药": "treated_by",
        "检查": "diagnosed_by",
        "诊断": "diagnosed_by",
        "并发": "complicates",
        "禁忌": "contraindicated_with",
        "科室": "belongs_to_department",
        "风险": "has_risk_factor",
    }
    wanted = {predicate for hint, predicate in relation_hints.items() if hint in question}

    for relation in graph.relations:
        touches_entity = not entities or any(
            entity.id in {relation.subject_id, relation.object_id} for entity in entities
        )
        touches_predicate = not wanted or relation.predicate in wanted
        if touches_entity and touches_predicate:
            evidence_relations.append(relation)

    if not evidence_relations and entities:
        for relation in graph.relations:
            if any(entity.id in {relation.subject_id, relation.object_id} for entity in entities):
                evidence_relations.append(relation)

    seen_triples: set[tuple[str, str, str]] = set()
    deduped = []
    for relation in evidence_relations:
        triple = (relation.subject_id, relation.predicate, relation.object_id)
        if triple in seen_triples:
            continue
        seen_triples.add(triple)
        deduped.append(relation)
    evidence_relations = sorted(
        deduped,
        key=lambda relation: (
            0 if any(entity.id == relation.subject_id for entity in entities) else 1,
            -relation.confidence,
            relation.predicate_label,
            relation.object_name,
        ),
    )[:8]
    if evidence_relations:
        grouped: dict[str, list[str]] = {}
        for relation in evidence_relations:
            key = f"{relation.subject_name}的{relation.predicate_label}"
            grouped.setdefault(key, [])
            if relation.object_name not in grouped[key]:
                grouped[key].append(relation.object_name)
        parts = [f"{key}包括{', '.join(values)}" for key, values in grouped.items()]
        answer = "；".join(parts) + "。"
        confidence = min(0.95, 0.62 + 0.04 * len(evidence_relations))
    else:
        answer = "当前图谱没有找到直接证据。建议补充相关病例、指南或结构化表后重新运行流水线。"
        confidence = 0.25

    return Answer(
        question=question,
        answer=answer,
        confidence=round(confidence, 4),
        evidence=[
            {
                "subject": relation.subject_name,
                "predicate": relation.predicate_label,
                "object": relation.object_name,
                "evidence": relation.evidence,
                "confidence": relation.confidence,
            }
            for relation in evidence_relations
        ],
        plan=plan,
    )
