from __future__ import annotations

from collections import Counter
from typing import Any

from medgraph_agent.core.models import GraphSnapshot
from medgraph_agent.operators.medical_extraction import ALLOWED_RELATIONS

FORBIDDEN_ENTITY_NAMES = {"糖尿病", "慢阻肺"}
KNOWN_FALSE_TRIPLES = {
    ("高血压", "treated_by", "阿司匹林"),
    ("高血压", "treated_by", "阿托伐他汀"),
    ("高血压", "has_symptom", "多尿"),
    ("高血压", "has_symptom", "乏力"),
}


def audit_graph(graph: GraphSnapshot) -> dict[str, Any]:
    entity_by_id = {entity.id: entity for entity in graph.entities}
    entity_names = [entity.name for entity in graph.entities]
    duplicate_names = sorted(name for name, count in Counter(entity_names).items() if count > 1)
    forbidden_names = sorted(name for name in entity_names if name in FORBIDDEN_ENTITY_NAMES)

    dangling_relations = []
    schema_violations = []
    empty_evidence = []
    false_triples = []
    unique_triples = set()

    for relation in graph.relations:
        subject = entity_by_id.get(relation.subject_id)
        obj = entity_by_id.get(relation.object_id)
        if not subject or not obj:
            dangling_relations.append(relation.id)
            continue
        allowed_subjects, allowed_objects = ALLOWED_RELATIONS[relation.predicate]
        if subject.type not in allowed_subjects or obj.type not in allowed_objects:
            schema_violations.append(
                {
                    "id": relation.id,
                    "subject": subject.name,
                    "subject_type": subject.type,
                    "predicate": relation.predicate,
                    "object": obj.name,
                    "object_type": obj.type,
                }
            )
        if not relation.evidence.strip():
            empty_evidence.append(relation.id)
        triple = (relation.subject_name, relation.predicate, relation.object_name)
        unique_triples.add(triple)
        if triple in KNOWN_FALSE_TRIPLES:
            false_triples.append({"subject": triple[0], "predicate": triple[1], "object": triple[2]})

    total_relations = len(graph.relations)
    evidence_coverage = 1.0 if total_relations == 0 else (total_relations - len(empty_evidence)) / total_relations
    passed = not any([duplicate_names, forbidden_names, dangling_relations, schema_violations, empty_evidence, false_triples])

    return {
        "passed": passed,
        "entity_count": len(graph.entities),
        "relation_count": total_relations,
        "duplicate_entity_names": duplicate_names,
        "forbidden_entity_names": forbidden_names,
        "dangling_relations": dangling_relations,
        "schema_violations": schema_violations,
        "empty_evidence_relations": empty_evidence,
        "known_false_triples": false_triples,
        "unique_relation_triples": len(unique_triples),
        "evidence_coverage": round(evidence_coverage, 4),
        "checks": {
            "no_duplicate_entity_names": not duplicate_names,
            "no_forbidden_alias_entities": not forbidden_names,
            "no_dangling_relations": not dangling_relations,
            "schema_valid": not schema_violations,
            "all_relations_have_evidence": not empty_evidence,
            "known_false_triples_absent": not false_triples,
        },
    }
