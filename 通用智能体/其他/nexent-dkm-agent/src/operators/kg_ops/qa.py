"""Graph-backed question answering for task 2.

Supports disease-centric, drug-centric, symptom-centric, examination-centric,
and treatment-centric queries.
"""

from __future__ import annotations

from typing import Any

QUESTION_INTENTS = {
    "symptom": ("症状", "表现", "主诉", "symptom"),
    "drug": ("用药", "药", "服用", "drug", "medication"),
    "examination": ("检查", "诊断", "化验", "exam"),
    "treatment": ("治疗", "处理", "建议", "treatment"),
    "complication": ("并发", "合并", "并发症", "complication"),
}

INTENT_RELATIONS = {
    "symptom": {"has_symptom"},
    "drug": {"treated_by"},
    "examination": {"diagnosed_by"},
    "treatment": {"treated_by", "recommended_treatment"},
    "complication": {"complication_of"},
}

# Entity types that can serve as the query center
_ENTITY_TYPES = ("Disease", "Drug", "Symptom", "Examination", "Treatment")

# Reverse relation mapping for non-Disease queries
# For Drug/Symptom/Exam/Treatment, we look for edges where the entity is the target
_REVERSE_RELATIONS = {
    "Drug": {"treated_by"},
    "Symptom": {"has_symptom"},
    "Examination": {"diagnosed_by"},
    "Treatment": {"treated_by", "recommended_treatment"},
}

_ENTITY_LABELS = {
    "Disease": "疾病",
    "Drug": "药物",
    "Symptom": "症状",
    "Examination": "检查",
    "Treatment": "治疗方式",
}


def answer_graph_question(question: str | None, graph: dict[str, Any]) -> dict[str, Any]:
    """Answer a question using graph evidence, supporting multiple entity types."""

    if not question:
        return {
            "status": "skipped",
            "question": question,
            "answer": "No question provided.",
            "evidence": [],
        }

    # Try disease-centric first (original behavior)
    entity, entity_type = _find_entity(question, graph)

    if not entity:
        return {
            "status": "unanswered",
            "question": question,
            "answer": "未在图谱中找到问题对应的实体。",
            "evidence": [],
        }

    if entity_type == "Disease":
        return _answer_disease_centric(question, graph, entity)

    return _answer_entity_centric(question, graph, entity, entity_type)


def _answer_disease_centric(
    question: str, graph: dict[str, Any], disease: dict[str, Any],
) -> dict[str, Any]:
    """Original disease-centric QA logic."""
    intents = _detect_intents(question)
    relation_filter = set()
    for intent in intents:
        relation_filter.update(INTENT_RELATIONS[intent])
    if not relation_filter:
        relation_filter = {
            "has_symptom", "treated_by", "diagnosed_by",
            "recommended_treatment", "complication_of",
        }

    evidence = [
        edge for edge in graph.get("edges", [])
        if _edge_matches_disease(edge, disease["id"], relation_filter)
    ]
    if not evidence:
        return {
            "status": "unanswered",
            "question": question,
            "answer": f"图谱中暂未找到{disease['name']}的相关关系。",
            "evidence": [],
        }

    grouped = _group_targets(evidence, graph, disease["id"])
    parts = []
    for label, values in grouped.items():
        if values:
            parts.append(f"{label}: {'、'.join(values)}")

    return {
        "status": "answered",
        "question": question,
        "answer": f"{disease['name']}相关信息：" + "；".join(parts) + "。",
        "evidence": evidence,
    }


def _answer_entity_centric(
    question: str, graph: dict[str, Any], entity: dict[str, Any], entity_type: str,
) -> dict[str, Any]:
    """Answer questions centered on non-Disease entities (Drug, Symptom, etc.)."""
    entity_id = entity["id"]
    entity_name = entity["name"]
    type_label = _ENTITY_LABELS.get(entity_type, entity_type)

    reverse_preds = _REVERSE_RELATIONS.get(entity_type, set())

    # Find all edges where this entity is the target
    related_edges = [
        edge for edge in graph.get("edges", [])
        if edge.get("target") == entity_id and edge.get("predicate") in reverse_preds
    ]

    # Also find edges where this entity is the source (e.g., complication_of direction)
    source_edges = [
        edge for edge in graph.get("edges", [])
        if edge.get("source") == entity_id
    ]

    all_evidence = related_edges + source_edges
    if not all_evidence:
        return {
            "status": "unanswered",
            "question": question,
            "answer": f"图谱中暂未找到{type_label}「{entity_name}」的相关关系。",
            "evidence": [],
        }

    # Build answer
    node_names = {node["id"]: node["name"] for node in graph.get("nodes", [])}
    node_types = {node["id"]: node.get("type", "") for node in graph.get("nodes", [])}
    relation_labels = {
        "has_symptom": "相关症状",
        "treated_by": "治疗疾病",
        "diagnosed_by": "诊断疾病",
        "recommended_treatment": "推荐治疗",
        "complication_of": "并发症",
    }

    parts = []
    for edge in all_evidence:
        pred = edge.get("predicate", "")
        if edge.get("target") == entity_id:
            # Entity is the target, source is the related node
            related_id = edge.get("source", "")
            related_name = node_names.get(related_id, related_id)
            related_type = node_types.get(related_id, "")
            pred_label = relation_labels.get(pred, pred)
            related_label = _ENTITY_LABELS.get(related_type, related_type)
            parts.append(f"{pred_label}：{related_label}「{related_name}」")
        elif edge.get("source") == entity_id:
            related_id = edge.get("target", "")
            related_name = node_names.get(related_id, related_id)
            related_type = node_types.get(related_id, "")
            pred_label = relation_labels.get(pred, pred)
            related_label = _ENTITY_LABELS.get(related_type, related_type)
            parts.append(f"{pred_label}：{related_label}「{related_name}」")

    if not parts:
        return {
            "status": "unanswered",
            "question": question,
            "answer": f"图谱中暂未找到{type_label}「{entity_name}」的相关信息。",
            "evidence": [],
        }

    return {
        "status": "answered",
        "question": question,
        "answer": f"{type_label}「{entity_name}」相关信息：" + "；".join(parts) + "。",
        "evidence": all_evidence,
    }


def _find_entity(
    question: str, graph: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Find the best matching entity from the graph across all types.

    Prioritizes Disease for backward compatibility, then searches
    Drug, Symptom, Examination, Treatment.
    """
    for entity_type in _ENTITY_TYPES:
        nodes = [n for n in graph.get("nodes", []) if n.get("type") == entity_type]
        for node in sorted(nodes, key=lambda n: len(n["name"]), reverse=True):
            name = node["name"]
            if name in question or question in name:
                return node, entity_type
            # Handle partial matches for common medical terms
            if "糖尿病" in question and "糖尿病" in name:
                return node, entity_type
            if "高血压" in question and "血压" in name:
                return node, entity_type
    return None, None


def _detect_intents(question: str) -> list[str]:
    intents = [
        intent
        for intent, triggers in QUESTION_INTENTS.items()
        if any(trigger in question.lower() for trigger in triggers)
    ]
    return intents


def _edge_matches_disease(edge: dict[str, Any], disease_id: str, relation_filter: set[str]) -> bool:
    predicate = edge.get("predicate")
    if predicate not in relation_filter:
        return False
    if predicate == "complication_of":
        return edge.get("target") == disease_id
    return edge.get("source") == disease_id


def _group_targets(
    evidence: list[dict[str, Any]],
    graph: dict[str, Any],
    disease_id: str,
) -> dict[str, list[str]]:
    node_names = {node["id"]: node["name"] for node in graph.get("nodes", [])}
    grouped = {
        "症状": [],
        "用药": [],
        "检查": [],
        "治疗建议": [],
        "并发症": [],
    }
    relation_labels = {
        "has_symptom": "症状",
        "treated_by": "用药",
        "diagnosed_by": "检查",
        "recommended_treatment": "治疗建议",
        "complication_of": "并发症",
    }
    for edge in evidence:
        label = relation_labels.get(edge["predicate"])
        target_id = edge["source"] if edge["predicate"] == "complication_of" and edge["target"] == disease_id else edge["target"]
        target_name = node_names.get(target_id, target_id)
        if label and target_name not in grouped[label]:
            grouped[label].append(target_name)
    return grouped
