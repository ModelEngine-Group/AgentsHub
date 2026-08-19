"""Quality reporting for task 2 medical KG runs."""

from __future__ import annotations

from typing import Any

from src.operators.kg_ops.relation_extractor import RELATION_SCHEMA


def build_kg_quality_report(
    extraction: dict[str, Any],
    validation: dict[str, Any],
    graph: dict[str, Any],
    qa: dict[str, Any],
    export: dict[str, Any],
) -> dict[str, Any]:
    """Summarize task-2 run evidence for demos and review."""

    stats = graph.get("statistics", {})
    node_count = stats.get("node_count", 0)
    edge_count = stats.get("edge_count", 0)
    entity_total = sum(extraction.get("entity_counts", {}).values())
    max_edges = node_count * max(node_count - 1, 0)
    graph_density = round(edge_count / max_edges, 4) if max_edges else 0.0
    evidence_edges = sum(1 for edge in graph.get("edges", []) if edge.get("evidence"))
    relation_types = {edge.get("predicate") for edge in graph.get("edges", []) if edge.get("predicate")}
    relation_schema_size = len(RELATION_SCHEMA)
    confidences = [
        float(edge.get("confidence", 0.0))
        for edge in graph.get("edges", [])
        if isinstance(edge.get("confidence"), int | float)
    ]
    average_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    readiness = {
        "records_loaded": extraction.get("record_count", 0) > 0,
        "entities_extracted": entity_total > 0,
        "triples_validated": validation.get("status") in {"passed", "warning"},
        "graph_built": node_count > 0 and edge_count > 0,
        "graph_exported": export.get("status") == "completed" and bool(export.get("output_path")),
        "qa_answered": qa.get("status") == "answered",
        "evidence_attached": evidence_edges == edge_count and edge_count > 0,
        "relation_schema_covered": len(relation_types) >= 3,
    }
    status = "passed" if all(readiness.values()) else "warning"
    if not readiness["records_loaded"] or not readiness["graph_built"]:
        status = "failed"

    return {
        "status": status,
        "metrics": {
            "record_count": extraction.get("record_count", 0),
            "entity_total": entity_total,
            "valid_triple_count": validation.get("valid_count", 0),
            "invalid_triple_count": validation.get("invalid_count", 0),
            "node_count": node_count,
            "edge_count": edge_count,
            "triple_count": stats.get("triple_count", 0),
            "graph_density": graph_density,
            "evidence_edge_count": evidence_edges,
            "relation_type_count": len(relation_types),
            "relation_coverage": round(len(relation_types) / relation_schema_size, 4),
            "average_confidence": average_confidence,
        },
        "readiness": readiness,
    }
