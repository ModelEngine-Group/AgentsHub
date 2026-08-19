"""Statistical graph analysis operators for task 3."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def generate_statistical_summary(graph: dict[str, Any]) -> dict[str, Any]:
    """Compute graph-level descriptive statistics."""

    entity_counts = Counter(node.get("type", "Unknown") for node in graph.get("nodes", []))
    relation_counts = Counter(edge.get("predicate", "unknown") for edge in graph.get("edges", []))
    degrees: dict[str, int] = defaultdict(int)
    confidence_values = []

    for edge in graph.get("edges", []):
        degrees[edge.get("source", "")] += 1
        degrees[edge.get("target", "")] += 1
        confidence_values.append(float(edge.get("confidence", 0.0)))

    node_names = {node["id"]: node["name"] for node in graph.get("nodes", [])}
    top_degree_nodes = [
        {
            "node_id": node_id,
            "name": node_names.get(node_id, node_id),
            "degree": degree,
        }
        for node_id, degree in sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:10]
        if node_id
    ]

    return {
        "status": "completed",
        "entity_type_counts": dict(sorted(entity_counts.items())),
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "top_degree_nodes": top_degree_nodes,
        "confidence": {
            "average": round(mean(confidence_values), 4) if confidence_values else 0.0,
            "min": min(confidence_values) if confidence_values else 0.0,
            "max": max(confidence_values) if confidence_values else 0.0,
        },
        "graph_size": {
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "record_count": graph.get("statistics", {}).get("record_count", 0),
        },
    }
