"""Record-sequence trend analysis operators for task 3."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def generate_trend_analysis(graph: dict[str, Any]) -> dict[str, Any]:
    """Summarize graph changes across record order.

    Task-2 samples do not include real timestamps, so record IDs are treated
    as a reproducible sequence axis.
    """

    record_edges: dict[str, Counter[str]] = defaultdict(Counter)
    record_entities: dict[str, Counter[str]] = defaultdict(Counter)

    for edge in graph.get("edges", []):
        for record_id in edge.get("record_ids", []):
            record_edges[record_id][edge.get("predicate", "unknown")] += 1

    for node in graph.get("nodes", []):
        for record_id in node.get("record_ids", []):
            record_entities[record_id][node.get("type", "Unknown")] += 1

    record_ids = sorted(set(record_edges) | set(record_entities), key=_record_sort_key)
    trends = []
    for record_id in record_ids:
        edge_count = sum(record_edges[record_id].values())
        entity_count = sum(record_entities[record_id].values())
        trends.append(
            {
                "record_id": record_id,
                "entity_count": entity_count,
                "edge_count": edge_count,
                "relation_counts": dict(sorted(record_edges[record_id].items())),
                "entity_type_counts": dict(sorted(record_entities[record_id].items())),
            }
        )

    return {
        "status": "completed",
        "axis": "record_sequence",
        "record_trends": trends,
        "peak_record": max(trends, key=lambda item: item["edge_count"], default=None),
    }


def _record_sort_key(record_id: str) -> tuple[int, str]:
    suffix = record_id.rsplit("_", 1)[-1]
    return (int(suffix), record_id) if suffix.isdigit() else (999999, record_id)
