"""Medical knowledge graph construction for task 2."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_medical_graph(
    triples: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deduplicated node/edge graph from valid triples."""

    record_lookup = {record["record_id"]: record for record in records}
    nodes: dict[str, dict[str, Any]] = {}
    edge_records: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    edge_confidence: dict[tuple[str, str, str], float] = {}
    edge_evidence: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for triple in triples:
        source_id = _node_id(triple["subject_type"], triple["subject"])
        target_id = _node_id(triple["object_type"], triple["object"])
        _touch_node(nodes, source_id, triple["subject"], triple["subject_type"], triple["record_id"])
        _touch_node(nodes, target_id, triple["object"], triple["object_type"], triple["record_id"])

        edge_key = (source_id, triple["predicate"], target_id)
        edge_records[edge_key].add(triple["record_id"])
        edge_confidence[edge_key] = max(edge_confidence.get(edge_key, 0.0), triple["confidence"])
        evidence = triple.get("evidence")
        if evidence and evidence not in edge_evidence[edge_key]:
            edge_evidence[edge_key].append(evidence)

    edges = [
        {
            "id": f"{source}-{predicate}-{target}",
            "source": source,
            "target": target,
            "predicate": predicate,
            "record_ids": sorted(records_),
            "confidence": round(edge_confidence[(source, predicate, target)], 2),
            "evidence": edge_evidence[(source, predicate, target)],
        }
        for (source, predicate, target), records_ in sorted(edge_records.items())
    ]

    sorted_nodes = sorted(nodes.values(), key=lambda item: (item["type"], item["name"]))
    for node in sorted_nodes:
        node["record_ids"] = sorted(node["record_ids"])

    return {
        "nodes": sorted_nodes,
        "edges": edges,
        "triples": triples,
        "records": [
            {
                "record_id": record_id,
                "text": record.get("text", ""),
            }
            for record_id, record in sorted(record_lookup.items())
        ],
        "statistics": {
            "node_count": len(sorted_nodes),
            "edge_count": len(edges),
            "triple_count": len(triples),
            "record_count": len(records),
        },
    }


def _touch_node(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    name: str,
    entity_type: str,
    record_id: str,
) -> None:
    if node_id not in nodes:
        nodes[node_id] = {
            "id": node_id,
            "name": name,
            "type": entity_type,
            "record_ids": set(),
            "mention_count": 0,
        }
    nodes[node_id]["record_ids"].add(record_id)
    nodes[node_id]["mention_count"] += 1


def _node_id(entity_type: str, name: str) -> str:
    return f"{entity_type}:{name}"
