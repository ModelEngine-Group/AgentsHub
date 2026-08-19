"""Read-only query operators for task 2 medical knowledge graphs."""

from __future__ import annotations

from typing import Any


def find_graph_entities(
    query: str,
    graph: dict[str, Any],
    entity_type: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Find graph nodes by exact or substring match."""

    normalized_query = _normalize(query)
    if not normalized_query:
        return {
            "status": "unmatched",
            "query": query,
            "entity_type": entity_type,
            "matches": [],
        }

    matches = []
    for node in graph.get("nodes", []):
        if entity_type and node.get("type") != entity_type:
            continue

        match_type = _match_type(normalized_query, node)
        if not match_type:
            continue

        matches.append(
            {
                "id": node["id"],
                "name": node["name"],
                "type": node["type"],
                "record_ids": node.get("record_ids", []),
                "mention_count": node.get("mention_count", 0),
                "match_type": match_type,
                "score": _match_score(match_type, normalized_query, node["name"]),
            }
        )

    matches = sorted(matches, key=lambda item: (-item["score"], item["type"], item["name"]))[:limit]
    return {
        "status": "matched" if matches else "unmatched",
        "query": query,
        "entity_type": entity_type,
        "matches": matches,
    }


def query_graph_neighbors(
    entity: str,
    graph: dict[str, Any],
    relation: str | None = None,
    direction: str = "out",
    limit: int = 20,
) -> dict[str, Any]:
    """Return neighboring nodes connected to a matched entity."""

    if direction not in {"out", "in", "both"}:
        return {
            "status": "failed",
            "entity": entity,
            "message": "direction must be one of: out, in, both",
            "neighbors": [],
        }

    entity_match = find_graph_entities(entity, graph, limit=1)
    if not entity_match["matches"]:
        return {
            "status": "unmatched",
            "entity": entity,
            "relation": relation,
            "direction": direction,
            "neighbors": [],
        }

    selected = entity_match["matches"][0]
    node_lookup = {node["id"]: node for node in graph.get("nodes", [])}
    neighbors = []
    for edge in graph.get("edges", []):
        if relation and edge.get("predicate") != relation:
            continue

        neighbor_direction = _edge_direction(edge, selected["id"], direction)
        if not neighbor_direction:
            continue

        neighbor_id = edge["target"] if neighbor_direction == "out" else edge["source"]
        neighbor_node = node_lookup.get(neighbor_id, {"id": neighbor_id, "name": neighbor_id, "type": "Unknown"})
        neighbors.append(
            {
                "edge_id": edge["id"],
                "predicate": edge["predicate"],
                "direction": neighbor_direction,
                "source": node_lookup.get(edge["source"], {"id": edge["source"], "name": edge["source"]}),
                "target": neighbor_node,
                "record_ids": edge.get("record_ids", []),
                "confidence": edge.get("confidence", 0.0),
                "evidence": edge.get("evidence", []),
            }
        )

    neighbors = sorted(
        neighbors,
        key=lambda item: (-item["confidence"], item["predicate"], item["target"]["name"]),
    )[:limit]
    return {
        "status": "matched" if neighbors else "unmatched",
        "entity": entity,
        "matched_entity": selected,
        "relation": relation,
        "direction": direction,
        "neighbors": neighbors,
    }


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _match_type(query: str, node: dict[str, Any]) -> str | None:
    name = _normalize(node.get("name"))
    node_id = _normalize(node.get("id"))
    if query in {name, node_id}:
        return "exact"
    if query in name or name in query:
        return "substring"
    return None


def _match_score(match_type: str, query: str, name: str) -> float:
    if match_type == "exact":
        return 1.0
    normalized_name = _normalize(name)
    if not normalized_name:
        return 0.0
    return round(min(len(query), len(normalized_name)) / max(len(query), len(normalized_name)), 3)


def _edge_direction(edge: dict[str, Any], node_id: str, direction: str) -> str | None:
    if direction in {"out", "both"} and edge.get("source") == node_id:
        return "out"
    if direction in {"in", "both"} and edge.get("target") == node_id:
        return "in"
    return None
