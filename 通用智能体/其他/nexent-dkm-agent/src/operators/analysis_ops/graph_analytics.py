"""Graph analytics operators for task 3.

Provides centrality analysis, shortest-path analysis, and community
detection over the in-memory medical KG.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)


def compute_degree_topk_npu_cached(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy bridge to the NPU top-k operator to avoid package import cycles."""

    from src.operators.npu_ops.graph_tensor_ops import compute_degree_topk_npu_cached as _impl

    return _impl(*args, **kwargs)


def prepare_graph_degree_tensor_cache(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy bridge to prepare reusable NPU graph-degree tensors."""

    from src.operators.npu_ops.graph_tensor_ops import prepare_graph_degree_tensor_cache as _impl

    return _impl(*args, **kwargs)


def compute_centrality(
    graph: dict[str, Any],
    prefer_device: str = "auto",
    degree_tensor_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute degree centrality for all nodes in the graph.

    Returns the top nodes by degree and type-level centrality summaries.
    """
    return compute_centrality_with_cache(
        graph,
        prefer_device=prefer_device,
        degree_tensor_cache=degree_tensor_cache,
    )


def compute_type_centrality_npu(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy bridge to the NPU type-centrality aggregation operator."""

    from src.operators.npu_ops.graph_tensor_ops import compute_type_centrality_npu as _impl

    return _impl(*args, **kwargs)


def compute_centrality_with_cache(
    graph: dict[str, Any],
    prefer_device: str = "auto",
    degree_tensor_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute centrality while optionally reusing a prepared NPU degree cache."""

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    n = len(nodes)
    max_possible = max(n - 1, 1)

    has_npu_cache = (
        degree_tensor_cache is not None
        and degree_tensor_cache.get("status") == "completed"
        and degree_tensor_cache.get("cache_reusable") is True
    )

    npu_top_hubs = {"status": "not_run"}
    try:
        npu_top_hubs = compute_degree_topk_npu_cached(
            degree_tensor_cache if degree_tensor_cache is not None else graph,
            prefer_device=prefer_device,
            top_k=10,
            kernel="bincount",
        )
    except Exception as exc:
        npu_top_hubs = {"status": "failed", "reason": str(exc) or type(exc).__name__}
        logger.debug("NPU top-hubs operator unavailable; using CPU result", exc_info=True)

    npu_type_result = {"status": "not_run"}
    if has_npu_cache and npu_top_hubs.get("status") == "completed":
        try:
            npu_type_result = compute_type_centrality_npu(
                degree_tensor_cache,
                prefer_device=prefer_device,
                kernel="bincount",
            )
        except Exception:
            npu_type_result = {"status": "failed"}
            logger.debug("NPU type centrality unavailable; falling back to CPU", exc_info=True)

    # Compute CPU degree counts only if needed for fallback
    degree: dict[str, int] | None = None
    if npu_type_result.get("status") != "completed":
        degree = _compute_degree_counts(nodes, edges)
        type_centrality = _build_type_centrality(nodes, degree)
    else:
        type_centrality = npu_type_result["type_centrality"]

    if npu_top_hubs.get("status") == "completed":
        top_hubs = npu_top_hubs.get("top_hubs", [])
        top_hubs_backend = npu_top_hubs.get("backend", "torch_npu")
        top_hubs_npu_reason = None
    else:
        if degree is None:
            degree = _compute_degree_counts(nodes, edges)
        top_hubs = _format_cpu_top_hubs(nodes, degree, max_possible, limit=10)
        top_hubs_backend = "python"
        top_hubs_npu_reason = npu_top_hubs.get("reason", npu_top_hubs.get("status"))

    return {
        "status": "completed",
        "node_count": n,
        "top_hubs": top_hubs,
        "top_hubs_backend": top_hubs_backend,
        "top_hubs_npu_reason": top_hubs_npu_reason,
        "type_centrality": type_centrality,
    }


def _compute_degree_counts(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    index = {node["id"] for node in nodes}
    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src in index and tgt in index:
            degree[src] += 1
            degree[tgt] += 1
    return degree


def _format_cpu_top_hubs(
    nodes: list[dict[str, Any]],
    degree: dict[str, int],
    max_possible: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    centrality_scores = []
    for node in nodes:
        nid = node["id"]
        deg = degree.get(nid, 0)
        centrality_scores.append({
            "id": nid,
            "name": node.get("name", ""),
            "type": node.get("type", ""),
            "degree": deg,
            "degree_centrality": round(deg / max_possible, 4),
        })
    centrality_scores.sort(key=lambda item: (-item["degree"], item["id"]))
    return centrality_scores[:limit]


def _build_type_centrality(
    nodes: list[dict[str, Any]],
    degree: dict[str, int],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_type = node.get("type", "")
        deg = degree.get(node["id"], 0)
        bucket = buckets.setdefault(
            node_type,
            {
                "count": 0,
                "degree_sum": 0,
                "max_degree": 0,
                "top_node": "",
                "top_degree": -1,
            },
        )
        bucket["count"] += 1
        bucket["degree_sum"] += deg
        bucket["max_degree"] = max(bucket["max_degree"], deg)
        if deg > bucket["top_degree"]:
            bucket["top_degree"] = deg
            bucket["top_node"] = node.get("name", "")

    type_centrality: dict[str, dict[str, Any]] = {}
    for node_type, bucket in buckets.items():
        count = bucket["count"]
        type_centrality[node_type] = {
            "count": count,
            "avg_degree": round(bucket["degree_sum"] / max(count, 1), 2),
            "max_degree": bucket["max_degree"],
            "top_node": bucket["top_node"],
        }
    return type_centrality


def compute_shortest_paths(
    graph: dict[str, Any],
    start_entity: str,
    end_entity: str | None = None,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Compute shortest paths from a start entity using BFS.

    If end_entity is given, returns the shortest path between them.
    Otherwise returns all entities reachable within max_depth.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_lookup = {nd["id"]: nd for nd in nodes}

    # Build directed adjacency
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        pred = edge.get("predicate", "")
        adj[src].append((tgt, pred))
        adj[tgt].append((src, pred))

    # Find start node
    start_id = None
    for nd in nodes:
        if nd.get("name", "") == start_entity or nd["id"] == start_entity:
            start_id = nd["id"]
            break

    if not start_id:
        return {
            "status": "unmatched",
            "start_entity": start_entity,
            "paths": [],
        }

    # BFS
    if end_entity:
        end_id = None
        for nd in nodes:
            if nd.get("name", "") == end_entity or nd["id"] == end_entity:
                end_id = nd["id"]
                break

        if not end_id:
            return {
                "status": "unmatched",
                "start_entity": start_entity,
                "end_entity": end_entity,
                "paths": [],
            }

        path = _bfs_path(adj, start_id, end_id, max_depth)
        if not path:
            return {
                "status": "no_path",
                "start_entity": start_entity,
                "end_entity": end_entity,
                "paths": [],
            }

        formatted = _format_path(path, node_lookup)
        return {
            "status": "path_found",
            "start_entity": start_entity,
            "end_entity": end_entity,
            "hop_count": len(path),
            "paths": [formatted],
        }

    # No end entity -- return all reachable within max_depth
    reachable = _bfs_reachable(adj, start_id, max_depth)
    result_nodes = []
    for nid, (depth, path_len) in reachable.items():
        nd = node_lookup.get(nid, {})
        result_nodes.append({
            "id": nid,
            "name": nd.get("name", ""),
            "type": nd.get("type", ""),
            "distance": depth,
        })
    result_nodes.sort(key=lambda x: x["distance"])

    return {
        "status": "reachable",
        "start_entity": start_entity,
        "reachable_count": len(result_nodes),
        "max_depth": max_depth,
        "nodes": result_nodes[:30],
    }


def detect_communities(graph: dict[str, Any]) -> dict[str, Any]:
    """Detect communities using label propagation on the graph.

    Returns community assignments and size distribution.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return {"status": "completed", "community_count": 0, "communities": []}

    # Build adjacency
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt:
            adj[src].add(tgt)
            adj[tgt].add(src)

    # Initialize: each node is its own community
    labels: dict[str, str] = {nd["id"]: nd["id"] for nd in nodes}
    node_ids = [nd["id"] for nd in nodes]

    # Run label propagation (5 iterations)
    for _ in range(5):
        changed = False
        for nid in node_ids:
            if not adj[nid]:
                continue
            # Count labels among neighbors
            label_counts: dict[str, int] = defaultdict(int)
            for neighbor in adj[nid]:
                label_counts[labels[neighbor]] += 1
            if label_counts:
                best_label = max(label_counts, key=lambda x: label_counts[x])
                if labels[nid] != best_label:
                    labels[nid] = best_label
                    changed = True
        if not changed:
            break

    # Group by community
    communities: dict[str, list[str]] = defaultdict(list)
    for nid, label in labels.items():
        communities[label].append(nid)

    # Build output
    node_lookup = {nd["id"]: nd for nd in nodes}
    community_list = []
    for cid, members in sorted(communities.items(), key=lambda x: -len(x[1])):
        member_nodes = []
        types: dict[str, int] = defaultdict(int)
        for mid in members:
            nd = node_lookup.get(mid, {})
            t = nd.get("type", "Unknown")
            types[t] += 1
            member_nodes.append({
                "id": mid,
                "name": nd.get("name", ""),
                "type": t,
            })
        community_list.append({
            "community_id": cid,
            "size": len(members),
            "type_distribution": dict(types),
            "members": member_nodes[:10],
        })

    return {
        "status": "completed",
        "community_count": len(community_list),
        "communities": community_list,
    }


def _bfs_path(
    adj: dict[str, list[tuple[str, str]]],
    start: str,
    end: str,
    max_depth: int,
) -> list[tuple[str, str, str]] | None:
    """BFS to find shortest path, returning list of (from, predicate, to)."""
    queue: deque[tuple[str, list[tuple[str, str, str]], set[str]]] = deque()
    queue.append((start, [], {start}))

    while queue:
        current, path, visited = queue.popleft()
        if len(path) >= max_depth:
            continue
        for neighbor, pred in adj.get(current, []):
            if neighbor == end:
                return path + [(current, pred, neighbor)]
            if neighbor not in visited:
                queue.append((
                    neighbor,
                    path + [(current, pred, neighbor)],
                    visited | {neighbor},
                ))
    return None


def _bfs_reachable(
    adj: dict[str, list[tuple[str, str]]],
    start: str,
    max_depth: int,
) -> dict[str, tuple[int, int]]:
    """BFS to find all reachable nodes with their distances."""
    result: dict[str, tuple[int, int]] = {start: (0, 0)}
    queue: deque[tuple[str, int]] = deque([(start, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor, _ in adj.get(current, []):
            if neighbor not in result:
                result[neighbor] = (depth + 1, depth + 1)
                queue.append((neighbor, depth + 1))
    return result


def _format_path(
    path: list[tuple[str, str, str]],
    node_lookup: dict[str, dict],
) -> dict[str, Any]:
    """Format a path into a readable structure."""
    steps = []
    for src_id, pred, tgt_id in path:
        src_name = node_lookup.get(src_id, {}).get("name", src_id)
        tgt_name = node_lookup.get(tgt_id, {}).get("name", tgt_id)
        steps.append({
            "source": src_name,
            "predicate": pred,
            "target": tgt_name,
        })
    entities = [s["source"] for s in steps] + ([steps[-1]["target"]] if steps else [])
    return {
        "hop_count": len(steps),
        "steps": steps,
        "entities": entities,
    }
