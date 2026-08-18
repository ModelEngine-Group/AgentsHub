"""Graph analytics beyond 1-3 hop traversal: PageRank, community detection,
degree centrality, shortest path.

Task 3 asks for "关联分析" (association analysis) alongside statistics/trends;
`AnalysisAgent`'s existing GRAPH route answers "what is X connected to"
(neighbor/path traversal) but has no notion of *global* graph structure --
which entities are structurally central, which cluster together, how two
entities relate through the shortest chain of edges. This module adds those
four, kept intentionally small (real analytics value per line of code, not
exhaustive algorithm coverage).

All functions take a `LocalGraphStore` and return plain dicts/lists (JSON- and
ECharts-ready), and accept an optional `entity_type` filter since a raw
whole-graph ranking is dominated by whichever node type happens to be most
numerous (Disease, in this project's graphs) rather than being useful on its
own.
"""
from __future__ import annotations

import networkx as nx


def _simple_undirected(store) -> nx.Graph:
    """Collapse the MultiDiGraph to a simple undirected Graph.

    PageRank and centrality are computed on the directed multigraph directly
    (they support it natively); only community detection needs this -- Louvain
    modularity is defined over undirected simple graphs, and collapsing parallel
    edges (multiple relations between the same pair) to one is the right
    behaviour for "are these two entities in the same cluster", not a
    simplification that changes what's being measured.
    """
    return nx.Graph(store.g)


def pagerank_ranking(store, top_n: int = 10, entity_type: str | None = None) -> list[dict]:
    """Top entities by PageRank (confidence-weighted).

    Edge confidence is used as the PageRank weight, so a low-confidence
    extracted edge contributes less to a node's importance than a high-
    confidence one -- otherwise a node with many uncertain edges could
    outrank one with fewer but well-evidenced ones.

    Directedness matters here: this project's edges point disease -> attribute
    (`disease -[has_symptom]-> symptom`, `disease -[recommend_drug]-> drug`, ...),
    so PageRank -- which accumulates rank at nodes that are *pointed to* --
    naturally surfaces heavily-shared attributes (a symptom common to many
    diseases) rather than the diseases themselves. That is a different, equally
    valid question from "which disease is best-connected" (answered by
    `degree_centrality_ranking`, which counts edges regardless of direction);
    filtering by `entity_type="Disease"` still works and is meaningful, but a
    disease's PageRank reflects its in-edges (e.g. `complication` edges from
    other diseases pointing to it), not its out-edges.
    """
    scores = nx.pagerank(store.g, weight="confidence")
    return _ranked(store, scores, top_n, entity_type, "pagerank")


def degree_centrality_ranking(store, top_n: int = 10, entity_type: str | None = None) -> list[dict]:
    """Top entities by degree centrality (in+out edges, normalised by graph size).

    Cheap structural-importance proxy independent of edge confidence -- pairs
    with `pagerank_ranking` to show a query-independent "hub" view distinct
    from PageRank's confidence-weighted, propagated-importance view.
    """
    scores = nx.degree_centrality(store.g)
    return _ranked(store, scores, top_n, entity_type, "degree_centrality")


def _ranked(store, scores: dict, top_n: int, entity_type: str | None, score_key: str) -> list[dict]:
    items = scores.items()
    if entity_type:
        items = (
            (node, score)
            for node, score in items
            if store.g.nodes[node].get("type") == entity_type
        )
    ranked = sorted(items, key=lambda kv: -kv[1])[:top_n]
    return [
        {
            "name": node,
            "type": store.g.nodes[node].get("type", "Unknown"),
            score_key: round(score, 6),
            "degree": store.g.degree[node],
        }
        for node, score in ranked
    ]


def detect_communities(store, top_n: int = 5, min_size: int = 3) -> list[dict]:
    """Top-N Louvain communities by size, each summarised by its most-central
    member entities (so a community reads as "the diabetes/nephropathy
    cluster" rather than an opaque node-id list).

    `min_size` drops singleton/near-singleton communities -- an entity with one
    edge trivially forms its own "community" under modularity optimisation,
    which is noise for this use case, not a meaningful cluster.
    """
    undirected = _simple_undirected(store)
    communities = nx.community.louvain_communities(undirected, weight="confidence", seed=20260730)
    ranked = sorted((c for c in communities if len(c) >= min_size), key=len, reverse=True)[:top_n]
    result = []
    for index, members in enumerate(ranked):
        subgraph = undirected.subgraph(members)
        centrality = nx.degree_centrality(subgraph)
        top_members = sorted(centrality, key=lambda n: -centrality[n])[:5]
        type_counts: dict[str, int] = {}
        for node in members:
            node_type = store.g.nodes[node].get("type", "Unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        result.append(
            {
                "community_id": index,
                "size": len(members),
                "representative_entities": top_members,
                "type_breakdown": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
            }
        )
    return result


def shortest_path(store, source: str, target: str) -> dict | None:
    """Shortest relation path between two entities (BFS, hop count as distance).

    Direction-agnostic (built on the undirected collapse): a clinically useful
    "how is X related to Y" answer should not depend on which direction the
    extractor happened to record the edge in.
    """
    undirected = _simple_undirected(store)
    if source not in undirected or target not in undirected:
        return None
    try:
        node_path = nx.shortest_path(undirected, source, target)
    except nx.NetworkXNoPath:
        return None
    steps = []
    for head, tail in zip(node_path, node_path[1:]):
        edge_data = store.g.get_edge_data(head, tail) or store.g.get_edge_data(tail, head) or {}
        first_edge = next(iter(edge_data.values()), {}) if edge_data else {}
        steps.append(
            {
                "head": head,
                "tail": tail,
                "relation": first_edge.get("relation", ""),
                "confidence": first_edge.get("confidence"),
            }
        )
    return {"source": source, "target": target, "hops": len(steps), "path": node_path, "steps": steps}
