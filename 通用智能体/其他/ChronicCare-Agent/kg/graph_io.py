from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from runtime_common.common import ensure_directory


def save_graph_json(path: Path, graph_data: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(graph_data, file, ensure_ascii=False, indent=2)


def load_graph_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_nodes_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    if pd is None:
        raise RuntimeError("pandas is required to write nodes.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def write_edges_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    if pd is None:
        raise RuntimeError("pandas is required to write edges.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def build_multidigraph(nodes: Iterable[Dict[str, Any]], edges: Iterable[Dict[str, Any]]) -> nx.MultiDiGraph:
    if nx is None:
        raise RuntimeError("networkx is required to build a MultiDiGraph")
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node["id"], **node)
    for index, edge in enumerate(edges):
        edge_key = (
            edge.get("id")
            or edge.get("edge_id")
            or f"{edge.get('source', 'unknown')}::{edge.get('relation') or edge.get('relation_type') or 'related_to'}::{edge.get('target', 'unknown')}::{index}"
        )
        graph.add_edge(edge["source"], edge["target"], key=edge_key, **edge)
    return graph


def adjacency_indexes(edges: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    outgoing: Dict[str, List[Dict[str, Any]]] = {}
    incoming: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        outgoing.setdefault(edge["source"], []).append(edge)
        incoming.setdefault(edge["target"], []).append(edge)
    return outgoing, incoming
