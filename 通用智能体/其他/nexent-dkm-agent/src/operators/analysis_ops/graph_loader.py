"""Graph loading helpers for task 3 analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_graph(path: str | Path) -> dict[str, Any]:
    """Load a task-2 graph JSON artifact and validate the minimal contract."""

    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    _validate_graph(graph, graph_path)
    return graph


def _validate_graph(graph: dict[str, Any], graph_path: Path) -> None:
    required = {"nodes", "edges", "statistics"}
    missing = required.difference(graph)
    if missing:
        raise ValueError(f"Graph file {graph_path} is missing fields: {sorted(missing)}")
    if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
        raise ValueError("Graph nodes and edges must be lists.")
    for node in graph["nodes"]:
        if not {"id", "name", "type"}.issubset(node):
            raise ValueError(f"Invalid graph node: {node}")
    for edge in graph["edges"]:
        if not {"source", "target", "predicate"}.issubset(edge):
            raise ValueError(f"Invalid graph edge: {edge}")
