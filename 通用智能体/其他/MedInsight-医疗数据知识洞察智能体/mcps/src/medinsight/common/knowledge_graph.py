from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass
class KnowledgeGraphNode:
    node_id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeGraphEdge:
    edge_id: str
    source: str
    target: str
    predicate: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeGraph:
    nodes: list[KnowledgeGraphNode]
    edges: list[KnowledgeGraphEdge]
    schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "schema": self.schema,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
