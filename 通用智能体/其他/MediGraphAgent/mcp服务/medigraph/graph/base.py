"""Abstract knowledge-graph store interface.

Two backends implement it: LocalGraphStore (NetworkX + JSON, default) and
Neo4jGraphStore (optional). Agents depend only on this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class GraphStore(ABC):
    @abstractmethod
    def upsert_entity(self, name: str, etype: str, **props) -> None:
        ...

    @abstractmethod
    def upsert_triple(
        self,
        head: str,
        head_type: str,
        relation: str,
        tail: str,
        tail_type: str,
        confidence: float = 1.0,
        source: str = "",
        operator_version: str = "",
        **props,
    ) -> None:
        ...

    @abstractmethod
    def neighbors(self, name: str, hops: int = 1) -> list[dict]:
        """Return triples within `hops` of the entity named `name`."""

    @abstractmethod
    def find_entities(self, names: list[str]) -> list[str]:
        """Resolve free-text names to existing entity node names (fuzzy)."""

    @abstractmethod
    def stats(self) -> dict:
        ...


def get_graph_store(config=None) -> GraphStore:
    """Factory: pick backend from GraphConfig."""
    from config.settings import get_graph_config

    cfg = config or get_graph_config()
    if cfg.backend == "neo4j":
        from medigraph.graph.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
    from medigraph.graph.local_store import LocalGraphStore

    return LocalGraphStore()
