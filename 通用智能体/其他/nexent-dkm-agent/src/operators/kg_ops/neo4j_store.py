"""Neo4j graph storage and retrieval operators for task 2.

Provides functions to write the in-memory medical KG dict into Neo4j
and read it back.  Uses the official ``neo4j`` Python driver.

Falls back gracefully when the driver or server is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PREDICATE_TO_REL_TYPE = {
    "has_symptom": "HAS_SYMPTOM",
    "treated_by": "TREATED_BY",
    "diagnosed_by": "DIAGNOSED_BY",
    "recommended_treatment": "RECOMMENDED_TREATMENT",
    "complication_of": "COMPLICATION_OF",
}

_REL_TYPE_TO_PREDICATE = {v: k for k, v in _PREDICATE_TO_REL_TYPE.items()}

_VALID_ENTITY_LABELS = frozenset({
    "Disease", "Symptom", "Drug", "Examination", "Treatment",
})

_VALID_REL_TYPES = frozenset(_PREDICATE_TO_REL_TYPE.values())


def graph_to_neo4j(
    graph: dict[str, Any],
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> dict[str, Any]:
    """Write the in-memory graph dict into Neo4j.

    Creates nodes with labels matching their entity type and relationships
    with types derived from the predicate.  All writes are idempotent
    (MERGE semantics).
    """
    if not password:
        return _credentials_required()
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "unavailable", "message": "Neo4j driver not available."}

    node_count = 0
    edge_count = 0

    try:
        with driver.session(database=database) as session:
            _create_constraints(session)

            for node in graph.get("nodes", []):
                session.execute_write(_write_node, node)
                node_count += 1

            for edge in graph.get("edges", []):
                session.execute_write(_write_edge, edge)
                edge_count += 1

        return {
            "status": "completed",
            "node_count": node_count,
            "edge_count": edge_count,
            "uri": uri,
            "database": database,
        }
    except Exception as exc:
        logger.error("Neo4j write failed: %s", exc, exc_info=True)
        return {"status": "failed", "message": str(exc)}
    finally:
        driver.close()


def neo4j_to_graph(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> dict[str, Any]:
    """Read the full graph from Neo4j back into the in-memory dict format."""
    if not password:
        return {
            **_credentials_required(),
            "nodes": [],
            "edges": [],
            "statistics": {},
        }
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "unavailable", "nodes": [], "edges": [], "statistics": {}}

    try:
        with driver.session(database=database) as session:
            nodes = _read_nodes(session)
            edges = _read_edges(session)

        return {
            "status": "completed",
            "nodes": nodes,
            "edges": edges,
            "triples": [],
            "records": [],
            "statistics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "triple_count": len(edges),
                "record_count": 0,
            },
        }
    except Exception as exc:
        logger.error("Neo4j read failed: %s", exc, exc_info=True)
        return {"status": "failed", "nodes": [], "edges": [], "statistics": {}}
    finally:
        driver.close()


def clear_neo4j_graph(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> dict[str, Any]:
    """Remove only task-2 labelled nodes and relationships from Neo4j.

    Only deletes nodes with labels in the task-2 entity whitelist
    (Disease, Symptom, Drug, Examination, Treatment).  Other data
    in the same Neo4j database is preserved.
    """
    if not password:
        return _credentials_required()
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "unavailable"}

    try:
        with driver.session(database=database) as session:
            label_list = ":" + "|".join(sorted(_VALID_ENTITY_LABELS))
            session.run(f"MATCH (n:{label_list}) DETACH DELETE n")
            info = session.run(
                f"MATCH (n:{label_list}) RETURN count(n) AS remaining"
            ).single()
            return {
                "status": "completed",
                "remaining_nodes": info["remaining"] if info else 0,
                "scope": "task2_labels_only",
            }
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    finally:
        driver.close()


def check_neo4j_connection(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
) -> dict[str, Any]:
    """Check whether Neo4j is reachable."""
    if not password:
        return _credentials_required()
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "driver_unavailable", "message": "neo4j Python package not installed."}
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            ok = result.single()
            return {
                "status": "connected" if ok and ok["ok"] == 1 else "error",
                "uri": uri,
            }
    except Exception as exc:
        return {"status": "connection_failed", "message": str(exc)}
    finally:
        driver.close()


def _get_driver(uri: str, user: str, password: str):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None
    try:
        return GraphDatabase.driver(uri, auth=(user, password))
    except Exception:
        logger.warning("Failed to create Neo4j driver.", exc_info=True)
        return None


def _credentials_required() -> dict[str, str]:
    return {
        "status": "credentials_required",
        "message": "Neo4j password must be provided explicitly.",
    }


def _create_constraints(session) -> None:
    entity_types = ["Disease", "Symptom", "Drug", "Examination", "Treatment"]
    for etype in entity_types:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{etype}) REQUIRE n.name IS UNIQUE"
        )


def _write_node(tx, node: dict[str, Any]) -> None:
    label = _sanitize_label(node.get("type", "Entity"))
    name = node.get("name", "")
    tx.run(
        f"MERGE (n:`{label}` {{name: $name}}) "
        "SET n.type = $type, "
        "    n.mention_count = $mention_count, "
        "    n.record_ids = $record_ids",
        type=label,
        name=name,
        mention_count=node.get("mention_count", 0),
        record_ids=node.get("record_ids", []),
    )


def _write_edge(tx, edge: dict[str, Any]) -> None:
    source_id = edge.get("source", "")
    target_id = edge.get("target", "")
    predicate = edge.get("predicate", "")
    rel_type = _PREDICATE_TO_REL_TYPE.get(predicate, predicate.upper())

    source_label = _sanitize_label(source_id.split(":")[0] if ":" in source_id else "Entity")
    target_label = _sanitize_label(target_id.split(":")[0] if ":" in target_id else "Entity")
    source_name = source_id.split(":", 1)[1] if ":" in source_id else source_id
    target_name = target_id.split(":", 1)[1] if ":" in target_id else target_id

    if rel_type not in _VALID_REL_TYPES:
        logger.warning("Skipping edge with unrecognised rel_type: %s", rel_type)
        return

    tx.run(
        f"MATCH (s:`{source_label}` {{name: $source_name}}) "
        f"MATCH (t:`{target_label}` {{name: $target_name}}) "
        f"MERGE (s)-[r:`{rel_type}`]->(t) "
        "SET r.confidence = $confidence, "
        "    r.evidence = $evidence, "
        "    r.record_ids = $record_ids",
        source_name=source_name,
        target_name=target_name,
        confidence=edge.get("confidence", 0.0),
        evidence=edge.get("evidence", []),
        record_ids=edge.get("record_ids", []),
    )


def _read_nodes(session) -> list[dict[str, Any]]:
    result = session.run("MATCH (n) RETURN n, labels(n) AS labels")
    nodes = []
    for record in result:
        node = dict(record["n"])
        labels = record.get("labels", [])
        # Prefer explicit 'type' property, fallback to first Neo4j label
        node_type = node.get("type") or (labels[0] if labels else "Entity")
        node_id = f"{node_type}:{node.get('name', '')}"
        node["id"] = node_id
        node["type"] = node_type
        nodes.append(node)
    return nodes


def _read_edges(session) -> list[dict[str, Any]]:
    result = session.run("MATCH (s)-[r]->(t) RETURN s, type(r) AS rel_type, r, t")
    edges = []
    for record in result:
        source_props = dict(record["s"])
        target_props = dict(record["t"])
        rel_type = record["rel_type"]
        rel_props = dict(record["r"])
        predicate = _REL_TYPE_TO_PREDICATE.get(rel_type, rel_type.lower())

        source_label = list(record["s"].labels)[0] if record["s"].labels else "Entity"
        target_label = list(record["t"].labels)[0] if record["t"].labels else "Entity"

        edges.append({
            "id": f"{source_label}:{source_props.get('name', '')}-{predicate}-{target_label}:{target_props.get('name', '')}",
            "source": f"{source_label}:{source_props.get('name', '')}",
            "target": f"{target_label}:{target_props.get('name', '')}",
            "predicate": predicate,
            "confidence": rel_props.get("confidence", 0.0),
            "evidence": rel_props.get("evidence", []),
            "record_ids": rel_props.get("record_ids", []),
        })
    return edges


def _sanitize_label(label: str) -> str:
    """Validate and sanitize a Cypher label against the task-2 whitelist.

    Only allows known entity type labels to prevent injection or accidental
    creation of arbitrary node labels.
    """
    if label in _VALID_ENTITY_LABELS:
        return label
    logger.warning("Unrecognised entity label '%s' replaced with 'Entity'.", label)
    return "Entity"
