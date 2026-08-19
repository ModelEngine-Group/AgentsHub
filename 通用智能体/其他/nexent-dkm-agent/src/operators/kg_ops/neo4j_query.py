"""Neo4j Cypher-based query operators for task 2.

Replaces the in-memory BFS traversal with native Cypher queries when
Neo4j is available.  Each function mirrors the signature and return
format of the corresponding in-memory operator in ``query.py`` and
``multi_hop_qa.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def neo4j_find_entities(
    query: str,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
    entity_type: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Find graph nodes by substring match using Cypher."""
    if not password:
        return _credentials_required(query=query, matches=[])
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "driver_unavailable", "query": query, "matches": []}

    try:
        with driver.session(database=database) as session:
            if entity_type:
                safe_type = _sanitize_cypher_label(entity_type)
                cypher = (
                    f"MATCH (n:`{safe_type}`) "
                    "WHERE n.name CONTAINS $search_term "
                    "RETURN n, labels(n) AS labels "
                    "LIMIT $limit"
                )
            else:
                cypher = (
                    "MATCH (n) "
                    "WHERE n.name CONTAINS $search_term "
                    "RETURN n, labels(n) AS labels "
                    "LIMIT $limit"
                )
            # ``query`` is reserved by neo4j-driver 6.x Session.run(); use search_term.
            result = session.run(cypher, search_term=query, limit=limit)
            matches = []
            for record in result:
                node = dict(record["n"])
                labels = record["labels"]
                node_type = labels[0] if labels else "Unknown"
                matches.append({
                    "id": f"{node_type}:{node.get('name', '')}",
                    "name": node.get("name", ""),
                    "type": node_type,
                    "record_ids": node.get("record_ids", []),
                    "mention_count": node.get("mention_count", 0),
                    "match_type": "substring",
                    "score": _score(query, node.get("name", "")),
                })

            matches.sort(key=lambda m: -m["score"])
            return {
                "status": "matched" if matches else "unmatched",
                "query": query,
                "entity_type": entity_type,
                "matches": matches,
            }
    except Exception as exc:
        logger.error("Neo4j entity search failed: %s", exc, exc_info=True)
        return {"status": "failed", "query": query, "matches": [], "message": str(exc)}
    finally:
        driver.close()


def neo4j_query_neighbors(
    entity: str,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
    relation: str | None = None,
    direction: str = "out",
    limit: int = 20,
) -> dict[str, Any]:
    """Query neighboring nodes connected to a matched entity via Cypher."""

    if not password:
        return _credentials_required(entity=entity, neighbors=[])
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "driver_unavailable", "entity": entity, "neighbors": []}

    try:
        with driver.session(database=database) as session:
            # Find the entity
            entity_result = session.run(
                "MATCH (n) WHERE n.name CONTAINS $entity "
                "RETURN n, labels(n) AS labels LIMIT 1",
                entity=entity,
            )
            entity_record = entity_result.single()
            if not entity_record:
                return {
                    "status": "unmatched",
                    "entity": entity,
                    "relation": relation,
                    "direction": direction,
                    "neighbors": [],
                }

            entity_node = dict(entity_record["n"])
            entity_labels = entity_record["labels"]
            entity_type = entity_labels[0] if entity_labels else "Unknown"
            entity_name = entity_node.get("name", "")
            matched_entity = {
                "id": f"{entity_type}:{entity_name}",
                "name": entity_name,
                "type": entity_type,
            }

            # Build relationship pattern
            rel_type = ""
            if relation:
                safe_rel_type = _sanitize_relationship_type(relation)
                if not safe_rel_type:
                    return {
                        "status": "failed",
                        "entity": entity,
                        "relation": relation,
                        "direction": direction,
                        "neighbors": [],
                        "message": "relation must be a supported task-2 predicate.",
                    }
                rel_type = f":`{safe_rel_type}`"

            if direction == "out":
                pattern = f"(s)-[r{rel_type}]->(t)"
            elif direction == "in":
                pattern = f"(t)-[r{rel_type}]->(s)"
            else:  # both
                pattern = f"(s)-[r{rel_type}]-(t)"

            cypher = (
                f"MATCH {pattern} "
                "WHERE s.name = $entity_name "
                "RETURN s, type(r) AS rel_type, r, t, labels(t) AS t_labels "
                "LIMIT $limit"
            )
            result = session.run(cypher, entity_name=entity_name, limit=limit)
            neighbors = []
            for record in result:
                source_props = dict(record["s"])
                source_labels = list(record["s"].labels)
                target_props = dict(record["t"])
                target_labels = record["t_labels"]
                rel_props = dict(record["r"])
                rel_type_str = record["rel_type"]

                from src.operators.kg_ops.neo4j_store import _REL_TYPE_TO_PREDICATE
                predicate = _REL_TYPE_TO_PREDICATE.get(rel_type_str, rel_type_str.lower())
                source_type = source_labels[0] if source_labels else "Unknown"
                target_type = target_labels[0] if target_labels else "Unknown"

                edge_direction = "out" if direction in ("out", "both") else "in"

                neighbors.append({
                    "edge_id": f"{source_type}:{source_props.get('name', '')}-{predicate}-{target_type}:{target_props.get('name', '')}",
                    "predicate": predicate,
                    "direction": edge_direction,
                    "source": {"id": f"{source_type}:{source_props.get('name', '')}", "name": source_props.get("name", ""), "type": source_type},
                    "target": {"id": f"{target_type}:{target_props.get('name', '')}", "name": target_props.get("name", ""), "type": target_type},
                    "record_ids": rel_props.get("record_ids", []),
                    "confidence": rel_props.get("confidence", 0.0),
                    "evidence": rel_props.get("evidence", []),
                })

            neighbors.sort(key=lambda n: (-n["confidence"], n["predicate"]))
            return {
                "status": "matched" if neighbors else "unmatched",
                "entity": entity,
                "matched_entity": matched_entity,
                "relation": relation,
                "direction": direction,
                "neighbors": neighbors,
            }
    except Exception as exc:
        logger.error("Neo4j neighbor query failed: %s", exc, exc_info=True)
        return {"status": "failed", "entity": entity, "neighbors": [], "message": str(exc)}
    finally:
        driver.close()


def neo4j_multi_hop(
    start_entity: str,
    target_entity: str | None = None,
    max_hops: int = 3,
    max_paths: int = 5,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> dict[str, Any]:
    """Find multi-hop paths using native Cypher path matching."""
    if not password:
        return _credentials_required(start_entity=start_entity, paths=[])
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "driver_unavailable", "start_entity": start_entity, "paths": []}

    try:
        with driver.session(database=database) as session:
            if target_entity:
                cypher = (
                    "MATCH path = (s)-[*..$max_hops]-(t) "
                    "WHERE s.name CONTAINS $start AND t.name CONTAINS $target "
                    "RETURN path "
                    "LIMIT $max_paths"
                )
                params = {
                    "start": start_entity, "target": target_entity,
                    "max_hops": max_hops, "max_paths": max_paths,
                }
            else:
                cypher = (
                    "MATCH path = (s)-[*..$max_hops]-(t) "
                    "WHERE s.name CONTAINS $start "
                    "RETURN path "
                    "LIMIT $max_paths"
                )
                params = {
                    "start": start_entity,
                    "max_hops": max_hops, "max_paths": max_paths,
                }

            result = session.run(cypher, **params)
            paths = []
            for record in result:
                path_obj = record["path"]
                steps = _extract_path_steps(path_obj)
                if steps:
                    paths.append(steps)

            return {
                "status": "matched" if paths else "unmatched",
                "start_entity": start_entity,
                "target_entity": target_entity,
                "path_count": len(paths),
                "paths": paths,
            }
    except Exception as exc:
        logger.error("Neo4j multi-hop failed: %s", exc, exc_info=True)
        return {"status": "failed", "start_entity": start_entity, "paths": [], "message": str(exc)}
    finally:
        driver.close()


def neo4j_answer_question(
    question: str,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
    database: str = "neo4j",
) -> dict[str, Any]:
    """Answer a question using Cypher queries against Neo4j."""
    from src.operators.kg_ops.qa import INTENT_RELATIONS, QUESTION_INTENTS

    if not password:
        return _credentials_required(question=question, answer="", evidence=[])
    driver = _get_driver(uri, user, password)
    if driver is None:
        return {"status": "driver_unavailable", "question": question, "answer": "Neo4j 不可用。", "evidence": []}

    try:
        with driver.session(database=database) as session:
            # Find disease in question
            disease_result = session.run(
                "MATCH (d:Disease) WHERE $question CONTAINS d.name "
                "RETURN d.name AS name ORDER BY size(d.name) DESC LIMIT 1",
                question=question,
            )
            disease_record = disease_result.single()
            if not disease_record:
                return {
                    "status": "unanswered",
                    "question": question,
                    "answer": "未在 Neo4j 图谱中找到对应的疾病实体。",
                    "evidence": [],
                }

            disease_name = disease_record["name"]

            # Detect intents
            intents = [
                intent for intent, triggers in QUESTION_INTENTS.items()
                if any(trigger in question.lower() for trigger in triggers)
            ]
            relation_filter = set()
            for intent in intents:
                relation_filter.update(INTENT_RELATIONS.get(intent, set()))

            if not relation_filter:
                relation_filter = {"HAS_SYMPTOM", "TREATED_BY", "DIAGNOSED_BY", "RECOMMENDED_TREATMENT", "COMPLICATION_OF"}

            from src.operators.kg_ops.neo4j_store import _PREDICATE_TO_REL_TYPE
            rel_types = [_PREDICATE_TO_REL_TYPE.get(r, r.upper()) for r in relation_filter]
            rel_labels = "|".join(f"`{rt}`" for rt in rel_types)

            cypher = (
                f"MATCH (d:Disease {{name: $disease}})-[r:{rel_labels}]->(t) "
                "RETURN type(r) AS rel_type, r.confidence AS confidence, "
                "       r.evidence AS evidence, t.name AS target_name, labels(t) AS t_labels "
                "ORDER BY r.confidence DESC"
            )
            result = session.run(cypher, disease=disease_name)

            grouped: dict[str, list[str]] = {"症状": [], "用药": [], "检查": [], "治疗建议": [], "并发症": []}
            rel_label_map = {
                "HAS_SYMPTOM": "症状", "TREATED_BY": "用药", "DIAGNOSED_BY": "检查",
                "RECOMMENDED_TREATMENT": "治疗建议", "COMPLICATION_OF": "并发症",
            }
            evidence = []
            for record in result:
                from src.operators.kg_ops.neo4j_store import _REL_TYPE_TO_PREDICATE
                predicate = _REL_TYPE_TO_PREDICATE.get(record["rel_type"], "")
                label = rel_label_map.get(record["rel_type"], "")
                target_name = record["target_name"]
                if label and target_name not in grouped[label]:
                    grouped[label].append(target_name)
                evidence.append({
                    "predicate": predicate,
                    "target": target_name,
                    "confidence": record.get("confidence", 0.0),
                })

            parts = [f"{label}: {'、'.join(values)}" for label, values in grouped.items() if values]
            if not parts:
                return {
                    "status": "unanswered",
                    "question": question,
                    "answer": f"Neo4j 图谱中暂未找到{disease_name}的相关关系。",
                    "evidence": [],
                }

            answer = f"{disease_name}相关信息（Neo4j）：{'；'.join(parts)}。"
            return {
                "status": "answered",
                "question": question,
                "answer": answer,
                "evidence": evidence,
                "backend": "neo4j",
            }
    except Exception as exc:
        logger.error("Neo4j QA failed: %s", exc, exc_info=True)
        return {"status": "failed", "question": question, "answer": str(exc), "evidence": []}
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


def _credentials_required(**context: Any) -> dict[str, Any]:
    return {
        "status": "credentials_required",
        "message": "Neo4j password must be provided explicitly.",
        **context,
    }


def _score(query: str, name: str) -> float:
    if not query or not name:
        return 0.0
    q, n = query.lower(), name.lower()
    if q == n:
        return 1.0
    if q in n or n in q:
        return round(min(len(q), len(n)) / max(len(q), len(n)), 3)
    return 0.0


def _extract_path_steps(path_obj) -> dict[str, Any]:
    """Extract path steps from a Neo4j path object."""
    from src.operators.kg_ops.neo4j_store import _REL_TYPE_TO_PREDICATE

    try:
        nodes = path_obj.nodes
        relationships = path_obj.relationships
        steps = []
        for i, rel in enumerate(relationships):
            source = nodes[i]
            target = nodes[i + 1]
            source_props = dict(source)
            target_props = dict(target)
            rel_type = rel.type
            predicate = _REL_TYPE_TO_PREDICATE.get(rel_type, rel_type.lower())

            steps.append({
                "source": source_props.get("name", ""),
                "predicate": predicate,
                "target": target_props.get("name", ""),
                "confidence": dict(rel).get("confidence", 0.0),
            })

        entities = [s["source"] for s in steps] + ([steps[-1]["target"]] if steps else [])
        return {
            "hop_count": len(steps),
            "steps": steps,
            "entities": entities,
        }
    except Exception:
        return None


def _sanitize_cypher_label(label: str) -> str:
    """Validate a Cypher label against the task-2 entity whitelist."""
    from src.operators.kg_ops.neo4j_store import _VALID_ENTITY_LABELS

    if label in _VALID_ENTITY_LABELS:
        return label
    logger.warning("Unrecognised Cypher label '%s' replaced with 'Entity'.", label)
    return "Entity"


def _sanitize_relationship_type(relation: str) -> str | None:
    """Map a task-2 predicate to a whitelisted Cypher relationship type."""
    from src.operators.kg_ops.neo4j_store import _PREDICATE_TO_REL_TYPE, _VALID_REL_TYPES

    rel_type = _PREDICATE_TO_REL_TYPE.get(relation, relation.upper())
    if rel_type in _VALID_REL_TYPES:
        return rel_type
    logger.warning("Rejected unrecognised Cypher relationship type: %s", relation)
    return None
