from __future__ import annotations

from collections import Counter
from typing import Any

from medinsight.common.knowledge_graph import stable_id
from medinsight.common.operators.base import OperatorSpec


class MedicalKnowledgeGraphBuilder:
    spec = OperatorSpec(
        name="medical_knowledge_graph_builder",
        description="Build reusable graph nodes and edges from validated medical triples.",
        input_schema={"type": "object", "properties": {"triples": {"type": "array"}, "schema": {"type": "array"}}},
        output_schema={"type": "object", "properties": {"graph": {"type": "object"}}},
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        triples = payload.get("triples") or []
        schema = payload.get("schema") or []
        nodes_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        predicate_counts = Counter()
        node_type_counts = Counter()

        def get_node(name: str, entity_type: str, evidence_doc: str | None) -> dict[str, Any]:
            key = (entity_type, name)
            if key not in nodes_by_key:
                node = {
                    "node_id": stable_id("node", entity_type, name),
                    "name": name,
                    "type": entity_type,
                    "aliases": [],
                    "source_docs": [],
                    "properties": {"task": "task2_medical_knowledge_graph"},
                }
                nodes_by_key[key] = node
                node_type_counts[entity_type] += 1
            node = nodes_by_key[key]
            if evidence_doc and evidence_doc not in node["source_docs"]:
                node["source_docs"].append(evidence_doc)
            return node

        for triple in triples:
            evidence = triple.get("evidence") or []
            first_doc = evidence[0].get("doc_id") if evidence else None
            subject_node = get_node(str(triple["subject"]), str(triple["subject_type"]), first_doc)
            object_node = get_node(str(triple["object"]), str(triple["object_type"]), first_doc)
            predicate = str(triple["predicate"])
            edge_key = (subject_node["node_id"], predicate, object_node["node_id"])
            predicate_counts[predicate] += 1
            if edge_key not in edges_by_key:
                edges_by_key[edge_key] = {
                    "edge_id": stable_id("edge", *edge_key),
                    "source": subject_node["node_id"],
                    "target": object_node["node_id"],
                    "predicate": predicate,
                    "evidence": [],
                    "properties": {
                        "subject": triple["subject"],
                        "subject_type": triple["subject_type"],
                        "object": triple["object"],
                        "object_type": triple["object_type"],
                        "source_dataset": triple.get("source_dataset"),
                        "confidence": triple.get("confidence", 1.0),
                        "triple_id": triple.get("triple_id"),
                    },
                }
            edge = edges_by_key[edge_key]
            edge["evidence"].extend(evidence[:3])

        metadata = {
            "name": "task2_medical_knowledge_graph",
            "source": "CBLUE CMeIE-V2 with CMeEE-V2 entity inventory",
            "node_count": len(nodes_by_key),
            "edge_count": len(edges_by_key),
            "node_type_counts": dict(node_type_counts),
            "predicate_counts": dict(predicate_counts),
        }
        graph = {
            "metadata": metadata,
            "schema": {
                "entity_types": sorted(node_type_counts),
                "relation_schemas": schema,
                "predicate_count": len({item.get("predicate") for item in schema}),
            },
            "nodes": list(nodes_by_key.values()),
            "edges": list(edges_by_key.values()),
        }
        return {"graph": graph, "summary": metadata}
