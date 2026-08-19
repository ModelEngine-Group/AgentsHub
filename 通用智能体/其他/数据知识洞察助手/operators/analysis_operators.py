"""Graph analysis operators for task 3."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import networkx as nx

from .base import BaseOperator


class GraphStatisticsOperator(BaseOperator):
    """Compute basic graph statistics."""

    name = "GraphStatisticsOperator"
    description = "统计图谱节点、边、实体类型和关系类型数量"
    input_schema = {"graph": "dict(nodes=list, edges=list)"}
    output_schema = {"statistics": "dict"}

    def run(self, graph: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        entity_type_count = Counter(str(node.get("type", "Unknown")) for node in nodes)
        relation_type_count = Counter(str(edge.get("relation", "unknown")) for edge in edges)

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entity_type_count": dict(entity_type_count),
            "relation_type_count": dict(relation_type_count),
        }


class DiseaseSymptomCountOperator(BaseOperator):
    """Count how many symptoms each disease has."""

    name = "DiseaseSymptomCountOperator"
    description = "统计每个疾病关联的症状数量"
    input_schema = {"graph": "dict(nodes=list, edges=list)"}
    output_schema = {"counts": "list[dict]"}

    def run(self, graph: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        symptoms_by_disease: dict[str, set[str]] = defaultdict(set)
        for edge in graph.get("edges", []):
            if edge.get("relation") == "has_symptom":
                symptoms_by_disease[str(edge["source"])].add(str(edge["target"]))

        result = [
            {"disease": disease, "symptom_count": len(symptoms)}
            for disease, symptoms in symptoms_by_disease.items()
        ]
        return sorted(result, key=lambda item: (-item["symptom_count"], item["disease"]))


class DrugDiseaseCountOperator(BaseOperator):
    """Count how many diseases each drug treats."""

    name = "DrugDiseaseCountOperator"
    description = "统计每个药物关联的疾病数量"
    input_schema = {"graph": "dict(nodes=list, edges=list)"}
    output_schema = {"counts": "list[dict]"}

    def run(self, graph: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        node_type_by_id = {node["id"]: node.get("type", "Unknown") for node in graph.get("nodes", [])}
        diseases_by_drug: dict[str, set[str]] = defaultdict(set)
        for edge in graph.get("edges", []):
            target = str(edge["target"])
            if edge.get("relation") == "treated_by" and node_type_by_id.get(target) == "Drug":
                diseases_by_drug[target].add(str(edge["source"]))

        result = [
            {"drug": drug, "disease_count": len(diseases), "diseases": sorted(diseases)}
            for drug, diseases in diseases_by_drug.items()
        ]
        return sorted(result, key=lambda item: (-item["disease_count"], item["drug"]))


class CentralityAnalysisOperator(BaseOperator):
    """Compute degree centrality for graph nodes."""

    name = "CentralityAnalysisOperator"
    description = "计算图谱节点中心性"
    input_schema = {"graph": "dict(nodes=list, edges=list)", "top_k": "int"}
    output_schema = {"centrality": "list[dict]"}

    def run(self, graph: dict[str, list[dict[str, Any]]], top_k: int = 10) -> list[dict[str, Any]]:
        nx_graph = nx.Graph()
        for node in graph.get("nodes", []):
            nx_graph.add_node(str(node["id"]), type=node.get("type", "Unknown"))
        for edge in graph.get("edges", []):
            nx_graph.add_edge(str(edge["source"]), str(edge["target"]))

        centrality = nx.degree_centrality(nx_graph)
        result = [
            {
                "node": node,
                "type": nx_graph.nodes[node].get("type", "Unknown"),
                "degree_centrality": round(score, 6),
            }
            for node, score in centrality.items()
        ]
        return sorted(result, key=lambda item: (-item["degree_centrality"], item["node"]))[:top_k]


class NL2AnalysisOperator(BaseOperator):
    """Map Chinese analysis questions to analysis task names."""

    name = "NL2AnalysisOperator"
    description = "将自然语言分析需求映射为分析任务名称"
    input_schema = {"question": "str"}
    output_schema = {"task_name": "str"}

    def run(self, question: str) -> str:
        text = question.strip()
        if any(word in text for word in ["中心", "重要", "核心节点"]):
            return "centrality_analysis"
        if any(word in text for word in ["药物", "用药"]):
            return "drug_disease_count"
        if any(word in text for word in ["症状最多", "疾病的症状", "症状数量"]):
            return "disease_symptom_count"
        if any(word in text for word in ["实体", "关系", "统计", "数量"]):
            return "graph_statistics"
        return "graph_statistics"
