"""Knowledge graph question-answering agent for task 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class KGQAAgent:
    """Answer simple medical questions by querying medical_kg.json."""

    def __init__(self, outputs_dir: Path | str) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.graph_path = self.outputs_dir / "medical_kg.json"

    def run(self, question: str) -> dict[str, Any]:
        """Answer one question with answers and supporting triples."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")

        graph = self._load_graph()
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        node_type_by_id = {node["id"]: node.get("type", "Unknown") for node in nodes}

        question_type = self._detect_question_type(normalized_question)
        if question_type == "disease_symptoms":
            entity = self._find_entity(normalized_question, node_type_by_id, "Disease")
            answers, evidence = self._outgoing(edges, entity, "has_symptom")
        elif question_type == "disease_drugs":
            entity = self._find_entity(normalized_question, node_type_by_id, "Disease")
            answers, evidence = self._outgoing(edges, entity, "treated_by", target_type="Drug", node_type_by_id=node_type_by_id)
        elif question_type == "symptom_diseases":
            entity = self._find_entity(normalized_question, node_type_by_id, "Symptom")
            answers, evidence = self._incoming(edges, entity, "has_symptom")
        elif question_type == "drug_diseases":
            entity = self._find_entity(normalized_question, node_type_by_id, "Drug")
            answers, evidence = self._incoming(edges, entity, "treated_by")
        else:
            return {
                "status": "unsupported",
                "question": question,
                "question_type": "unknown",
                "answers": [],
                "answer": "暂时只支持疾病症状、疾病用药、症状反查疾病、药物反查疾病这四类问题。",
                "evidence": [],
            }

        answer_text = self._format_answer(entity, question_type, answers)
        return {
            "status": "success" if answers else "not_found",
            "question": question,
            "question_type": question_type,
            "entity": entity,
            "answers": answers,
            "answer": answer_text,
            "evidence": evidence,
        }

    def _load_graph(self) -> dict[str, list[dict[str, str]]]:
        if not self.graph_path.exists():
            raise FileNotFoundError(f"知识图谱文件不存在: {self.graph_path}")
        with open(self.graph_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _detect_question_type(self, question: str) -> str:
        if "哪些疾病" in question and any(word in question for word in ["治疗", "药"]):
            return "drug_diseases"
        if "哪些疾病" in question:
            return "symptom_diseases"
        if any(word in question for word in ["什么药", "用什么药", "药物", "治疗"]):
            return "disease_drugs"
        if any(word in question for word in ["症状", "表现"]):
            return "disease_symptoms"
        return "unknown"

    def _find_entity(self, question: str, node_type_by_id: dict[str, str], entity_type: str) -> str:
        candidates = [name for name, node_type in node_type_by_id.items() if node_type == entity_type and name in question]
        if not candidates:
            return ""
        return max(candidates, key=len)

    def _outgoing(
        self,
        edges: list[dict[str, str]],
        source: str,
        relation: str,
        target_type: str | None = None,
        node_type_by_id: dict[str, str] | None = None,
    ) -> tuple[list[str], list[dict[str, str]]]:
        answers: list[str] = []
        evidence: list[dict[str, str]] = []
        seen: set[str] = set()
        for edge in edges:
            target = edge["target"]
            if edge["source"] != source or edge["relation"] != relation:
                continue
            if target_type and node_type_by_id and node_type_by_id.get(target) != target_type:
                continue
            if target in seen:
                continue
            seen.add(target)
            answers.append(target)
            evidence.append({"subject": source, "predicate": relation, "object": target})
        return answers, evidence

    def _incoming(
        self,
        edges: list[dict[str, str]],
        target: str,
        relation: str,
    ) -> tuple[list[str], list[dict[str, str]]]:
        answers: list[str] = []
        evidence: list[dict[str, str]] = []
        seen: set[str] = set()
        for edge in edges:
            source = edge["source"]
            if edge["target"] != target or edge["relation"] != relation:
                continue
            if source in seen:
                continue
            seen.add(source)
            answers.append(source)
            evidence.append({"subject": source, "predicate": relation, "object": target})
        return answers, evidence

    def _format_answer(self, entity: str, question_type: str, answers: list[str]) -> str:
        if not entity:
            return "没有在知识图谱中识别到问题里的医疗实体。"
        if not answers:
            return f"没有在知识图谱中找到与 {entity} 相关的答案。"

        answer_text = "、".join(answers)
        templates = {
            "disease_symptoms": f"{entity} 的相关症状包括：{answer_text}。",
            "disease_drugs": f"{entity} 的相关治疗药物包括：{answer_text}。",
            "symptom_diseases": f"与 {entity} 相关的疾病包括：{answer_text}。",
            "drug_diseases": f"{entity} 在知识图谱中关联的疾病包括：{answer_text}。",
        }
        return templates[question_type]
