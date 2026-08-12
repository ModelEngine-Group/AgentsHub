"""Knowledge graph agent for task 2.

The MVP agent reads the medical text, extracts entities and relations paragraph
by paragraph, validates triples, builds a graph, and writes reproducible outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from operators import (
    EntityRecognitionOperator,
    GraphBuildOperator,
    ReadTextOperator,
    RelationExtractionOperator,
    SaveJsonOperator,
    SplitParagraphOperator,
    TripleGenerationOperator,
    TripleValidationOperator,
)


class KnowledgeGraphAgent:
    """Task 2 medical knowledge graph construction agent."""

    def __init__(self, data_dir: Path | str, outputs_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path(outputs_dir)
        self.medical_text_path = self.data_dir / "medical_text.txt"
        self.triples_path = self.outputs_dir / "medical_triples.json"
        self.graph_path = self.outputs_dir / "medical_kg.json"
        self.graphml_path = self.outputs_dir / "medical_kg.graphml"
        self.html_path = self.outputs_dir / "medical_kg.html"
        self.steps: list[dict[str, Any]] = []

    def plan(self, task_description: str) -> list[dict[str, str]]:
        """Create the fixed task 2 operator plan."""
        if not task_description.strip():
            raise ValueError("任务描述不能为空")

        return [
            {"step": "read_text", "operator": "ReadTextOperator", "goal": "读取医疗文本"},
            {"step": "split_paragraph", "operator": "SplitParagraphOperator", "goal": "按疾病段落切分"},
            {"step": "recognize_entities", "operator": "EntityRecognitionOperator", "goal": "识别医疗实体"},
            {"step": "extract_relations", "operator": "RelationExtractionOperator", "goal": "抽取实体关系"},
            {"step": "generate_triples", "operator": "TripleGenerationOperator", "goal": "生成并校验三元组"},
            {"step": "build_graph", "operator": "GraphBuildOperator", "goal": "构建图谱节点和边"},
            {"step": "save_outputs", "operator": "SaveJsonOperator/networkx", "goal": "保存 JSON、GraphML、HTML"},
        ]

    def run(self, task_description: str) -> dict[str, Any]:
        """Run the complete knowledge graph construction workflow."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        plan = self.plan(task_description)

        self._record("read_text", "running", f"读取文本: {self.medical_text_path}")
        text = ReadTextOperator().execute(str(self.medical_text_path))
        self._record("read_text", "success", f"读取完成，字符数 {len(text)}")

        self._record("split_paragraph", "running", "按疾病章节切分文本")
        paragraphs = SplitParagraphOperator().execute(text, min_length=20)
        self._record("split_paragraph", "success", f"切分完成，共 {len(paragraphs)} 段")

        all_entities: list[dict[str, str]] = []
        all_triples: list[dict[str, str]] = []
        all_relations: list[dict[str, str]] = []

        for paragraph in paragraphs:
            entities = EntityRecognitionOperator().execute(paragraph)
            relations = RelationExtractionOperator().execute(paragraph)
            triples = TripleGenerationOperator().execute(paragraph)
            all_entities.extend(entities)
            all_relations.extend(relations)
            all_triples.extend(triples)

        entities = self._dedupe_entities(all_entities)
        triples = TripleValidationOperator().execute(all_triples)
        graph = GraphBuildOperator().execute(triples, entities)

        self._record("recognize_entities", "success", f"识别实体 {len(entities)} 个")
        self._record("extract_relations", "success", f"抽取关系 {len(all_relations)} 条")
        self._record("generate_triples", "success", f"生成有效三元组 {len(triples)} 条")
        self._record("build_graph", "success", f"图谱包含 {len(graph['nodes'])} 个节点、{len(graph['edges'])} 条边")

        SaveJsonOperator().execute(triples, str(self.triples_path))
        SaveJsonOperator().execute(graph, str(self.graph_path))
        self._save_graphml(graph)
        self._save_html(graph)
        self._record("save_outputs", "success", f"输出已保存到 {self.outputs_dir}")

        return {
            "task": "task2_knowledge_graph_agent",
            "task_description": task_description,
            "status": "success",
            "plan": plan,
            "entity_count": len(entities),
            "triple_count": len(triples),
            "graph": graph,
            "outputs": {
                "triples": str(self.triples_path),
                "graph": str(self.graph_path),
                "graphml": str(self.graphml_path),
                "html": str(self.html_path),
            },
            "execution_state": {"steps": self.steps},
        }

    def _record(self, step: str, status: str, detail: str) -> None:
        self.steps.append({"step": step, "status": status, "detail": detail})
        print(f"[kg-agent] {step}: {status} - {detail}")

    def _dedupe_entities(self, entities: list[dict[str, str]]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for entity in entities:
            name = entity.get("name", "").strip()
            entity_type = entity.get("type", "Unknown").strip() or "Unknown"
            key = (name, entity_type)
            if name and key not in seen:
                seen.add(key)
                result.append({"name": name, "type": entity_type})
        return result

    def _save_graphml(self, graph: dict[str, list[dict[str, str]]]) -> None:
        nx_graph = nx.DiGraph()
        for node in graph["nodes"]:
            nx_graph.add_node(node["id"], label=node["label"], type=node["type"])
        for edge in graph["edges"]:
            nx_graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
        nx.write_graphml(nx_graph, self.graphml_path)

    def _save_html(self, graph: dict[str, list[dict[str, str]]]) -> None:
        node_items = "\n".join(
            f"<li><strong>{node['label']}</strong> <span>{node['type']}</span></li>"
            for node in graph["nodes"]
        )
        edge_items = "\n".join(
            f"<li>{edge['source']} -- {edge['relation']} --&gt; {edge['target']}</li>"
            for edge in graph["edges"]
        )
        html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>医疗知识图谱</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.6; }}
    h1 {{ color: #1f4f75; }}
    section {{ margin-bottom: 24px; }}
    span {{ color: #666; margin-left: 8px; }}
  </style>
</head>
<body>
  <h1>医疗知识图谱</h1>
  <section>
    <h2>节点</h2>
    <ul>{node_items}</ul>
  </section>
  <section>
    <h2>关系</h2>
    <ul>{edge_items}</ul>
  </section>
</body>
</html>
"""
        self.html_path.write_text(html, encoding="utf-8")
