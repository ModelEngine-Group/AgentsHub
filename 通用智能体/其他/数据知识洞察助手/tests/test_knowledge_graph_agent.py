import json
from pathlib import Path

from agents import KnowledgeGraphAgent


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def test_knowledge_graph_agent_builds_graph_outputs():
    agent = KnowledgeGraphAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR)

    result = agent.run("请从医疗文本中抽取实体关系，并生成医疗知识图谱")

    assert result["status"] == "success"
    assert result["triple_count"] > 0
    assert result["graph"]["nodes"]
    assert result["graph"]["edges"]

    triples_path = OUTPUTS_DIR / "medical_triples.json"
    kg_path = OUTPUTS_DIR / "medical_kg.json"
    graphml_path = OUTPUTS_DIR / "medical_kg.graphml"
    html_path = OUTPUTS_DIR / "medical_kg.html"

    assert triples_path.exists()
    assert kg_path.exists()
    assert graphml_path.exists()
    assert html_path.exists()

    with open(triples_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    assert {"subject": "肺炎", "predicate": "has_symptom", "object": "发热"} in triples

    with open(kg_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    assert "nodes" in graph
    assert "edges" in graph
    assert {"id": "肺炎", "label": "肺炎", "type": "Disease"} in graph["nodes"]
