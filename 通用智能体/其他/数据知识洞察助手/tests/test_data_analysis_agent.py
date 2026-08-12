from pathlib import Path

import matplotlib

from agents import DataAnalysisAgent, KnowledgeGraphAgent


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def prepare_graph() -> None:
    KnowledgeGraphAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR).run(
        "请从医疗文本中抽取实体关系，并生成医疗知识图谱"
    )


def test_data_analysis_uses_non_interactive_chart_backend():
    assert matplotlib.get_backend().lower() == "agg"


def test_data_analysis_agent_generates_charts_and_report():
    prepare_graph()
    agent = DataAnalysisAgent(outputs_dir=OUTPUTS_DIR)

    result = agent.run("请分析医疗知识图谱，生成统计结果、图表和报告")

    assert result["status"] == "success"
    assert result["statistics"]["node_count"] > 0
    assert result["statistics"]["edge_count"] > 0
    assert result["disease_symptom_counts"]
    assert result["drug_disease_counts"]
    assert result["centrality"]
    assert result["sqlite_result"]["row_count"] == 15
    assert result["nl2sql_metrics"]["accuracy"] >= 0.85

    assert (OUTPUTS_DIR / "charts" / "disease_symptom_count.png").exists()
    assert (OUTPUTS_DIR / "charts" / "drug_disease_count.png").exists()
    assert (OUTPUTS_DIR / "medical_records.db").exists()
    assert (OUTPUTS_DIR / "analysis_report.md").exists()
    assert (OUTPUTS_DIR / "analysis_summary.json").exists()
