import json
from pathlib import Path

from adapters import nexent_mcp_tools
from adapters.nexent_mcp_tools import analyze_medical_data_tool, ask_medical_kg_tool


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def use_tmp_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(nexent_mcp_tools, "DEFAULT_OUTPUTS_DIR", tmp_path)
    return tmp_path


def test_process_medical_data_tool_returns_json_serializable_result(monkeypatch, tmp_path):
    use_tmp_outputs(monkeypatch, tmp_path)
    result = nexent_mcp_tools.process_medical_data_tool(
        "请清洗医疗CSV数据，并处理医疗文本",
        data_dir=str(DATA_DIR),
        outputs_dir=str(tmp_path),
    )

    assert result["status"] == "success"
    assert "structured_result" in result
    assert "text_result" in result
    assert "execution_state" in result
    assert result["execution_state"]["errors"] == []
    json.dumps(result, ensure_ascii=False)


def test_build_medical_knowledge_graph_tool_returns_json_serializable_result(monkeypatch, tmp_path):
    use_tmp_outputs(monkeypatch, tmp_path)
    result = nexent_mcp_tools.build_medical_knowledge_graph_tool(
        "请从医疗文本中抽取实体关系，并生成医疗知识图谱",
        data_dir=str(DATA_DIR),
        outputs_dir=str(tmp_path),
    )

    assert result["status"] == "success"
    assert result["triple_count"] > 0
    assert result["graph"]["nodes"]
    assert result["graph"]["edges"]
    assert (tmp_path / "medical_kg.json").exists()
    json.dumps(result, ensure_ascii=False)


def test_ask_medical_kg_tool_reports_missing_graph_prerequisite(monkeypatch, tmp_path):
    use_tmp_outputs(monkeypatch, tmp_path)
    result = ask_medical_kg_tool("肺炎有哪些症状？", outputs_dir=str(tmp_path))

    assert result["status"] == "missing_prerequisite"
    assert result["tool"] == "ask_medical_kg"
    assert str(tmp_path / "medical_kg.json") in result["required"]
    assert "build_medical_knowledge_graph" in result["message"]
    json.dumps(result, ensure_ascii=False)


def test_ask_medical_kg_tool_answers_after_graph_exists(monkeypatch, tmp_path):
    use_tmp_outputs(monkeypatch, tmp_path)
    nexent_mcp_tools.build_medical_knowledge_graph_tool(
        "请从医疗文本中抽取实体关系，并生成医疗知识图谱",
        data_dir=str(DATA_DIR),
        outputs_dir=str(tmp_path),
    )

    result = ask_medical_kg_tool("肺炎有哪些症状？", outputs_dir=str(tmp_path))

    assert result["status"] == "success"
    assert "发热" in result["answers"]
    assert result["question_type"] == "disease_symptoms"
    json.dumps(result, ensure_ascii=False)


def test_analyze_medical_data_tool_reports_missing_prerequisites(monkeypatch, tmp_path):
    use_tmp_outputs(monkeypatch, tmp_path)
    result = analyze_medical_data_tool("请分析医疗知识图谱", outputs_dir=str(tmp_path))

    assert result["status"] == "missing_prerequisite"
    assert result["tool"] == "analyze_medical_data"
    assert str(tmp_path / "medical_kg.json") in result["required"]
    assert str(tmp_path / "cleaned_records.csv") in result["required"]
    assert "process_medical_data" in result["message"]
    assert "build_medical_knowledge_graph" in result["message"]
    json.dumps(result, ensure_ascii=False)


def test_analyze_medical_data_tool_returns_json_serializable_result(monkeypatch, tmp_path):
    use_tmp_outputs(monkeypatch, tmp_path)
    nexent_mcp_tools.process_medical_data_tool(
        "请清洗医疗CSV数据，并处理医疗文本",
        data_dir=str(DATA_DIR),
        outputs_dir=str(tmp_path),
    )
    nexent_mcp_tools.build_medical_knowledge_graph_tool(
        "请从医疗文本中抽取实体关系，并生成医疗知识图谱",
        data_dir=str(DATA_DIR),
        outputs_dir=str(tmp_path),
    )

    result = analyze_medical_data_tool("请分析医疗知识图谱，生成统计结果、图表和报告", outputs_dir=str(tmp_path))

    assert result["status"] == "success"
    assert result["statistics"]["node_count"] > 0
    assert result["statistics"]["edge_count"] > 0
    assert result["sqlite_result"]["row_count"] == 15
    assert (tmp_path / "analysis_summary.json").exists()
    json.dumps(result, ensure_ascii=False)
