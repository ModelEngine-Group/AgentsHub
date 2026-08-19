from pathlib import Path

from agents.data_processing_agent import DataProcessingAgent


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def test_data_processing_agent_creates_plan_from_task_description():
    agent = DataProcessingAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR)

    plan = agent.plan("请清洗医疗CSV数据，并处理医疗文本")

    assert plan[0]["step"] == "read_csv"
    assert [item["step"] for item in plan] == [
        "read_csv",
        "clean_missing",
        "normalize_text",
        "extract_fields",
        "save_csv",
        "read_text",
        "split_paragraph",
        "save_text_json",
    ]


def test_data_processing_agent_runs_structured_and_text_pipeline():
    agent = DataProcessingAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR)

    result = agent.run("请清洗医疗CSV数据，并处理医疗文本")

    assert result["status"] == "success"
    assert result["structured_result"]["row_count"] == 15
    assert result["text_result"]["paragraph_count"] == 15
    assert (OUTPUTS_DIR / "cleaned_records.csv").exists()
    assert (OUTPUTS_DIR / "cleaned_text.json").exists()
    assert (OUTPUTS_DIR / "task1_summary.json").exists()
    assert all(step["status"] in {"running", "success"} for step in result["execution_state"]["steps"])
