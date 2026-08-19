from pathlib import Path

from agents.data_processing_agent import DataProcessingAgent, StepExecutionPolicy


def test_data_processing_agent_returns_failed_summary_for_missing_required_csv(tmp_path):
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    data_dir.mkdir()

    agent = DataProcessingAgent(data_dir=data_dir, outputs_dir=outputs_dir)

    result = agent.run("只清洗医疗CSV结构化数据")

    assert result["status"] == "failed"
    assert result["execution_state"]["errors"] == [
        {
            "step": "read_csv",
            "error": f"[ReadCsvOperator] 执行失败: 文件不存在: {Path(data_dir / 'medical_records.csv')}",
        }
    ]
    assert result["execution_state"]["steps"][-1]["status"] == "failed"
    assert (outputs_dir / "task1_summary.json").exists()


def test_data_processing_agent_execute_step_retries_before_success(tmp_path):
    agent = DataProcessingAgent(data_dir=tmp_path / "data", outputs_dir=tmp_path / "outputs")
    attempts = {"count": 0}

    def flaky_operation():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("temporary failure")
        return "ok"

    result = agent._execute_step("flaky", flaky_operation, retries=1)

    assert result == "ok"
    assert attempts["count"] == 2
    assert [step["status"] for step in agent.steps] == ["running", "retrying", "success"]


def test_data_processing_agent_can_skip_optional_text_failure(tmp_path):
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    data_dir.mkdir()

    source_csv = Path(__file__).parent.parent / "data" / "medical_records.csv"
    (data_dir / "medical_records.csv").write_text(source_csv.read_text(encoding="utf-8"), encoding="utf-8")

    agent = DataProcessingAgent(
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        skip_optional_text_errors=True,
    )

    result = agent.run("请清洗医疗CSV数据，并处理医疗文本")

    assert result["status"] == "partial_success"
    assert result["structured_result"]["row_count"] == 15
    assert result["text_result"] is None
    assert result["execution_state"]["errors"][0]["step"] == "read_text"
    assert result["execution_state"]["steps"][-1]["status"] == "skipped"


def test_data_processing_agent_uses_step_execution_policy_for_retries(tmp_path):
    agent = DataProcessingAgent(
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
        step_policies={"flaky": StepExecutionPolicy(retries=2)},
    )
    attempts = {"count": 0}

    def flaky_operation():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary failure")
        return "ok"

    result = agent._execute_step("flaky", flaky_operation)

    assert result == "ok"
    assert attempts["count"] == 3
    assert [step["status"] for step in agent.steps] == ["running", "retrying", "retrying", "success"]


def test_data_processing_agent_uses_step_execution_policy_for_optional_skip(tmp_path):
    agent = DataProcessingAgent(
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
        step_policies={"optional": StepExecutionPolicy(required=False)},
    )

    result = agent._execute_step("optional", lambda: (_ for _ in ()).throw(RuntimeError("optional failed")))

    assert result is None
    assert agent.errors == [{"step": "optional", "error": "optional failed"}]
    assert [step["status"] for step in agent.steps] == ["running", "skipped"]
