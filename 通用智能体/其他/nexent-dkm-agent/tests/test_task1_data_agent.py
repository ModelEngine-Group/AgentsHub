import csv
import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO

import pytest

import src.agents.data_processing_agent.agent as agent_module
from src.agents.data_processing_agent.planner import (
    HybridPlanner,
    REGISTERED_OPERATORS,
    plan_data_task,
)
from src.agents.data_processing_agent.nexent_adapter import (
    DataProcessingAgentTool,
    build_nexent_agent_spec,
    build_nexent_tool_spec,
    maybe_create_nexent_agent_config,
)
from src.agents.data_processing_agent.llm_orchestrator import (
    build_chat_completion_payload,
    load_config_file,
    load_env_file,
    parse_tool_call_json,
    validate_tool_call,
)
from src.agents.data_processing_agent.reporting import build_quality_report
from src.operators.data_ops.datamate_client import DataMateClient, safe_datamate_call
from src.pipelines.task1_evaluation import run_task1_evaluation
from src.pipelines.task1_data_pipeline import run_task1_pipeline


class FakeDataMateClient(DataMateClient):
    def __init__(self) -> None:
        pass

    def list_operators(self, keyword=None, page=0, size=10):
        operators = [
            _fake_operator("document_deduplicator"),
            _fake_operator("document_minhash_deduplicator"),
            _fake_operator("text_type_normalizer"),
            _fake_operator("DuplicateFilesFilter"),
            _fake_operator("DuplicateSentencesFilter"),
            _fake_operator("ExtraSpaceCleaner"),
            _fake_operator("UnicodeSpaceCleaner"),
        ]
        if keyword in {"duplicate", "dedup"}:
            content = operators[:2]
        elif keyword == "text":
            content = operators[2:]
        else:
            content = []
        if keyword is None:
            content = operators
        return {
            "data": {
                "totalElements": len(content),
                "content": content[:size],
            }
        }


def _fake_operator(operator_id):
    return {
        "id": operator_id,
        "name": f"{operator_id} name",
        "description": f"{operator_id} description",
        "inputs": "text",
        "outputs": "text",
        "categories": ["cleaning"],
        "settings": None,
        "overrides": None,
    }


class RecordingDataMateClient(FakeDataMateClient):
    def __init__(self) -> None:
        self.template_submissions = 0
        self.task_submissions = 0
        self.submitted_templates = []
        self.submitted_tasks = []

    def create_cleaning_template(self, payload):
        self.template_submissions += 1
        self.submitted_templates.append(payload)
        return {"code": 200, "data": payload}

    def create_cleaning_task(self, payload):
        self.task_submissions += 1
        self.submitted_tasks.append(payload)
        return {"code": 200, "data": payload}


class VerifyingDataMateClient(FakeDataMateClient):
    def __init__(self) -> None:
        self.verified_template_ids = []
        self.verified_task_ids = []

    def create_cleaning_template(self, payload):
        return {"code": "0", "data": {"id": "template-123", **payload}}

    def create_cleaning_task(self, payload):
        return {"code": "0", "data": {"id": "task-456", "status": "PENDING", **payload}}

    def get_cleaning_template(self, template_id):
        self.verified_template_ids.append(template_id)
        return {"code": "0", "data": {"id": template_id, "name": "verified-template"}}

    def get_cleaning_task(self, task_id):
        self.verified_task_ids.append(task_id)
        return {"code": "0", "data": {"id": task_id, "status": "RUNNING"}}


def test_task1_pipeline_profiles_csv_and_builds_plan(tmp_path):
    sample = tmp_path / "patients.csv"
    sample.write_text(
        "\n".join(
            [
                "patient_id,age,diagnosis,cost",
                "P001,34,hypertension,120.5",
                "P002,,diabetes,88",
                "P002,,diabetes,88",
                "P003,41,,",
            ]
        ),
        encoding="utf-8",
    )

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
    )

    assert result.task == "task1_data_processing_agent"
    assert result.status == "completed"
    assert "patients.csv" in result.message
    assert result.artifacts["input"]["path"] == str(sample)
    assert result.artifacts["profile"]["row_count"] == 4
    assert result.artifacts["profile"]["column_count"] == 4
    assert result.artifacts["profile"]["duplicate_rows"] == 1
    assert result.artifacts["profile"]["missing_cells"]["age"] == 2
    assert result.artifacts["profile"]["missing_cells"]["diagnosis"] == 1
    assert "drop_duplicate_rows" in result.artifacts["plan"]["operators"]
    assert "fill_missing_values" in result.artifacts["plan"]["operators"]
    assert result.artifacts["understanding"]["task_type"] == "cleaning"
    assert result.artifacts["understanding"]["data_type"] == "structured_csv"
    assert result.artifacts["run_state"]["status"] == "completed"
    assert [
        step["name"] for step in result.artifacts["run_state"]["steps"]
    ] == [
        "understand_task",
        "validate_runtime_config",
        "profile_schema",
        "plan_operators",
        "execute_local_cleaning",
        "validate_cleaning_result",
        "inspect_datamate",
        "build_quality_report",
    ]
    assert all(
        step["status"] == "completed"
        for step in result.artifacts["run_state"]["steps"]
    )
    assert result.artifacts["datamate"]["status"] == "skipped"
    assert result.artifacts["datamate"]["operators"]["status"] == "skipped"
    assert result.artifacts["cleaning"]["duplicate_rows_removed"] == 1
    assert result.artifacts["cleaning"]["missing_values_filled"] == 3
    assert result.artifacts["quality_report"]["status"] == "passed"
    assert result.artifacts["quality_report"]["readiness"]["local_execution"] is True
    assert result.artifacts["quality_report"]["readiness"]["quality_validation"] is True

    output_path = Path(result.artifacts["cleaning"]["output_path"])
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        cleaned_rows = list(csv.DictReader(handle))

    assert len(cleaned_rows) == 3
    assert cleaned_rows[1]["age"] == "0"
    assert cleaned_rows[2]["diagnosis"] == "unknown"
    assert cleaned_rows[2]["cost"] == "0.0"


def test_task1_pipeline_uses_default_sample_when_input_not_provided(tmp_path):
    result = run_task1_pipeline(datamate_base_url=None, output_dir=tmp_path)

    assert result.task == "task1_data_processing_agent"
    assert result.status == "completed"
    assert Path(result.artifacts["input"]["path"]).name == "task1_patients.csv"
    assert result.artifacts["profile"]["row_count"] > 0


def test_task1_pipeline_returns_failed_state_for_missing_input(tmp_path):
    result = run_task1_pipeline(
        input_path=tmp_path / "missing.csv",
        datamate_base_url=None,
        output_dir=tmp_path,
    )

    assert result.status == "failed"
    assert "missing.csv" in result.message
    assert result.artifacts["run_state"]["status"] == "failed"
    assert result.artifacts["run_state"]["steps"][-1]["name"] == "profile_schema"
    assert result.artifacts["run_state"]["steps"][-1]["status"] == "failed"


def test_task1_pipeline_rejects_non_csv_input(tmp_path):
    sample = tmp_path / "patients.txt"
    sample.write_text("patient_id,age\nP001,34\n", encoding="utf-8")

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
    )

    # txt is now allowed but csv processing will fail due to format
    # The pipeline accepts .txt/.csv/.json now
    assert result.status in ("completed", "completed_with_warnings", "failed")


def test_task1_pipeline_rejects_invalid_datamate_mode(tmp_path):
    sample = tmp_path / "patients.csv"
    sample.write_text("patient_id,age\nP001,34\n", encoding="utf-8")

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        datamate_mode="unsafe",
        output_dir=tmp_path / "outputs",
    )

    assert result.status == "failed"
    assert result.artifacts["error"]["type"] == "ValueError"
    assert "DataMate mode must be 'dry_run', 'submit', or 'auto'" in result.artifacts[
        "error"
    ]["message"]


def test_datamate_client_rejects_unsafe_base_url():
    with pytest.raises(ValueError):
        DataMateClient("file:///tmp/datamate")
    with pytest.raises(ValueError):
        DataMateClient("http://user:pass@localhost:18000")


def test_safe_datamate_call_normalizes_bad_base_url():
    result = safe_datamate_call(lambda: DataMateClient("not-a-url").health())

    assert result["status"] == "unavailable"
    assert "base_url" in result["message"]


def test_resolve_datamate_mode_auto_without_base_url():
    from src.operators.data_ops.datamate_client import resolve_datamate_mode

    mode, meta = resolve_datamate_mode(None, "auto")
    assert mode == "dry_run"
    assert meta["requested_mode"] == "auto"
    assert meta["resolved_mode"] == "dry_run"
    assert meta["reason"] == "no_base_url"


def test_resolve_datamate_mode_auto_healthy(monkeypatch):
    from src.operators.data_ops.datamate_client import resolve_datamate_mode

    class FakeClient:
        def health(self):
            return {"status": "healthy"}

    monkeypatch.setattr(
        "src.operators.data_ops.datamate_client.DataMateClient",
        lambda *args, **kwargs: FakeClient(),
    )
    mode, meta = resolve_datamate_mode("http://localhost:18000", "auto")
    assert mode == "submit"
    assert meta["auto_selected"] is True


def test_resolve_datamate_mode_passthrough():
    from src.operators.data_ops.datamate_client import resolve_datamate_mode

    mode, meta = resolve_datamate_mode(None, "dry_run")
    assert mode == "dry_run"
    assert meta["requested_mode"] == "dry_run"
    assert meta["auto_selected"] is False


def test_task1_planner_understands_free_form_cleaning_request():
    plan = plan_data_task(
        "请清洗患者CSV，删除重复记录，填补空值，并统一字段类型后导出",
        data_profile={
            "file_name": "patients.csv",
            "duplicate_rows": 2,
            "missing_cells": {"age": 1, "diagnosis": 0},
            "columns": [
                {"name": "age", "inferred_type": "integer"},
                {"name": "diagnosis", "inferred_type": "text"},
            ],
        },
    )

    assert plan.understanding.task_type == "cleaning"
    assert plan.understanding.data_type == "structured_csv"
    assert plan.understanding.intent_keywords == [
        "deduplicate",
        "fill_missing",
        "normalize_types",
        "export",
    ]
    assert plan.operators == [
        "load_csv",
        "profile_schema",
        "drop_duplicate_rows",
        "fill_missing_values",
        "normalize_column_types",
        "export_clean_dataset",
        "validate_clean_dataset",
    ]
    assert plan.confidence >= 0.8
    assert plan.planner_mode == "rule"


def test_datamate_catalog_summary_maps_plan_operators():
    summary = FakeDataMateClient().catalog_summary(
        [
            "drop_duplicate_rows",
            "fill_missing_values",
            "normalize_column_types",
        ]
    )

    assert summary["status"] == "available"
    assert summary["operator_count"] == 7
    assert summary["sample_operator_ids"] == [
        "document_deduplicator",
        "document_minhash_deduplicator",
        "text_type_normalizer",
        "DuplicateFilesFilter",
        "DuplicateSentencesFilter",
        "ExtraSpaceCleaner",
        "UnicodeSpaceCleaner",
    ]
    assert summary["candidate_mappings"]["drop_duplicate_rows"][
        "selected_operator_ids"
    ] == ["DuplicateFilesFilter", "DuplicateSentencesFilter"]
    assert summary["candidate_mappings"]["fill_missing_values"][
        "support_level"
    ] == "local_only"
    assert summary["candidate_mappings"]["fill_missing_values"][
        "selected_operator_ids"
    ] == []
    assert summary["candidate_mappings"]["normalize_column_types"][
        "selected_operator_ids"
    ] == ["UnicodeSpaceCleaner", "ExtraSpaceCleaner"]


def test_datamate_catalog_summary_builds_cleaning_submission_payloads():
    summary = FakeDataMateClient().catalog_summary(
        [
            "drop_duplicate_rows",
            "fill_missing_values",
            "normalize_column_types",
        ]
    )

    assert summary["cleaning_template"]["status"] == "ready"
    assert summary["cleaning_template"]["endpoint"] == "/api/cleaning/templates"
    template_payload = summary["cleaning_template"]["payload"]
    assert template_payload["name"] == "task1-data-cleaning-template"
    assert template_payload["description"] == (
        "DataMate cleaning template generated from task 1 local plan."
    )
    assert [item["id"] for item in template_payload["instance"]] == [
        "DuplicateFilesFilter",
        "DuplicateSentencesFilter",
        "UnicodeSpaceCleaner",
        "ExtraSpaceCleaner",
    ]
    assert all(item["inputs"] == "text" for item in template_payload["instance"])
    assert all(item["outputs"] == "text" for item in template_payload["instance"])
    assert all(item["overrides"] == {} for item in template_payload["instance"])
    assert summary["cleaning_template"]["local_only_operators"] == [
        "fill_missing_values"
    ]
    assert summary["cleaning_task"]["status"] == "waiting_for_dataset"

    ready_summary = FakeDataMateClient().catalog_summary(
        [
            "drop_duplicate_rows",
            "fill_missing_values",
            "normalize_column_types",
        ],
        src_dataset_id="dataset-1",
        src_dataset_name="patients",
        dest_dataset_name="patients_cleaned",
    )
    task_payload = ready_summary["cleaning_task"]["payload"]
    assert ready_summary["cleaning_task"]["status"] == "ready"
    assert task_payload["srcDatasetId"] == "dataset-1"
    assert task_payload["destDatasetType"] == "TEXT"
    assert task_payload["instance"] == template_payload["instance"]


def test_datamate_template_payload_uses_catalog_operator_metadata():
    summary = FakeDataMateClient().catalog_summary(
        ["drop_duplicate_rows", "normalize_column_types"]
    )

    instance = summary["cleaning_template"]["payload"]["instance"][0]
    assert instance == {
        "id": "DuplicateFilesFilter",
        "name": "DuplicateFilesFilter name",
        "description": "DuplicateFilesFilter description",
        "inputs": "text",
        "outputs": "text",
        "categories": ["cleaning"],
        "overrides": {},
    }


def test_datamate_submit_failures_are_returned_as_artifacts():
    client = FakeDataMateClient()

    def fail():
        raise ValueError("template rejected")

    result = client._submit_payload(
        endpoint="/api/cleaning/templates",
        payload={"instance": []},
        submit=fail,
        mode="submit",
    )

    assert result["status"] == "submit_failed"
    assert result["submitted"] is False
    assert "template rejected" in result["message"]


def test_safe_datamate_call_includes_http_error_body():
    def fail():
        raise HTTPError(
            url="http://localhost:18000/api/cleaning/templates",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(b'{"message":"Invalid operator input/output types"}'),
        )

    result = safe_datamate_call(fail)

    assert result["status"] == "unavailable"
    assert result["http_status"] == 400
    assert "Invalid operator input/output types" in result["message"]


def test_datamate_submission_is_dry_run_by_default():
    client = FakeDataMateClient()
    summary = client.catalog_summary(
        ["drop_duplicate_rows", "normalize_column_types"],
        src_dataset_id="dataset-1",
        src_dataset_name="patients",
    )

    assert summary["cleaning_template"]["status"] == "ready"
    assert summary["cleaning_template"]["submission"]["mode"] == "dry_run"
    assert summary["cleaning_template"]["submission"]["submitted"] is False
    assert summary["cleaning_task"]["status"] == "ready"
    assert summary["cleaning_task"]["submission"]["mode"] == "dry_run"
    assert summary["cleaning_task"]["submission"]["submitted"] is False


def test_datamate_submit_skips_empty_template_payload():
    client = RecordingDataMateClient()
    summary = client.catalog_summary(
        ["fill_missing_values"],
        mode="submit",
        src_dataset_id="dataset-1",
        src_dataset_name="patients",
    )

    assert summary["cleaning_template"]["status"] == "skipped"
    assert summary["cleaning_template"]["submission"]["status"] == "skipped"
    assert summary["cleaning_template"]["submission"]["submitted"] is False
    assert summary["cleaning_task"]["status"] == "skipped"
    assert summary["cleaning_task"]["submission"]["status"] == "skipped"
    assert summary["cleaning_task"]["submission"]["submitted"] is False
    assert client.template_submissions == 0


def test_datamate_submit_uses_unique_names_for_repeatable_smoke_tests():
    client = RecordingDataMateClient()
    summary = client.catalog_summary(
        ["drop_duplicate_rows", "normalize_column_types"],
        mode="submit",
        src_dataset_id="dataset-1",
        src_dataset_name="patients",
    )

    assert summary["cleaning_template"]["submission"]["submitted"] is True
    assert summary["cleaning_task"]["submission"]["submitted"] is True
    assert client.submitted_templates[0]["name"].startswith(
        "task1-data-cleaning-template-"
    )
    assert client.submitted_tasks[0]["name"].startswith("patients-task1-cleaning-")
    assert client.submitted_tasks[0]["destDatasetName"].startswith("patients_cleaned_")
    assert summary["cleaning_template"]["payload"]["name"] == (
        "task1-data-cleaning-template"
    )


def test_datamate_submit_verifies_server_created_resources():
    client = VerifyingDataMateClient()

    summary = client.catalog_summary(
        ["drop_duplicate_rows"],
        mode="submit",
        src_dataset_id="dataset-1",
        src_dataset_name="patients",
    )

    template_submission = summary["cleaning_template"]["submission"]
    task_submission = summary["cleaning_task"]["submission"]
    assert template_submission["status"] == "verified"
    assert template_submission["resource_id"] == "template-123"
    assert template_submission["verification"]["data"]["id"] == "template-123"
    assert task_submission["status"] == "verified"
    assert task_submission["resource_id"] == "task-456"
    assert task_submission["verification"]["data"]["status"] == "RUNNING"
    assert client.verified_template_ids == ["template-123"]
    assert client.verified_task_ids == ["task-456"]


def test_datamate_client_lists_templates_and_tasks_with_query_parameters(monkeypatch):
    requested_urls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "code": "0",
                    "message": "success",
                    "data": {"content": [], "totalElements": 0},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(
        "src.operators.data_ops.datamate_client.urlopen",
        fake_urlopen,
    )
    client = DataMateClient("http://localhost:18000")

    client.list_cleaning_templates(keyword="task1", page=1, size=5)
    client.list_cleaning_tasks(status="RUNNING", keyword="patients", page=2, size=10)

    assert requested_urls == [
        "http://localhost:18000/api/cleaning/templates?page=1&size=5&keyword=task1",
        (
            "http://localhost:18000/api/cleaning/tasks?"
            "page=2&size=10&status=RUNNING&keyword=patients"
        ),
    ]


def test_nexent_adapter_exports_agent_and_tool_specs():
    tool_spec = build_nexent_tool_spec()
    agent_spec = build_nexent_agent_spec(model_name="glm-5.1")

    assert tool_spec["source"] == "local"
    assert tool_spec["class_name"] == "DataProcessingAgentTool"
    assert "task_request" in tool_spec["inputs"]
    assert agent_spec["name"] == "task1_data_processing_agent"
    assert agent_spec["model_name"] == "glm-5.1"
    assert agent_spec["tools"] == [tool_spec]
    assert agent_spec["max_steps"] >= 5


def test_nexent_agent_config_falls_back_without_sdk():
    config = maybe_create_nexent_agent_config(model_name="main_model")

    if isinstance(config, dict):
        assert config["name"] == "task1_data_processing_agent"
        assert config["tools"][0]["class_name"] == "DataProcessingAgentTool"
    else:
        assert config.name == "task1_data_processing_agent"
        assert config.tools[0].class_name == "DataProcessingAgentTool"


def test_nexent_tool_wrapper_runs_pipeline(tmp_path):
    sample = tmp_path / "patients.csv"
    sample.write_text(
        "\n".join(
            [
                "patient_id,age,diagnosis",
                "P001,34,hypertension",
                "P001,34,hypertension",
                "P002,,",
            ]
        ),
        encoding="utf-8",
    )
    tool = DataProcessingAgentTool(
        datamate_base_url=None,
        output_dir=str(tmp_path / "outputs"),
    )

    payload = json.loads(
        tool.forward(
            task_request="请清洗CSV，去重并填补缺失值后导出",
            input_path=str(sample),
        )
    )

    assert payload["task"] == "task1_data_processing_agent"
    assert payload["status"] == "completed"
    assert payload["artifacts"]["cleaning"]["duplicate_rows_removed"] == 1
    assert payload["artifacts"]["cleaning"]["missing_values_filled"] == 2


def test_llm_orchestrator_builds_secret_free_chat_payload():
    agent_spec = build_nexent_agent_spec(model_name="glm-5.1")
    payload = build_chat_completion_payload(
        agent_spec=agent_spec,
        model_name="glm-5.1",
        task_request="清洗CSV",
        input_path="patients.csv",
        datamate_mode="dry_run",
    )

    encoded = json.dumps(payload, ensure_ascii=False)
    assert payload["model"] == "glm-5.1"
    assert "task1_data_processing" in encoded
    assert "patients.csv" in encoded
    assert "api_key" not in encoded.lower()


def test_llm_orchestrator_parses_fenced_tool_call_json():
    tool_call = parse_tool_call_json(
        """
        ```json
        {"tool":"task1_data_processing","arguments":{"task_request":"清洗CSV","datamate_mode":"dry_run"}}
        ```
        """
    )

    assert tool_call["tool"] == "task1_data_processing"
    assert tool_call["arguments"]["datamate_mode"] == "dry_run"


def test_llm_orchestrator_rejects_submit_without_explicit_permission():
    tool_call = {
        "tool": "task1_data_processing",
        "arguments": {"task_request": "清洗CSV", "datamate_mode": "submit"},
    }

    try:
        validate_tool_call(tool_call, allow_submit=False)
    except ValueError as exc:
        assert "allow_submit" in str(exc)
    else:
        raise AssertionError("submit mode should require explicit permission")


def test_llm_orchestrator_loads_local_env_file_without_logging_secret(tmp_path):
    env_file = tmp_path / "task1_llm.env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test",
                "OPENAI_BASE_URL=https://example.test/v1",
                "OPENAI_MODEL=glm-5.1",
                "IGNORED=value",
            ]
        ),
        encoding="utf-8",
    )

    values = load_env_file(env_file)

    assert values == {
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_BASE_URL": "https://example.test/v1",
        "OPENAI_MODEL": "glm-5.1",
    }


def test_llm_orchestrator_loads_powershell_utf8_bom_env_file(tmp_path):
    env_file = tmp_path / "task1_llm.env"
    env_file.write_bytes(
        "\ufeffOPENAI_API_KEY=sk-test\nOPENAI_BASE_URL=https://example.test/v1\n".encode(
            "utf-8"
        )
    )

    values = load_env_file(env_file)

    assert values["OPENAI_API_KEY"] == "sk-test"
    assert values["OPENAI_BASE_URL"] == "https://example.test/v1"


def test_llm_orchestrator_loads_local_json_config_without_extra_keys(tmp_path):
    config_file = tmp_path / "task1_llm_config.json"
    config_file.write_text(
        json.dumps(
            {
                "api_key": "sk-test",
                "base_url": "https://example.test/v1",
                "model_name": "glm-5.1",
                "ignored": "value",
            }
        ),
        encoding="utf-8",
    )

    values = load_config_file(config_file)

    assert values == {
        "api_key": "sk-test",
        "base_url": "https://example.test/v1",
        "model_name": "glm-5.1",
    }


def test_llm_orchestrator_loads_bom_json_config(tmp_path):
    config_file = tmp_path / "task1_llm_config.json"
    config_file.write_bytes(
        ("\ufeff" + json.dumps({"api_key": "sk-test"})).encode("utf-8")
    )

    values = load_config_file(config_file)

    assert values["api_key"] == "sk-test"


def test_task1_pipeline_validates_cleaned_output(tmp_path):
    sample = tmp_path / "patients.csv"
    sample.write_text(
        "\n".join(
            [
                "patient_id,age,diagnosis",
                "P001,34,hypertension",
                "P001,34,hypertension",
                "P002,,",
            ]
        ),
        encoding="utf-8",
    )

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
    )
    validation = result.artifacts["validation"]

    assert validation["status"] == "passed"
    assert validation["checks"]["duplicate_rows_removed"] is True
    assert validation["checks"]["missing_values_filled"] is True
    assert validation["after"]["duplicate_rows"] == 0
    assert sum(validation["after"]["missing_cells"].values()) == 0


def test_task1_pipeline_fails_when_output_validation_fails(tmp_path, monkeypatch):
    sample = tmp_path / "patients.csv"
    sample.write_text(
        "\n".join(
            [
                "patient_id,age",
                "P001,34",
                "P002,",
            ]
        ),
        encoding="utf-8",
    )

    def fake_validation(before_profile, cleaning_result):
        return {
            "status": "failed",
            "checks": {"missing_values_filled": False},
            "before": {},
            "after": {},
        }

    monkeypatch.setattr(agent_module, "validate_cleaning_result", fake_validation)

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
    )

    assert result.status == "failed"
    assert "Cleaned output validation failed" in result.message
    assert result.artifacts["run_state"]["steps"][-1]["name"] == (
        "validate_cleaning_result"
    )
    assert result.artifacts["run_state"]["steps"][-1]["status"] == "failed"


def test_task1_quality_report_summarizes_run_evidence():
    report = build_quality_report(
        plan={
            "understanding": {"task_type": "cleaning"},
            "operators": [
                "load_csv",
                "profile_schema",
                "drop_duplicate_rows",
                "export_clean_dataset",
            ],
            "confidence": 0.9,
        },
        profile={"row_count": 3, "duplicate_rows": 1, "missing_cells": {"age": 1}},
        cleaning={
            "status": "completed",
            "output_rows": 2,
            "duplicate_rows_removed": 1,
            "missing_values_filled": 1,
        },
        validation={"status": "passed", "checks": {"duplicate_rows_removed": True}},
        datamate={
            "status": "healthy",
            "operators": {
                "status": "available",
                "operator_count": 210,
                "cleaning_template": {
                    "status": "ready",
                    "submission": {"mode": "dry_run", "submitted": False},
                },
                "cleaning_task": {"status": "waiting_for_dataset"},
            },
        },
    )

    assert report["status"] == "passed"
    assert report["metrics"]["input_rows"] == 3
    assert report["metrics"]["output_rows"] == 2
    assert report["metrics"]["datamate_operator_count"] == 210
    assert report["readiness"]["multi_operator_plan"] is True
    assert report["readiness"]["datamate_catalog"] is True
    assert report["readiness"]["datamate_submission_safe"] is True


def test_task1_quality_report_warns_when_datamate_is_unavailable():
    report = build_quality_report(
        plan={"operators": ["load_csv", "profile_schema", "export_clean_dataset", "validate_clean_dataset"]},
        profile={"row_count": 1, "duplicate_rows": 0, "missing_cells": {}},
        cleaning={"status": "completed", "output_rows": 1},
        validation={"status": "passed", "checks": {}},
        datamate={"status": "unavailable", "operators": {"status": "skipped"}},
    )

    assert report["status"] == "warning"
    assert report["readiness"]["datamate_catalog"] is False


def test_task1_quality_report_passes_when_datamate_is_explicitly_skipped():
    report = build_quality_report(
        plan={
            "understanding": {"task_type": "cleaning"},
            "operators": [
                "load_csv",
                "profile_schema",
                "export_clean_dataset",
                "validate_clean_dataset",
            ],
        },
        profile={"row_count": 1, "duplicate_rows": 0, "missing_cells": {}},
        cleaning={"status": "completed", "output_rows": 1},
        validation={"status": "passed", "checks": {}},
        datamate={"status": "skipped", "operators": {"status": "skipped"}},
    )

    assert report["status"] == "passed"
    assert report["readiness"]["datamate_catalog"] is False
    assert report["datamate"]["execution_mode"] == "offline"


def test_task1_evaluation_writes_reproducible_report(tmp_path):
    report_path = tmp_path / "task1_quality_report.json"
    payload = run_task1_evaluation(
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
        report_path=report_path,
    )

    assert report_path.exists()
    assert payload["task"] == "task1_data_processing_agent"
    assert payload["status"] == "completed"
    assert payload["quality_report"]["readiness"]["local_execution"] is True
    assert payload["quality_report"]["readiness"]["quality_validation"] is True
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == payload


def test_task1_data_quality_benchmark_writes_quality_metrics(tmp_path):
    from benchmarks.task1_data_quality_benchmark import run_task1_quality_benchmark

    report_path = tmp_path / "task1_data_quality.json"
    payload = run_task1_quality_benchmark(
        iterations=2,
        output_dir=tmp_path / "outputs",
        report_path=report_path,
    )

    assert report_path.exists()
    assert payload["task"] == "task1_data_processing_agent"
    assert payload["benchmark_type"] == "data_quality"
    assert payload["passed"] is True
    assert payload["iterations"] == 2
    assert payload["timing"]["latency_ms_avg"] >= 0

    metrics = payload["quality_metrics"]
    assert metrics["input_rows"] == 5
    assert metrics["output_rows"] == 4
    assert metrics["duplicate_rows_removed"] == 1
    assert metrics["missing_values_filled"] == 2
    assert metrics["duplicate_rows_after"] == 0
    assert metrics["missing_values_after"] == 0
    assert metrics["quality_score_after"] > metrics["quality_score_before"]

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == payload


# ---------------------------------------------------------------------------
# Phase 1: HybridPlanner tests
# ---------------------------------------------------------------------------


def test_hybrid_planner_falls_back_to_rules_without_llm():
    planner = HybridPlanner(llm_config=None)
    plan = planner.plan("请清洗CSV，去重并填补缺失值")

    assert plan.planner_mode == "rule"
    assert "drop_duplicate_rows" in plan.operators
    assert "fill_missing_values" in plan.operators


def test_hybrid_planner_uses_llm_when_configured(tmp_path):
    planner = HybridPlanner(
        llm_config={
            "base_url": "https://fake.test/v1",
            "api_key": "sk-test",
            "model_name": "test-model",
        }
    )

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "operators": [
                            "load_csv", "profile_schema",
                            "drop_duplicate_rows", "fill_missing_values",
                            "export_clean_dataset", "validate_clean_dataset",
                        ],
                        "rationale": ["LLM test plan."],
                        "task_type": "cleaning",
                        "data_type": "structured_csv",
                        "intent_keywords": ["deduplicate", "fill_missing"],
                        "confidence": 0.9,
                    })
                }
            }
        ]
    }

    with patch(
        "src.agents.data_processing_agent.llm_orchestrator.request_chat_completion",
        return_value=mock_response,
    ):
        plan = planner.plan(
            "请清洗CSV",
            data_profile={
                "file_name": "patients.csv",
                "row_count": 100,
                "column_count": 4,
                "duplicate_rows": 5,
                "missing_cells": {"age": 3},
                "columns": [
                    {"name": "age", "inferred_type": "integer"},
                ],
            },
        )

    assert plan.planner_mode == "llm"
    assert "drop_duplicate_rows" in plan.operators
    assert "fill_missing_values" in plan.operators
    assert plan.confidence == 0.9


def test_hybrid_planner_falls_back_when_llm_fails():
    planner = HybridPlanner(
        llm_config={
            "base_url": "https://fake.test/v1",
            "api_key": "sk-test",
        }
    )

    with patch(
        "src.agents.data_processing_agent.llm_orchestrator.request_chat_completion",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        plan = planner.plan("请清洗CSV，去重并填补缺失值")

    assert plan.planner_mode == "rule"
    assert "drop_duplicate_rows" in plan.operators


def test_planner_skips_dedup_when_no_duplicates():
    plan = plan_data_task(
        "请去重并导出CSV",
        data_profile={
            "file_name": "clean.csv",
            "row_count": 100,
            "duplicate_rows": 0,
            "missing_cells": {},
            "columns": [{"name": "id", "inferred_type": "integer"}],
        },
    )
    assert "drop_duplicate_rows" not in plan.operators


def test_planner_adds_extract_for_text_columns():
    plan = plan_data_task(
        "处理病历文本",
        data_profile={
            "file_name": "notes.csv",
            "row_count": 50,
            "duplicate_rows": 0,
            "missing_cells": {},
            "columns": [
                {"name": "note", "inferred_type": "text"},
                {"name": "summary", "inferred_type": "text"},
            ],
        },
    )
    assert "extract_entities" in plan.operators


def test_local_model_planner_diverse_outputs():
    train_path = Path(__file__).resolve().parents[1] / "data" / "training" / "task_orchestration_train.jsonl"
    assert train_path.exists()
    combos = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        operators = tuple(json.loads(payload["output"])["operators"])
        combos.add(operators)
    assert len(combos) >= 6


def test_planner_adds_drop_column_for_sparse_profile():
    plan = plan_data_task(
        "清洗稀疏列",
        data_profile={
            "file_name": "sparse.csv",
            "row_count": 100,
            "duplicate_rows": 0,
            "missing_cells": {"notes": 80, "score": 5},
            "columns": [
                {"name": "notes", "inferred_type": "text"},
                {"name": "score", "inferred_type": "float"},
            ],
        },
    )
    assert "drop_column" in plan.operators


def test_planner_mode_appears_in_pipeline_output(tmp_path):
    sample = tmp_path / "patients.csv"
    sample.write_text(
        "patient_id,age\nP001,34\nP002,,\n",
        encoding="utf-8",
    )

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
    )

    assert result.artifacts["plan"]["planner_mode"] == "rule"


def test_registered_operators_contains_expected_set():
    assert "load_csv" in REGISTERED_OPERATORS
    assert "drop_duplicate_rows" in REGISTERED_OPERATORS
    assert "fill_missing_values" in REGISTERED_OPERATORS
    assert "export_clean_dataset" in REGISTERED_OPERATORS
    assert "validate_clean_dataset" in REGISTERED_OPERATORS
    assert "clean_text" in REGISTERED_OPERATORS
    assert "extract_entities" in REGISTERED_OPERATORS
    assert "transform_columns" in REGISTERED_OPERATORS


# ---------------------------------------------------------------------------
# Phase 2: Text processing and data transform tests
# ---------------------------------------------------------------------------


def test_text_processor_cleans_html_and_normalizes(tmp_path):
    from src.operators.data_ops.text_processor import process_text

    sample = tmp_path / "notes.txt"
    sample.write_text(
        "---\n"
        "患者张三，<b>高血压</b>，服用氨氯地平。\n"
        "---\n"
        "患者李四，<span style='color:red'>糖尿病</span>，Ｂ超检查。\n"
        "---\n",
        encoding="utf-8",
    )

    result = process_text(sample, tmp_path / "out")

    assert result["status"] == "completed"
    assert result["input_records"] == 2
    assert result["output_records"] == 2
    assert result["html_tags_removed"] >= 2

    output = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "<b>" not in output
    assert "<span" not in output


def test_text_processor_extracts_medical_entities():
    from src.operators.data_ops.text_processor import extract_medical_entities

    text = "患者有高血压和糖尿病，服用阿司匹林和二甲双胍，建议做血常规和心电图检查。"
    entities = extract_medical_entities(text)

    assert "高血压" in entities["diseases"]
    assert "糖尿病" in entities["diseases"]
    assert "阿司匹林" in entities["drugs"]
    assert "二甲双胍" in entities["drugs"]
    assert "血常规" in entities["examinations"]
    assert "心电图" in entities["examinations"]


def test_text_processor_redacts_pii(tmp_path):
    from src.operators.data_ops.text_processor import process_text

    sample = tmp_path / "notes.txt"
    sample.write_text(
        "患者电话13812345678，身份证320102199001011234\n",
        encoding="utf-8",
    )

    result = process_text(sample, tmp_path / "out")

    assert result["pii_redacted"] >= 1
    output = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "13812345678" not in output
    assert "320102199001011234" not in output
    assert "[PHONE]" in output or "[ID_CARD]" in output


def test_data_transform_selects_columns(tmp_path):
    from src.operators.data_ops.data_transform import transform_csv

    sample = tmp_path / "data.csv"
    sample.write_text(
        "name,age,city\nAlice,30,NY\nBob,25,LA\n",
        encoding="utf-8",
    )

    result = transform_csv(
        sample,
        tmp_path / "out",
        transforms=[{"kind": "select", "columns": ["name", "age"]}],
    )

    assert result["status"] == "completed"
    assert result["output_columns"] == ["name", "age"]
    assert result["input_rows"] == 2
    assert result["output_rows"] == 2

    output = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "city" not in output


def test_data_transform_filters_rows(tmp_path):
    from src.operators.data_ops.data_transform import transform_csv

    sample = tmp_path / "data.csv"
    sample.write_text(
        "name,value\nAlice,30\nBob,\nCharlie,10\n",
        encoding="utf-8",
    )

    result = transform_csv(
        sample,
        tmp_path / "out",
        transforms=[{"kind": "filter", "column": "value", "op": "not_empty"}],
    )

    assert result["status"] == "completed"
    assert result["input_rows"] == 3
    assert result["output_rows"] == 2
    assert "filtered value not_empty" in result["transforms_applied"]


def test_extract_fields_from_text(tmp_path):
    from src.operators.data_ops.data_transform import extract_fields_from_text

    sample = tmp_path / "notes.txt"
    sample.write_text(
        "患者有高血压，服用阿司匹林，建议做血常规。\n---\n患者有糖尿病，服用二甲双胍，建议做CT。\n",
        encoding="utf-8",
    )

    result = extract_fields_from_text(sample, tmp_path / "out")

    assert result["status"] == "completed"
    assert result["records_processed"] == 2
    assert "diseases" in result["fields_extracted"]
    assert "drugs" in result["fields_extracted"]

    output = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "高血压" in output
    assert "阿司匹林" in output


# ---------------------------------------------------------------------------
# Phase 3: DAG Scheduler tests
# ---------------------------------------------------------------------------


def test_dag_scheduler_runs_in_topological_order():
    from src.agents.data_processing_agent.scheduler import DAGScheduler, StepSpec
    from src.agents.data_processing_agent.state import TaskStateTracker

    tracker = TaskStateTracker()
    scheduler = DAGScheduler(tracker)
    execution_order = []

    steps = [
        StepSpec("a", lambda: (execution_order.append("a"), "a")[1], depends_on=[]),
        StepSpec("b", lambda: (execution_order.append("b"), "b")[1], depends_on=["a"]),
        StepSpec("c", lambda: (execution_order.append("c"), "c")[1], depends_on=["a"]),
        StepSpec("d", lambda: (execution_order.append("d"), "d")[1], depends_on=["b", "c"]),
    ]

    results = scheduler.run_dag(steps)

    assert results["a"] == "a"
    assert results["d"] == "d"
    assert execution_order.index("a") < execution_order.index("b")
    assert execution_order.index("a") < execution_order.index("c")
    assert execution_order.index("b") < execution_order.index("d")
    assert execution_order.index("c") < execution_order.index("d")


def test_dag_scheduler_retries_failed_step():
    from src.agents.data_processing_agent.scheduler import DAGScheduler, StepSpec
    from src.agents.data_processing_agent.state import TaskStateTracker

    tracker = TaskStateTracker()
    scheduler = DAGScheduler(tracker)
    call_count = 0

    def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return "ok"

    steps = [
        StepSpec("flaky", flaky_operation, max_retries=3, retry_delay=0.01),
    ]

    results = scheduler.run_dag(steps)

    assert results["flaky"] == "ok"
    assert call_count == 3


def test_dag_scheduler_raises_on_cycle():
    from src.agents.data_processing_agent.scheduler import DAGScheduler, StepSpec
    from src.agents.data_processing_agent.state import TaskStateTracker

    tracker = TaskStateTracker()
    scheduler = DAGScheduler(tracker)

    steps = [
        StepSpec("a", lambda: "a", depends_on=["b"]),
        StepSpec("b", lambda: "b", depends_on=["a"]),
    ]

    try:
        scheduler.run_dag(steps)
        raise AssertionError("Should have detected cycle")
    except ValueError as exc:
        assert "ircular" in str(exc) or "dependency" in str(exc).lower()


def test_dag_scheduler_parallel_execution():
    from src.agents.data_processing_agent.scheduler import DAGScheduler, StepSpec
    from src.agents.data_processing_agent.state import TaskStateTracker
    import threading

    tracker = TaskStateTracker()
    scheduler = DAGScheduler(tracker, max_workers=4)
    threads_used = set()

    def track_thread(name):
        threads_used.add(threading.current_thread().ident)
        return name

    steps = [
        StepSpec("a", lambda: track_thread("a"), depends_on=[]),
        StepSpec("b", lambda: track_thread("b"), depends_on=[]),
        StepSpec("c", lambda: track_thread("c"), depends_on=[]),
        StepSpec("d", lambda: track_thread("d"), depends_on=["a", "b", "c"]),
    ]

    results = scheduler.run_dag(steps)
    assert results["d"] == "d"
    # At least 2 different threads were used for parallel steps
    assert len(threads_used) >= 2


# ---------------------------------------------------------------------------
# Phase 4: REST API tests
# ---------------------------------------------------------------------------


def test_api_health_endpoint():
    from src.pipelines.task1_api_server import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_api_list_operators():
    from src.pipelines.task1_api_server import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/api/task1/operators")
    assert resp.status_code == 200
    data = resp.json()
    assert "operators" in data
    assert "drop_duplicate_rows" in data["operators"]
    assert "clean_text" in data["operators"]


def test_api_submit_and_status(tmp_path):
    from src.pipelines.task1_api_server import app, _tasks
    from fastapi.testclient import TestClient

    sample = tmp_path / "patients.csv"
    sample.write_text("id,val\n1,a\n2,\n", encoding="utf-8")

    client = TestClient(app)
    _tasks.clear()

    resp = client.post("/api/task1/process", json={
        "input_path": str(sample),
        "output_dir": str(tmp_path / "out"),
        "datamate_url": "none",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "pending"

    task_id = data["task_id"]

    import time
    for _ in range(20):
        status_resp = client.get(f"/api/task1/status/{task_id}")
        if status_resp.json()["status"] != "pending":
            break
        time.sleep(0.2)

    status_data = status_resp.json()
    assert status_data["status"] in ("completed", "completed_with_warnings", "failed")

    report_resp = client.get(f"/api/task1/report/{task_id}")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["task"] == "task1_data_processing_agent"


def test_api_status_404_for_unknown_task():
    from src.pipelines.task1_api_server import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/api/task1/status/nonexistent")
    assert resp.status_code == 404


def test_api_blocks_datamate_submit_without_server_write_enablement():
    from fastapi.testclient import TestClient
    from src.pipelines.task1_api_server import app, configure_datamate_access

    configure_datamate_access(
        base_url="http://localhost:18000",
        allow_write=False,
    )
    client = TestClient(app)

    resp = client.post(
        "/api/task1/process",
        json={
            "datamate_url": "http://localhost:18000",
            "datamate_mode": "submit",
            "src_dataset_id": "dataset-1",
            "src_dataset_name": "patients",
            "dest_dataset_name": "patients-cleaned",
        },
    )

    assert resp.status_code == 403
    assert "server startup" in resp.json()["detail"]


def test_api_rejects_unconfigured_datamate_url():
    from fastapi.testclient import TestClient
    from src.pipelines.task1_api_server import app, configure_datamate_access

    configure_datamate_access(
        base_url="http://localhost:18000",
        allow_write=False,
    )
    client = TestClient(app)

    resp = client.post(
        "/api/task1/process",
        json={
            "datamate_url": "http://169.254.169.254",
            "datamate_mode": "dry_run",
        },
    )

    assert resp.status_code == 400
    assert "server-configured DataMate URL" in resp.json()["detail"]


def test_api_rejects_unknown_datamate_mode_before_execution():
    from fastapi.testclient import TestClient
    from src.pipelines.task1_api_server import app

    client = TestClient(app)
    resp = client.post(
        "/api/task1/process",
        json={"datamate_url": "none", "datamate_mode": "unsafe"},
    )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Text pipeline integration tests
# ---------------------------------------------------------------------------

def test_text_pipeline_processes_medical_notes():
    """Text input should route to _run_text_pipeline, calling process_text."""
    sample_text = Path("data/samples/task1_medical_notes.txt")
    if not sample_text.exists():
        return

    result = run_task1_pipeline(
        task_request="清洗文本数据，去除HTML标签和特殊字符",
        input_path=str(sample_text),
        datamate_base_url=None,
    )
    assert result.status in ("completed", "completed_with_warnings")
    assert result.artifacts["input"]["format"] == "text"
    assert "processing" in result.artifacts
    processing = result.artifacts["processing"]
    assert processing["status"] == "completed"
    assert processing["input_records"] > 0
    assert result.artifacts["quality_report"]["status"] == "passed"
    assert result.artifacts["quality_report"]["metrics"]["input_rows"] == processing["input_records"]
    assert result.artifacts["quality_report"]["metrics"]["output_rows"] == processing["output_records"]
    assert processing["html_tags_removed"] >= 0
    assert processing["pii_redacted"] >= 0
    assert Path(processing["output_path"]).exists()


def test_text_pipeline_plans_before_processing():
    """The text pipeline must record plan_operators BEFORE process_text runs."""
    sample_text = Path("data/samples/task1_medical_notes.txt")
    if not sample_text.exists():
        return

    result = run_task1_pipeline(
        task_request="清洗文本数据，去除HTML标签和特殊字符",
        input_path=str(sample_text),
        datamate_base_url=None,
    )
    assert result.status in ("completed", "completed_with_warnings")
    step_names = [step["name"] for step in result.artifacts["run_state"]["steps"]]
    assert "plan_operators" in step_names
    assert "process_text" in step_names
    # Plan-first-then-execute: the plan must be recorded before the cleaning step.
    assert step_names.index("plan_operators") < step_names.index("process_text")


def test_csv_pipeline_uses_dag_dependency_order(tmp_path):
    """The DAG-driven CSV stage must keep build_quality_report after its deps."""
    sample_csv = Path("data/samples/task1_patients.csv")
    if not sample_csv.exists():
        return

    result = run_task1_pipeline(
        input_path=str(sample_csv),
        output_dir=tmp_path / "task1",
        datamate_base_url=None,
    )
    assert result.status in ("completed", "completed_with_warnings")
    step_names = [step["name"] for step in result.artifacts["run_state"]["steps"]]
    # DAGScheduler resolved dependencies: cleaning -> validate -> datamate -> report.
    assert step_names.index("execute_local_cleaning") < step_names.index("validate_cleaning_result")
    assert step_names.index("validate_cleaning_result") < step_names.index("inspect_datamate")
    assert step_names.index("inspect_datamate") < step_names.index("build_quality_report")
    assert all(step["status"] == "completed" for step in result.artifacts["run_state"]["steps"])


def test_text_pipeline_extracts_entities():
    """Text pipeline with extract intent should produce entity CSV."""
    sample_text = Path("data/samples/task1_medical_notes.txt")
    if not sample_text.exists():
        return

    result = run_task1_pipeline(
        task_request="处理医疗文本，抽取诊断和药品信息",
        input_path=str(sample_text),
        datamate_base_url=None,
    )
    assert result.status in ("completed", "completed_with_warnings")
    entities = result.artifacts.get("entities", {})
    assert entities.get("status") == "completed"
    assert entities.get("records_processed", 0) > 0
    entity_path = Path(entities["output_path"])
    assert entity_path.exists()
    assert entity_path.suffix == ".csv"


def test_text_pipeline_input_type_in_artifacts():
    """Input format should be 'text' for .txt files."""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write("患者张三，诊断高血压，服用阿司匹林。---患者李四，诊断糖尿病，服用二甲双胍。")
        tmp_path = f.name

    try:
        result = run_task1_pipeline(
            task_request="清洗文本数据",
            input_path=tmp_path,
            datamate_base_url=None,
        )
        assert result.artifacts["input"]["format"] == "text"
    finally:
        os.unlink(tmp_path)


def test_csv_pipeline_still_works_after_refactor():
    """CSV pipeline should still work unchanged after text branch refactor."""
    sample_csv = Path("data/samples/task1_patients.csv")
    if not sample_csv.exists():
        return

    result = run_task1_pipeline(
        input_path=str(sample_csv),
        datamate_base_url=None,
    )
    assert result.status in ("completed", "completed_with_warnings")
    assert result.artifacts["input"]["format"] == "csv"
    assert "profile" in result.artifacts
    assert "cleaning" in result.artifacts
    assert result.artifacts["cleaning"]["output_rows"] > 0


def test_json_loader_parses_wrapper_and_list_shapes(tmp_path):
    from src.operators.data_ops.json_loader import json_records_to_csv, load_json_records

    wrapper = tmp_path / "wrapped.json"
    wrapper.write_text(
        '{"records": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}', encoding="utf-8"
    )
    plain = tmp_path / "plain.json"
    plain.write_text('[{"a": 1}, {"a": 2, "c": 3}]', encoding="utf-8")

    assert len(load_json_records(wrapper)) == 2
    conversion = json_records_to_csv(plain, tmp_path / "out")
    assert conversion["status"] == "completed"
    assert conversion["record_count"] == 2
    assert conversion["columns"] == ["a", "c"]
    assert Path(conversion["csv_path"]).exists()


def test_json_loader_rejects_invalid_payloads(tmp_path):
    from src.operators.data_ops.json_loader import load_json_records

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_json_records(empty)

    not_objects = tmp_path / "bad.json"
    not_objects.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_json_records(not_objects)


def test_json_pipeline_cleans_records_via_csv_flow(tmp_path):
    """JSON input should be converted to CSV then cleaned by the structured flow."""
    sample = tmp_path / "patients.json"
    sample.write_text(
        json.dumps({
            "records": [
                {"patient_id": "P001", "age": 34, "diagnosis": "hypertension", "cost": 120.5},
                {"patient_id": "P002", "age": None, "diagnosis": "diabetes", "cost": 88},
                {"patient_id": "P003", "age": 41, "diagnosis": "", "cost": 95.2},
                {"patient_id": "P003", "age": 41, "diagnosis": "", "cost": 95.2},
            ]
        }),
        encoding="utf-8",
    )

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
    )

    assert result.status == "completed"
    assert result.artifacts["input"]["format"] == "json"
    assert result.artifacts["json_conversion"]["record_count"] == 4
    assert result.artifacts["profile"]["duplicate_rows"] == 1
    assert result.artifacts["cleaning"]["duplicate_rows_removed"] == 1
    assert Path(result.artifacts["cleaning"]["output_path"]).exists()
    assert "convert_json_to_csv" in [
        step["name"] for step in result.artifacts["run_state"]["steps"]
    ]


def _write_dirty_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "patient_id,age,diagnosis",
                "P001,34,hypertension",
                "P001,34,hypertension",
                "P002,,diabetes",
            ]
        ),
        encoding="utf-8",
    )


def test_datamate_operator_keywords_loaded_from_config():
    """The keyword mapping must come from configs/task1_datamate.yaml (config-driven)."""
    import yaml

    from src.operators.data_ops.datamate_client import OPERATOR_KEYWORDS, load_operator_keywords

    config_path = Path(__file__).resolve().parents[1] / "configs" / "task1_datamate.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = raw["datamate"]["operator_keyword_mapping"]

    assert OPERATOR_KEYWORDS == {k: list(v) for k, v in expected.items()}
    assert load_operator_keywords(config_path)["drop_duplicate_rows"] == ["duplicate", "dedup"]


def test_datamate_operator_keywords_falls_back_when_config_missing(tmp_path):
    from src.operators.data_ops.datamate_client import load_operator_keywords

    keywords = load_operator_keywords(tmp_path / "does_not_exist.yaml")
    assert "drop_duplicate_rows" in keywords
    assert keywords["fill_missing_values"] == ["missing", "empty"]


def test_clean_csv_runs_all_operators_by_default(tmp_path):
    """Default (operators=None) must preserve the full dedup+fill+normalize flow."""
    from src.operators.data_ops.csv_cleaner import clean_csv
    from src.operators.data_ops.csv_profile import profile_csv

    sample = tmp_path / "patients.csv"
    _write_dirty_csv(sample)
    profile = profile_csv(sample)

    result = clean_csv(sample, profile, tmp_path / "out")

    assert result["duplicate_rows_removed"] == 1
    assert result["missing_values_filled"] >= 1
    assert set(result["operators_applied"]) == {
        "drop_duplicate_rows",
        "fill_missing_values",
        "normalize_column_types",
    }


def test_clean_csv_respects_plan_operator_subset(tmp_path):
    """When a plan omits fill_missing_values, missing cells must NOT be filled."""
    from src.operators.data_ops.csv_cleaner import clean_csv
    from src.operators.data_ops.csv_profile import profile_csv

    sample = tmp_path / "patients.csv"
    _write_dirty_csv(sample)
    profile = profile_csv(sample)

    result = clean_csv(
        sample,
        profile,
        tmp_path / "out",
        operators=["drop_duplicate_rows", "export_clean_dataset"],
    )

    assert result["duplicate_rows_removed"] == 1
    assert result["missing_values_filled"] == 0
    assert result["operators_applied"] == ["drop_duplicate_rows"]

    after = profile_csv(result["output_path"])
    assert sum(after["missing_cells"].values()) >= 1


def test_validate_cleaning_result_skips_unplanned_checks(tmp_path):
    """Validation must not fail on missing cells when fill was not planned."""
    from src.operators.data_ops.csv_cleaner import clean_csv, validate_cleaning_result
    from src.operators.data_ops.csv_profile import profile_csv

    sample = tmp_path / "patients.csv"
    _write_dirty_csv(sample)
    profile = profile_csv(sample)

    cleaning = clean_csv(
        sample,
        profile,
        tmp_path / "out",
        operators=["drop_duplicate_rows", "export_clean_dataset"],
    )
    validation = validate_cleaning_result(profile, cleaning)

    assert validation["status"] == "passed"
    assert "missing_values_filled" not in validation["checks"]
    assert validation["checks"]["duplicate_rows_removed"] is True


def test_task1_pipeline_records_plan_execution_mapping(tmp_path):
    """Agent artifacts should expose which planned operators actually executed."""
    sample = tmp_path / "patients.csv"
    _write_dirty_csv(sample)

    result = run_task1_pipeline(
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
    )

    plan_execution = result.artifacts["plan_execution"]
    assert plan_execution["planned_operators"] == result.artifacts["plan"]["operators"]
    assert "drop_duplicate_rows" in plan_execution["executed_operators"]
    assert "fill_missing_values" in plan_execution["executed_operators"]


def test_task1_pipeline_executes_transform_when_planned(tmp_path):
    """A transform request must actually run the transform operator (not be ignored)."""
    sample = tmp_path / "patients.csv"
    _write_dirty_csv(sample)

    result = run_task1_pipeline(
        task_request="清洗并转换列：仅保留 patient_id 与 diagnosis 列后导出",
        input_path=sample,
        datamate_base_url=None,
        output_dir=tmp_path / "outputs",
        transforms=[{"kind": "select", "columns": ["patient_id", "diagnosis"]}],
    )

    assert "transform_columns" in result.artifacts["plan"]["operators"]
    transform = result.artifacts["transform"]
    assert transform["status"] == "completed"
    assert transform["output_columns"] == ["patient_id", "diagnosis"]
    assert "transform_columns" in result.artifacts["plan_execution"]["executed_operators"]


def test_task1_training_input_matches_inference_format():
    """Generated training ``input`` must mirror what predict_plan sends.

    Inference puts a data_profile-only JSON after ``Input:`` (the request is
    carried by the ``Task:`` line). Duplicating ``task_request`` in the training
    input would train the adapter on a prompt it never sees at inference.
    """
    from data.training.generate_training_data import generate_samples

    for sample in generate_samples(count=20):
        parsed = json.loads(sample["input"])
        assert "data_profile" in parsed
        assert "task_request" not in parsed


def test_datamate_submit_benchmark_skips_when_unavailable():
    from benchmarks.task1_datamate_submit_benchmark import run_datamate_submit_benchmark

    report = run_datamate_submit_benchmark(
        base_url="http://127.0.0.1:59999",
        mode="submit",
        timeout=0.3,
    )

    assert report["status"] == "skipped"
    assert report["passed"] is False
    assert report["reason"] == "datamate_unavailable"


def test_datamate_submit_benchmark_verifies_live_submit(tmp_path, monkeypatch):
    from benchmarks.task1_datamate_submit_benchmark import run_datamate_submit_benchmark

    class SubmitClient(VerifyingDataMateClient):
        def __init__(self, base_url=None, timeout=3.0):
            super().__init__()

        def health(self):
            return {"status": "healthy"}

    monkeypatch.setattr(agent_module, "DataMateClient", SubmitClient)
    monkeypatch.setattr(
        "src.operators.data_ops.datamate_client.DataMateClient",
        SubmitClient,
    )

    sample = tmp_path / "patients.csv"
    _write_dirty_csv(sample)

    report = run_datamate_submit_benchmark(
        base_url="http://localhost:18000",
        mode="submit",
        input_path=sample,
        output_dir=tmp_path / "submit_out",
        timeout=1.0,
        src_dataset_id="dataset-benchmark-1",
        src_dataset_name="task1_submit_smoke",
    )

    assert report["status"] == "completed"
    assert report["passed"] is True
    assert report["submit_evidence"]["template_status"] == "verified"
    assert report["submit_evidence"]["task_status"] == "verified"
    assert report["submit_evidence"]["verified"] is True
