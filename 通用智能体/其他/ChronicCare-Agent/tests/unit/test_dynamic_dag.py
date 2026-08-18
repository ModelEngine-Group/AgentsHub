from __future__ import annotations

import json

import pytest

from orchestration.dag import engine


@pytest.fixture
def dag_input(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "patients.csv").write_text("patient_id\nP001\n", encoding="utf-8")
    (root / "notes.md").write_text("随访记录", encoding="utf-8")
    (root / "data_manifest.json").write_text(
        json.dumps({"data_version": "test", "patient_count": 1, "content_sha256": "abc"}),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def isolated_runs(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(engine, "RUNS", runs)
    monkeypatch.setattr(engine.time, "sleep", lambda _: None)
    return runs


def test_goal_normalization_accepts_aliases_and_rejects_unknown() -> None:
    assert engine.normalize_goal("只重建知识图谱") == "kg"
    assert engine.normalize_goal("请运行完整主线") == "full"
    with pytest.raises(ValueError, match="unsupported goal"):
        engine.normalize_goal("导出病历")


def test_profile_input_reports_manifest_and_content(dag_input) -> None:
    profile = engine.profile_input(str(dag_input), ["full"], use_npu=True)
    assert profile["exists"] is True
    assert profile["file_count"] == 3
    assert profile["contains_text"] is True
    assert profile["needs_graph"] is True
    assert profile["needs_sqlite"] is True
    assert profile["needs_npu"] is True
    assert profile["patient_count"] == 1
    assert len(profile["input_hash"]) == 64


def test_path_hash_changes_with_input_content(dag_input) -> None:
    before = engine.path_content_hash(dag_input)
    (dag_input / "patients.csv").write_text("patient_id\nP002\n", encoding="utf-8")
    assert engine.path_content_hash(dag_input) != before
    assert (
        engine.path_content_hash(dag_input / "missing") == engine.hash_value("")
        or len(engine.path_content_hash(dag_input / "missing")) == 64
    )


def test_build_plan_selects_cpu_and_npu_operator_variants(dag_input) -> None:
    cpu = engine.build_plan("kg", str(dag_input), use_npu=False)
    npu = engine.build_plan("kg", str(dag_input), use_npu=True)
    cpu_names = {item["name"] for item in cpu["nodes"]}
    npu_names = {item["name"] for item in npu["nodes"]}
    assert "chronic_entity_extract" in cpu_names
    assert "chronic_relation_extract" in cpu_names
    assert "chronic_entity_extract_model_npu" in npu_names
    assert "chronic_relation_extract_model_npu" in npu_names
    assert npu["estimated_resources"]["npu"] == 1
    assert npu["validation"]["acyclic"] is True


def test_build_plan_detects_cycle(monkeypatch, dag_input) -> None:
    deps = {key: list(value) for key, value in engine.DEPS.items()}
    deps["chronic_file_ingest"] = ["chronic_table_clean"]
    monkeypatch.setattr(engine, "DEPS", deps)
    with pytest.raises(ValueError, match="DAG_CYCLE"):
        engine.build_plan("clean", str(dag_input))


def test_dag_dry_run_does_not_write(dag_input, isolated_runs) -> None:
    plan = engine.build_plan("clean", str(dag_input))
    result = engine.DagEngine(lambda *_: {"status": "success"}).run(plan, dry_run=True)
    assert result["status"] == "validated"
    assert result["writes_performed"] is False
    assert not isolated_runs.exists()


def test_dag_rejects_missing_input(isolated_runs, tmp_path) -> None:
    plan = engine.build_plan("clean", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        engine.DagEngine(lambda *_: {"status": "success"}).run(plan)


class RecordingRunner:
    def __init__(self, status: str = "success"):
        self.status = status
        self.calls: list[str] = []

    def __call__(self, name, context):
        self.calls.append(name)
        return {
            "status": self.status,
            "artifact_hash": engine.hash_value({"name": name, "input": context["input_hash"]}),
        }

    def finalize(self, run_id):
        return {"artifact_root": f"runs/{run_id}/artifacts"}


def test_dag_executes_nodes_and_materializes(dag_input, isolated_runs) -> None:
    runner = RecordingRunner()
    plan = engine.build_plan("clean", str(dag_input))
    result = engine.DagEngine(runner).run(plan, resume_run_id="run-success")
    assert result["status"] == "succeeded"
    assert runner.calls == [item["name"] for item in plan["nodes"]]
    assert all(item["state"] == "succeeded" for item in result["nodes"])
    assert result["materialization"]["artifact_root"].endswith("/artifacts")
    assert engine.get_run("run-success")["status"] == "succeeded"


def test_dag_resume_reuses_valid_cache(dag_input, isolated_runs) -> None:
    runner = RecordingRunner()
    plan = engine.build_plan("clean", str(dag_input))
    executor = engine.DagEngine(runner)
    executor.run(plan, resume_run_id="run-cache")
    runner.calls.clear()
    resumed = executor.run(plan, resume_run_id="run-cache")
    assert runner.calls == []
    assert resumed["cache_hits"] == [item["name"] for item in plan["nodes"]]
    assert all(item["state"] == "skipped" for item in resumed["nodes"])


def test_dag_resume_from_forces_selected_node_and_downstream(dag_input, isolated_runs) -> None:
    runner = RecordingRunner()
    plan = engine.build_plan("clean", str(dag_input))
    executor = engine.DagEngine(runner)
    executor.run(plan, resume_run_id="run-resume")
    runner.calls.clear()
    resumed = executor.run(
        plan,
        resume_run_id="run-resume",
        resume_from="chronic_table_clean",
    )
    assert resumed["cache_hits"] == ["chronic_file_ingest"]
    assert runner.calls == ["chronic_table_clean", "chronic_field_normalize"]


def test_dag_retries_timeout_then_succeeds(dag_input, isolated_runs) -> None:
    runner = RecordingRunner()
    plan = engine.build_plan("clean", str(dag_input))
    result = engine.DagEngine(runner).run(
        plan,
        resume_run_id="run-retry",
        fail_node="chronic_file_ingest",
        fail_attempts=1,
    )
    assert result["status"] == "succeeded"
    assert result["nodes"][0]["attempts"] == 2
    assert result["nodes"][0]["state"] == "succeeded"


def test_dag_stops_after_non_retryable_operator_failure(dag_input, isolated_runs) -> None:
    plan = engine.build_plan("clean", str(dag_input))
    result = engine.DagEngine(RecordingRunner(status="failed")).run(
        plan,
        resume_run_id="run-failed",
    )
    assert result["status"] == "failed"
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["state"] == "failed"
    assert result["nodes"][0]["error"]["type"] == "RuntimeError"


def test_get_run_returns_not_found(isolated_runs) -> None:
    assert engine.get_run("missing") == {
        "status": "not_found",
        "state": "not_found",
        "run_id": "missing",
    }
