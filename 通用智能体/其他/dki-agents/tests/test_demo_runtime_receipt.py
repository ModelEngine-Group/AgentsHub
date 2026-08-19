from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from dki import cli


ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"
RID = "20260807T120000Z-g1234567-t000000000000"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_manifest(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "dki" / "cli.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# sealed executable fixture\n", encoding="utf-8")
    entry = {
        "path": "dki/cli.py",
        "size": source.stat().st_size,
        "sha256": _sha256(source),
    }
    digest = hashlib.sha256()
    digest.update(entry["path"].encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(entry["size"]).encode("ascii"))
    digest.update(b"\0")
    digest.update(entry["sha256"].encode("ascii"))
    digest.update(b"\n")
    tree = digest.hexdigest()
    manifest = tmp_path / "evidence" / "included_tree_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "rid": RID,
        "git_head": "1" * 40,
        "algorithm": ALGORITHM,
        "included_tree_sha256": tree,
        "entries": [entry],
    }), encoding="utf-8")
    return manifest, tree


def _write_agent_observation(
    output: Path,
    *,
    index: int,
    agent: str,
    calls: list[dict],
    offline: bool = True,
) -> None:
    path = output / "agent_observations" / f"{index:03d}-{agent}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "dki_demo_agent_observation",
        "agent": agent,
        "status": "completed",
        "offline": offline,
        "model_backend": "ScriptedModel" if offline else "OpenAIModel",
        "model_implementation": "smolagents.models.ScriptedFixture",
        "core_agent": "nexent.core.agents.CoreAgent",
        "task_sha256": str(index) * 64,
        "task_length": 20,
        "trace_source": "instrumented_tool_forward",
        "instrumented_tool_count": 2,
        "instrumentation_failures": [],
        "tool_calls": calls,
        "privacy": {
            "contains_full_user_input": False,
            "contains_tool_arguments": False,
            "contains_tool_results": False,
            "contains_secrets": False,
        },
    }), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    output = tmp_path / "outputs"
    processed = output / "processed"
    tasks = output / "tasks"
    processed.mkdir(parents=True)
    tasks.mkdir(parents=True)
    report = tasks / "task-1.json"
    report.write_text(json.dumps({
        "schema_version": 3,
        "task_id": "task-1",
        "status": "COMPLETED",
        "n_input": 2,
        "n_output": 2,
        "n_failed": 0,
        "n_filtered": 0,
        "duration_s": 0.25,
        "op_metrics": [{"op_name": "MedicalTextCleaner", "wall_s": 0.2}],
        "recovery_events": [{"op_name": "DemoTransientOnce", "applied": True}],
    }), encoding="utf-8")
    manifest = processed / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "task_id": "task-1",
        "source": {
            "raw_text_files": [
                {"path": "data/raw/a.txt", "bytes": 3, "sha256": "a" * 64},
                {"path": "data/raw/b.txt", "bytes": 4, "sha256": "b" * 64},
            ]
        },
        "task_report": {
            "path": "outputs/tasks/task-1.json",
            "bytes": report.stat().st_size,
            "sha256": _sha256(report),
        },
    }), encoding="utf-8")

    artifact_paths = {
        "kg": output / "kg" / "medical_kg.json",
        "kg_provenance": output / "kg" / "medical_kg.provenance.json",
        "database": output / "insight.db",
        "database_provenance": output / "insight.provenance.json",
        "bi_report": output / "reports" / "insight_report.html",
    }
    for name, path in artifact_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{name}\n", encoding="utf-8")

    call = lambda tool, attempt=1: {
        "tool": tool,
        "attempt": attempt,
        "retry": attempt > 1,
        "observed": True,
        "outcome": "reported_success",
        "duration_ms": 1.25,
    }
    _write_agent_observation(
        output,
        index=1,
        agent="data",
        calls=[call("list_operators"), call("get_task_status"), call("get_task_status", 2)],
    )
    _write_agent_observation(output, index=2, agent="kg", calls=[call("kg_query")])
    _write_agent_observation(
        output,
        index=3,
        agent="insight",
        calls=[call("nl2sql_query")],
    )

    source_manifest, tree = _write_source_manifest(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_DIR", output)
    monkeypatch.setattr(cli, "PROCESSED_DIR", processed)
    monkeypatch.setenv("DKI_INCLUDED_TREE_MANIFEST", str(source_manifest))
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", tree)
    monkeypatch.setenv("DKI_INCLUDED_TREE_MANIFEST_SHA256", _sha256(source_manifest))
    return output, manifest, report, source_manifest


def _rewrite_report_and_rebind(report: Path, manifest: Path, payload: dict) -> None:
    report.write_text(json.dumps(payload), encoding="utf-8")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["task_report"]["bytes"] = report.stat().st_size
    manifest_payload["task_report"]["sha256"] = _sha256(report)
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")


def test_demo_runtime_receipt_is_structured_hash_bound_and_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, manifest, _report, source_manifest = _fixture(tmp_path, monkeypatch)
    path = cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))
    payload = json.loads(path.read_text(encoding="utf-8"))

    source = payload["source"]
    assert payload["source_tree_sha256"] == source["included_tree_sha256"]
    assert source["manifest_sha256"] == _sha256(source_manifest)
    assert payload["evidence_eligible"] is True
    assert payload["task1_manifest"]["sha256"] == _sha256(manifest)
    expected_input = hashlib.sha256()
    for relative, size, digest in sorted([
        ("data/raw/a.txt", 3, "a" * 64),
        ("data/raw/b.txt", 4, "b" * 64),
    ]):
        expected_input.update(relative.encode("utf-8"))
        expected_input.update(b"\0")
        expected_input.update(str(size).encode("ascii"))
        expected_input.update(b"\0")
        expected_input.update(digest.encode("ascii"))
        expected_input.update(b"\n")
    assert payload["input_snapshot"] == {
        "algorithm": ALGORITHM,
        "n_files": 2,
        "sha256": expected_input.hexdigest(),
    }
    assert payload["runtime"]["model_backend"] == "ScriptedModel"
    assert payload["runtime"]["core_agent"] == "nexent.core.agents.CoreAgent"
    assert payload["runtime"]["datamate_mode"] in {"real-datamate", "compat-shim"}
    assert [item["tool"] for item in payload["agent_trace"]["tool_calls"]] == [
        "list_operators", "get_task_status", "get_task_status", "kg_query", "nl2sql_query",
    ]
    assert payload["agent_trace"]["retry_count"] == 1
    assert len(payload["agent_trace"]["observations"]) == 3
    assert set(payload["artifacts"]) == {
        "kg", "kg_provenance", "database", "database_provenance", "bi_report",
    }
    assert payload["privacy"]["contains_private_blind_data"] is False
    assert path == output / "demo_runtime.json"

    with pytest.raises(FileExistsError, match="create-once"):
        cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))


def test_demo_runtime_naked_tree_hash_is_not_evidence_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _output, _manifest, _report, _source_manifest = _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("DKI_INCLUDED_TREE_MANIFEST")
    monkeypatch.delenv("DKI_INCLUDED_TREE_MANIFEST_SHA256")

    path = cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] is None
    assert payload["source_tree_sha256"] is None
    assert payload["evidence_eligible"] is False


def test_demo_runtime_ui_child_requires_manifest_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("DKI_INCLUDED_TREE_MANIFEST")
    monkeypatch.setenv("DKI_DEMO_SERVER_RID", "ui-run")

    with pytest.raises(RuntimeError, match="lacks authoritative"):
        cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))


def test_demo_runtime_receipt_rejects_task_report_outside_isolated_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _output, manifest, _report, _source = _fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    external = tmp_path / "external.json"
    external.write_text('{"status":"COMPLETED","task_id":"task-1"}', encoding="utf-8")
    payload["task_report"] = {
        "path": "external.json",
        "bytes": external.stat().st_size,
        "sha256": _sha256(external),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the isolated demo output"):
        cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))


@pytest.mark.parametrize("field", ["bytes", "sha256"])
def test_demo_runtime_receipt_rejects_task_report_size_or_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    _output, manifest, _report, _source = _fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["task_report"][field] = 0 if field == "bytes" else "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="binding failed"):
        cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))


@pytest.mark.parametrize(
    "report_update",
    [
        {"task_id": "task-1", "status": "FAILED"},
        {"task_id": "other-task", "status": "COMPLETED"},
    ],
)
def test_demo_runtime_receipt_rejects_noncompleted_or_wrong_task_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_update: dict,
) -> None:
    _output, manifest, report, _source = _fixture(tmp_path, monkeypatch)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload.update(report_update)
    _rewrite_report_and_rebind(report, manifest, payload)

    with pytest.raises(ValueError, match="not a completed manifest match"):
        cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))


def test_demo_runtime_receipt_rejects_mode_mismatched_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _manifest, _report, _source = _fixture(tmp_path, monkeypatch)
    trace = output / "agent_observations" / "002-kg.json"
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["offline"] = False
    payload["model_backend"] = "OpenAIModel"
    trace.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mode binding failed"):
        cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))


def test_demo_runtime_receipt_rejects_missing_required_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _manifest, _report, _source = _fixture(tmp_path, monkeypatch)
    (output / "insight.provenance.json").unlink()

    with pytest.raises(ValueError, match="database_provenance is missing"):
        cli._publish_demo_runtime(argparse.Namespace(agent=True, offline=True))


def test_agent_tool_trace_records_only_actual_calls_outcomes_and_retries() -> None:
    class Tool:
        name = "actual_tool"

        def forward(self, *, secret: str):
            return json.dumps({"ok": True, "status": "COMPLETED", "echo": secret})

    class Agent:
        tools = {"actual_tool": Tool()}

    calls: list[dict] = []
    instrumented, failures = cli._instrument_agent_tools(Agent(), calls)
    Agent.tools["actual_tool"].forward(secret="must-not-be-recorded")
    Agent.tools["actual_tool"].forward(secret="must-not-be-recorded")

    assert instrumented == 1
    assert failures == []
    assert calls == [
        {
            "tool": "actual_tool", "attempt": 1, "repeated_call": False, "retry": False,
            "observed": True, "outcome": "reported_success", "status": "COMPLETED",
            "disposition": None, "terminal": None, "is_running": None,
            "duration_ms": calls[0]["duration_ms"],
        },
        {
            "tool": "actual_tool", "attempt": 2, "repeated_call": True, "retry": False,
            "observed": True, "outcome": "reported_success", "status": "COMPLETED",
            "disposition": None, "terminal": None, "is_running": None,
            "duration_ms": calls[1]["duration_ms"],
        },
    ]
    assert "must-not-be-recorded" not in json.dumps(calls)
