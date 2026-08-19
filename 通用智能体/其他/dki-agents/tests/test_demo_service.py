from __future__ import annotations

import io
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import pytest

from dki.demo.service import (
    DemoRunService,
    _publish_terminal,
    _write_once,
    iter_run_artifacts,
)


TREE_SHA256 = "a" * 64


class SuccessfulProcess:
    def __init__(self, stdout: str = "pipeline started\npipeline completed\n", stderr: str = ""):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_success_artifacts(
    run_dir: Path,
    source_tree: str = TREE_SHA256,
    *,
    agent_enabled: bool = False,
    offline: bool = True,
) -> None:
    clean_dir = run_dir / "processed" / "runs" / "task-1" / "clean"
    structured_dir = run_dir / "processed" / "runs" / "task-1" / "structured"
    task_dir = run_dir / "tasks"
    for directory in (clean_dir, structured_dir, task_dir, run_dir / "kg", run_dir / "reports"):
        directory.mkdir(parents=True, exist_ok=True)

    raw = run_dir / "input" / "medical-a.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("患者张三因高血压就诊。", encoding="utf-8")
    clean_text = clean_dir / "medical-a.txt"
    clean_text.write_text("患者张三因高血压就诊。", encoding="utf-8")
    patients = structured_dir / "patients.csv"
    patients.write_text(
        "patient_id,name,gender,age,region\nP001,张三,男,55,北京\n",
        encoding="utf-8",
    )
    visits = structured_dir / "visits.csv"
    visits.write_text(
        "visit_id,patient_id,visit_date,department,disease,drug_prescribed,exam,cost,los_days\n"
        "V001,P001,2026-01-01,心内科,高血压,氨氯地平,血压,100,0\n",
        encoding="utf-8",
    )
    report = task_dir / "task-1.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "task_id": "task-1",
                "status": "COMPLETED",
                "n_input": 1,
                "n_output": 1,
                "n_failed": 0,
                "n_filtered": 0,
                "duration_s": 0.1,
                "op_metrics": [],
                "recovery_events": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    task1 = run_dir / "processed" / "manifest.json"
    manifest = {
        "schema_version": 1,
        "task_id": "task-1",
        "created_at": "2026-08-07T00:00:00Z",
        "source": {
            "raw_text_dir": str(raw.parent.resolve()),
            "raw_text_count": 1,
            "raw_text_files": [_binding(raw)],
            "structured_source_dir": str(structured_dir.resolve()),
        },
        "clean_text": {
            "directory": str(clean_dir.resolve()),
            "count": 1,
            "files": [_binding(clean_text)],
        },
        "structured_tables": {
            "patients": {
                "path": str(patients.resolve()),
                "sha256": _sha256(patients),
                "columns": ["patient_id", "name", "gender", "age", "region"],
                "row_count": 1,
            },
            "visits": {
                "path": str(visits.resolve()),
                "sha256": _sha256(visits),
                "columns": [
                    "visit_id", "patient_id", "visit_date", "department", "disease",
                    "drug_prescribed", "exam", "cost", "los_days",
                ],
                "row_count": 1,
            },
        },
        "task_report": _binding(report),
    }
    task1.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    kg = run_dir / "kg" / "medical_kg.json"
    kg.write_text('{"entities": {}, "triples": []}\n', encoding="utf-8")
    kg_provenance = run_dir / "kg" / "medical_kg.provenance.json"
    kg_provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2026-08-07T00:00:00Z",
                "task_id": "task-2",
                "task1_task_id": "task-1",
                "task1_manifest_path": str(task1.resolve()),
                "task1_manifest_sha256": _sha256(task1),
                "kg_path": str(kg.resolve()),
                "kg_sha256": _sha256(kg),
            }
        ),
        encoding="utf-8",
    )

    database = run_dir / "insight.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            "CREATE TABLE patients(patient_id TEXT);"
            "CREATE TABLE visits(visit_id TEXT);"
            "CREATE TABLE kg_entities(name TEXT);"
            "CREATE TABLE kg_triples(head TEXT);"
        )
        connection.execute("INSERT INTO patients VALUES ('P001')")
        connection.execute("INSERT INTO visits VALUES ('V001')")
        connection.commit()
    finally:
        connection.close()
    db_provenance = run_dir / "insight.provenance.json"
    db_provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task1_manifest_path": str(task1.resolve()),
                "task1_manifest_sha256": _sha256(task1),
                "task2_kg_path": str(kg.resolve()),
                "task2_kg_sha256": _sha256(kg),
                "database_path": str(database.resolve()),
                "database_sha256": _sha256(database),
            }
        ),
        encoding="utf-8",
    )

    bi_report = run_dir / "reports" / "insight_report.html"
    bi_report.write_text(
        "<!DOCTYPE html><html><head><title>DKI</title></head><body>"
        "<div id='chart_1'>医疗分析</div><script>const CHARTS = []; "
        "if (typeof echarts !== 'undefined') { echarts.init(document.getElementById('chart_1')); }"
        "</script></body></html>",
        encoding="utf-8",
    )

    artifact_paths = {
        "kg": kg,
        "kg_provenance": kg_provenance,
        "database": database,
        "database_provenance": db_provenance,
        "bi_report": bi_report,
    }
    tool_calls = (
        [{"agent": "data", "tool": "run_pipeline"}]
        if agent_enabled and offline
        else []
    )
    runtime = {
        "schema_version": 1,
        "kind": "dki_demo_runtime_receipt",
        "status": "completed",
        "source_tree_sha256": source_tree,
        "evidence_eligible": True,
        "created_at": "2026-08-07T00:00:00Z",
        "input_snapshot": DemoRunService._input_snapshot_from_manifest(manifest),
        "task1_manifest": _binding(task1),
        "task1_report": _binding(report),
        "artifacts": {name: _binding(path) for name, path in artifact_paths.items()},
        "runtime": {
            "datamate_mode": "compat-shim",
            "datamate_available": False,
            "agent_enabled": agent_enabled,
            "core_agent": "nexent.core.agents.CoreAgent" if agent_enabled else None,
            "model_backend": (
                "ScriptedModel" if agent_enabled and offline else
                ("OpenAIModel" if agent_enabled else None)
            ),
            "offline": offline,
        },
        "agent_trace": {"trace_source": "fixture", "tool_calls": tool_calls},
        "pipeline": {
            "status": "COMPLETED",
            "n_input": 1,
            "n_output": 1,
            "n_failed": 0,
            "n_filtered": 0,
            "duration_s": 0.1,
            "operators": [],
            "recovery_events": [],
        },
        "privacy": {
            "contains_secrets": False,
            "contains_private_blind_data": False,
            "contains_full_user_input": False,
        },
    }
    (run_dir / "demo_runtime.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )


def _successful_popen(
    *args,
    fake_stdout: str = "pipeline started\npipeline completed\n",
    fake_stderr: str = "",
    **kwargs,
):
    run_dir = Path(kwargs["env"]["DKI_OUTPUT_DIR"])
    command = list(args[0])
    _write_success_artifacts(
        run_dir,
        kwargs["env"].get("DKI_INCLUDED_TREE_SHA256", TREE_SHA256).lower(),
        agent_enabled="--agent" in command,
        offline="--offline" in command,
    )
    return SuccessfulProcess(stdout=fake_stdout, stderr=fake_stderr)


def test_write_once_is_create_only(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    _write_once(target, {"status": "started"})
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "started"
    with pytest.raises(FileExistsError):
        _write_once(target, {"status": "overwritten"})
    assert "overwritten" not in target.read_text(encoding="utf-8")


def test_write_once_never_exposes_partial_json_when_atomic_publish_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "receipt.json"

    def fail_link(source, destination):
        raise OSError("simulated publish crash")

    monkeypatch.setattr("dki.demo.service.os.link", fail_link)
    with pytest.raises(OSError, match="simulated publish crash"):
        _write_once(target, {"status": "completed", "large": "x" * 100_000})

    assert not target.exists()
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []


def test_terminal_state_is_mutually_exclusive_and_strict_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    completed = {"schema_version": 1, "run_id": "run", "status": "completed"}
    failed = {"schema_version": 1, "run_id": "run", "status": "failed"}

    _publish_terminal(run_dir, "completed.json", completed)
    with pytest.raises(FileExistsError, match="already published"):
        _publish_terminal(run_dir, "failed.json", failed)

    assert json.loads((run_dir / "completed.json").read_text(encoding="utf-8")) == completed
    assert not (run_dir / "failed.json").exists()
    assert (run_dir / "terminal.json").read_bytes() == (run_dir / "completed.json").read_bytes()


@pytest.mark.parametrize("value", [None, "unbound", "f" * 63, "z" * 64])
def test_demo_service_source_tree_binding_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("DKI_INCLUDED_TREE_SHA256", raising=False)
    else:
        monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", value)
    service = DemoRunService(tmp_path / "server-rid", python="python")

    with pytest.raises(ValueError, match="64-hex included-tree"):
        service.run(agent=False, offline=True, run_id="must-not-exist")

    assert not (service.root / "runs" / "must-not-exist").exists()


def test_demo_service_run_directory_is_unique_and_receipt_is_public(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_SHA256.upper())
    service = DemoRunService(tmp_path / "server-rid", python="python")
    monkeypatch.setattr(
        "dki.demo.service.subprocess.Popen",
        lambda *args, **kwargs: _successful_popen(
            *args, fake_stderr="pipeline diagnostic\n", **kwargs
        ),
    )
    first = service.run(agent=False, offline=True, run_id="run-a")
    second = service.run(agent=False, offline=True, run_id="run-b")
    assert first.status == second.status == "completed"
    assert first.source_tree_sha256 == TREE_SHA256
    assert first.run_dir != second.run_dir
    assert (first.run_dir / "started.json").is_file()
    assert (first.run_dir / "completed.json").is_file()
    assert (first.run_dir / "events.jsonl").read_text(encoding="utf-8").count("pipeline_log") == 3
    assert (first.run_dir / "stderr.log").read_text(encoding="utf-8") == "pipeline diagnostic\n"
    completed = json.loads((first.run_dir / "completed.json").read_text(encoding="utf-8"))
    assert completed["source_tree_sha256"] == TREE_SHA256
    assert completed["stderr_sha256"]
    assert DemoRunService.read_receipt(first.run_dir)["_service_verified"] is True
    assert "_service_verified" not in completed
    assert (second.run_dir / "completed.json").is_file()
    with pytest.raises(FileExistsError):
        service.run(agent=False, offline=True, run_id="run-a")


def test_demo_service_popen_exception_seals_failed_receipt_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_SHA256)
    service = DemoRunService(tmp_path / "server-rid", python="python")

    def fail_to_start(*args, **kwargs):
        raise OSError("child launch failed")

    monkeypatch.setattr("dki.demo.service.subprocess.Popen", fail_to_start)
    receipt = service.run(agent=True, offline=True, run_id="launch-failure")

    assert receipt.status == "failed"
    for name in ("started.json", "events.jsonl", "stdout.log", "stderr.log", "failed.json"):
        assert (receipt.run_dir / name).is_file()
    assert not (receipt.run_dir / "completed.json").exists()
    payload = json.loads((receipt.run_dir / "failed.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["source_tree_sha256"] == TREE_SHA256
    assert payload["error_summary"]["failure_kind"] == "process_launch_or_runtime"
    assert payload["error_summary"]["exception_type"] == "OSError"
    assert "runtime_exception" in (receipt.run_dir / "events.jsonl").read_text(encoding="utf-8")


def test_demo_service_stream_read_exception_is_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_SHA256)
    service = DemoRunService(tmp_path / "server-rid", python="python")

    class BrokenStream:
        def __iter__(self):
            raise OSError("stdout pipe failed")
            yield ""  # pragma: no cover - makes this an iterator

        def close(self):
            pass

    process = SuccessfulProcess()
    process.stdout = BrokenStream()
    monkeypatch.setattr("dki.demo.service.subprocess.Popen", lambda *args, **kwargs: process)

    receipt = service.run(agent=False, offline=True, run_id="stream-failure")
    payload = json.loads((receipt.run_dir / "failed.json").read_text(encoding="utf-8"))
    assert receipt.status == "failed"
    assert payload["error_summary"]["failure_kind"] == "stdout_read"
    assert payload["error_summary"]["exception_type"] == "OSError"
    assert not (receipt.run_dir / "completed.json").exists()


def test_demo_service_timeout_does_not_wait_for_stdout_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_SHA256)
    released = threading.Event()

    class BlockingStream:
        def __iter__(self):
            yield "child is still running\n"
            released.wait(timeout=30)

        def close(self):
            released.set()

    class TimeoutProcess:
        def __init__(self):
            self.stdout = BlockingStream()
            self.stderr = io.StringIO("")
            self.returncode = None
            self.terminated = False

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("demo-all", timeout)
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15
            released.set()

        def kill(self):
            self.returncode = -9
            released.set()

    process = TimeoutProcess()
    monkeypatch.setattr("dki.demo.service.subprocess.Popen", lambda *args, **kwargs: process)
    service = DemoRunService(tmp_path / "server-rid", python="python", timeout=1)

    receipt = service.run(agent=False, offline=True, run_id="timeout")
    payload = json.loads((receipt.run_dir / "failed.json").read_text(encoding="utf-8"))
    assert receipt.status == "failed"
    assert process.terminated is True
    assert released.is_set()
    assert payload["error_summary"]["failure_kind"] == "timeout"
    assert '"kind": "timeout"' in (receipt.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert not (receipt.run_dir / "completed.json").exists()


def test_demo_service_summary_exception_is_sealed_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_SHA256)
    service = DemoRunService(tmp_path / "server-rid", python="python")
    monkeypatch.setattr(
        "dki.demo.service.subprocess.Popen", lambda *args, **kwargs: SuccessfulProcess()
    )

    def broken_summary(run_dir: Path):
        raise OSError("cannot summarize artifacts")

    monkeypatch.setattr(service, "_artifact_summary", broken_summary)
    receipt = service.run(agent=False, offline=True, run_id="summary-failure")
    payload = json.loads((receipt.run_dir / "failed.json").read_text(encoding="utf-8"))

    assert receipt.status == "failed"
    assert payload["error_summary"]["failure_kind"] == "summary"
    assert any(
        item["stage"] == "artifact_summary"
        for item in payload["error_summary"]["diagnostics"]
    )
    assert not (receipt.run_dir / "completed.json").exists()


def test_demo_artifact_summary_and_export_include_database_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_success_artifacts(run_dir)

    summary = DemoRunService._artifact_summary(run_dir)
    runtime = DemoRunService._runtime_summary(
        run_dir,
        TREE_SHA256,
        expected_mode="deterministic",
        artifacts=summary,
    )
    exported = dict(iter_run_artifacts(run_dir))
    assert summary["database_provenance"]["path"].endswith("insight.provenance.json")
    assert summary["database_provenance"]["sha256"]
    assert runtime["input_snapshot"]["n_files"] == 1
    assert exported["insight.provenance.json"] == run_dir / "insight.provenance.json"
    assert exported["demo_runtime.json"] == run_dir / "demo_runtime.json"


@pytest.mark.parametrize(
    "relative",
    [
        "processed/manifest.json",
        "kg/medical_kg.json",
        "kg/medical_kg.provenance.json",
        "insight.db",
        "insight.provenance.json",
        "reports/insight_report.html",
        "demo_runtime.json",
    ],
)
def test_demo_artifact_summary_fails_closed_when_any_required_artifact_is_missing(
    tmp_path: Path,
    relative: str,
) -> None:
    run_dir = tmp_path / "run"
    _write_success_artifacts(run_dir)
    (run_dir / relative).unlink()

    with pytest.raises(FileNotFoundError, match="missing required artifacts"):
        DemoRunService._artifact_summary(run_dir)


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        ("processed/manifest.json", b'{"schema_version":1}'),
        ("kg/medical_kg.json", b'{"entities":{},"triples":[{}]}'),
        ("kg/medical_kg.provenance.json", b'{"schema_version":1}'),
        ("insight.db", b"not a sqlite database"),
        ("insight.provenance.json", b'{"schema_version":1}'),
        ("reports/insight_report.html", b"<html>not an ECharts report</html>"),
        ("demo_runtime.json", b'{"schema_version":1}'),
    ],
)
def test_demo_validation_rejects_each_tampered_required_artifact(
    tmp_path: Path,
    relative: str,
    replacement: bytes,
) -> None:
    run_dir = tmp_path / "run"
    _write_success_artifacts(run_dir)
    (run_dir / relative).write_bytes(replacement)

    with pytest.raises((FileNotFoundError, KeyError, OSError, ValueError, sqlite3.DatabaseError)):
        artifacts = DemoRunService._artifact_summary(run_dir)
        DemoRunService._runtime_summary(
            run_dir,
            TREE_SHA256,
            expected_mode="deterministic",
            artifacts=artifacts,
        )


def test_demo_runtime_summary_rejects_source_tree_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_success_artifacts(run_dir, source_tree="c" * 64)

    with pytest.raises(ValueError, match="public schema"):
        DemoRunService._runtime_summary(run_dir, TREE_SHA256)


def test_demo_runtime_summary_recomputes_input_snapshot_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_success_artifacts(run_dir)
    runtime_path = run_dir / "demo_runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["input_snapshot"]["sha256"] = "b" * 64
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="public schema"):
        DemoRunService._runtime_summary(run_dir, TREE_SHA256)


def test_read_receipt_revokes_completed_trust_after_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DKI_INCLUDED_TREE_SHA256", TREE_SHA256)
    service = DemoRunService(tmp_path / "server-rid", python="python")
    monkeypatch.setattr(
        "dki.demo.service.subprocess.Popen",
        lambda *args, **kwargs: _successful_popen(*args, **kwargs),
    )
    receipt = service.run(agent=False, offline=True, run_id="tamper-after-complete")
    assert DemoRunService.read_receipt(receipt.run_dir)["_service_verified"] is True

    (receipt.run_dir / "reports" / "insight_report.html").write_text(
        "<html>tampered</html>", encoding="utf-8"
    )

    assert DemoRunService.read_receipt(receipt.run_dir) is None


def _write_started(
    run_dir: Path,
    *,
    started_at: str = "2000-01-01T00:00:00Z",
    hostname: str | None = None,
    service_pid: int = 10,
    service_token: str = "service-old",
) -> None:
    run_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": started_at,
        "source_tree_sha256": TREE_SHA256,
        "mode": "agent-offline",
        "hostname": hostname or socket.gethostname(),
        "service_pid": service_pid,
        "service_start_token": service_token,
    }
    (run_dir / "started.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_child(run_dir: Path, *, pid: int = 20, token: str = "child-old") -> None:
    (run_dir / "child-started.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "child_pid": pid,
                "child_start_token": token,
            }
        ),
        encoding="utf-8",
    )


def test_demo_service_reconciles_only_stale_dead_local_orphans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "orphan"
    _write_started(run_dir, service_pid=999999)
    _write_child(run_dir, pid=999998)
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: False))

    DemoRunService(root, timeout=1)

    payload = json.loads((run_dir / "failed.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_summary"]["failure_kind"] == "orphaned_after_service_restart"
    assert not (run_dir / "completed.json").exists()


def test_demo_service_dead_child_is_orphan_even_when_service_is_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "dead-child"
    _write_started(run_dir)
    _write_child(run_dir)
    monkeypatch.setattr(
        DemoRunService,
        "_pid_alive",
        staticmethod(lambda pid: pid == 10),
    )
    monkeypatch.setattr(
        DemoRunService,
        "_pid_start_token",
        staticmethod(lambda pid: "service-old" if pid == 10 else None),
    )

    DemoRunService(root, timeout=1)

    assert (run_dir / "failed.json").is_file()


def test_demo_service_live_child_identity_is_not_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "live-child"
    _write_started(run_dir)
    _write_child(run_dir)
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: pid == 20))
    monkeypatch.setattr(
        DemoRunService,
        "_pid_start_token",
        staticmethod(lambda pid: "child-old" if pid == 20 else None),
    )

    DemoRunService(root, timeout=1)

    assert not (run_dir / "failed.json").exists()


def test_demo_service_pid_reuse_does_not_keep_stale_child_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "reused-pid"
    _write_started(run_dir)
    _write_child(run_dir)
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: pid in {10, 20}))
    monkeypatch.setattr(
        DemoRunService,
        "_pid_start_token",
        staticmethod(lambda pid: "new-process" if pid == 20 else "service-old"),
    )

    DemoRunService(root, timeout=1)

    assert (run_dir / "failed.json").is_file()


def test_demo_service_fresh_dead_run_is_not_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "fresh"
    _write_started(run_dir, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _write_child(run_dir)
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: False))

    DemoRunService(root, timeout=60)

    assert not (run_dir / "failed.json").exists()


def test_demo_service_live_service_without_child_is_not_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "live-service"
    _write_started(run_dir)
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: pid == 10))
    monkeypatch.setattr(DemoRunService, "_pid_start_token", staticmethod(lambda pid: "service-old"))

    DemoRunService(root, timeout=1)

    assert not (run_dir / "failed.json").exists()


def test_demo_service_foreign_host_run_is_not_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "foreign"
    _write_started(run_dir, hostname="definitely-another-host")
    _write_child(run_dir)
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: False))

    DemoRunService(root, timeout=1)

    assert not (run_dir / "failed.json").exists()


def test_demo_service_existing_terminal_is_never_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "terminal"
    _write_started(run_dir)
    _publish_terminal(
        run_dir,
        "failed.json",
        {"schema_version": 1, "run_id": run_dir.name, "status": "failed"},
    )
    before = (run_dir / "failed.json").read_bytes()
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: False))

    DemoRunService(root, timeout=1)

    assert (run_dir / "failed.json").read_bytes() == before
    assert not (run_dir / "completed.json").exists()


def test_demo_service_stale_malformed_started_json_is_sealed_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "server-rid"
    run_dir = root / "runs" / "malformed"
    run_dir.mkdir(parents=True)
    started = run_dir / "started.json"
    started.write_text('{"run_id":', encoding="utf-8")
    os.utime(started, (1, 1))
    monkeypatch.setattr(DemoRunService, "_pid_alive", staticmethod(lambda pid: False))

    DemoRunService(root, timeout=1)

    payload = json.loads((run_dir / "failed.json").read_text(encoding="utf-8"))
    assert payload["error_summary"]["failure_kind"] == "orphaned_invalid_started_receipt"
    assert DemoRunService.read_receipt(run_dir)["_service_verified"] is False


def test_demo_service_read_receipt_rejects_malformed_terminal_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad-terminal"
    run_dir.mkdir()
    (run_dir / "completed.json").write_text('{"status":', encoding="utf-8")

    assert DemoRunService.read_receipt(run_dir) is None
