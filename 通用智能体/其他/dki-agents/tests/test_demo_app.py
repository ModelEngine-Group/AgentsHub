from __future__ import annotations

import json
import sys
import types
from pathlib import Path


# The shared NPU environment intentionally keeps Streamlit in RID-local
# demo dependencies.  Pure UI-contract tests must therefore remain importable
# before that optional dependency is added to PYTHONPATH.
if "streamlit" not in sys.modules:
    try:
        __import__("streamlit")
    except ModuleNotFoundError:
        sys.modules["streamlit"] = types.ModuleType("streamlit")

from dki.demo import app


SHA_A = "a" * 64
SHA_B = "b" * 64


def _completed_receipt() -> dict:
    return {
        "_service_verified": True,
        "status": "completed",
        "returncode": 0,
        "source_tree_sha256": SHA_A,
        "mode": "agent-offline",
        "artifacts": {
            name: {"bytes": 1, "sha256": SHA_B}
            for name in app._REQUIRED_DEMO_ARTIFACTS
        },
    }


def _passed_component() -> dict:
    return {
        "status": "PASSED",
        "gate_passed": True,
        "available": True,
        "authority": "final_status",
        "sha256": SHA_A,
        "source_tree_sha256": SHA_A,
    }


def test_statuses_fail_closed_and_require_authoritative_release() -> None:
    assert {
        name: item["status"]
        for name, item in app._status_summary(None, {}).items()
    } == {
        "DEMO_READY": "BLOCKED",
        "QUALITY_GATES_PASSED": "BLOCKED",
        "READY_FOR_SUBMISSION": "BLOCKED",
    }

    receipt = _completed_receipt()
    registry = {
        "cmeie": _passed_component(),
        "nl2sql": _passed_component(),
    }
    statuses = app._status_summary(receipt, registry)
    assert statuses["DEMO_READY"]["status"] == "PASS"
    assert statuses["QUALITY_GATES_PASSED"]["status"] == "PASS"
    assert statuses["READY_FOR_SUBMISSION"]["status"] == "BLOCKED"

    registry["release"] = {
        "status": "READY_FOR_SUBMISSION",
        "ready_for_submission": True,
        "quality_gates_passed": True,
        "available": True,
        "hash_bound": True,
        "final_status_sha256": SHA_A,
        "source_tree_sha256": SHA_A,
    }
    assert app._status_summary(receipt, registry)["READY_FOR_SUBMISSION"]["status"] == "PASS"

    unverified = dict(receipt)
    unverified["_service_verified"] = False
    statuses = app._status_summary(unverified, registry)
    assert statuses["DEMO_READY"]["status"] == "BLOCKED"
    assert statuses["READY_FOR_SUBMISSION"]["status"] == "BLOCKED"


def test_non_authoritative_quality_claim_cannot_promote_status() -> None:
    component = _passed_component()
    component["authority"] = "latest_directory"
    statuses = app._status_summary(
        _completed_receipt(), {"cmeie": component, "nl2sql": _passed_component()}
    )
    assert statuses["QUALITY_GATES_PASSED"]["status"] == "BLOCKED"
    assert statuses["READY_FOR_SUBMISSION"]["status"] == "BLOCKED"


def test_demo_ready_requires_structured_runtime_artifact() -> None:
    receipt = _completed_receipt()
    receipt["artifacts"].pop("demo_runtime")

    assert app._status_summary(receipt, {})["DEMO_READY"]["status"] == "BLOCKED"


def test_npu_binding_uses_get_and_fails_closed() -> None:
    assert app._npu_is_bound(None) is False
    assert app._npu_is_bound({"status": "bound"}) is False
    evidence = {
        "status": "bound",
        "hash_bound": True,
        "acceptance_sha256": SHA_A,
        "expected_acceptance_sha256": SHA_A,
    }
    assert app._npu_is_bound(evidence) is True
    evidence["expected_acceptance_sha256"] = SHA_B
    assert app._npu_is_bound(evidence) is False


def _device_snapshot_v2(*, detected: bool) -> dict:
    return {
        "status": "available",
        "available": True,
        "schema_version": 2,
        "kind": "dki_demo_device_snapshot",
        "created_at": "2026-08-07T00:00:00Z",
        "requested_logical_device": "npu:0",
        "detected_available": detected,
        "detected_device_name": "Ascend910_9382" if detected else None,
        "environment_label": "910C 验收环境" if detected else None,
        "detection_sources": ["torch_npu", "npu-smi"] if detected else [],
        "commands": {
            "npu_smi": {
                "returncode": 0 if detected else None,
                "summary_lines": ["Ascend 910"] if detected else [],
            },
            "python_runtime": {
                "torch_npu": "2.7.1.post6" if detected else None,
                "npu_available": detected,
            },
        },
    }


def test_live_device_snapshot_v2_keeps_requested_and_detected_separate() -> None:
    detected = app._live_device_snapshot(_device_snapshot_v2(detected=True))
    assert detected is not None
    assert detected["requested_logical_device"] == "npu:0"
    assert detected["detected_device_name"] == "Ascend910_9382"
    assert detected["environment_label"] == "910C 验收环境"

    unavailable = app._live_device_snapshot(_device_snapshot_v2(detected=False))
    assert unavailable is not None
    assert unavailable["detected_available"] is False
    assert unavailable["detected_device_name"] is None
    assert unavailable["environment_label"] is None


def test_live_device_snapshot_rejects_v1_and_fabricated_detection() -> None:
    assert app._live_device_snapshot({"status": "available", "schema_version": 1}) is None
    forged = _device_snapshot_v2(detected=False)
    forged["environment_label"] = "910C 验收环境"
    assert app._live_device_snapshot(forged) is None
    forged = _device_snapshot_v2(detected=True)
    forged["detection_sources"] = []
    assert app._live_device_snapshot(forged) is None


def test_safe_receipt_strips_paths_messages_and_row_level_material() -> None:
    receipt = {
        **_completed_receipt(),
        "run_id": "run-1",
        "question": "private question",
        "gold_sql": "select secret",
        "error_summary": {
            "error_line_count": 2,
            "failure_kind": "runtime",
            "exception_type": "ValueError",
            "exception_message": "contains full private input",
        },
        "artifacts": {
            "database": {"path": "outputs/private/holdout.db", "bytes": 2, "sha256": SHA_A}
        },
    }
    safe = app._safe_receipt_download(receipt)
    encoded = json.dumps(safe, ensure_ascii=False)
    assert "private question" not in encoded
    assert "gold_sql" not in encoded
    assert "exception_message" not in encoded
    assert "outputs/private" not in encoded
    assert safe["artifacts"]["database"] == {"bytes": 2, "sha256": SHA_A}


def test_aggregate_registry_download_is_whitelist_only() -> None:
    registry = {
        "cmeie": {
            **_passed_component(),
            "f1": 0.51,
            "question": "must not leave",
            "gold_spo": [["a", "b", "c"]],
        },
        "nl2sql": {
            **_passed_component(),
            "n": 120,
            "correct": 103,
            "predicted_sql": "select secret",
            "details": [{"id": 1}],
        },
    }
    safe = app._safe_registry_download(registry, app._status_summary(None, registry))
    encoded = json.dumps(safe, ensure_ascii=False).casefold()
    for forbidden in ("must not leave", "gold_spo", "predicted_sql", "select secret", "details"):
        assert forbidden not in encoded


def test_manifest_download_is_constructed_without_paths_or_unknown_fields() -> None:
    normal = {
        "schema_version": 1,
        "task_id": "task-1",
        "created_at": "2026-08-07T00:00:00Z",
        "source": {
            "raw_text_count": 1,
            "raw_text_dir": "outputs/private/v29/holdout",
            "raw_text_files": [
                {"path": "uploads/patient-name.txt", "bytes": 7, "sha256": SHA_A}
            ],
        },
        "clean_text": {
            "directory": "outputs/private/v29/clean",
            "count": 1,
            "files": [{"path": "secret.txt", "bytes": 5, "sha256": SHA_B}],
        },
        "structured_tables": {
            "patients": {
                "path": "outputs/private/v29/patients.csv",
                "source_path": "uploads/patients.csv",
                "sha256": SHA_A,
                "row_count": 1,
                "columns": ["patient_id"],
            }
        },
        "task_report": {"path": "outputs/tasks/task-1.json", "bytes": 9, "sha256": SHA_B},
        "goldSql": "SELECT secret FROM holdout",
        "reference_sql": "SELECT secret FROM holdout",
        "note": "outputs/private/v29/holdout.jsonl",
    }
    safe = app._safe_manifest_download(normal)
    assert safe is not None
    encoded = json.dumps(safe, ensure_ascii=False).casefold()
    for forbidden in (
        "upload",
        "outputs/private",
        "holdout",
        "goldsql",
        "reference_sql",
        "select secret",
    ):
        assert forbidden not in encoded
    assert "raw_text_dir" not in encoded
    assert "source_path" not in encoded
    assert safe["source"]["raw_text_files"] == [{"bytes": 7, "sha256": SHA_A}]
    assert safe["structured_tables"]["patients"]["row_count"] == 1


def test_manifest_download_fails_closed_for_unbound_entries() -> None:
    incomplete = {
        "schema_version": 1,
        "source": {"raw_text_count": 1, "raw_text_files": [{}]},
        "clean_text": {"count": 1, "files": []},
        "structured_tables": {},
    }
    assert app._safe_manifest_download(incomplete) is None


def test_deterministic_query_infers_once_executes_once_and_keeps_full_rows(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {"inference": 0, "execute": 0}

    class FakeNL2SQL:
        def __init__(self, *, db_path: Path):
            assert db_path == tmp_path / "insight.db"

        def rule_sql(self, question: str) -> str:
            calls["inference"] += 1
            assert question == "各科室就诊量是多少？"
            return "SELECT department, COUNT(*) AS n FROM visits GROUP BY department"

    nl2sql_module = types.ModuleType("dki.analysis.nl2sql")
    nl2sql_module.NL2SQL = FakeNL2SQL
    protocol_module = types.ModuleType("dki.analysis.nl2sql_protocol")

    def execute_complete_readonly(sql: str, db_path: Path):
        calls["execute"] += 1
        assert sql.startswith("SELECT")
        assert db_path == tmp_path / "insight.db"
        return ["department", "n"], [(f"科室{i}", i) for i in range(250)]

    protocol_module.execute_complete_readonly = execute_complete_readonly
    monkeypatch.setitem(sys.modules, "dki.analysis.nl2sql", nl2sql_module)
    monkeypatch.setitem(sys.modules, "dki.analysis.nl2sql_protocol", protocol_module)
    db_path = tmp_path / "insight.db"
    db_path.write_bytes(b"sealed database")

    result = app._deterministic_rule_query("各科室就诊量是多少？", db_path)
    assert calls == {"inference": 1, "execute": 1}
    assert result["validated"] is True
    assert len(result["rows"]) == 250
    assert result["route"] == "deterministic_rule_only"
    assert result["provenance"]["llm_used"] is False
    assert result["provenance"]["inference_passes"] == 1


def test_deterministic_query_abstains_without_execution(monkeypatch, tmp_path: Path) -> None:
    calls = {"inference": 0, "execute": 0}

    class FakeNL2SQL:
        def __init__(self, *, db_path: Path):
            pass

        def rule_sql(self, question: str) -> str | None:
            calls["inference"] += 1
            return None

    nl2sql_module = types.ModuleType("dki.analysis.nl2sql")
    nl2sql_module.NL2SQL = FakeNL2SQL
    protocol_module = types.ModuleType("dki.analysis.nl2sql_protocol")

    def forbidden_execute(*args, **kwargs):
        calls["execute"] += 1
        raise AssertionError("abstention must not execute SQL")

    protocol_module.execute_complete_readonly = forbidden_execute
    monkeypatch.setitem(sys.modules, "dki.analysis.nl2sql", nl2sql_module)
    monkeypatch.setitem(sys.modules, "dki.analysis.nl2sql_protocol", protocol_module)
    db_path = tmp_path / "insight.db"
    db_path.write_bytes(b"sealed database")

    result = app._deterministic_rule_query("unsupported", db_path, run_id="run-1")
    assert calls == {"inference": 1, "execute": 0}
    assert result["disposition"] == "abstained"
    assert result["validated"] is False
    assert result["run_id"] == "run-1"


def test_deterministic_query_discards_rows_if_database_changes(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeNL2SQL:
        def __init__(self, *, db_path: Path):
            pass

        def rule_sql(self, question: str) -> str:
            return "SELECT 1"

    nl2sql_module = types.ModuleType("dki.analysis.nl2sql")
    nl2sql_module.NL2SQL = FakeNL2SQL
    protocol_module = types.ModuleType("dki.analysis.nl2sql_protocol")
    db_path = tmp_path / "insight.db"
    db_path.write_bytes(b"before")
    before = app._sha256(db_path)

    def execute_and_replace(sql: str, path: Path):
        path.write_bytes(b"after")
        return ["value"], [(1,)]

    protocol_module.execute_complete_readonly = execute_and_replace
    monkeypatch.setitem(sys.modules, "dki.analysis.nl2sql", nl2sql_module)
    monkeypatch.setitem(sys.modules, "dki.analysis.nl2sql_protocol", protocol_module)

    result = app._deterministic_rule_query(
        "test",
        db_path,
        expected_database_sha256=before,
        run_id="run-1",
    )
    assert result["disposition"] == "blocked"
    assert result["validated"] is False
    assert result["rows"] == []


def test_csv_export_neutralizes_spreadsheet_formulas() -> None:
    data = app._csv_bytes(["name"], [["=HYPERLINK(\"https://example\")"]]).decode("utf-8-sig")
    assert "'=HYPERLINK" in data


def test_graph_json_cannot_close_script_tag() -> None:
    graph = {
        "nodes": [{"name": "</script><script>alert(1)</script>"}],
        "links": [],
        "categories": [],
    }
    rendered = app._graph_html(graph, "window.echarts={init:()=>({setOption:()=>{},resize:()=>{}})}")
    assert "</script><script>alert(1)</script>" not in rendered
    assert "<\\/script><script>alert(1)<\\/script>" in rendered


def test_competition_map_has_all_points_and_required_columns() -> None:
    rows = app._competition_rows({}, {})
    assert len(rows) == 13
    assert sum(row["分值"] for row in rows) == 110
    required = {
        "任务",
        "评分项",
        "分值",
        "当前实现",
        "演示步骤",
        "数据角色",
        "证据SHA-256",
        "状态",
        "已知限制",
    }
    assert all(set(row) == required for row in rows)
    assert all(row["状态"] in {"PASS", "PARTIAL", "BLOCKED"} for row in rows)
    assert next(row for row in rows if row["评分项"] == "关系与三元组抽取")["状态"] == "BLOCKED"
