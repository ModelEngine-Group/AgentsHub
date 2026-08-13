"""Chinese-first, evidence-bound Streamlit competition demonstration."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from dki.demo.evidence import public_registry
from dki.demo.service import DemoRunService


ROOT = Path(__file__).resolve().parents[2]
SERVER_RID = os.getenv("DKI_DEMO_SERVER_RID") or (
    time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-ui"
)
DEFAULT_ROOT = Path(
    os.getenv("DKI_DEMO_ROOT", str(ROOT / "outputs" / "demo_runs" / SERVER_RID))
)

_REQUIRED_DEMO_ARTIFACTS = {
    "task1_manifest",
    "kg",
    "kg_provenance",
    "database",
    "database_provenance",
    "bi_report",
    "demo_runtime",
}
_AUTHORITATIVE_COMPONENTS = {"final_status", "explicit_prefinal"}
_FORBIDDEN_DOWNLOAD_KEYS = {
    "question",
    "questions",
    "gold_sql",
    "gold_spo",
    "predicted_sql",
    "prediction_sql",
    "predictions",
    "details",
    "detailed_results",
    "error_samples",
    "holdout_rows",
    "development_rows",
}
_PRIVATE_PATH_PARTS = {
    "private",
    "holdout",
    ".env",
    "checkpoints",
    "models",
    "uploads",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_registry() -> dict[str, Any]:
    try:
        value = public_registry()
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _artifact_is_bound(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and _is_sha256(value.get("sha256"))
        and type(value.get("bytes")) is int
        and value.get("bytes", -1) >= 0
    )


def _component_gate_is_authoritative(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("gate_passed") is True
        and value.get("available") is True
        and value.get("authority") in _AUTHORITATIVE_COMPONENTS
        and _is_sha256(value.get("sha256"))
        and _is_sha256(value.get("source_tree_sha256"))
    )


def _release_is_authoritative(value: Any) -> bool:
    """Accept only the release verdict produced from a bound FINAL_STATUS."""

    return (
        isinstance(value, dict)
        and value.get("status") == "READY_FOR_SUBMISSION"
        and value.get("ready_for_submission") is True
        and value.get("quality_gates_passed") is True
        and value.get("available") is True
        and value.get("hash_bound") is True
        and _is_sha256(value.get("final_status_sha256"))
        and _is_sha256(value.get("source_tree_sha256"))
    )


def _npu_is_bound(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    actual = value.get("acceptance_sha256")
    return (
        value.get("status") == "bound"
        and value.get("hash_bound") is True
        and _is_sha256(actual)
        and actual == value.get("expected_acceptance_sha256")
    )


def _live_device_snapshot(value: Any) -> dict[str, Any] | None:
    """Return the small v2 device view only when its detection claims agree."""

    if not isinstance(value, dict) or not (
        value.get("status") == "available"
        and value.get("available") is True
        and value.get("schema_version") == 2
        and value.get("kind") == "dki_demo_device_snapshot"
        and value.get("requested_logical_device") == "npu:0"
    ):
        return None
    detected = value.get("detected_available")
    name = value.get("detected_device_name")
    environment_label = value.get("environment_label")
    sources = value.get("detection_sources")
    commands = value.get("commands")
    if (
        not isinstance(detected, bool)
        or (name is not None and (not isinstance(name, str) or not name.strip()))
        or not isinstance(sources, list)
        or len(sources) != len(set(sources))
        or any(source not in {"torch_npu", "npu-smi"} for source in sources)
        or not isinstance(commands, dict)
    ):
        return None
    if detected != bool(sources):
        return None
    if detected:
        if environment_label != "910C 验收环境":
            return None
    elif environment_label is not None or name is not None:
        return None
    runtime = commands.get("python_runtime")
    npu_smi = commands.get("npu_smi")
    if not isinstance(runtime, dict) or not isinstance(npu_smi, dict):
        return None
    torch_detected = runtime.get("npu_available")
    if not isinstance(torch_detected, bool) or (
        ("torch_npu" in sources) != torch_detected
    ):
        return None
    if "npu-smi" in sources and npu_smi.get("returncode") != 0:
        return None
    summary_lines = npu_smi.get("summary_lines")
    if (
        not isinstance(summary_lines, list)
        or len(summary_lines) > 80
        or any(not isinstance(line, str) or len(line) > 500 for line in summary_lines)
    ):
        return None
    return {
        "requested_logical_device": "npu:0",
        "detected_available": detected,
        "detected_device_name": name,
        "environment_label": environment_label,
        "detection_sources": list(sources),
        "torch_npu": runtime.get("torch_npu"),
        "created_at": value.get("created_at"),
        "summary_lines": list(summary_lines),
    }


def _status_summary(
    receipt: dict[str, Any] | None,
    registry: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Return conservative UI statuses; missing authority is always BLOCKED."""

    receipt = receipt if isinstance(receipt, dict) else {}
    registry = registry if isinstance(registry, dict) else {}
    artifacts = receipt.get("artifacts")
    demo_ready = (
        receipt.get("_service_verified") is True
        and receipt.get("status") == "completed"
        and receipt.get("returncode") == 0
        and _is_sha256(receipt.get("source_tree_sha256"))
        and isinstance(artifacts, dict)
        and _REQUIRED_DEMO_ARTIFACTS.issubset(artifacts)
        and all(
            _artifact_is_bound(artifacts.get(name))
            for name in _REQUIRED_DEMO_ARTIFACTS
        )
    )
    cmeie = registry.get("cmeie")
    nl2sql = registry.get("nl2sql")
    cmeie_source = cmeie.get("source_tree_sha256") if isinstance(cmeie, dict) else None
    nl2sql_source = nl2sql.get("source_tree_sha256") if isinstance(nl2sql, dict) else None
    quality_ready = (
        _component_gate_is_authoritative(cmeie)
        and _component_gate_is_authoritative(nl2sql)
        and cmeie_source == nl2sql_source
    )
    release = registry.get("release")
    release_source = release.get("source_tree_sha256") if isinstance(release, dict) else None
    submission_ready = (
        demo_ready
        and quality_ready
        and _release_is_authoritative(release)
        and release_source == cmeie_source
        and receipt.get("source_tree_sha256") == release_source
    )
    return {
        "DEMO_READY": {
            "status": "PASS" if demo_ready else "BLOCKED",
            "reason": (
                "七项流水线产物已由 completed receipt 与现场复验绑定"
                if demo_ready
                else "缺少完整、哈希绑定的演示回执"
            ),
        },
        "QUALITY_GATES_PASSED": {
            "status": "PASS" if quality_ready else "BLOCKED",
            "reason": (
                "CMeIE 与 NL2SQL 权威聚合证据均通过"
                if quality_ready
                else "双质量门缺少权威通过证据"
            ),
        },
        "READY_FOR_SUBMISSION": {
            "status": "PASS" if submission_ready else "BLOCKED",
            "reason": (
                "FINAL_STATUS 明确声明就绪，且演示与质量门均通过"
                if submission_ready
                else "没有完整、哈希绑定的 READY_FOR_SUBMISSION 声明"
            ),
        },
    }


def _safe_public_path(value: Any, *, root: Path = ROOT) -> Path | None:
    """Resolve a public repository text artifact without allowing traversal."""

    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/")
    if any(part.casefold() in _PRIVATE_PATH_PARTS for part in normalized.split("/")):
        return None
    candidate = Path(value)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True) if candidate.is_absolute() else (resolved_root / candidate).resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _read_preview(value: Any, *, limit: int = 1600) -> tuple[str, str] | None:
    path = _safe_public_path(value)
    if path is None or path.suffix.casefold() != ".txt":
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    suffix = "\n……（仅显示前 1600 个字符）" if len(text) > limit else ""
    return str(value), text[:limit] + suffix


def _manifest_preview(manifest: dict[str, Any]) -> dict[str, tuple[str, str]]:
    source = manifest.get("source") if isinstance(manifest, dict) else None
    clean = manifest.get("clean_text") if isinstance(manifest, dict) else None
    source_files = source.get("raw_text_files") if isinstance(source, dict) else None
    clean_files = clean.get("files") if isinstance(clean, dict) else None
    raw_entry = source_files[0] if isinstance(source_files, list) and source_files else None
    clean_entry = clean_files[0] if isinstance(clean_files, list) and clean_files else None
    raw = _read_preview(raw_entry.get("path") if isinstance(raw_entry, dict) else None)
    cleaned = _read_preview(clean_entry.get("path") if isinstance(clean_entry, dict) else None)
    result: dict[str, tuple[str, str]] = {}
    if raw is not None:
        result["before"] = raw
    if cleaned is not None:
        result["after"] = cleaned
    return result


def _event_summary(events: list[dict[str, Any]]) -> dict[str, int]:
    recovery = 0
    errors = 0
    tool_calls = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        text = f"{event.get('kind', '')} {event.get('content', '')}".casefold()
        recovery += int(any(token in text for token in ("recovery", "retry", "恢复", "重试")))
        errors += int(any(token in text for token in ("error", "failed", "exception", "失败")))
        tool_calls += int(any(token in text for token in ("tool", "算子", "operator")))
    return {
        "events": len(events),
        "recovery": recovery,
        "errors": errors,
        "tool_calls": tool_calls,
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _safe_receipt_download(receipt: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    raw_artifacts = receipt.get("artifacts")
    if isinstance(raw_artifacts, dict):
        for name, item in raw_artifacts.items():
            if isinstance(name, str) and isinstance(item, dict):
                artifacts[name] = {
                    key: item.get(key)
                    for key in ("bytes", "sha256")
                    if key in item
                }
    trace = receipt.get("agent_trace")
    safe_trace: dict[str, Any] = {}
    if isinstance(trace, dict):
        for key in ("event_count", "recovery_event_count", "tool_call_count"):
            if type(trace.get(key)) is int:
                safe_trace[key] = trace.get(key)
    errors = receipt.get("error_summary")
    safe_errors: dict[str, Any] = {}
    if isinstance(errors, dict):
        for key in ("error_line_count", "failure_kind", "exception_type"):
            value = errors.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                safe_errors[key] = value
    allowed = (
        "schema_version",
        "run_id",
        "started_at",
        "completed_at",
        "status",
        "returncode",
        "source_tree_sha256",
        "mode",
        "datamate_mode",
        "input_data_sha256",
        "stdout_sha256",
        "stderr_sha256",
    )
    result = {key: receipt.get(key) for key in allowed if key in receipt}
    result.update(
        {
            "agent_trace": safe_trace,
            "error_summary": safe_errors,
            "artifacts": artifacts,
        }
    )
    return result


def _contains_forbidden_download_material(value: Any, *, key: str = "") -> bool:
    if key.casefold() in _FORBIDDEN_DOWNLOAD_KEYS:
        return True
    if isinstance(value, dict):
        return any(
            _contains_forbidden_download_material(child, key=str(child_key))
            for child_key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_download_material(child, key=key) for child in value)
    if isinstance(value, str) and key.casefold() in {"path", "directory", "source_path"}:
        parts = value.replace("\\", "/").split("/")
        return any(part.casefold() in _PRIVATE_PATH_PARTS for part in parts)
    return False


def _safe_manifest_download(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Construct an aggregate Task1 manifest; never forward arbitrary fields.

    Runtime manifests contain local source/output paths.  Those paths are useful
    to the pipeline but are unnecessary in a public browser download and could
    expose an upload filename.  Keep only counts, schemas and content hashes.
    """

    if not isinstance(manifest, dict):
        return None

    def binding(value: Any, *, allow_table_shape: bool = False) -> dict[str, Any] | None:
        if not isinstance(value, dict) or not _is_sha256(value.get("sha256")):
            return None
        result: dict[str, Any] = {"sha256": value["sha256"]}
        size = value.get("bytes")
        if type(size) is int and size >= 0:
            result["bytes"] = size
        if allow_table_shape:
            row_count = value.get("row_count")
            columns = value.get("columns")
            if type(row_count) is int and row_count >= 0:
                result["row_count"] = row_count
            if (
                isinstance(columns, list)
                and len(columns) <= 128
                and all(isinstance(column, str) and len(column) <= 128 for column in columns)
            ):
                result["columns"] = list(columns)
        return result

    source = manifest.get("source")
    clean = manifest.get("clean_text")
    tables = manifest.get("structured_tables")
    if not isinstance(source, dict) or not isinstance(clean, dict) or not isinstance(tables, dict):
        return None
    raw_rows = source.get("raw_text_files")
    clean_rows = clean.get("files")
    if not isinstance(raw_rows, list) or not isinstance(clean_rows, list):
        return None
    safe_raw = [binding(row) for row in raw_rows]
    safe_clean = [binding(row) for row in clean_rows]
    safe_tables = {
        str(name): binding(value, allow_table_shape=True)
        for name, value in sorted(tables.items(), key=lambda item: str(item[0]))
        if isinstance(name, str)
    }
    if (
        any(row is None for row in safe_raw)
        or any(row is None for row in safe_clean)
        or not safe_tables
        or any(value is None for value in safe_tables.values())
    ):
        return None
    task_report = binding(manifest.get("task_report"))
    payload: dict[str, Any] = {
        "schema_version": manifest.get("schema_version"),
        "kind": "dki_public_task1_manifest_aggregate",
        "task_id": manifest.get("task_id"),
        "created_at": manifest.get("created_at"),
        "source": {
            "raw_text_count": source.get("raw_text_count"),
            "raw_text_files": safe_raw,
        },
        "clean_text": {
            "count": clean.get("count"),
            "files": safe_clean,
        },
        "structured_tables": safe_tables,
        "privacy": {
            "contains_paths": False,
            "contains_user_text": False,
            "contains_private_blind_data": False,
        },
    }
    if task_report is not None:
        payload["task_report"] = task_report
    return None if _contains_forbidden_download_material(payload) else payload


def _component_aggregate(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"status": "unavailable", "gate_passed": False}
    return {field: value.get(field) for field in fields if field in value}


def _safe_registry_download(
    registry: dict[str, Any], statuses: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "statuses": statuses,
        "cmeee": _component_aggregate(
            registry.get("cmeee"),
            ("status", "dataset_role", "f1", "acceptance_sha256"),
        ),
        "cmeie": _component_aggregate(
            registry.get("cmeie"),
            (
                "status",
                "gate_status",
                "gate_passed",
                "dataset_role",
                "model_family",
                "f1",
                "metrics",
                "authority",
                "sha256",
                "source_tree_sha256",
            ),
        ),
        "nl2sql": _component_aggregate(
            registry.get("nl2sql"),
            (
                "status",
                "gate_status",
                "gate_passed",
                "n",
                "correct",
                "accuracy",
                "coverage",
                "hard_n",
                "hard_correct",
                "hard_accuracy",
                "authority",
                "sha256",
                "source_tree_sha256",
            ),
        ),
        "npu": _component_aggregate(
            registry.get("npu"),
            (
                "status",
                "environment_label",
                "reported_device",
                "official_dev_f1",
                "npu_docs_per_second",
                "npu_cpu_speedup",
                "acceptance_sha256",
                "expected_acceptance_sha256",
            ),
        ),
        "release": _component_aggregate(
            registry.get("release"),
            (
                "status",
                "available",
                "hash_bound",
                "quality_gates_passed",
                "ready_for_submission",
                "final_status_sha256",
                "source_tree_sha256",
            ),
        ),
    }
    if _contains_forbidden_download_material(payload):
        raise ValueError("aggregate registry unexpectedly contains forbidden detail fields")
    return payload


def _safe_csv_cell(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _csv_bytes(columns: list[Any], rows: list[Any]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([_safe_csv_cell(str(column)) for column in columns])
    for row in rows:
        writer.writerow([_safe_csv_cell(value) for value in row])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def _deterministic_rule_query(
    question: str,
    db_path: Path,
    *,
    expected_database_sha256: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run exactly one deterministic rule inference and one complete execution."""

    from dki.analysis.nl2sql import NL2SQL
    from dki.analysis.nl2sql_protocol import execute_complete_readonly

    database_sha256 = _sha256(db_path)
    question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    if expected_database_sha256 is not None and database_sha256 != expected_database_sha256:
        return {
            "sql": "",
            "engine": "rule",
            "disposition": "blocked",
            "validated": False,
            "columns": [],
            "rows": [],
            "route": "deterministic_rule_only",
            "question_sha256": question_sha256,
            "database_sha256": database_sha256,
            "run_id": run_id,
            "provenance": {
                "database_sha256": database_sha256,
                "expected_database_sha256": expected_database_sha256,
                "llm_used": False,
                "inference_passes": 0,
                "execution_mode": "not_started",
            },
            "error": "演示数据库与 completed receipt 的哈希不一致。",
        }
    engine = NL2SQL(db_path=db_path)
    sql = engine.rule_sql(question)
    provenance = {
        "database_sha256": database_sha256,
        "question_sha256": question_sha256,
        "run_id": run_id,
        "llm_used": False,
        "inference_passes": 1,
        "execution_mode": "complete_readonly",
    }
    if not sql:
        return {
            "sql": "",
            "engine": "rule",
            "disposition": "abstained",
            "validated": False,
            "columns": [],
            "rows": [],
            "route": "deterministic_rule_only",
            "question_sha256": question_sha256,
            "database_sha256": database_sha256,
            "run_id": run_id,
            "provenance": provenance,
            "error": "确定性规则未覆盖该问题；本页面不会自动调用 LLM。",
        }
    try:
        columns, rows = execute_complete_readonly(sql, db_path)
    except Exception as exc:
        return {
            "sql": sql,
            "engine": "rule",
            "disposition": "blocked",
            "validated": False,
            "columns": [],
            "rows": [],
            "route": "deterministic_rule_only",
            "question_sha256": question_sha256,
            "database_sha256": database_sha256,
            "run_id": run_id,
            "provenance": {**provenance, "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest()},
            "error": f"SQL 未通过只读完整执行门：{type(exc).__name__}",
        }
    database_sha256_after = _sha256(db_path)
    if database_sha256_after != database_sha256:
        return {
            "sql": sql,
            "engine": "rule",
            "disposition": "blocked",
            "validated": False,
            "columns": [],
            "rows": [],
            "route": "deterministic_rule_only",
            "question_sha256": question_sha256,
            "database_sha256": database_sha256_after,
            "run_id": run_id,
            "provenance": {
                **provenance,
                "database_sha256_after": database_sha256_after,
                "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            },
            "error": "SQL 执行期间数据库发生变化，结果已丢弃。",
        }
    full_rows = [list(row) for row in rows]
    return {
        "sql": sql,
        "engine": "rule",
        "disposition": "sql",
        "validated": True,
        "columns": list(columns),
        "rows": full_rows,
        "route": "deterministic_rule_only",
        "question_sha256": question_sha256,
        "database_sha256": database_sha256,
        "run_id": run_id,
        "provenance": {
            **provenance,
            "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "result_rows": len(full_rows),
        },
    }


def _graph_html(graph: dict[str, Any], echarts_script: str) -> str:
    encoded = (
        json.dumps(graph, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return f"""
    <div id="kg-core-graph" style="width:100%;height:520px;background:#f7faf9;border:1px solid #dce7e4;border-radius:16px"></div>
    <script>{echarts_script}</script>
    <script>
    const graph = {encoded};
    const chart = echarts.init(document.getElementById('kg-core-graph'));
    chart.setOption({{
      color:['#147d72','#3f748b','#6e62a6','#c1833d','#5d8f7f','#7d8b8f'],
      tooltip: {{backgroundColor:'#ffffff',borderColor:'#dce7e4',textStyle:{{color:'#1b2b30'}}}},
      legend: [{{data: graph.categories.map(x => x.name), textStyle: {{color:'#5f7075'}}}}],
      series: [{{type:'graph', layout:'force', roam:true, draggable:true,
        data:graph.nodes, links:graph.links, categories:graph.categories,
        label:{{show:true,color:'#1b2b30',fontWeight:600,backgroundColor:'rgba(255,255,255,.88)',borderRadius:4,padding:[2,4]}},
        edgeLabel:{{show:true,formatter:'{{c}}',color:'#5f7075',backgroundColor:'rgba(247,250,249,.82)',borderRadius:3,padding:[1,3]}},
        force:{{repulsion:180,edgeLength:[60,140]}},
        lineStyle:{{color:'source',curveness:0.12,opacity:0.52}}}}]
    }});
    window.addEventListener('resize', () => chart.resize());
    </script>
    """


def _evidence_sha(value: Any, *keys: str) -> str:
    if not isinstance(value, dict):
        return "—"
    for key in keys:
        candidate = value.get(key)
        if _is_sha256(candidate):
            return candidate
    return "—"


def _artifact_sha(receipt: dict[str, Any], name: str) -> str:
    artifacts = receipt.get("artifacts") if isinstance(receipt, dict) else None
    artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
    return _evidence_sha(artifact, "sha256")


def _competition_rows(
    registry: dict[str, Any], receipt: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Map the 110-point competition rubric to implementation and evidence."""

    receipt = receipt if isinstance(receipt, dict) else {}
    statuses = _status_summary(receipt, registry)
    demo_pass = statuses.get("DEMO_READY", {}).get("status") == "PASS"
    artifacts = receipt.get("artifacts")

    def artifact_status(name: str) -> str:
        value = artifacts.get(name) if isinstance(artifacts, dict) else None
        return "PASS" if demo_pass and _artifact_is_bound(value) else "BLOCKED"

    cmeie = registry.get("cmeie")
    nl2sql = registry.get("nl2sql")
    npu = registry.get("npu")
    cmeee = registry.get("cmeee")
    npu_bound = _npu_is_bound(npu)
    agent_mode = receipt.get("mode") in {"agent", "agent-offline"}
    datamate_mode = receipt.get("datamate_mode")
    kg_ready = artifact_status("kg") == "PASS"
    database_ready = artifact_status("database") == "PASS"
    cmeie_ready = _component_gate_is_authoritative(cmeie)
    nl2sql_ready = _component_gate_is_authoritative(nl2sql)
    rows = [
        ("任务一", "自然语言任务与 Agent 规划", 5, "Nexent CoreAgent 调度现有 CLI 流水线", "总览→运行完整闭环；任务一→Agent 轨迹", "controlled synthetic / demo", _artifact_sha(receipt, "task1_manifest"), "PASS" if demo_pass and agent_mode else "PARTIAL" if demo_pass else "BLOCKED", "离线演示使用 ScriptedModel；在线模型需显式配置"),
        ("任务一", "DataMate 数据处理算子", 10, "真实 DataMate 或兼容 shim，模式写入回执", "任务一→查看 DataMate 实际模式和 manifest", "controlled synthetic / demo", _artifact_sha(receipt, "task1_manifest"), "PASS" if demo_pass and datamate_mode == "real-datamate" else "PARTIAL" if demo_pass and datamate_mode == "compat-shim" else "BLOCKED", "shim 模式明确标为 PARTIAL，不能冒充真实 DataMate"),
        ("任务一", "状态、产物与可复现性", 10, "create-once run、manifest、输入与源码哈希", "总览→选择运行；任务一→查看运行证据", "public demo receipt", _artifact_sha(receipt, "task1_manifest"), artifact_status("task1_manifest"), "只有 completed 且七项产物通过现场血缘/哈希复验才判 DEMO_READY"),
        ("任务一", "异常恢复与执行轨迹", 5, "events.jsonl 与恢复/错误聚合摘要", "任务一→事件摘要与轨迹", "public aggregate trace", _evidence_sha(receipt, "stdout_sha256"), "PASS" if demo_pass else "BLOCKED", "页面不导出异常原文，避免泄露输入或路径"),
        ("任务二", "医疗实体抽取", 10, "KG 构建链路与实体统计；CMeEE 独立展示", "任务二→实体规模；质量边界", "controlled synthetic + official dev historical", _evidence_sha(cmeee, "acceptance_sha256"), "PASS" if kg_ready and npu_bound else "PARTIAL" if kg_ready else "BLOCKED", "当前 demo KG 与历史 CMeEE 证据必须分栏；CMeEE 不能替代 CMeIE"),
        ("任务二", "关系与三元组抽取", 10, "规则默认链路；联合模型通过显式配置启用", "任务二→关系/三元组与核心子图", "controlled synthetic; CMeIE quality gate separate", _evidence_sha(cmeie, "sha256"), "PASS" if kg_ready and cmeie_ready else "PARTIAL" if kg_ready else "BLOCKED", "完整 text-to-SPO 权威门未过时只能证明演示链路可运行"),
        ("任务二", "Schema 校验、去重与可追溯存储", 10, "KG JSON 与 provenance 哈希绑定", "任务二→统计、证据与子图", "controlled synthetic / pipeline", _artifact_sha(receipt, "kg_provenance"), artifact_status("kg_provenance"), "演示正确性不等同独立质量评测"),
        ("任务二", "知识图谱问答与可视化", 10, "结构化问答、confidence、原文证据、ECharts 子图", "任务二→输入问题并查看结构化答案", "controlled synthetic / demo", _artifact_sha(receipt, "kg"), artifact_status("kg"), "问答模板覆盖范围有限"),
        ("任务二", "Ascend NPU 性能加分", 10, "历史正式 910C 验收环境证据", "NPU 性能证据→核对设备与指标", "historical official dev and NPU acceptance", _evidence_sha(npu, "acceptance_sha256"), "PASS" if npu_bound else "BLOCKED", "展示驱动报告设备名，不伪造硬件型号"),
        ("任务三", "复用任务一、二数据血缘", 5, "Task1 manifest、KG、SQLite provenance 串联", "总览→完整闭环；任务三→provenance", "controlled synthetic / demo", _artifact_sha(receipt, "database_provenance"), artifact_status("database_provenance"), "任一上游哈希缺失即不视为完成"),
        ("任务三", "自然语言到安全 SQL", 10, "确定性 rule 路由、只读完整执行与严格验证", "任务三→生成并执行 SQL", "demo; independent gate separate", _evidence_sha(nl2sql, "sha256"), "PASS" if database_ready and nl2sql_ready else "PARTIAL" if database_ready else "BLOCKED", "演示命中不替代 prospective independent blind 门槛"),
        ("任务三", "统计分析与完整结果", 10, "同一次 SQL 的完整结果供 CSV，页面仅预览 200 行", "任务三→结果预览与完整 CSV", "controlled synthetic / demo", _artifact_sha(receipt, "database"), artifact_status("database"), "下载结果取决于当前演示数据库"),
        ("任务三", "BI 图表与图文说明", 5, "嵌入 ECharts BI 报告并绑定哈希", "任务三→打开 BI 报告", "controlled synthetic / demo", _artifact_sha(receipt, "bi_report"), artifact_status("bi_report"), "静态 BI 不代表盲测质量"),
    ]
    columns = (
        "任务",
        "评分项",
        "分值",
        "当前实现",
        "演示步骤",
        "数据角色",
        "证据SHA-256",
        "状态",
        "已知限制",
    )
    return [dict(zip(columns, row)) for row in rows]


def _run_dir() -> Path | None:
    selected = st.session_state.get("selected_run")
    if (
        not isinstance(selected, str)
        or not selected
        or selected in {".", ".."}
        or "/" in selected
        or "\\" in selected
    ):
        return None
    runs_root = (DEFAULT_ROOT / "runs").resolve()
    try:
        path = (runs_root / selected).resolve(strict=True)
        path.relative_to(runs_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return path if path.is_dir() else None


def _verified_run(
    service: DemoRunService,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Return a run only after the service revalidates every artifact/lineage."""

    run_dir = _run_dir()
    receipt = service.read_receipt(run_dir) if run_dir is not None else None
    if not isinstance(receipt, dict) or receipt.get("_service_verified") is not True:
        return None, None
    return run_dir, receipt


def _metric(label: str, value: Any, note: str = "") -> None:
    st.markdown(
        '<div class="metric"><div class="metric-label">'
        + html.escape(str(label))
        + '</div><div class="metric-value">'
        + html.escape(str(value))
        + '</div><div class="metric-note">'
        + html.escape(str(note))
        + "</div></div>",
        unsafe_allow_html=True,
    )


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary-color:#147d72;
            --background-color:#f5f8f7;
            --secondary-background-color:#ffffff;
            --text-color:#1b2b30;
            --canvas:#f5f8f7;
            --panel:#ffffff;
            --panel-soft:#f8fbfa;
            --ink:#1b2b30;
            --muted:#65757a;
            --line:#dce7e4;
            --teal:#147d72;
            --teal-strong:#0f665d;
            --teal-soft:#e9f4f1;
            --blue:#3f748b;
            --amber:#9a6a1a;
            --red:#bd4d5b;
            --shadow:0 16px 42px -34px rgba(38,63,66,.42);
        }
        html, body, [class*="st-"] {
            font-family:"Segoe UI Variable Text","PingFang SC","Microsoft YaHei UI","Noto Sans CJK SC",sans-serif;
        }
        .stApp {
            color:var(--ink);
            background:var(--canvas);
        }
        [data-testid="stHeader"] {
            background:rgba(245,248,247,.92);
            border-bottom:1px solid rgba(220,231,228,.86);
            backdrop-filter:blur(12px);
        }
        [data-testid="stToolbar"] button { color:var(--muted); }
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width:1440px;
            padding-top:2rem;
            padding-bottom:4rem;
        }
        [data-testid="stSidebar"] {
            background:#f9fbfa;
            border-right:1px solid var(--line);
        }
        [data-testid="stSidebarContent"] { padding-top:1rem; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color:var(--muted);
        }
        [data-testid="stSidebar"] [role="radiogroup"] { gap:.3rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding:.65rem .72rem;
            border:1px solid transparent;
            border-radius:10px;
            color:#46585d;
            transition:background-color .16s ease,border-color .16s ease,color .16s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            color:var(--teal-strong);
            background:#f0f6f4;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            color:var(--teal-strong);
            background:var(--teal-soft);
            border-color:#d3e8e3;
            font-weight:650;
        }
        input[type="radio"], input[type="checkbox"] { accent-color:var(--teal); }
        .hero {
            position:relative;
            overflow:hidden;
            margin-bottom:2.2rem;
            padding:2.15rem 2.35rem 2.2rem;
            border:1px solid var(--line);
            border-radius:20px;
            background:var(--panel);
            box-shadow:var(--shadow);
        }
        .hero::before {
            content:"";
            position:absolute;
            inset:0 0 auto;
            height:4px;
            background:linear-gradient(90deg,#147d72 0 33.333%,#3f748b 33.333% 66.666%,#8aa79f 66.666% 100%);
        }
        .hero-kicker {
            color:var(--teal-strong);
            font-size:.73rem;
            font-weight:700;
            letter-spacing:.13em;
            text-transform:uppercase;
        }
        .hero-title {
            margin:.58rem 0 .45rem;
            color:var(--ink);
            font-family:"Segoe UI Variable Display","Aptos Display","PingFang SC","Microsoft YaHei UI",sans-serif;
            font-size:clamp(2rem,3vw,2.55rem);
            font-weight:720;
            letter-spacing:-.035em;
            line-height:1.15;
        }
        .hero-copy {
            max-width:58rem;
            color:var(--muted);
            font-size:1rem;
            line-height:1.72;
        }
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color:var(--ink);
            font-family:"Segoe UI Variable Display","Aptos Display","PingFang SC","Microsoft YaHei UI",sans-serif;
            letter-spacing:-.018em;
        }
        [data-testid="stMarkdownContainer"] h3 { margin-top:1.75rem; }
        [data-testid="stMarkdownContainer"] h4 { margin-top:1.45rem; }
        [data-testid="stMarkdownContainer"] p { line-height:1.68; }
        [data-testid="stCaptionContainer"] { color:var(--muted); }
        hr { border-color:var(--line); }
        .metric {
            position:relative;
            min-height:6.8rem;
            padding:1.12rem 1.15rem 1rem;
            border:1px solid var(--line);
            border-radius:15px;
            background:var(--panel);
            box-shadow:0 10px 28px -27px rgba(38,63,66,.48);
        }
        .metric::before {
            content:"";
            position:absolute;
            top:0;
            left:1.15rem;
            width:2.5rem;
            height:3px;
            border-radius:0 0 999px 999px;
            background:var(--teal);
        }
        .metric-label {
            color:var(--muted);
            font-size:.76rem;
            font-weight:650;
            letter-spacing:.025em;
        }
        .metric-value {
            margin-top:.42rem;
            color:var(--ink);
            font-family:"Segoe UI Variable Display","Aptos Display","PingFang SC","Microsoft YaHei UI",sans-serif;
            font-size:1.28rem;
            font-weight:720;
            line-height:1.25;
            word-break:break-word;
        }
        .metric-note {
            margin-top:.32rem;
            color:var(--muted);
            font-size:.72rem;
            line-height:1.4;
        }
        .status-pass { color:var(--teal); font-weight:700; }
        .status-fail { color:var(--red); font-weight:700; }
        [data-testid="stForm"] {
            padding:1.15rem 1.25rem;
            border:1px solid var(--line);
            border-radius:15px;
            background:var(--panel);
            box-shadow:0 12px 32px -31px rgba(38,63,66,.5);
        }
        [data-testid="stExpander"] {
            overflow:hidden;
            border:1px solid var(--line);
            border-radius:13px;
            background:var(--panel);
        }
        [data-testid="stExpander"] summary:hover { color:var(--teal-strong); }
        [data-testid="stAlert"] {
            border-radius:13px;
            color:var(--ink);
            box-shadow:none;
        }
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div {
            border-color:#cad9d5 !important;
            border-radius:10px !important;
            background:var(--panel) !important;
            box-shadow:none !important;
        }
        [data-baseweb="input"]:focus-within > div,
        [data-baseweb="textarea"]:focus-within > div,
        [data-baseweb="select"]:focus-within > div {
            border-color:var(--teal) !important;
            box-shadow:0 0 0 3px rgba(20,125,114,.12) !important;
        }
        input, textarea { color:var(--ink) !important; }
        [data-baseweb="popover"], [role="listbox"] {
            border-color:var(--line) !important;
            border-radius:12px !important;
            background:var(--panel) !important;
            box-shadow:0 18px 46px -27px rgba(38,63,66,.46) !important;
        }
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            min-height:2.6rem;
            border:1px solid #c8d7d3;
            border-radius:10px;
            color:var(--ink);
            background:var(--panel);
            box-shadow:0 8px 20px -20px rgba(38,63,66,.56);
            font-weight:650;
            transition:transform .16s ease,border-color .16s ease,background-color .16s ease,box-shadow .16s ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            border-color:#a8cec5;
            color:var(--teal-strong);
            background:var(--teal-soft);
            transform:translateY(-1px);
        }
        [data-testid="stBaseButton-primary"],
        button[kind="primary"] {
            border-color:var(--teal) !important;
            color:#ffffff !important;
            background:var(--teal) !important;
            box-shadow:0 12px 24px -17px rgba(20,125,114,.72) !important;
        }
        [data-testid="stBaseButton-primary"]:hover,
        button[kind="primary"]:hover {
            border-color:var(--teal-strong) !important;
            color:#ffffff !important;
            background:var(--teal-strong) !important;
        }
        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [tabindex]:focus-visible {
            outline:3px solid var(--teal) !important;
            outline-offset:2px;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            overflow:hidden;
            border:1px solid var(--line);
            border-radius:13px;
            background:var(--panel);
        }
        [data-testid="stJson"],
        [data-testid="stCodeBlock"],
        .stCodeBlock,
        .react-json-view {
            overflow:hidden;
            border:1px solid var(--line);
            border-radius:13px;
            color:var(--ink) !important;
            background:#f7faf9 !important;
        }
        code, pre {
            font-family:"Cascadia Mono","SFMono-Regular",Consolas,"Liberation Mono",monospace !important;
        }
        iframe { border-radius:16px; }
        @media (max-width:900px) {
            .main .block-container,
            [data-testid="stMainBlockContainer"] { padding-top:1.25rem; }
            .hero { padding:1.75rem 1.8rem; }
        }
        @media (max-width:640px) {
            .main .block-container,
            [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
            .hero { margin-bottom:1.55rem; padding:1.4rem; border-radius:17px; }
            .hero-title { font-size:1.85rem; }
            .metric { min-height:6.2rem; border-radius:13px; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation:none !important; transition:none !important; scroll-behavior:auto !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">ModelEngine · dki-agents</div>'
        '<div class="hero-title">数据 → 知识 → 洞察</div>'
        '<div class="hero-copy">面向评审的可复现医疗数据智能体闭环。页面只显示哈希绑定证据；缺少权威证据时保持 BLOCKED。</div></div>',
        unsafe_allow_html=True,
    )


def _service() -> DemoRunService:
    return DemoRunService(DEFAULT_ROOT, python=os.getenv("DKI_DEMO_PYTHON"))


_FAILURE_HINTS = {
    "log_initialization": "无法在运行目录创建日志文件；请检查运行目录权限或磁盘空间。",
    "process_launch_or_runtime": "demo-all 子进程未能启动或运行中断；请确认解释器与依赖可用。",
    "timeout": "闭环执行超时；请重试或改用更小的数据规模。",
    "subprocess_exit": "demo-all 子进程非零退出，产物未齐全；这是 fail-closed 拦截，未验证的结果不会展示。",
    "stdout_read": "读取子进程标准输出失败。",
    "stderr_read": "读取子进程标准错误失败。",
    "summary": "产物汇总或血缘校验未通过；这是 fail-closed 拦截。",
    "completed_receipt_publish": "终态回执写入失败；create-once 目录可能已存在同名回执。",
}


def _render_failure_reason(service: DemoRunService, receipt_object: Any) -> None:
    """Explain a failed run in Chinese; the fail-closed verdict itself is unchanged."""

    payload = service.read_receipt(receipt_object.run_dir) or {}
    errors = payload.get("error_summary")
    errors = errors if isinstance(errors, dict) else {}
    kind = str(errors.get("failure_kind") or "unknown")
    exception_type = str(errors.get("exception_type") or "未知异常")
    hint = _FAILURE_HINTS.get(kind, "运行未通过完整性校验；产物保持 fail-closed，不会展示未验证结果。")
    st.error(f"闭环未完成 · 阶段 {kind} · {exception_type}\n\n{hint}")
    st.caption(
        f"完整回溯：{receipt_object.run_dir / 'failed.json'} 与 {receipt_object.run_dir / 'stderr.log'}"
    )


def _render_statuses(
    receipt: dict[str, Any] | None, registry: dict[str, Any]
) -> dict[str, dict[str, str]]:
    statuses = _status_summary(receipt, registry)
    columns = st.columns(3)
    for column, name in zip(columns, ("DEMO_READY", "QUALITY_GATES_PASSED", "READY_FOR_SUBMISSION")):
        item = statuses.get(name, {})
        with column:
            _metric(name, item.get("status", "BLOCKED"), item.get("reason", "证据不足"))
    return statuses


def _overview(service: DemoRunService) -> None:
    st.subheader("项目总览")
    st.caption(f"服务运行目录：{DEFAULT_ROOT} · server RID：{SERVER_RID}")
    with st.form("run_demo"):
        st.write("使用仓库内置的确定性合成医疗记录，执行 Task1 → Task2 → Task3 完整链路。")
        agent = st.checkbox("展示真实 Nexent CoreAgent 工具轨迹", value=True)
        offline = st.checkbox("使用可复现 ScriptedModel（推荐）", value=True)
        submitted = st.form_submit_button("运行一次完整闭环", type="primary")
    if submitted:
        try:
            with st.status("正在执行数据—知识—洞察闭环……", expanded=True) as status:
                receipt_object = service.run(agent=agent, offline=offline)
                st.session_state["selected_run"] = receipt_object.run_id
                completed = receipt_object.status == "completed"
                status.update(
                    label=f"运行{'完成' if completed else '失败'}：{receipt_object.run_id}",
                    state="complete" if completed else "error",
                )
            if completed:
                st.rerun()
            # A rerun would discard the status box, so the reason is rendered here.
            _render_failure_reason(service, receipt_object)
        except Exception as exc:
            st.error(f"运行启动失败：{type(exc).__name__}")

    runs = service.list_runs()
    if runs:
        labels = [path.name for path in runs]
        current = st.session_state.get("selected_run")
        selected = st.selectbox(
            "查看运行",
            labels,
            index=labels.index(current) if current in labels else 0,
        )
        st.session_state["selected_run"] = selected
    run_dir = _run_dir()
    receipt = service.read_receipt(run_dir) if run_dir else None
    registry = _safe_registry()
    _render_statuses(receipt, registry)

    columns = st.columns(4)
    with columns[0]:
        _metric("当前运行", (receipt or {}).get("run_id", "—"), "create-once")
    with columns[1]:
        _metric("流水线状态", (receipt or {}).get("status", "未运行"), "pipeline receipt")
    with columns[2]:
        _metric("运行模式", (receipt or {}).get("mode", "—"), "deterministic / agent")
    with columns[3]:
        _metric("DataMate", (receipt or {}).get("datamate_mode", "—"), "真实或兼容模式")

    if receipt:
        st.markdown("#### 哈希绑定摘要")
        st.json(_safe_receipt_download(receipt))


def _task1(service: DemoRunService) -> None:
    st.subheader("任务一 · 数据处理智能体")
    st.info("自然语言任务：清洗本次合成医疗文本，发布不可变 manifest，并将同一快照交给知识图谱与分析任务。")
    run_dir, receipt = _verified_run(service)
    if not run_dir:
        st.info("请先在项目总览完成一条通过现场血缘与哈希复验的完整闭环。")
        return
    assert receipt is not None
    manifest = _load_json(run_dir / "processed" / "manifest.json") or {}
    runtime_receipt = _load_json(run_dir / "demo_runtime.json") or {}
    runtime = (
        runtime_receipt.get("runtime")
        if isinstance(runtime_receipt.get("runtime"), dict)
        else {}
    )
    agent_trace = (
        runtime_receipt.get("agent_trace")
        if isinstance(runtime_receipt.get("agent_trace"), dict)
        else {}
    )
    pipeline = (
        runtime_receipt.get("pipeline")
        if isinstance(runtime_receipt.get("pipeline"), dict)
        else {}
    )
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    clean = manifest.get("clean_text") if isinstance(manifest.get("clean_text"), dict) else {}
    columns = st.columns(4)
    with columns[0]:
        _metric("原始文本", source.get("raw_text_count", "—"), "source snapshot")
    with columns[1]:
        _metric("清洗文本", clean.get("count", "—"), "Task1 manifest")
    with columns[2]:
        _metric("运行模式", receipt.get("mode", "—"), "Nexent / deterministic")
    with columns[3]:
        _metric("DataMate 实际模式", receipt.get("datamate_mode", "—"), "real-datamate / compat-shim")
    if manifest:
        st.success("Task1 manifest 已发布；Task2/Task3 只消费此运行快照。")
        st.caption(
            f"task_id={manifest.get('task_id', '—')} · created_at={manifest.get('created_at', '—')}"
        )
    else:
        st.warning("Task1 manifest 尚未生成或无法解析。")

    st.markdown("#### 结构化规划、工具与算子执行")
    runtime_columns = st.columns(4)
    with runtime_columns[0]:
        _metric("Agent 框架", runtime.get("core_agent") or "deterministic", "runtime receipt")
    with runtime_columns[1]:
        _metric("模型后端", runtime.get("model_backend") or "未启用", "ScriptedModel / online")
    tool_calls = agent_trace.get("tool_calls")
    with runtime_columns[2]:
        _metric("工具调用", len(tool_calls) if isinstance(tool_calls, list) else 0, "structured trace")
    recovery_events = pipeline.get("recovery_events")
    with runtime_columns[3]:
        _metric("恢复动作", len(recovery_events) if isinstance(recovery_events, list) else 0, "task report")
    tasks = agent_trace.get("natural_language_tasks")
    if isinstance(tasks, list) and tasks:
        st.markdown("**Agent 任务规划**")
        st.dataframe(
            [{"阶段": index + 1, "自然语言任务": task} for index, task in enumerate(tasks)],
            use_container_width=True,
            hide_index=True,
        )
    if isinstance(tool_calls, list) and tool_calls:
        st.markdown("**工具调用轨迹**")
        st.dataframe(tool_calls, use_container_width=True, hide_index=True)
    operators = pipeline.get("operators")
    if isinstance(operators, list) and operators:
        st.markdown("**算子耗时与结果**")
        st.dataframe(operators, use_container_width=True, hide_index=True)
    if isinstance(recovery_events, list) and recovery_events:
        with st.expander("异常恢复证据", expanded=True):
            st.dataframe(recovery_events, use_container_width=True, hide_index=True)

    events = service.tail_events(run_dir)
    summary = _event_summary(events)
    st.markdown("#### 执行与恢复摘要")
    columns = st.columns(4)
    with columns[0]:
        _metric("事件", summary.get("events", 0), "events.jsonl")
    with columns[1]:
        _metric("工具/算子事件", summary.get("tool_calls", 0), "keyword aggregate")
    with columns[2]:
        _metric("恢复/重试", summary.get("recovery", 0), "keyword aggregate")
    with columns[3]:
        _metric("错误事件", summary.get("errors", 0), "keyword aggregate")

    previews = _manifest_preview(manifest)
    if previews:
        st.markdown("#### 清洗前后对比（首份可用文本）")
        before_column, after_column = st.columns(2)
        with before_column:
            st.caption(f"清洗前 · {previews.get('before', ('不可用', ''))[0]}")
            st.code(previews.get("before", ("", "无可安全预览的源文本"))[1], language="text")
        with after_column:
            st.caption(f"清洗后 · {previews.get('after', ('不可用', ''))[0]}")
            st.code(previews.get("after", ("", "无可安全预览的清洗文本"))[1], language="text")
    else:
        st.caption("manifest 未提供可在仓库公共路径内安全读取的前后文本。")

    st.markdown("#### Agent / 流水线事件")
    if events:
        st.dataframe(events, use_container_width=True, hide_index=True)
    else:
        st.info("暂无事件记录。")


def _task2(service: DemoRunService) -> None:
    st.subheader("任务二 · 医疗知识图谱与问答")
    run_dir, receipt = _verified_run(service)
    kg_path = run_dir / "kg" / "medical_kg.json" if run_dir else None
    if kg_path is None or not kg_path.is_file():
        st.info("请先完成一条通过现场血缘与哈希复验的完整闭环。")
        return
    assert receipt is not None
    from dki.kg.qa import answer_question
    from dki.kg.store import KnowledgeGraph

    kg = KnowledgeGraph.load(kg_path)
    stats = kg.stats()
    relations = stats.get("triples_by_relation")
    columns = st.columns(4)
    with columns[0]:
        _metric("实体", stats.get("n_entities", "—"), "知识图谱")
    with columns[1]:
        _metric("三元组", stats.get("n_triples", "—"), "Schema 校验后")
    with columns[2]:
        _metric("关系类型", len(relations) if isinstance(relations, dict) else "—", "实际激活")
    with columns[3]:
        _metric("平均置信度", stats.get("avg_confidence", "—"), "聚合 confidence")

    question = st.text_input("图谱问答", value="高血压有哪些症状？")
    if st.button("查询图谱", type="primary"):
        result = answer_question(kg, question)
        st.write(result.get("answer", "未返回答案"))
        structured = result.get("structured")
        if isinstance(structured, list) and structured:
            st.markdown("#### 结构化答案与置信度")
            st.dataframe(structured, use_container_width=True, hide_index=True)
        evidence = result.get("evidence")
        if isinstance(evidence, list) and evidence:
            st.markdown("#### 原文证据")
            st.dataframe(evidence, use_container_width=True, hide_index=True)

    st.markdown("#### 核心子图")
    script_path = ROOT / "dki" / "analysis" / "static" / "echarts.min.js"
    try:
        script = script_path.read_text(encoding="utf-8")
        graph = kg.to_echarts_graph(max_nodes=90)
    except (OSError, UnicodeError, ValueError):
        st.warning("核心子图资源不可用。")
    else:
        import streamlit.components.v1 as components

        components.html(_graph_html(graph, script), height=540, scrolling=False)

    with st.expander("质量指标边界"):
        st.info("controlled synthetic、CMeEE NER、CMeIE text-to-SPO 是不同数据角色；页面不会把条件化或开发指标冒充端到端质量。")
        registry = _safe_registry()
        st.json(
            {
                "controlled_synthetic": "pipeline correctness only",
                "cmeee_official_dev": registry.get("cmeee", {}),
                "cmeie_text_to_spo": registry.get("cmeie", {}),
            }
        )


def _task3(service: DemoRunService) -> None:
    st.subheader("任务三 · NL2SQL 与 BI 洞察")
    run_dir, receipt = _verified_run(service)
    db_path = run_dir / "insight.db" if run_dir else None
    if db_path is None or not db_path.is_file():
        st.info("请先完成一条通过现场血缘与哈希复验的完整闭环。")
        return
    assert receipt is not None
    st.caption("固定路由：deterministic rule。此页不启用 LLM fallback；验证和 CSV 共用同一次 SQL 推理与完整执行结果。")
    question = st.text_input("自然语言分析问题", value="各科室就诊量是多少？")
    state_key = f"nl2sql_result:{run_dir.name}"
    database_binding = receipt.get("artifacts", {}).get("database", {})
    expected_database_sha256 = database_binding.get("sha256")
    current_question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    if st.button("生成并执行 SQL", type="primary"):
        with st.spinner("正在进行确定性规则推理和只读完整执行……"):
            st.session_state[state_key] = _deterministic_rule_query(
                question,
                db_path,
                expected_database_sha256=(
                    expected_database_sha256
                    if _is_sha256(expected_database_sha256)
                    else None
                ),
                run_id=run_dir.name,
            )

    result = st.session_state.get(state_key)
    if isinstance(result, dict) and (
        result.get("question_sha256") != current_question_sha256
        or result.get("run_id") != run_dir.name
        or result.get("database_sha256") != expected_database_sha256
    ):
        st.info("问题、运行或数据库已经变化；请重新点击生成，旧结果不会显示或下载。")
        result = None
    if isinstance(result, dict):
        st.code(result.get("sql", ""), language="sql")
        st.caption(
            " · ".join(
                (
                    f"engine={result.get('engine', 'rule')}",
                    f"route={result.get('route', 'deterministic_rule_only')}",
                    f"disposition={result.get('disposition', 'blocked')}",
                    f"validated={result.get('validated', False)}",
                )
            )
        )
        error = result.get("error")
        if isinstance(error, str) and error:
            st.warning(error)
        columns = result.get("columns")
        rows = result.get("rows")
        if isinstance(columns, list) and isinstance(rows, list) and columns:
            preview_rows = rows[:200]
            st.dataframe(
                [dict(zip(columns, row)) for row in preview_rows],
                use_container_width=True,
                hide_index=True,
            )
            if len(rows) > 200:
                st.caption(f"页面仅预览 200/{len(rows)} 行；CSV 包含同一次执行的全部结果。")
            st.download_button(
                "下载完整结果 CSV",
                data=_csv_bytes(columns, rows),
                file_name=f"{run_dir.name}-nl2sql-complete.csv",
                mime="text/csv",
            )
        provenance = result.get("provenance")
        if isinstance(provenance, dict):
            st.markdown("#### 路由与执行 provenance")
            st.json(provenance)

    report = run_dir / "reports" / "insight_report.html"
    if report.is_file():
        with st.expander("打开 ECharts BI 报告", expanded=True):
            import streamlit.components.v1 as components

            components.html(
                report.read_text(encoding="utf-8", errors="replace"),
                height=720,
                scrolling=True,
            )


def _npu_evidence() -> None:
    st.subheader("Ascend NPU 性能证据")
    registry = _safe_registry()
    raw = registry.get("npu")
    evidence = raw if isinstance(raw, dict) else {}
    actual = evidence.get("acceptance_sha256")
    bound = _npu_is_bound(evidence)
    if bound:
        st.success("权威 NPU 验收文档哈希匹配。")
    else:
        st.error("NPU 验收证据缺失或哈希不匹配；数值保持不可采信。")

    def trusted_number(key: str, digits: int, suffix: str = "") -> str:
        value = evidence.get(key)
        if not bound or isinstance(value, bool) or not isinstance(value, (int, float)):
            return "—"
        return f"{float(value):.{digits}f}{suffix}"

    columns = st.columns(3)
    with columns[0]:
        _metric("官方 dev F1", trusted_number("official_dev_f1", 6), "FP16 OM")
    with columns[1]:
        _metric("NPU 微批吞吐", trusted_number("npu_docs_per_second", 3), "docs/s")
    with columns[2]:
        _metric("NPU/CPU", trusted_number("npu_cpu_speedup", 4, "×"), "速度比")

    st.markdown("#### 设备标识")
    device_columns = st.columns(2)
    with device_columns[0]:
        _metric("用户环境称谓", evidence.get("environment_label", "未绑定") if bound else "未绑定", "environment_label")
    with device_columns[1]:
        _metric("驱动报告设备", evidence.get("reported_device", "未绑定") if bound else "未绑定", "reported_device")
    if bound:
        details = {
            "driver": evidence.get("driver"),
            "cann": evidence.get("cann"),
            "torch": evidence.get("torch"),
            "torch_npu": evidence.get("torch_npu"),
            "acceptance_sha256": actual,
        }
        st.json(details)
        acceptance_path = _safe_public_path(evidence.get("acceptance_path"))
        canonical = ROOT / "docs" / "remote_npu_acceptance_20260725.md"
        if acceptance_path is not None and acceptance_path == canonical.resolve():
            with st.expander("查看绑定验收摘要"):
                st.markdown(acceptance_path.read_text(encoding="utf-8", errors="replace"))

    st.markdown("#### 当前服务设备快照（展示用，不替代历史正式指标）")
    current = _live_device_snapshot(evidence.get("current_device_snapshot"))
    if current is not None:
        snapshot_columns = st.columns(3)
        with snapshot_columns[0]:
            _metric("请求逻辑设备", current["requested_logical_device"], "实时快照")
        with snapshot_columns[1]:
            detected_name = current.get("detected_device_name")
            if current["detected_available"] and not detected_name:
                detected_name = "已检测（运行时未返回型号）"
            _metric("实测设备名", detected_name or "未检测到 NPU", "原始运行时值")
        with snapshot_columns[2]:
            _metric("torch_npu", current.get("torch_npu") or "unavailable", current.get("created_at", ""))
        if current["detected_available"]:
            st.caption(
                f"环境称谓：{current['environment_label']}；检测来源："
                + ", ".join(current["detection_sources"])
            )
        else:
            st.caption("快照有效，但当前进程未检测到 NPU；不会借用历史设备名称。")
        summary_lines = current["summary_lines"]
        if summary_lines:
            with st.expander("npu-smi 当前摘要"):
                st.code("\n".join(str(line) for line in summary_lines), language="text")
    else:
        st.caption("当前设备快照不可用；页面不会用历史设备名替代实时值。")


def _score_map(service: DemoRunService) -> None:
    st.subheader("比赛任务点对照与证据导出")
    run_dir = _run_dir()
    receipt = service.read_receipt(run_dir) if run_dir else None
    receipt = receipt if isinstance(receipt, dict) else {}
    manifest = _load_json(run_dir / "processed" / "manifest.json") if run_dir else None
    registry = _safe_registry()
    statuses = _render_statuses(receipt, registry)
    rows = _competition_rows(registry, receipt)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(f"评分项共 {sum(int(row.get('分值', 0)) for row in rows)} 分（含 NPU 加分）；状态来自绑定证据，不是页面自行打分。")

    st.markdown("#### 安全公开下载")
    st.caption("仅导出 receipt 摘要、Task1 manifest 和 aggregate-only registry；不包含 private/detail/gold/prediction。")
    download_columns = st.columns(3)
    run_id = str(receipt.get("run_id", "no-run"))
    with download_columns[0]:
        if receipt:
            st.download_button(
                "下载演示 receipt",
                data=_json_bytes(_safe_receipt_download(receipt)),
                file_name=f"{run_id}-demo-receipt.json",
                mime="application/json",
            )
        else:
            st.caption("暂无 receipt")
    with download_columns[1]:
        safe_manifest = _safe_manifest_download(manifest) if manifest else None
        if safe_manifest is not None:
            st.download_button(
                "下载 Task1 manifest",
                data=_json_bytes(safe_manifest),
                file_name=f"{run_id}-task1-manifest.json",
                mime="application/json",
            )
        else:
            st.caption("manifest 不可用或未通过公开内容检查")
    with download_columns[2]:
        st.download_button(
            "下载聚合验收摘要",
            data=_json_bytes(_safe_registry_download(registry, statuses)),
            file_name=f"{run_id}-aggregate-evidence.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(
        page_title="dki-agents 证据驾驶舱",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()
    _header()
    service = _service()
    pages = [
        "项目总览",
        "任务一 · 数据处理",
        "任务二 · 知识图谱",
        "任务三 · NL2SQL/BI",
        "NPU 性能证据",
        "比赛任务点对照",
    ]
    page = st.sidebar.radio("导航", pages)
    st.sidebar.caption("每次运行独立目录 · 不覆盖历史证据")
    if page == pages[0]:
        _overview(service)
    elif page == pages[1]:
        _task1(service)
    elif page == pages[2]:
        _task2(service)
    elif page == pages[3]:
        _task3(service)
    elif page == pages[4]:
        _npu_evidence()
    else:
        _score_map(service)


if __name__ == "__main__":
    main()
