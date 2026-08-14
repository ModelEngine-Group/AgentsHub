"""DataMate/Nexent integration evidence collector."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import socket
import time
import uuid
from typing import Any
from urllib import error, parse, request

from demo.task1_openapi import TASK1_OPENAPI_OPERATION_ID, TASK1_OPENAPI_SERVICE_NAME, build_task1_openapi_spec
from demo.task2_openapi import (
    TASK2_ASK_OPERATION_ID,
    TASK2_BUILD_OPERATION_ID,
    TASK2_OPENAPI_SERVICE_NAME,
    build_task2_openapi_spec,
)
from demo.task3_openapi import (
    TASK3_ANALYZE_OPERATION_ID,
    TASK3_CONFIG_OPERATION_ID,
    TASK3_INSIGHT_OPERATION_ID,
    TASK3_OPENAPI_SERVICE_NAME,
    build_task3_openapi_spec,
)


SECRET_MARKERS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "access_key",
    "authorization",
)

DEFAULT_DATAMATE_URL = "http://localhost:30000"
DEFAULT_DATAMATE_API_BASE = "http://localhost:8080"
DEFAULT_DATAMATE_RUNTIME_API = "http://localhost:18000"
DEFAULT_NEXENT_URL = "http://localhost:3000"
DEFAULT_NEXENT_CONFIG_API = "http://localhost:5010"
DEFAULT_NEXENT_RUNTIME_API = "http://localhost:5014"
DEFAULT_NEXENT_MCP_MANAGEMENT_API = "http://localhost:5015"
DEFAULT_DEMO_API_URL = "http://localhost:8088"
DEFAULT_MODEL_API_BASE = "http://host.docker.internal:18089/v1"
DEFAULT_DATAMATE_PROVIDER = "OpenAI"
DEFAULT_DATAMATE_EMBEDDING_CANDIDATES = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)
DEFAULT_DATAMATE_DATASET_TYPE = "TEXT"
DEFAULT_DATAMATE_KB_TYPE = "DOCUMENT"
DEFAULT_DATAMATE_PROCESS_TYPE = "DEFAULT_CHUNK"
DATAMATE_KB_SUPPORTED_SUFFIXES = {
    ".txt",
    ".json",
    ".csv",
    ".md",
    ".markdown",
    ".log",
    ".xml",
    ".html",
    ".htm",
    ".yaml",
    ".yml",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".pdf",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SECRET_MARKERS)


def _redact_value(key: str, value: str | None) -> str | None:
    if value is None:
        return None
    if _is_secret_key(key):
        return "***REDACTED***" if value else ""
    return value


def _redact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _redact_payload(_redact_value(str(key), value) if isinstance(value, str) else value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_redact_payload(item) for item in payload]
    return payload


def _env_value(name: str, default: str, env_files: list[dict[str, str]]) -> str:
    if os.getenv(name):
        return str(os.getenv(name))
    for env_file in env_files:
        if env_file.get(name):
            return env_file[name]
    return default


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _base_url_variants(base_url: str) -> list[str]:
    normalized = _normalize_base_url(base_url)
    variants = [normalized]
    if normalized.endswith("/v1"):
        variants.append(normalized[: -len("/v1")].rstrip("/"))
    elif normalized:
        variants.append(normalized + "/v1")
    return [item for index, item in enumerate(variants) if item and item not in variants[:index]]


def _container_accessible_url(url: str) -> str:
    parsed = parse.urlsplit(url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return url
    host = os.getenv("DOCKER_HOST_GATEWAY", "host.docker.internal")
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    return parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _base_url_matches(candidate: str, reference: str) -> bool:
    return _normalize_base_url(candidate) in _base_url_variants(reference)


def _json_body(body: dict[str, Any] | None) -> bytes | None:
    if body is None:
        return None
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _build_multipart_form_data(
    fields: dict[str, Any],
    files: list[dict[str, Any]],
) -> tuple[bytes, str]:
    boundary = f"----ModelEngineBoundary{uuid.uuid4().hex}"
    payload = bytearray()

    def append_line(text: str | bytes) -> None:
        if isinstance(text, bytes):
            payload.extend(text)
        else:
            payload.extend(text.encode("utf-8"))

    for name, value in fields.items():
        if value is None:
            continue
        append_line(f"--{boundary}\r\n")
        append_line(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        append_line(str(value))
        append_line("\r\n")

    for file_item in files:
        field_name = str(file_item.get("field_name", "file"))
        filename = str(file_item.get("filename", "upload.bin")).replace('"', "")
        content = file_item.get("content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")
        content_type = str(
            file_item.get("content_type")
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )
        append_line(f"--{boundary}\r\n")
        append_line(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        )
        append_line(f"Content-Type: {content_type}\r\n\r\n")
        append_line(content)
        append_line("\r\n")

    append_line(f"--{boundary}--\r\n")
    return bytes(payload), f"multipart/form-data; boundary={boundary}"


def _decode_json(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _decode_sse_content(text: str) -> str:
    """Concatenate text/event-stream content chunks for marker detection."""
    chunks: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = _decode_json(line[len("data:"):].strip())
        if not isinstance(payload, dict):
            continue
        content = payload.get("content")
        if isinstance(content, str):
            chunks.append(content)
    return "".join(chunks)


def _unwrap_response_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        data = payload.get("data")
        if data is not None:
            return data
    return payload


def _extract_payload_items(payload: Any) -> list[dict[str, Any]]:
    payload = _unwrap_response_payload(payload)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("content", "items", "records", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("content", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _extract_json_objects_from_text(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        objects.append(payload)
        index = start + end
    return objects


def _find_task1_result_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("summary"), dict) and isinstance(payload.get("plan"), dict):
            return payload
        for value in payload.values():
            found = _find_task1_result_payload(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_task1_result_payload(item)
            if found:
                return found
    elif isinstance(payload, str):
        for item in _extract_json_objects_from_text(payload):
            found = _find_task1_result_payload(item)
            if found:
                return found
    return {}


def _task1_v3_markers_from_payload(payload: Any, output_root: Path | None = None) -> dict[str, Any]:
    result = _find_task1_result_payload(payload)
    if not result and output_root is not None:
        summary_path = output_root / "data" / "processed" / "datamate_processing_summary.json"
        plan_path = output_root / "data" / "processed" / "datamate_processing_plan.json"
        performance_path = output_root / "data" / "processed" / "datamate_performance_report.json"
        if summary_path.exists() or plan_path.exists():
            result = {
                "summary": _read_json(summary_path),
                "plan": _read_json(plan_path),
                "performance_report": _read_json(performance_path),
            }
    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    plan = result.get("plan", {}) if isinstance(result.get("plan"), dict) else {}
    performance = result.get("performance_report", {}) if isinstance(result.get("performance_report"), dict) else {}
    if not performance and output_root is not None:
        performance = _read_json(output_root / "data" / "processed" / "datamate_performance_report.json")
    task_understanding = plan.get("task_understanding", {}) if isinstance(plan.get("task_understanding"), dict) else {}
    semantic_graph = (
        task_understanding.get("semantic_constraint_graph", {})
        if isinstance(task_understanding.get("semantic_constraint_graph"), dict)
        else {}
    )
    autonomous_plan = plan.get("autonomous_plan", {}) if isinstance(plan.get("autonomous_plan"), dict) else {}
    workflow_plan = plan.get("workflow_plan", {}) if isinstance(plan.get("workflow_plan"), dict) else {}
    etl_contract = (
        workflow_plan.get("etl_contract", {})
        if isinstance(workflow_plan.get("etl_contract"), dict)
        else {}
    )
    scorecard = etl_contract.get("quality_scorecard", {}) if isinstance(etl_contract.get("quality_scorecard"), dict) else {}
    operator_planning = plan.get("operator_planning", {}) if isinstance(plan.get("operator_planning"), dict) else {}
    policy_model = (
        plan.get("policy_model_prediction", {})
        if isinstance(plan.get("policy_model_prediction"), dict)
        else {}
    )
    return {
        "status": summary.get("status") or result.get("status"),
        "cleaning_task_status": summary.get("cleaning_task_status"),
        "task_understanding_version": summary.get("task_understanding_version") or task_understanding.get("understanding_version"),
        "semantic_constraint_graph_version": semantic_graph.get("graph_version"),
        "autonomous_planner": summary.get("autonomous_planner") or autonomous_plan.get("planner"),
        "operator_planner": summary.get("operator_planner") or operator_planning.get("planner"),
        "structured_etl_contract_version": etl_contract.get("contract_version"),
        "structured_etl_quality_score": scorecard.get("overall_score"),
        "capability_coverage_ratio": summary.get("capability_coverage_ratio") or performance.get("capability_coverage_ratio"),
        "quality_preserved": summary.get("quality_preserved_under_performance_optimization") if "quality_preserved_under_performance_optimization" in summary else performance.get("quality_preserved"),
        "uncovered_capabilities": summary.get("operator_uncovered_capabilities") or performance.get("uncovered_capabilities", []),
        "target_file_count": summary.get("target_file_count") or performance.get("target_file_count"),
        "operator_ids": summary.get("operator_ids", []),
        "candidate_frontier_count": len(operator_planning.get("candidate_frontier", [])) if isinstance(operator_planning.get("candidate_frontier"), list) else 0,
        "policy_model_version": policy_model.get("policy_version"),
        "policy_model_source": policy_model.get("source"),
        "policy_model_status": policy_model.get("status", "loaded" if policy_model.get("source") == "trained_task1_orchestration_policy_model" else None),
        "policy_model_confidence": policy_model.get("confidence"),
        "policy_model_workflow_type": policy_model.get("workflow_type"),
    }


def _task1_v3_ready(markers: dict[str, Any]) -> bool:
    target_file_count = markers.get("target_file_count") or 0
    post_quality_status = str(markers.get("post_execution_quality_status") or "").lower()
    return bool(
        markers.get("status") == "completed"
        and markers.get("cleaning_task_status") == "COMPLETED"
        and markers.get("task_understanding_version") == "task_semantics_v3_constraint_graph"
        and markers.get("semantic_constraint_graph_version") == "task_semantic_constraint_graph_v3"
        and markers.get("autonomous_planner") == "autonomous_task_planner_v3_semantic_constraint_dag"
        and markers.get("operator_planner") == "operator_planner_v3_multi_objective_beam_search"
        and int(target_file_count) > 0
        and post_quality_status in {"passed", "pass", "ok", "success", "completed", ""}
    )


def _find_task2_result_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if (
            payload.get("agent") == "MedicalKGAgent"
            and isinstance(payload.get("kg_validation_report"), dict)
            and isinstance(payload.get("kg_operator_profile"), dict)
        ):
            return payload
        for value in payload.values():
            found = _find_task2_result_payload(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_task2_result_payload(item)
            if found:
                return found
    elif isinstance(payload, str):
        for item in _extract_json_objects_from_text(payload):
            found = _find_task2_result_payload(item)
            if found:
                return found
    return {}


def _task2_kg_markers_from_payload(payload: Any, output_root: Path | None = None) -> dict[str, Any]:
    result = _find_task2_result_payload(payload)
    validation = result.get("kg_validation_report", {}) if isinstance(result.get("kg_validation_report"), dict) else {}
    operator_profile = result.get("kg_operator_profile", {}) if isinstance(result.get("kg_operator_profile"), dict) else {}
    quality_eval = result.get("kg_quality_eval", {}) if isinstance(result.get("kg_quality_eval"), dict) else {}
    if output_root is not None:
        if not validation:
            validation = _read_json(output_root / "graph" / "kg_validation_report.json")
        if not operator_profile:
            operator_profile = _read_json(output_root / "graph" / "kg_operator_profile.json")
        if not quality_eval:
            quality_eval = _read_json(output_root / "graph" / "kg_quality_eval.json")
    operators = operator_profile.get("operators", []) if isinstance(operator_profile.get("operators"), list) else []
    operator_names = [str(item.get("name", "")) for item in operators if isinstance(item, dict)]
    artifact_paths = result.get("artifact_paths", {}) if isinstance(result.get("artifact_paths"), dict) else {}
    return {
        "status": result.get("status"),
        "agent": result.get("agent"),
        "documents": result.get("documents"),
        "entities": result.get("entities"),
        "triples": result.get("triples"),
        "qa_count": result.get("qa_count"),
        "accepted_triples": validation.get("accepted_triples"),
        "rejected_triples": validation.get("rejected_triples"),
        "duplicate_triples": validation.get("duplicate_triples"),
        "conflict_filtered_triples": validation.get("conflict_filtered_triples"),
        "operator_count": len(operators) or validation.get("operator_profile", {}).get("operator_count"),
        "operator_names": operator_names,
        "gold_quality_enabled": bool(quality_eval.get("enabled")),
        "gold_quality_f1": (quality_eval.get("overall") or {}).get("f1") if isinstance(quality_eval.get("overall"), dict) else None,
        "has_entities_artifact": bool(artifact_paths.get("entities")) or bool(output_root and (output_root / "graph" / "entities.jsonl").exists()),
        "has_triples_artifact": bool(artifact_paths.get("triples")) or bool(output_root and (output_root / "graph" / "triples.jsonl").exists()),
        "has_qa_trace_artifact": bool(artifact_paths.get("qa_trace")) or bool(output_root and (output_root / "demo" / "qa_trace.jsonl").exists()),
    }


def _task2_kg_ready(markers: dict[str, Any]) -> bool:
    required_operators = {
        "entity_extract",
        "relation_extract",
        "triple_normalize",
        "triple_validate_arbitrate",
        "entity_link",
    }
    operator_names = {str(item) for item in markers.get("operator_names", [])}
    return bool(
        markers.get("status") == "completed"
        and markers.get("agent") == "MedicalKGAgent"
        and int(markers.get("documents") or 0) > 0
        and int(markers.get("entities") or 0) > 0
        and int(markers.get("triples") or 0) > 0
        and int(markers.get("accepted_triples") or 0) > 0
        and required_operators.issubset(operator_names)
        and markers.get("has_entities_artifact")
        and markers.get("has_triples_artifact")
        and markers.get("has_qa_trace_artifact")
    )


def _task2_agent_run_summary_markers(text: str) -> dict[str, Any]:
    lowered = text.lower()

    def has_any(*values: str) -> bool:
        return any(value in text or value.lower() in lowered for value in values)

    return {
        "tool_called": (
            TASK2_BUILD_OPERATION_ID in text
            or "modelengine_task2_medical_kg_agent_build_medical_kg_agent" in text
        ),
        "final_passed": has_any('"phase": "final_pass"', '"phase":"final_pass"', "最终自检通过"),
        "kg_build_answered": has_any(
            "医疗知识图谱构建结果",
            "医疗知识图谱",
            "Medical KG",
            "Task-2 MedicalKGAgent",
            "MedicalKGAgent",
        ),
        "entity_count_reported": has_any("4,468", "4468", "实体数", "总实体数"),
        "triple_count_reported": has_any("3,403", "3403", "最终采纳三元组", "三元组"),
        "accepted_triples_reported": has_any("3,403", "3403", "accepted_triples", "最终采纳三元组"),
        "operator_summary_reported": has_any(
            "实体抽取",
            "关系抽取",
            "三元组校验",
            "operator_names",
            "entity_extract",
            "relation_extract",
            "triple_validate_arbitrate",
        ),
        "quality_reported": has_any("97.24%", "0.9724", "F1", "Precision", "Recall"),
        "artifact_paths_reported": has_any("产物路径", "documents.jsonl", "combined_triples.jsonl", "cmeie_documents.jsonl"),
        "npu_not_claimed": not has_any("npu 优化完成", "完成 npu", "npu optimized", "npu acceleration enabled"),
        "no_runtime_error": not has_any("Run Agent Error", "Missing credentials", "工具调用失败", "无法执行工具"),
    }


def _task2_agent_run_summary_ready(markers: dict[str, Any]) -> bool:
    return bool(
        markers.get("tool_called")
        and markers.get("final_passed")
        and markers.get("kg_build_answered")
        and markers.get("entity_count_reported")
        and markers.get("triple_count_reported")
        and markers.get("accepted_triples_reported")
        and markers.get("operator_summary_reported")
        and markers.get("quality_reported")
        and markers.get("artifact_paths_reported")
        and markers.get("npu_not_claimed")
        and markers.get("no_runtime_error")
    )


def _task1_agent_run_summary_markers(text: str) -> dict[str, Any]:
    lowered = text.lower()

    def has_any(*values: str) -> bool:
        return any(value in text or value.lower() in lowered for value in values)

    capability_coverage_preserved = has_any(
        "capability_coverage_ratio = 1.0",
        "capability_coverage_ratio=1.0",
        "capability_coverage_ratio: 1.0",
        "capability_coverage_ratio：1.0",
        '"capability_coverage_ratio": 1.0',
        '"capability_coverage_ratio":1.0',
        "'capability_coverage_ratio': 1.0",
        "'capability_coverage_ratio':1.0",
        "算子覆盖全部",
        "operator 覆盖全部",
        "覆盖全部要求能力",
        "全部 7 项必需能力",
        "全部7项必需能力",
        "能力覆盖率 100%",
        "能力覆盖率：100%",
        "能力全覆盖",
    ) or (
        has_any("capability_coverage_ratio")
        and has_any("1.0", "100%")
        and has_any("无能力缺口", "uncovered_capabilities = []", '"uncovered_capabilities": []')
    )
    final_passed = has_any('"phase": "final_pass"', '"phase":"final_pass"', "最终自检通过")
    final_format_only_failure = has_any(
        '"phase": "final_fail"',
        '"phase":"final_fail"',
        '"phase": "blocked"',
        '"phase":"blocked"',
    ) and has_any("no_unresolved_raw_tags", "internal execution/display tags")
    hard_runtime_error = has_any(
        "Run Agent Error",
        "Missing credentials",
        "Reached max steps",
        "无法执行工具",
    )
    unresolved_tool_error = has_any("工具调用失败", "tool execution failed") and not has_any(
        "并非工具调用失败",
        "非工具调用失败",
        "不是工具调用失败",
        "工具已成功执行",
    )

    return {
        "tool_called": TASK1_OPENAPI_OPERATION_ID in text or "modelengine_task1_data_processing_agent_run_datamate_processing_agent" in text,
        "status_completed": has_any('"status":"completed"', '"status": "completed"', "status = \"completed\"", "status: completed", "状态为 `\"status\":\"completed\"`"),
        "cleaning_task_completed": has_any(
            '"cleaning_task_status":"COMPLETED"',
            '"cleaning_task_status": "COMPLETED"',
            "cleaning_task_status = \"COMPLETED\"",
            "cleaning_task_status=COMPLETED",
            "cleaning_task_status: COMPLETED",
            "cleaning_task_status: \"COMPLETED\"",
            "`cleaning_task_status: \"COMPLETED\"`",
            "`cleaning_task_status` | `COMPLETED`",
            "`cleaning_task_status`|`COMPLETED`",
            "cleaning_task_status` 为 `\"COMPLETED\"`",
            "状态**：`COMPLETED`",
            "状态：`COMPLETED`",
            "状态: `COMPLETED`",
        ),
        "datamate_status_reported": has_any(
            "DataMate 状态",
            "DataMate 清洗工作流",
            "DataMate 执行状态",
            "DataMate 清洗任务",
            "DataMate 清洗任务终态成功",
            "成功创建并完成 DataMate 清洗任务",
            "cleaning_task_status",
        ) and "COMPLETED" in text,
        "quality_summary_reported": has_any("质量摘要", "structured_etl_contract_status", "post_execution_quality_status", "全部 `passed`", "全部 passed"),
        "planning_summary_reported": has_any("处理计划关键证据", "8 阶段", "DAG", "operator 覆盖", "capability_coverage_ratio"),
        "capability_coverage_preserved": capability_coverage_preserved,
        "final_passed": final_passed,
        "final_format_only_selfcheck_failed": final_format_only_failure,
        "no_runtime_error": final_passed or not (hard_runtime_error or unresolved_tool_error),
    }


def _task1_agent_run_summary_ready(markers: dict[str, Any]) -> bool:
    return bool(
        markers.get("tool_called")
        and markers.get("status_completed")
        and markers.get("cleaning_task_completed")
        and markers.get("datamate_status_reported")
        and markers.get("quality_summary_reported")
        and markers.get("planning_summary_reported")
        and markers.get("capability_coverage_preserved")
        and markers.get("no_runtime_error")
    )


def _find_task3_result_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if (
            payload.get("agent") == "AnalysisAgent"
            and (
                isinstance(payload.get("analysis_summary"), dict)
                or isinstance(payload.get("nl2sql_plan"), dict)
            )
        ):
            return payload
        if isinstance(payload.get("analysis_summary"), dict) and isinstance(payload.get("nl2sql_eval"), dict):
            return payload
        for value in payload.values():
            found = _find_task3_result_payload(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_task3_result_payload(item)
            if found:
                return found
    elif isinstance(payload, str):
        for item in _extract_json_objects_from_text(payload):
            found = _find_task3_result_payload(item)
            if found:
                return found
    return {}


def _task3_analysis_markers_from_payload(payload: Any, output_root: Path | None = None) -> dict[str, Any]:
    result = _find_task3_result_payload(payload)
    analysis_summary = result.get("analysis_summary", {}) if isinstance(result.get("analysis_summary"), dict) else {}
    nl2sql_eval = result.get("nl2sql_eval", {}) if isinstance(result.get("nl2sql_eval"), dict) else {}
    lineage = result.get("task3_lineage", {}) if isinstance(result.get("task3_lineage"), dict) else {}
    graph_understanding = result.get("graph_understanding", {}) if isinstance(result.get("graph_understanding"), dict) else {}
    visualization_manifest = result.get("visualization_manifest", {}) if isinstance(result.get("visualization_manifest"), dict) else {}
    closed_loop = result.get("task3_closed_loop_status", {}) if isinstance(result.get("task3_closed_loop_status"), dict) else {}
    artifact_paths = result.get("artifact_paths", {}) if isinstance(result.get("artifact_paths"), dict) else {}
    if output_root is not None:
        if not analysis_summary:
            analysis_summary = _read_json(output_root / "demo" / "analysis_summary.json")
        if not nl2sql_eval:
            nl2sql_eval = _read_json(output_root / "demo" / "nl2sql_eval_report.json")
        if not lineage:
            lineage = _read_json(output_root / "demo" / "task3_lineage.json")
        if not graph_understanding:
            graph_understanding = _read_json(output_root / "demo" / "graph_understanding.json")
        if not visualization_manifest:
            visualization_manifest = _read_json(output_root / "demo" / "visualization_manifest.json")
        if not closed_loop:
            closed_loop = _read_json(output_root / "demo" / "task3_closed_loop_status.json")

    nl2sql_plan = result.get("nl2sql_plan", {}) if isinstance(result.get("nl2sql_plan"), dict) else {}
    sql_validation = nl2sql_plan.get("sql_validation", {}) if isinstance(nl2sql_plan.get("sql_validation"), dict) else {}
    query = result.get("query", {}) if isinstance(result.get("query"), dict) else {}
    supported_modes = graph_understanding.get("supported_analysis_modes", [])
    output_root_ready = output_root is not None
    return {
        "status": result.get("status"),
        "agent": result.get("agent"),
        "query_count": result.get("query_count") or analysis_summary.get("query_count"),
        "subgraph_count": result.get("subgraph_count") or analysis_summary.get("subgraph_count"),
        "nl2sql_accuracy": nl2sql_eval.get("accuracy") or analysis_summary.get("nl2sql_accuracy"),
        "nl2sql_total": nl2sql_eval.get("total"),
        "nl2sql_meets_threshold": nl2sql_eval.get("meets_threshold") if "meets_threshold" in nl2sql_eval else analysis_summary.get("nl2sql_meets_threshold"),
        "task3_lineage_ready": result.get("task3_lineage_ready") if "task3_lineage_ready" in result else lineage.get("ready"),
        "task3_closed_loop_ready": result.get("task3_closed_loop_ready") if "task3_closed_loop_ready" in result else closed_loop.get("ready"),
        "bi_chart_count": visualization_manifest.get("bi_chart_count") or analysis_summary.get("bi_chart_count"),
        "graph_visual_count": visualization_manifest.get("graph_visual_count") or analysis_summary.get("graph_visual_count"),
        "graph_understanding_version": graph_understanding.get("understanding_version") or analysis_summary.get("graph_understanding_version"),
        "supported_analysis_mode_count": len(supported_modes) if isinstance(supported_modes, list) else 0,
        "interactive_question": result.get("question"),
        "interactive_query_id": query.get("query_id"),
        "interactive_row_count": result.get("row_count"),
        "interactive_sql_valid": sql_validation.get("valid"),
        "interactive_chart_type": (result.get("chart") or {}).get("chart_type") if isinstance(result.get("chart"), dict) else None,
        "has_analysis_report": bool(artifact_paths.get("analysis_report")) or bool(output_root_ready and (output_root / "demo" / "analysis_report.md").exists()),
        "has_analysis_queries": bool(artifact_paths.get("analysis_queries")) or bool(output_root_ready and (output_root / "demo" / "analysis_queries.jsonl").exists()),
        "has_task3_lineage": bool(artifact_paths.get("task3_lineage")) or bool(output_root_ready and (output_root / "demo" / "task3_lineage.json").exists()),
        "has_graph_understanding": bool(artifact_paths.get("graph_understanding")) or bool(output_root_ready and (output_root / "demo" / "graph_understanding.json").exists()),
        "has_visualization_manifest": bool(artifact_paths.get("visualization_manifest")) or bool(output_root_ready and (output_root / "demo" / "visualization_manifest.json").exists()),
        "has_closed_loop_status": bool(artifact_paths.get("task3_closed_loop_status")) or bool(output_root_ready and (output_root / "demo" / "task3_closed_loop_status.json").exists()),
        "has_interactive_trace": bool(artifact_paths.get("interactive_analysis_trace")) or bool(output_root_ready and (output_root / "demo" / "interactive_analysis_trace.jsonl").exists()),
    }


def _task3_analysis_ready(markers: dict[str, Any]) -> bool:
    accuracy = float(markers.get("nl2sql_accuracy") or 0.0)
    return bool(
        markers.get("status") in {"ok", "completed", None}
        and int(markers.get("query_count") or 0) >= 8
        and int(markers.get("subgraph_count") or 0) >= 1
        and accuracy >= 0.85
        and markers.get("nl2sql_meets_threshold") in {True, None}
        and markers.get("task3_lineage_ready") is True
        and markers.get("task3_closed_loop_ready") is True
        and int(markers.get("bi_chart_count") or 0) >= 3
        and int(markers.get("graph_visual_count") or 0) >= 1
        and markers.get("graph_understanding_version") == "task3_graph_understanding_v1"
        and int(markers.get("supported_analysis_mode_count") or 0) >= 4
        and markers.get("has_analysis_report")
        and markers.get("has_analysis_queries")
        and markers.get("has_task3_lineage")
        and markers.get("has_graph_understanding")
        and markers.get("has_visualization_manifest")
        and markers.get("has_closed_loop_status")
    )


def _task3_interactive_ready(markers: dict[str, Any]) -> bool:
    return bool(
        markers.get("status") in {"ok", "completed", None}
        and markers.get("agent") in {"AnalysisAgent", None}
        and markers.get("interactive_question")
        and markers.get("interactive_query_id")
        and int(markers.get("interactive_row_count") or 0) >= 1
        and markers.get("interactive_sql_valid") is True
        and markers.get("interactive_chart_type")
        and markers.get("has_interactive_trace")
    )


def _task3_agent_run_summary_markers(text: str) -> dict[str, Any]:
    lowered = text.lower()

    def has_any(*values: str) -> bool:
        return any(value in text or value.lower() in lowered for value in values)

    return {
        "tool_called": (
            TASK3_INSIGHT_OPERATION_ID in text
            or TASK3_ANALYZE_OPERATION_ID in text
            or "modelengine_task3_analysis_agent_run_interactive_analysis" in text
            or "modelengine_task3_analysis_agent_run_analysis_agent" in text
        ),
        "final_passed": has_any('"phase": "final_pass"', '"phase":"final_pass"', "最终自检通过"),
        "analysis_answered": has_any("AnalysisAgent", "数据分析", "图谱分析", "NL2SQL", "BI", "可视化"),
        "query_id_reported": has_any("disease_complication_counts", "source_document_contribution", "query_id"),
        "row_count_reported": has_any("row_count", "结果行数", "行数"),
        "nl2sql_reported": has_any("sql_valid", "sql_validation", "NL2SQL", "准确率", "85%"),
        "chart_reported": has_any("chart_type", "bar", "图表", "BI"),
        "closed_loop_reported": has_any("task3_closed_loop_ready", "闭环", "数据 → 知识 → 洞察", "数据到知识到洞察"),
        "no_runtime_error": not has_any("Run Agent Error", "Missing credentials", "无法执行工具", "工具调用失败"),
    }


def _task3_agent_run_summary_ready(markers: dict[str, Any]) -> bool:
    return bool(
        markers.get("tool_called")
        and markers.get("analysis_answered")
        and markers.get("query_id_reported")
        and markers.get("row_count_reported")
        and markers.get("nl2sql_reported")
        and markers.get("chart_reported")
        and markers.get("closed_loop_reported")
        and markers.get("no_runtime_error")
    )


def _extract_first_text(payload: Any, keys: tuple[str, ...]) -> str | None:
    payload = _unwrap_response_payload(payload)
    if isinstance(payload, str):
        stripped = payload.strip()
        return stripped or None
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, str):
        stripped = data.strip()
        return stripped or None
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_object_id(payload: Any, keys: tuple[str, ...] = ("id",)) -> str | None:
    payload = _unwrap_response_payload(payload)
    if isinstance(payload, str):
        stripped = payload.strip()
        return stripped or None
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, str):
        stripped = data.strip()
        return stripped or None
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _http_call(
    *,
    name: str,
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 1.5,
    preview_chars: int = 2000,
) -> dict[str, Any]:
    request_headers = dict(headers or {})
    data = raw_body if raw_body is not None else _json_body(body)
    if data is not None:
        request_headers.setdefault("Content-Type", "application/json")
    req = request.Request(url=url, data=data, headers=request_headers, method=method.upper())

    started_at = _now()
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read(200_000).decode("utf-8", errors="replace")
            payload = _decode_json(raw)
            return {
                "name": name,
                "method": method.upper(),
                "url": url,
                "started_at": started_at,
                "status_code": response.status,
                "reachable": True,
                "ok": 200 <= response.status < 300,
                "content_type": response.headers.get("Content-Type", ""),
                "body_json": _redact_payload(payload),
                "body_preview": raw[:preview_chars] if payload is None else "",
                "error": "",
            }
    except error.HTTPError as exc:
        raw = exc.read(200_000).decode("utf-8", errors="replace")
        payload = _decode_json(raw)
        return {
            "name": name,
            "method": method.upper(),
            "url": url,
            "started_at": started_at,
            "status_code": exc.code,
            "reachable": True,
            "ok": 200 <= exc.code < 300,
            "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
            "body_json": _redact_payload(payload),
            "body_preview": raw[:preview_chars] if payload is None else "",
            "error": str(exc),
        }
    except (TimeoutError, socket.timeout) as exc:
        return _http_error_payload(name, method, url, started_at, "timeout", exc)
    except error.URLError as exc:
        return _http_error_payload(name, method, url, started_at, "url_error", exc.reason)
    except OSError as exc:
        return _http_error_payload(name, method, url, started_at, "os_error", exc)


def _http_error_payload(name: str, method: str, url: str, started_at: str, error_type: str, exc: Any) -> dict[str, Any]:
    return {
        "name": name,
        "method": method.upper(),
        "url": url,
        "started_at": started_at,
        "status_code": None,
        "reachable": False,
        "ok": False,
        "content_type": "",
        "body_json": None,
        "body_preview": "",
        "error": f"{error_type}: {exc}",
    }


def _extract_datamate_kbs(response: dict[str, Any]) -> list[dict[str, Any]]:
    payload = response.get("body_json")
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("content"), list):
        return [item for item in data["content"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(payload.get("content"), list):
        return [item for item in payload["content"] if isinstance(item, dict)]
    return []


def _collect_config_snapshot(output_root: Path, env_files: list[dict[str, str]]) -> dict[str, Any]:
    upstream_root = output_root.parent
    nexent_env = _load_env_file(upstream_root / "upstream" / "nexent" / "docker" / ".env")
    selected_names = [
        "OPENAI_API_BASE",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "MEDICAL_DEMO_USE_LLM",
        "DATAMATE_URL",
        "DATAMATE_API_BASE",
        "DATAMATE_RUNTIME_API",
        "NEXENT_URL",
        "NEXENT_CONFIG_API",
        "NEXENT_RUNTIME_API",
        "NEXENT_MCP_MANAGEMENT_API",
        "MODELENGINE_DEMO_API_URL",
        "MODELENGINE_DEMO_API_NEXENT_URL",
        "MODEL_ENGINE_ENABLED",
        "ENABLE_NPU",
    ]
    merged: dict[str, str] = {}
    for env_file in env_files:
        merged.update(env_file)
    merged.update({key: value for key, value in os.environ.items() if key in selected_names})

    return {
        "project_env": {name: _redact_value(name, merged.get(name)) for name in selected_names if merged.get(name) is not None},
        "nexent_env": {
            name: _redact_value(name, nexent_env.get(name))
            for name in [
                "MODEL_ENGINE_ENABLED",
                "DATAMATE_URL",
                "MINIO_API_PORT",
                "MINIO_CONSOLE_PORT",
                "ELASTICSEARCH_HTTP_PORT",
                "ELASTICSEARCH_TRANSPORT_PORT",
            ]
            if nexent_env.get(name) is not None
        },
    }


def _candidate_model_pairs(embedding_model: str | None, chat_model: str | None) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    chat_candidates = [chat_model] if chat_model else []
    chat_candidates.extend(["gpt-5.4", "gpt-4o-mini", "gpt-4o", "gpt-4"])
    embedding_candidates = [embedding_model] if embedding_model else []
    embedding_candidates.extend(DEFAULT_DATAMATE_EMBEDDING_CANDIDATES)

    for emb in embedding_candidates:
        if not emb:
            continue
        for chat in chat_candidates:
            if not chat:
                continue
            pair = (str(emb).strip(), str(chat).strip())
            if pair[0] and pair[1] and pair not in candidates:
                candidates.append(pair)
    return candidates


def _collect_models_api_evidence(
    urls: dict[str, str],
    *,
    model_api_base: str,
    embedding_model: str | None,
    chat_model: str | None,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    calls: list[dict[str, Any]] = []
    selected: dict[str, str] = {}

    for base_url in _base_url_variants(urls["datamate_api_base"]):
        list_call = _http_call(
            name=f"DataMate model list via {base_url}",
            method="GET",
            url=_join_url(base_url, "/api/models/list?page=0&size=2000"),
            timeout_seconds=timeout_seconds,
        )
        calls.append(list_call)
        model_items = _extract_payload_items(list_call.get("body_json"))
        if not model_items:
            continue

        embedding_matches: list[dict[str, Any]] = []
        chat_matches: list[dict[str, Any]] = []
        for item in model_items:
            model_type = str(item.get("type", "")).upper()
            model_name = str(item.get("modelName") or item.get("model_name") or "").strip()
            if not model_name:
                continue
            if model_type == "EMBEDDING":
                embedding_matches.append(item)
            elif model_type == "CHAT":
                chat_matches.append(item)

        chosen_embedding = None
        chosen_chat = None
        for candidate_embedding, candidate_chat in _candidate_model_pairs(embedding_model, chat_model):
            for item in embedding_matches:
                item_base_url = str(item.get("baseUrl") or item.get("base_url") or "").strip()
                if str(item.get("modelName") or item.get("model_name") or "").strip() == candidate_embedding and (
                    not item_base_url
                    or _base_url_matches(item_base_url, model_api_base)
                ):
                    chosen_embedding = item
                    break
            if chosen_embedding is not None:
                for item in chat_matches:
                    item_base_url = str(item.get("baseUrl") or item.get("base_url") or "").strip()
                    if str(item.get("modelName") or item.get("model_name") or "").strip() == candidate_chat and (
                        not item_base_url
                        or _base_url_matches(item_base_url, model_api_base)
                    ):
                        chosen_chat = item
                        break
            if chosen_embedding and chosen_chat:
                break

        if chosen_embedding is None and embedding_matches:
            chosen_embedding = embedding_matches[0]
        if chosen_chat is None and chat_matches:
            chosen_chat = chat_matches[0]

        if chosen_embedding:
            selected["embedding_model_id"] = str(chosen_embedding.get("id", "")).strip()
            selected["embedding_model_name"] = str(chosen_embedding.get("modelName") or chosen_embedding.get("model_name") or "").strip()
        if chosen_chat:
            selected["chat_model_id"] = str(chosen_chat.get("id", "")).strip()
            selected["chat_model_name"] = str(chosen_chat.get("modelName") or chosen_chat.get("model_name") or "").strip()

        if selected.get("embedding_model_id") and selected.get("chat_model_id"):
            break

    if not selected.get("embedding_model_name"):
        selected["embedding_model_name"] = embedding_model or DEFAULT_DATAMATE_EMBEDDING_CANDIDATES[0]
    if not selected.get("chat_model_name"):
        selected["chat_model_name"] = chat_model or "gpt-5.4"
    return calls, selected


def _ensure_model_exists(
    base_url: str,
    *,
    model_base_url: str,
    model_name: str,
    provider: str,
    model_type: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str | None]:
    create_body = {
        "modelName": model_name,
        "provider": provider,
        "baseUrl": model_base_url,
        "apiKey": api_key,
        "type": model_type,
    }
    create_call = _http_call(
        name=f"DataMate create {model_type.lower()} model {model_name}",
        method="POST",
        url=_join_url(base_url, "/api/models/create"),
        body=create_body,
        timeout_seconds=timeout_seconds,
    )
    if create_call.get("ok"):
        model_id = _extract_object_id(create_call.get("body_json"), ("id",))
        if not model_id:
            model_id = _extract_object_id(create_call.get("body_json"), ("data", "id"))
        return create_call, model_id

    existing_list = _http_call(
        name=f"DataMate list {model_type.lower()} models after create failure {model_name}",
        method="GET",
        url=_join_url(base_url, "/api/models/list?page=0&size=2000"),
        timeout_seconds=timeout_seconds,
    )
    payload_items = _extract_payload_items(existing_list.get("body_json"))
    for item in payload_items:
        if str(item.get("modelName") or item.get("model_name") or "").strip() != model_name:
            continue
        if str(item.get("type", "")).upper() != model_type.upper():
            continue
        model_id = str(item.get("id", "")).strip() or None
        if model_id:
            return create_call, model_id
    return create_call, None


def _ensure_datamate_models(
    urls: dict[str, str],
    env_files: list[dict[str, str]],
    *,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    calls: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    api_key = _env_value("OPENAI_API_KEY", "", env_files)
    model_api_base = _env_value("DATAMATE_MODEL_API_BASE", DEFAULT_MODEL_API_BASE, env_files)
    embedding_model = _env_value("EMBEDDING_MODEL", "", env_files).strip()
    llm_model = _env_value("LLM_MODEL", "gpt-5.4", env_files).strip()
    provider = DEFAULT_DATAMATE_PROVIDER

    model_list_calls, model_selection = _collect_models_api_evidence(
        urls,
        model_api_base=model_api_base,
        embedding_model=embedding_model,
        chat_model=llm_model,
        timeout_seconds=timeout_seconds,
    )
    calls.extend(model_list_calls)
    selected.update(model_selection)

    model_base_candidates = _base_url_variants(model_api_base)

    for base_url in _base_url_variants(urls["datamate_api_base"]):
        if not selected.get("embedding_model_id"):
            for candidate_model_base in model_base_candidates:
                create_call, model_id = _ensure_model_exists(
                    base_url,
                    model_base_url=candidate_model_base,
                    model_name=selected.get("embedding_model_name") or embedding_model or DEFAULT_DATAMATE_EMBEDDING_CANDIDATES[0],
                    provider=provider,
                    model_type="EMBEDDING",
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
                calls.append(create_call)
                if model_id:
                    selected["embedding_model_id"] = model_id
                    selected["embedding_model_base_url"] = candidate_model_base
                    break
        if not selected.get("chat_model_id"):
            for candidate_model_base in model_base_candidates:
                create_call, model_id = _ensure_model_exists(
                    base_url,
                    model_base_url=candidate_model_base,
                    model_name=selected.get("chat_model_name") or llm_model,
                    provider=provider,
                    model_type="CHAT",
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
                calls.append(create_call)
                if model_id:
                    selected["chat_model_id"] = model_id
                    selected["chat_model_base_url"] = candidate_model_base
                    break
        if selected.get("embedding_model_id") and selected.get("chat_model_id"):
            break

    return calls, selected


def _ensure_datamate_dataset(
    urls: dict[str, str],
    *,
    dataset_name: str,
    description: str,
    dataset_type: str,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    dataset_info: dict[str, Any] = {}
    for base_url in _base_url_variants(urls["datamate_api_base"]):
        create_call = _http_call(
            name=f"DataMate create dataset {dataset_name}",
            method="POST",
            url=_join_url(base_url, "/api/data-management/datasets"),
            body={
                "name": dataset_name,
                "description": description,
                "datasetType": dataset_type,
                "status": "DRAFT",
            },
            timeout_seconds=timeout_seconds,
        )
        calls.append(create_call)
        if create_call.get("ok"):
            dataset_id = _extract_object_id(create_call.get("body_json"), ("id",))
            if dataset_id:
                dataset_info = {"id": dataset_id, "name": dataset_name, "base_url": base_url}
                return calls, dataset_info

        list_call = _http_call(
            name=f"DataMate list dataset {dataset_name}",
            method="GET",
            url=_join_url(base_url, f"/api/data-management/datasets?size=100&page=0"),
            timeout_seconds=timeout_seconds,
        )
        calls.append(list_call)
        datasets = _extract_payload_items(list_call.get("body_json"))
        for item in datasets:
            if str(item.get("name", "")).strip() == dataset_name:
                dataset_id = str(item.get("id", "")).strip()
                if dataset_id:
                    dataset_info = {"id": dataset_id, "name": dataset_name, "base_url": base_url}
                    return calls, dataset_info
    return calls, dataset_info


def _upload_file_to_datamate_dataset(
    *,
    base_url: str,
    dataset_id: str,
    file_path: Path,
    prefix: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    file_bytes = file_path.read_bytes()
    total_size = len(file_bytes)
    upload_name = f"DataMate upload {file_path.name}"

    pre_upload_call = _http_call(
        name=f"{upload_name} pre-upload",
        method="POST",
        url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}/files/upload/pre-upload"),
        body={
            "hasArchive": False,
            "totalFileNum": 1,
            "totalSize": total_size,
            "prefix": prefix,
        },
        timeout_seconds=timeout_seconds,
    )
    calls.append(pre_upload_call)
    req_id = _extract_first_text(pre_upload_call.get("body_json"), ("data", "id"))
    if not req_id:
        req_id = _extract_first_text(pre_upload_call.get("body_json"), ("id",))
    if not req_id:
        return calls

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    form_body, content_type_header = _build_multipart_form_data(
        {
            "reqId": req_id,
            "fileNo": 0,
            "fileName": file_path.name,
            "totalChunkNum": 1,
            "chunkNo": 1,
            "checkSumHex": hashlib.sha256(file_bytes).hexdigest(),
        },
        [
            {
                "field_name": "file",
                "filename": file_path.name,
                "content": file_bytes,
                "content_type": content_type,
            }
        ],
    )
    upload_call = _http_call(
        name=f"{upload_name} chunk-upload",
        method="POST",
        url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}/files/upload/chunk"),
        headers={"Content-Type": content_type_header},
        raw_body=form_body,
        timeout_seconds=timeout_seconds * 2,
    )
    upload_call["body_preview"] = f"multipart({len(form_body)} bytes)"
    calls.append(upload_call)
    return calls


def _create_datamate_knowledge_base(
    urls: dict[str, str],
    *,
    knowledge_base_name: str,
    description: str,
    embedding_model_id: str,
    chat_model_id: str | None,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    kb_info: dict[str, Any] = {}
    for base_url in _base_url_variants(urls["datamate_api_base"]):
        create_call = _http_call(
            name=f"DataMate create knowledge-base {knowledge_base_name}",
            method="POST",
            url=_join_url(base_url, "/api/knowledge-base/create"),
            body={
                "name": knowledge_base_name,
                "description": description,
                "type": DEFAULT_DATAMATE_KB_TYPE,
                "embeddingModel": embedding_model_id,
                "chatModel": chat_model_id,
            },
            timeout_seconds=timeout_seconds,
        )
        calls.append(create_call)
        kb_id = _extract_object_id(create_call.get("body_json"), ("data", "id"))
        if not kb_id:
            kb_id = _extract_object_id(create_call.get("body_json"), ("id",))
        if kb_id:
            kb_info = {"id": kb_id, "name": knowledge_base_name, "base_url": base_url}
            return calls, kb_info

        list_call = _http_call(
            name=f"DataMate list knowledge-base {knowledge_base_name}",
            method="POST",
            url=_join_url(base_url, "/api/knowledge-base/list"),
            body={"page": 1, "size": 100, "name": knowledge_base_name, "type": DEFAULT_DATAMATE_KB_TYPE},
            timeout_seconds=timeout_seconds,
        )
        calls.append(list_call)
        payload = _unwrap_response_payload(list_call.get("body_json"))
        candidates: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                candidates = _extract_payload_items(data)
            else:
                candidates = _extract_payload_items(payload)
        else:
            candidates = _extract_payload_items(payload)
        for item in candidates:
            if str(item.get("name", "")).strip() == knowledge_base_name:
                kb_id = str(item.get("id", "")).strip()
                if kb_id:
                    kb_info = {"id": kb_id, "name": knowledge_base_name, "base_url": base_url}
                    return calls, kb_info
    return calls, kb_info


def _attach_files_to_knowledge_base(
    *,
    base_url: str,
    knowledge_base_id: str,
    dataset_file_ids: list[dict[str, str]],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    attach_call = _http_call(
        name="DataMate attach files to knowledge base",
        method="POST",
        url=_join_url(base_url, f"/api/knowledge-base/{knowledge_base_id}/files"),
        body={
            "knowledgeBaseId": knowledge_base_id,
            "processType": DEFAULT_DATAMATE_PROCESS_TYPE,
            "chunkSize": 500,
            "overlapSize": 50,
            "files": dataset_file_ids,
        },
        timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 1),
    )
    calls.append(attach_call)
    return calls


def _datamate_kb_supported_files(file_ids: list[dict[str, str]]) -> list[dict[str, str]]:
    supported: list[dict[str, str]] = []
    for item in file_ids:
        suffix = Path(str(item.get("fileName", ""))).suffix.lower()
        if suffix in DATAMATE_KB_SUPPORTED_SUFFIXES:
            supported.append(item)
    return supported


def _wait_for_knowledge_base_files(
    *,
    base_url: str,
    knowledge_base_id: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(10.0, min(90.0, timeout_seconds * 8))
    attempt = 0
    while True:
        attempt += 1
        call = _http_call(
            name=f"DataMate wait knowledge-base files attempt {attempt}",
            method="GET",
            url=_join_url(base_url, f"/api/knowledge-base/{knowledge_base_id}/files?page=1&size=100"),
            timeout_seconds=timeout_seconds,
        )
        calls.append(call)
        payload = _unwrap_response_payload(call.get("body_json"))
        items = _extract_payload_items(payload)
        statuses = {str(item.get("status", "")).upper() for item in items}
        active_items = [item for item in items if str(item.get("status", "")).upper() in {"COMPLETED", "ACTIVE", "SUCCESS"}]
        processing = statuses & {"PROCESSING", "PENDING", "RUNNING"}
        if active_items or (items and not processing):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(min(3.0, max(0.5, timeout_seconds / 2)))
    return calls


def _collect_dataset_file_ids(
    *,
    base_url: str,
    dataset_id: str,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    calls: list[dict[str, Any]] = []
    file_ids: list[dict[str, str]] = []
    page = 0
    while True:
        call = _http_call(
            name=f"DataMate list dataset files page {page}",
            method="GET",
            url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}/files?page={page}&size=100"),
            timeout_seconds=timeout_seconds,
        )
        calls.append(call)
        payload = _unwrap_response_payload(call.get("body_json"))
        files: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                content = data.get("content")
                if isinstance(content, list):
                    files = [item for item in content if isinstance(item, dict)]
            elif isinstance(data, list):
                files = [item for item in data if isinstance(item, dict)]
            else:
                content = payload.get("content")
                if isinstance(content, list):
                    files = [item for item in content if isinstance(item, dict)]
        if not files:
            break
        for item in files:
            file_id = str(item.get("id", "")).strip()
            file_name = str(item.get("fileName") or item.get("file_name") or item.get("originalName") or "").strip()
            if file_id and file_name:
                file_ids.append({"id": file_id, "fileName": file_name})
        if len(files) < 100:
            break
        page += 1
    return calls, file_ids


def _collect_service_probes(
    urls: dict[str, str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    probes = [
        ("DataMate UI", "HEAD", urls["datamate_url"]),
        ("DataMate Gateway", "HEAD", urls["datamate_api_base"]),
        ("DataMate Python Runtime", "HEAD", urls["datamate_runtime_api"]),
        ("Nexent UI", "HEAD", urls["nexent_url"]),
        ("Nexent Config API", "HEAD", urls["nexent_config_api"]),
        ("Nexent Runtime API", "HEAD", urls["nexent_runtime_api"]),
        ("Nexent MCP Management API", "HEAD", urls["nexent_mcp_management_api"]),
        ("ModelEngine Demo API", "HEAD", urls["demo_api_url"]),
    ]
    return [
        _http_call(name=name, method=method, url=url, timeout_seconds=timeout_seconds)
        for name, method, url in probes
    ]


def _collect_datamate_api_evidence(
    urls: dict[str, str],
    *,
    query: str,
    knowledge_base_id: str | None,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    api_calls: list[dict[str, Any]] = []
    datamate_search_trace: list[dict[str, Any]] = []
    candidate_bases = []
    for base_url in [urls["datamate_api_base"], urls["datamate_url"]]:
        if base_url and base_url not in candidate_bases:
            candidate_bases.append(base_url)

    discovered_ids: list[str] = []
    best_base = candidate_bases[0] if candidate_bases else urls["datamate_api_base"]
    for base_url in candidate_bases:
        list_call = _http_call(
            name=f"DataMate knowledge-base list via {base_url}",
            method="POST",
            url=_join_url(base_url, "/api/knowledge-base/list"),
            body={"page": 1, "size": 20},
            timeout_seconds=timeout_seconds,
        )
        api_calls.append(list_call)
        kbs = _extract_datamate_kbs(list_call)
        if kbs:
            best_base = base_url
        for kb in kbs:
            kb_id = kb.get("id")
            kb_type = kb.get("type")
            if kb_id and str(kb_id) not in discovered_ids and (kb_type in (None, "DOCUMENT") or not kb_type):
                discovered_ids.append(str(kb_id))

    selected_ids = [knowledge_base_id] if knowledge_base_id else discovered_ids[:1]
    selected_ids = [item for item in selected_ids if item]

    if selected_ids:
        kb_id = selected_ids[0]
        api_calls.append(
            _http_call(
                name="DataMate knowledge-base detail",
                method="GET",
                url=_join_url(best_base, f"/api/knowledge-base/{kb_id}"),
                timeout_seconds=timeout_seconds,
            )
        )
        api_calls.append(
            _http_call(
                name="DataMate knowledge-base files",
                method="GET",
                url=_join_url(best_base, f"/api/knowledge-base/{kb_id}/files?page=1&size=10"),
                timeout_seconds=timeout_seconds,
            )
        )
        retrieve_call: dict[str, Any] | None = None
        deadline = time.monotonic() + max(1.0, min(12.0, timeout_seconds * 6))
        attempt = 0
        while True:
            attempt += 1
            retrieve_call = _http_call(
                name=f"DataMate knowledge-base v2 retrieve attempt {attempt}",
                method="POST",
                url=_join_url(best_base, "/api/knowledge-base/v2/retrieve"),
                body={
                    "query": query,
                    "topK": 5,
                    "threshold": 0.2,
                    "knowledgeBaseIds": selected_ids,
                },
                timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 1),
            )
            api_calls.append(retrieve_call)
            if retrieve_call.get("ok") and _result_count(retrieve_call) > 0:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(min(1.0, max(0.2, timeout_seconds / 2)))
        datamate_search_trace.append(
            _build_datamate_search_trace(
                query=query,
                knowledge_base_ids=selected_ids,
                endpoint=retrieve_call["url"],
                call=retrieve_call,
            )
        )
    else:
        datamate_search_trace.append(
            {
                "agent": "nexent_medical_agent",
                "tool": "datamate_search",
                "status": "skipped",
                "query": query,
                "knowledge_base_ids": [],
                "reason": "No DataMate knowledge base id was provided or discovered.",
                "created_at": _now(),
            }
        )

    return api_calls, selected_ids, datamate_search_trace


def _result_count(call: dict[str, Any]) -> int:
    payload = call.get("body_json")
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict) and isinstance(data.get("content"), list):
            return len(data["content"])
    return 0


def _build_datamate_search_trace(
    *,
    query: str,
    knowledge_base_ids: list[str],
    endpoint: str,
    call: dict[str, Any],
) -> dict[str, Any]:
    result_count = _result_count(call)
    return {
        "agent": "nexent_medical_agent",
        "tool": "datamate_search",
        "status": "success" if call.get("ok") and result_count > 0 else "failed",
        "query": query,
        "knowledge_base_ids": knowledge_base_ids,
        "tool_input": {
            "query": query,
            "index_names": knowledge_base_ids,
            "top_k": 5,
            "threshold": 0.2,
        },
        "backend": {
            "endpoint": endpoint,
            "method": "POST",
            "source": "DataMate /api/knowledge-base/v2/retrieve",
        },
        "response_status": call.get("status_code"),
        "result_count": result_count,
        "error": call.get("error", ""),
        "created_at": _now(),
    }


def _authorization_headers() -> dict[str, str]:
    raw = os.getenv("NEXENT_AUTH_TOKEN") or os.getenv("NEXENT_AUTHORIZATION") or ""
    if not raw:
        return {}
    value = raw if raw.lower().startswith("bearer ") else f"Bearer {raw}"
    return {"Authorization": value}


def _normalize_nexent_models_config(models: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    required = ("llm", "embedding", "multiEmbedding", "rerank", "vlm", "vlm2", "vlm3", "stt", "tts")
    source = models if isinstance(models, dict) else {}
    for key in required:
        item = source.get(key) if isinstance(source.get(key), dict) else {}
        api_config = item.get("apiConfig") if isinstance(item.get("apiConfig"), dict) else {}
        model_config: dict[str, Any] = {
            "modelName": item.get("modelName") or item.get("name") or "",
            "displayName": item.get("displayName") or "",
            "apiConfig": {
                "apiKey": api_config.get("apiKey") or "",
                "modelUrl": api_config.get("modelUrl") or "",
            },
        }
        for optional_key in ("dimension", "modelFactory", "modelAppid", "accessToken"):
            if item.get(optional_key) is not None:
                model_config[optional_key] = item.get(optional_key)
        normalized[key] = model_config
    return normalized


def _collect_nexent_api_evidence(
    urls: dict[str, str],
    *,
    datamate_url: str,
    knowledge_base_ids: list[str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    headers = _authorization_headers()
    config_base = urls["nexent_config_api"]
    calls: list[dict[str, Any]] = []
    load_config = _http_call(
        name="Nexent load_config before DataMate save",
        method="GET",
        url=_join_url(config_base, "/config/load_config"),
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    calls.append(load_config)
    config_payload = load_config.get("body_json") if isinstance(load_config.get("body_json"), dict) else {}
    current_config = config_payload.get("config") if isinstance(config_payload, dict) else {}
    if isinstance(current_config, dict):
        app_config = dict(current_config.get("app") or {})
        icon_config = app_config.get("icon") if isinstance(app_config.get("icon"), dict) else {}
        app_body = {
            "appName": app_config.get("name") or app_config.get("appName") or "Nexent 智能体",
            "appDescription": app_config.get("description") or app_config.get("appDescription") or "",
            "iconType": icon_config.get("type") or app_config.get("iconType") or "preset",
            "iconKey": icon_config.get("iconKey") or app_config.get("iconKey") or "search",
            "customIconUrl": icon_config.get("customUrl") or app_config.get("customIconUrl") or "",
            "avatarUri": icon_config.get("avatarUri") or app_config.get("avatarUri") or "",
            "modelEngineEnabled": True,
            "datamateUrl": datamate_url,
        }
        calls.append(
            _http_call(
                name="Nexent save_config DataMate URL",
                method="POST",
                url=_join_url(config_base, "/config/save_config"),
                body={
                    "app": app_body,
                    "models": _normalize_nexent_models_config(current_config.get("models")),
                },
                headers=headers,
                timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 1),
            )
        )
    calls.extend(
        [
            _http_call(
                name="Nexent DataMate test_connection",
                method="POST",
                url=_join_url(config_base, "/datamate/test_connection"),
                body={"datamate_url": datamate_url},
                headers=headers,
                timeout_seconds=timeout_seconds,
            ),
            _http_call(
                name="Nexent sync_datamate_knowledges",
                method="POST",
                url=_join_url(config_base, "/datamate/sync_datamate_knowledges"),
                body={"datamate_url": datamate_url},
                headers=headers,
                timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 1),
            ),
        ]
    )
    if knowledge_base_ids:
        calls.append(
            _http_call(
                name="Nexent DataMate knowledge-base files",
                method="GET",
                url=_join_url(config_base, f"/datamate/{knowledge_base_ids[0]}/files"),
                headers=headers,
                timeout_seconds=timeout_seconds,
            )
        )
    return calls


def _extract_registered_openapi_service(payload: Any, service_name: str) -> dict[str, Any] | None:
    items = _extract_payload_items(payload)
    for item in items:
        if (
            item.get("service_name") == service_name
            or item.get("mcp_service_name") == service_name
            or item.get("name") == service_name
        ):
            return item
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        for key in ("services", "registered_services", "openapi_services"):
            nested_items = data.get(key)
            if isinstance(nested_items, list):
                for item in nested_items:
                    if isinstance(item, dict) and (
                        item.get("service_name") == service_name
                        or item.get("mcp_service_name") == service_name
                        or item.get("name") == service_name
                    ):
                        return item
    return None


def _extract_datamate_task_ids_from_text(text: str) -> list[str]:
    task_ids: list[str] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return task_ids

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            summary = value.get("summary")
            if isinstance(summary, dict):
                task_id = summary.get("cleaning_task_id")
                if isinstance(task_id, str) and task_id:
                    task_ids.append(task_id)
            cleaning_task = value.get("cleaning_task")
            if isinstance(cleaning_task, dict):
                task_id = cleaning_task.get("id")
                if isinstance(task_id, str) and task_id:
                    task_ids.append(task_id)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return [task_id for index, task_id in enumerate(task_ids) if task_id not in task_ids[:index]]


def _run_nexent_mcp_task1_tool(
    *,
    timeout_seconds: float,
    output_root: Path | None = None,
) -> dict[str, Any]:
    script = r'''
import asyncio
import json
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:5011/sse", timeout=180) as client:
        tools = await client.list_tools()
        tool_names = [getattr(tool, "name", "") for tool in tools]
        result = await client.call_tool(
            "modelengine_task1_data_processing_agent_run_datamate_processing_agent",
            {
                "input_path": "data/raw/medical_demo_structured.csv",
                "input_paths": ["data/raw/medical_demo_notes.txt"],
                "output_root": ".",
                "task": "基于DataMate和Nexent官方链路，对批量医疗结构化表和非结构化文本做ETL、清洗、字段抽取、严格质量校验、去重、异常恢复和性能优化；要求多目标算子调度、任务语义约束规划和结构化ETL质量分数，不允许本地fallback。",
                "wait_seconds": 120,
                "datamate_api_base": "http://127.0.0.1:8080",
            },
            timeout=180,
        )
        texts = [getattr(item, "text", "") for item in getattr(result, "content", []) if getattr(item, "type", "") == "text"]
        payload = {"tool_names": tool_names, "texts": texts, "is_error": bool(getattr(result, "is_error", False))}
        print(json.dumps(payload, ensure_ascii=False))

asyncio.run(main())
'''
    raw = ""
    try:
        with request.urlopen(
            request.Request(
                "http://localhost:5015/tools/openapi_service?tenant_id=tenant_id",
                method="GET",
            ),
            timeout=timeout_seconds,
        ):
            pass
    except Exception:
        pass
    try:
        import subprocess

        completed = subprocess.run(
            [
                "docker",
                "exec",
                "nexent-mcp",
                "sh",
                "-lc",
                f"/opt/backend/.venv/bin/python - <<'PY'\n{script}\nPY",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(180.0, timeout_seconds * 12),
        )
        raw = completed.stdout.strip()
        lines = [line for line in raw.splitlines() if line.strip().startswith("{")]
        payload = json.loads(lines[-1]) if lines else {}
        text_payloads = payload.get("texts") if isinstance(payload.get("texts"), list) else []
        task_ids: list[str] = []
        completed_status = False
        for text in text_payloads:
            if isinstance(text, str):
                task_ids.extend(_extract_datamate_task_ids_from_text(text))
                completed_status = completed_status or '"status":"completed"' in text or '"cleaning_task_status":"COMPLETED"' in text
        tool_names = payload.get("tool_names") if isinstance(payload.get("tool_names"), list) else []
        v3_markers = _task1_v3_markers_from_payload(text_payloads)
        v3_ready = _task1_v3_ready(v3_markers)
        return {
            "name": "Nexent MCP call Task-1 DataProcessingAgent tool",
            "method": "MCP_SSE",
            "url": "http://localhost:5011/sse",
            "started_at": _now(),
            "status_code": completed.returncode,
            "reachable": completed.returncode == 0,
            "ok": completed.returncode == 0 and completed_status and bool(task_ids) and v3_ready and not payload.get("is_error"),
            "content_type": "application/json",
            "body_json": {
                "tool_visible_in_mcp": "modelengine_task1_data_processing_agent_run_datamate_processing_agent" in tool_names,
                "tool_count": len(tool_names),
                "cleaning_task_ids": [task_id for index, task_id in enumerate(task_ids) if task_id not in task_ids[:index]],
                "completed_status_seen": completed_status,
                "task1_v3_ready": v3_ready,
                "task1_v3_markers": v3_markers,
                "is_error": bool(payload.get("is_error")),
            },
            "body_preview": raw[:2000],
            "error": completed.stderr.strip()[:2000] if completed.returncode != 0 else "",
        }
    except Exception as exc:
        return {
            "name": "Nexent MCP call Task-1 DataProcessingAgent tool",
            "method": "MCP_SSE",
            "url": "http://localhost:5011/sse",
            "started_at": _now(),
            "status_code": None,
            "reachable": False,
            "ok": False,
            "content_type": "",
            "body_json": {},
            "body_preview": raw[:2000],
            "error": str(exc),
        }


def _run_nexent_mcp_task2_tool(
    *,
    timeout_seconds: float,
    output_root: Path | None = None,
) -> dict[str, Any]:
    script = r'''
import asyncio
import json
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:5011/sse", timeout=240) as client:
        tools = await client.list_tools()
        tool_names = [getattr(tool, "name", "") for tool in tools]
        result = await client.call_tool(
            "modelengine_task2_medical_kg_agent_build_medical_kg_agent",
            {
                "input_path": "data/raw/medical_demo.jsonl",
                "output_root": ".",
                "questions": ["高血压有哪些并发症？", "2型糖尿病有哪些症状和治疗方式？"],
                "kg_backend": "rule",
                "use_llm": False,
                "external_documents": "data/external/converted/cblue/cmeie_documents.jsonl",
                "external_triples": "data/external/converted/cblue/combined_triples.jsonl",
                "external_entities": "data/external/converted/cblue/cmeee_entities.jsonl",
            },
            timeout=240,
        )
        texts = [getattr(item, "text", "") for item in getattr(result, "content", []) if getattr(item, "type", "") == "text"]
        payload = {"tool_names": tool_names, "texts": texts, "is_error": bool(getattr(result, "is_error", False))}
        print(json.dumps(payload, ensure_ascii=False))

asyncio.run(main())
'''
    raw = ""
    try:
        with request.urlopen(
            request.Request(
                "http://localhost:5015/tools/openapi_service?tenant_id=tenant_id",
                method="GET",
            ),
            timeout=timeout_seconds,
        ):
            pass
    except Exception:
        pass
    try:
        import subprocess

        completed = subprocess.run(
            [
                "docker",
                "exec",
                "nexent-mcp",
                "sh",
                "-lc",
                f"/opt/backend/.venv/bin/python - <<'PY'\n{script}\nPY",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(240.0, timeout_seconds * 16),
        )
        raw = completed.stdout.strip()
        lines = [line for line in raw.splitlines() if line.strip().startswith("{")]
        payload = json.loads(lines[-1]) if lines else {}
        text_payloads = payload.get("texts") if isinstance(payload.get("texts"), list) else []
        tool_names = payload.get("tool_names") if isinstance(payload.get("tool_names"), list) else []
        markers = _task2_kg_markers_from_payload(text_payloads, output_root=output_root)
        ready = _task2_kg_ready(markers)
        return {
            "name": "Nexent MCP call Task-2 MedicalKGAgent tool",
            "method": "MCP_SSE",
            "url": "http://localhost:5011/sse",
            "started_at": _now(),
            "status_code": completed.returncode,
            "reachable": completed.returncode == 0,
            "ok": completed.returncode == 0 and ready and not payload.get("is_error"),
            "content_type": "application/json",
            "body_json": {
                "tool_visible_in_mcp": "modelengine_task2_medical_kg_agent_build_medical_kg_agent" in tool_names,
                "tool_count": len(tool_names),
                "task2_kg_ready": ready,
                "task2_kg_markers": markers,
                "is_error": bool(payload.get("is_error")),
            },
            "body_preview": raw[:2000],
            "error": completed.stderr.strip()[:2000] if completed.returncode != 0 else "",
        }
    except Exception as exc:
        return {
            "name": "Nexent MCP call Task-2 MedicalKGAgent tool",
            "method": "MCP_SSE",
            "url": "http://localhost:5011/sse",
            "started_at": _now(),
            "status_code": None,
            "reachable": False,
            "ok": False,
            "content_type": "",
            "body_json": {},
            "body_preview": raw[:2000],
            "error": str(exc),
        }


def _run_nexent_mcp_task3_tool(
    *,
    timeout_seconds: float,
    output_root: Path | None = None,
) -> dict[str, Any]:
    script = r'''
import asyncio
import json
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:5011/sse", timeout=180) as client:
        tools = await client.list_tools()
        tool_names = [getattr(tool, "name", "") for tool in tools]
        analysis_result = await client.call_tool(
            "modelengine_task3_analysis_agent_run_analysis_agent",
            {"output_root": "."},
            timeout=180,
        )
        insight_result = await client.call_tool(
            "modelengine_task3_analysis_agent_run_interactive_analysis",
            {"output_root": ".", "question": "统计每种疾病的并发症数量"},
            timeout=180,
        )
        analysis_texts = [getattr(item, "text", "") for item in getattr(analysis_result, "content", []) if getattr(item, "type", "") == "text"]
        insight_texts = [getattr(item, "text", "") for item in getattr(insight_result, "content", []) if getattr(item, "type", "") == "text"]
        payload = {
            "tool_names": tool_names,
            "analysis_texts": analysis_texts,
            "insight_texts": insight_texts,
            "analysis_is_error": bool(getattr(analysis_result, "is_error", False)),
            "insight_is_error": bool(getattr(insight_result, "is_error", False)),
        }
        print(json.dumps(payload, ensure_ascii=False))

asyncio.run(main())
'''
    raw = ""
    try:
        with request.urlopen(
            request.Request(
                "http://localhost:5015/tools/openapi_service?tenant_id=tenant_id",
                method="GET",
            ),
            timeout=timeout_seconds,
        ):
            pass
    except Exception:
        pass
    try:
        import subprocess

        completed = subprocess.run(
            [
                "docker",
                "exec",
                "nexent-mcp",
                "sh",
                "-lc",
                f"/opt/backend/.venv/bin/python - <<'PY'\n{script}\nPY",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(240.0, timeout_seconds * 16),
        )
        raw = completed.stdout.strip()
        lines = [line for line in raw.splitlines() if line.strip().startswith("{")]
        payload = json.loads(lines[-1]) if lines else {}
        analysis_texts = payload.get("analysis_texts") if isinstance(payload.get("analysis_texts"), list) else []
        insight_texts = payload.get("insight_texts") if isinstance(payload.get("insight_texts"), list) else []
        tool_names = payload.get("tool_names") if isinstance(payload.get("tool_names"), list) else []
        analysis_markers = _task3_analysis_markers_from_payload(analysis_texts, output_root=output_root)
        insight_markers = _task3_analysis_markers_from_payload(insight_texts, output_root=output_root)
        analysis_ready = _task3_analysis_ready(analysis_markers)
        insight_ready = _task3_interactive_ready(insight_markers)
        return {
            "name": "Nexent MCP call Task-3 AnalysisAgent tools",
            "method": "MCP_SSE",
            "url": "http://localhost:5011/sse",
            "started_at": _now(),
            "status_code": completed.returncode,
            "reachable": completed.returncode == 0,
            "ok": (
                completed.returncode == 0
                and analysis_ready
                and insight_ready
                and not payload.get("analysis_is_error")
                and not payload.get("insight_is_error")
            ),
            "content_type": "application/json",
            "body_json": {
                "analyze_tool_visible_in_mcp": "modelengine_task3_analysis_agent_run_analysis_agent" in tool_names,
                "insight_tool_visible_in_mcp": "modelengine_task3_analysis_agent_run_interactive_analysis" in tool_names,
                "tool_count": len(tool_names),
                "task3_analysis_ready": analysis_ready,
                "task3_analysis_markers": analysis_markers,
                "task3_interactive_ready": insight_ready,
                "task3_interactive_markers": insight_markers,
                "analysis_is_error": bool(payload.get("analysis_is_error")),
                "insight_is_error": bool(payload.get("insight_is_error")),
            },
            "body_preview": raw[:2000],
            "error": completed.stderr.strip()[:2000] if completed.returncode != 0 else "",
        }
    except Exception as exc:
        return {
            "name": "Nexent MCP call Task-3 AnalysisAgent tools",
            "method": "MCP_SSE",
            "url": "http://localhost:5011/sse",
            "started_at": _now(),
            "status_code": None,
            "reachable": False,
            "ok": False,
            "content_type": "",
            "body_json": {},
            "body_preview": raw[:2000],
            "error": str(exc),
        }


def _extract_nexent_model_id(payload: Any, model_name: str, model_type: str = "llm") -> int | None:
    for item in _extract_payload_items(payload):
        if str(item.get("model_type") or "").lower() != model_type.lower():
            continue
        if str(item.get("model_name") or item.get("display_name") or "").strip() != model_name:
            continue
        value = item.get("model_id") or item.get("id")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _collect_task1_nexent_runtime_evidence(
    urls: dict[str, str],
    *,
    output_root: Path,
    integration_dir: Path,
    timeout_seconds: float,
    env_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    headers = _authorization_headers()
    config_base = urls["nexent_config_api"]
    mcp_management_api = urls["nexent_mcp_management_api"]
    demo_api_url = urls["demo_api_url"]
    nexent_server_url = urls["demo_api_nexent_url"]
    env_files = env_files or []
    llm_model = _env_value("LLM_MODEL", "gpt-5.4", env_files).strip() or "gpt-5.4"
    spec = build_task1_openapi_spec(nexent_server_url)
    spec_path = integration_dir / "task1_nexent_openapi.json"
    trace_path = integration_dir / "nexent_task1_tool_trace.jsonl"
    runtime_path = integration_dir / "nexent_task1_runtime_evidence.json"
    _write_json(spec_path, spec)

    registration_body = {
        "service_name": TASK1_OPENAPI_SERVICE_NAME,
        "server_url": nexent_server_url,
        "openapi_json": spec,
        "service_description": spec["info"]["description"],
        "force_update": True,
    }
    calls = [
        _http_call(
            name="Nexent import Task-1 DataProcessingAgent OpenAPI service",
            method="POST",
            url=_join_url(config_base, "/tool/openapi_service"),
            body=registration_body,
            headers=headers,
            timeout_seconds=max(timeout_seconds * 4, timeout_seconds + 5),
        ),
        _http_call(
            name="Nexent list OpenAPI services after Task-1 import",
            method="GET",
            url=_join_url(config_base, "/tool/openapi_services"),
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
        _http_call(
            name="Nexent MCP list registered OpenAPI services",
            method="GET",
            url=_join_url(mcp_management_api, "/tools/openapi_service?tenant_id=tenant_id"),
            timeout_seconds=timeout_seconds,
        ),
        _http_call(
            name="Nexent scan tools after Task-1 import",
            method="GET",
            url=_join_url(config_base, "/tool/scan_tool"),
            headers=headers,
            timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 2),
        ),
        _http_call(
            name="Nexent list tools after Task-1 scan",
            method="GET",
            url=_join_url(config_base, "/tool/list"),
            headers=headers,
            timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 2),
        ),
        _http_call(
            name="Nexent list models before Task-1 agent update",
            method="GET",
            url=_join_url(config_base, "/model/list"),
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
    ]
    import_call = calls[0]
    list_call = calls[1]
    mcp_call = calls[2]
    scan_call = calls[3]
    tool_list_call = calls[4]
    model_list_call = calls[5]
    llm_model_id = _extract_nexent_model_id(model_list_call.get("body_json"), llm_model)
    registered_service = _extract_registered_openapi_service(list_call.get("body_json"), TASK1_OPENAPI_SERVICE_NAME)
    mcp_registered_service = _extract_registered_openapi_service(mcp_call.get("body_json"), TASK1_OPENAPI_SERVICE_NAME)
    scanned_tools = _extract_payload_items(tool_list_call.get("body_json")) or _extract_payload_items(scan_call.get("body_json"))
    scanned_tool = next(
        (
            item for item in scanned_tools
            if item.get("origin_name") == TASK1_OPENAPI_OPERATION_ID
            or item.get("name") == TASK1_OPENAPI_OPERATION_ID
            or TASK1_OPENAPI_OPERATION_ID in str(item.get("name", ""))
            or TASK1_OPENAPI_OPERATION_ID in str(item.get("origin_name", ""))
            or TASK1_OPENAPI_SERVICE_NAME in str(item.get("class_name", ""))
        ),
        None,
    )
    tool_id = None
    if isinstance(scanned_tool, dict):
        for key in ("tool_id", "id"):
            value = scanned_tool.get(key)
            if isinstance(value, int):
                tool_id = value
                break
            if isinstance(value, str) and value.isdigit():
                tool_id = int(value)
                break
    agent_id: int | None = None
    if tool_id is not None:
        task1_tool_callable = "modelengine_task1_data_processing_agent_run_datamate_processing_agent"
        task1_call_example = (
            "<code>modelengine_task1_data_processing_agent_run_datamate_processing_agent("
            "task=\"Run the official DataMate-backed Task-1 medical data ETL, cleaning, extraction, validation, recovery, and performance-aware scheduling workflow\", "
            "input_path=\"data/raw/medical_demo_structured.csv\", "
            "input_paths=[\"data/raw/medical_demo_notes.txt\"], "
            "output_root=\".\", "
            "wait_seconds=120, "
            "datamate_api_base=\"http://127.0.0.1:8080\""
            ")</code>"
        )
        task1_agent_instructions = (
            "你是比赛任务一数据处理智能体，必须基于 Nexent 的 MCP 工具链调用 DataMate-backed "
            "DataProcessingAgent。收到数据处理请求后，第一步必须调用工具 "
            f"`{task1_tool_callable}`，不得调用短名 `run_datamate_processing_agent`，不得输出 JSON 工具声明，"
            "不得只解释计划。Nexent 执行器只会执行 `<code>...</code>` 中的合法 Python，因此工具调用必须严格写成 "
            f"`{task1_tool_callable}(...)` 的 Python 关键字参数函数调用。"
            "\n\n固定调用模板如下：\n"
            f"{task1_call_example}\n\n"
            "工具返回后，再基于返回的 status、summary、plan、trace、api_calls、cleaning_task 和 output_paths "
            "总结任务理解、算子调度、DataMate 执行状态、异常处理和质量结果。"
            "工具一旦返回 status=completed，下一步必须直接给最终答案，不要再次输出工具调用、不要继续规划、"
            "不要声明需要重试。最终答案只能使用简单 Markdown：标题、短句、普通短横线列表；不要使用 Markdown 表格、"
            "JSON、代码块、复杂反引号列表或内部执行标签。不要回显 `<code>`、`</code>`、`<MCP_START>`、"
            "`<MCP_END>` 或原始 Python 调用块。最终答案必须直接列出 DataMate 状态、处理计划关键证据、"
            "质量摘要、输出结果和结论。最终答案还必须保留一个名为“验收字段”的普通 Markdown 小节，"
            "用短横线逐行列出这些原始字段名和值：status、cleaning_task_status、post_execution_quality_status、"
            "structured_etl_contract_status、capability_coverage_ratio、operator_uncovered_capabilities、target_file_count。"
            "不要把这些字段只改写成中文概括；必须保留字段名，便于比赛验收复核。"
        )
        agent_update_call = _http_call(
            name="Nexent create/update Task-1 DataProcessingAgent",
            method="POST",
            url=_join_url(config_base, "/agent/update"),
            body={
                "name": "modelengine_task1_data_processing_agent",
                "display_name": "ModelEngine 任务一数据处理智能体",
                "description": "基于 Nexent OpenAPI/MCP 工具运行本项目任务一 DataProcessingAgent。",
                "business_description": (
                    "接收结构化或非结构化医疗数据路径，完成任务理解、规划、DAG 调度、清洗、质量检查、"
                    "异常恢复和比赛任务一标准产物输出。"
                ),
                "author": "ModelEngineBySoldierBoy",
                "model_name": llm_model,
                "model_id": llm_model_id,
                "business_logic_model_name": llm_model,
                "business_logic_model_id": llm_model_id,
                "max_steps": 3,
                "provide_run_summary": True,
                "instructions": task1_agent_instructions,
                "duty_prompt": (
                    "你是比赛任务一数据处理智能体。必须调用完整 MCP 工具名 "
                    f"{task1_tool_callable} 处理用户给出的数据路径，并基于工具返回的 plan、summary、trace、"
                    "api_calls 和 cleaning_task 回答。"
                ),
                "constraint_prompt": (
                    "不得绕过 Nexent 工具链生成虚假处理结果；没有工具结果时必须说明阻塞原因。"
                    f"首个动作必须是 `<code>` 包裹的 `{task1_tool_callable}(...)` Python 函数调用；"
                    "不要使用 Markdown 代码块、JSON 字典、短工具名或自然语言工具声明。"
                    "工具执行完成后的最终回答不得包含 `<code>`、`</code>`、`<MCP_START>`、"
                    "`<MCP_END>` 或 Python 函数调用原文；只输出面向用户的 Markdown 摘要。"
                    "工具返回 completed 后必须直接最终回答，不得再次进入代码动作。最终 Markdown 必须使用短横线列表，"
                    "不得使用表格或代码块；必须包含“验收字段”小节，并保留 cleaning_task_status、"
                    "capability_coverage_ratio、post_execution_quality_status 等原始字段名。"
                ),
            "few_shots_prompt": (
                    "示例：当用户要求处理 data/raw/medical_demo_notes.txt 时，先输出这一行，不要换行：\n"
                    f"{task1_call_example}"
                ),
                "enabled": True,
                "enabled_tool_ids": [tool_id],
            },
            headers=headers,
            timeout_seconds=max(timeout_seconds * 3, timeout_seconds + 3),
        )
        calls.append(agent_update_call)
        agent_payload = _unwrap_response_payload(agent_update_call.get("body_json"))
        if isinstance(agent_payload, dict):
            value = agent_payload.get("agent_id") or agent_payload.get("id")
            if isinstance(value, int):
                agent_id = value
            elif isinstance(value, str) and value.isdigit():
                agent_id = int(value)
    if agent_id is not None:
        agent_run_call = _http_call(
            name="Nexent run Task-1 DataProcessingAgent",
            method="POST",
            url=_join_url(urls["nexent_runtime_api"], "/agent/run"),
            body={
                "query": (
                    "请通过 Nexent MCP 工具链处理 input_path=data/raw/medical_demo_structured.csv，并同时处理 input_paths=[data/raw/medical_demo_notes.txt]。"
                    "第一步必须逐字输出下面这一行合法 Python 工具调用，不要换行、不要改参数、不要改地址："
                    f"{task1_call_example}"
                    "工具完成后返回处理计划、DataMate状态和质量摘要；最终回答不要包含 `<code>`、`</code>`、"
                    "`<MCP_START>`、`<MCP_END>` 或 Python 调用原文。"
                    "工具返回 completed 后立刻输出最终答案，不要再次执行动作；最终答案只用短横线列表，"
                    "不要使用表格、代码块或 JSON。必须包含“验收字段”小节，逐行保留 status、"
                    "cleaning_task_status、post_execution_quality_status、structured_etl_contract_status、"
                    "capability_coverage_ratio、operator_uncovered_capabilities、target_file_count。"
                ),
                "agent_id": agent_id,
                "conversation_id": None,
                "is_debug": True,
            },
            headers=headers,
            timeout_seconds=max(120.0, timeout_seconds * 12, timeout_seconds + 10),
            preview_chars=80_000,
        )
        calls.append(agent_run_call)
    else:
        agent_run_call = None
    mcp_tool_call = _run_nexent_mcp_task1_tool(timeout_seconds=timeout_seconds, output_root=output_root)
    calls.append(mcp_tool_call)
    imported = bool(import_call.get("ok"))
    listed = bool(registered_service)
    mcp_registered = bool(mcp_registered_service) or (
        bool(mcp_call.get("ok")) and TASK1_OPENAPI_SERVICE_NAME in json.dumps(mcp_call.get("body_json"), ensure_ascii=False)
    )
    tool_visible = bool(scanned_tool)
    agent_runtime_run_ok = False
    agent_runtime_blocker = ""
    if agent_run_call is not None:
        body_text = json.dumps(agent_run_call.get("body_json"), ensure_ascii=False)
        if not agent_run_call.get("body_json"):
            body_text = str(agent_run_call.get("body_preview") or "")
        decoded_body_text = _decode_sse_content(body_text)
        marker_text = body_text + "\n" + decoded_body_text
        agent_run_v3_markers = _task1_v3_markers_from_payload(marker_text)
        agent_run_v3_ready = _task1_v3_ready(agent_run_v3_markers)
        agent_run_summary_markers = _task1_agent_run_summary_markers(marker_text)
        agent_run_summary_ready = _task1_agent_run_summary_ready(agent_run_summary_markers)
        agent_runtime_blocker = marker_text[:2000]
        success_markers = (
            '"status":"completed"',
            '"status": "completed"',
            '"cleaning_task_status":"COMPLETED"',
            '"cleaning_task_status": "COMPLETED"',
            'status:"completed"',
            "status: completed",
            "cleaning_task_status: COMPLETED",
            'cleaning_task_status` 为 `"COMPLETED"`',
            "cleaning_task_id",
        )
        hard_blocker_markers = (
            "Run Agent Error",
            "Missing credentials",
            "无法被用户以任何已知方式触发",
            "不可执行",
        )
        recoverable_blocker_markers = (
            "Reached max steps",
            '"phase": "blocked"',
            '"phase":"blocked"',
            "Rewrite the action as valid Python",
        )
        has_success_marker = any(marker in marker_text for marker in success_markers)
        has_final_pass = bool(agent_run_summary_markers.get("final_passed"))
        has_hard_blocker = any(marker in marker_text for marker in hard_blocker_markers)
        has_unrecovered_blocker = (
            any(marker in marker_text for marker in recoverable_blocker_markers)
            and not has_final_pass
            and not agent_run_summary_ready
        )
        agent_runtime_run_ok = bool(
            agent_run_call.get("ok")
            and (has_success_marker or agent_run_v3_ready or agent_run_summary_ready)
            and not has_hard_blocker
            and not has_unrecovered_blocker
            and TASK1_OPENAPI_OPERATION_ID in marker_text
            and (agent_run_v3_ready or agent_run_summary_ready)
        )
    else:
        agent_run_v3_markers = {}
        agent_run_v3_ready = False
        agent_run_summary_markers = {}
        agent_run_summary_ready = False
    mcp_v3_markers = (mcp_tool_call.get("body_json") or {}).get("task1_v3_markers", {})
    mcp_v3_ready = bool((mcp_tool_call.get("body_json") or {}).get("task1_v3_ready"))
    task1_latest_v3_markers = agent_run_v3_markers if agent_run_v3_ready else mcp_v3_markers
    openapi_mcp_tool_chain_ready = imported and listed and mcp_registered and tool_visible and bool(mcp_tool_call.get("ok"))
    summary = {
        "service_name": TASK1_OPENAPI_SERVICE_NAME,
        "operation_id": TASK1_OPENAPI_OPERATION_ID,
        "demo_api_url": demo_api_url,
        "nexent_server_url": nexent_server_url,
        "openapi_spec": str(spec_path.relative_to(output_root)),
        "registered_via_nexent_openapi_api": imported,
        "listed_by_nexent": listed,
        "registered_in_nexent_mcp": mcp_registered,
        "visible_in_nexent_tool_scan": tool_visible,
        "nexent_agent_tool_id": tool_id,
        "nexent_agent_id": agent_id,
        "nexent_agent_model": llm_model,
        "nexent_agent_model_id": llm_model_id,
        "nexent_runtime_run_attempted": agent_id is not None,
        "nexent_runtime_run_http_ok": bool(agent_run_call and agent_run_call.get("ok")),
        "nexent_runtime_run_ok": agent_runtime_run_ok,
        "nexent_runtime_run_blocker": "" if agent_runtime_run_ok else agent_runtime_blocker,
        "nexent_runtime_run_v3_ready": agent_run_v3_ready,
        "nexent_runtime_run_v3_markers": agent_run_v3_markers,
        "nexent_runtime_run_summary_ready": agent_run_summary_ready,
        "nexent_runtime_run_summary_markers": agent_run_summary_markers,
        "nexent_mcp_tool_call_ok": bool(mcp_tool_call.get("ok")),
        "nexent_mcp_tool_call_v3_ready": mcp_v3_ready,
        "nexent_mcp_tool_call_v3_markers": mcp_v3_markers,
        "nexent_mcp_tool_cleaning_task_ids": (mcp_tool_call.get("body_json") or {}).get("cleaning_task_ids", []),
        "nexent_openapi_mcp_tool_chain_ready": openapi_mcp_tool_chain_ready,
        "nexent_agent_runtime_autonomous_ready": agent_runtime_run_ok,
        "nexent_runtime_ready": openapi_mcp_tool_chain_ready,
        "task1_latest_v3_markers": task1_latest_v3_markers,
        "task1_latest_v3_ready": _task1_v3_ready(task1_latest_v3_markers),
        "registered_service": registered_service or {},
        "mcp_registered_service": mcp_registered_service or {},
        "scanned_tool": scanned_tool or {},
    }
    evidence = {
        "created_at": _now(),
        "goal": "Register the Task-1 DataProcessingAgent as an official Nexent OpenAPI/MCP tool.",
        "summary": summary,
        "calls": calls,
        "artifact_paths": {
            "openapi_spec": str(spec_path),
            "trace": str(trace_path),
            "runtime_evidence": str(runtime_path),
        },
    }
    _write_jsonl(trace_path, calls)
    _write_json(runtime_path, evidence)
    return evidence


def _collect_task2_nexent_runtime_evidence(
    urls: dict[str, str],
    *,
    output_root: Path,
    integration_dir: Path,
    timeout_seconds: float,
    env_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    headers = _authorization_headers()
    config_base = urls["nexent_config_api"]
    mcp_management_api = urls["nexent_mcp_management_api"]
    demo_api_url = urls["demo_api_url"]
    nexent_server_url = urls["demo_api_nexent_url"]
    env_files = env_files or []
    llm_model = _env_value("LLM_MODEL", "gpt-5.4", env_files).strip() or "gpt-5.4"
    spec = build_task2_openapi_spec(nexent_server_url)
    spec_path = integration_dir / "task2_nexent_openapi.json"
    trace_path = integration_dir / "nexent_task2_tool_trace.jsonl"
    runtime_path = integration_dir / "nexent_task2_runtime_evidence.json"
    _write_json(spec_path, spec)

    registration_body = {
        "service_name": TASK2_OPENAPI_SERVICE_NAME,
        "server_url": nexent_server_url,
        "openapi_json": spec,
        "service_description": spec["info"]["description"],
        "force_update": True,
    }
    calls = [
        _http_call(
            name="Nexent import Task-2 MedicalKGAgent OpenAPI service",
            method="POST",
            url=_join_url(config_base, "/tool/openapi_service"),
            body=registration_body,
            headers=headers,
            timeout_seconds=max(timeout_seconds * 4, timeout_seconds + 5),
        ),
        _http_call(
            name="Nexent list OpenAPI services after Task-2 import",
            method="GET",
            url=_join_url(config_base, "/tool/openapi_services"),
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
        _http_call(
            name="Nexent MCP list registered OpenAPI services after Task-2 import",
            method="GET",
            url=_join_url(mcp_management_api, "/tools/openapi_service?tenant_id=tenant_id"),
            timeout_seconds=timeout_seconds,
        ),
        _http_call(
            name="Nexent scan tools after Task-2 import",
            method="GET",
            url=_join_url(config_base, "/tool/scan_tool"),
            headers=headers,
            timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 2),
        ),
        _http_call(
            name="Nexent list tools after Task-2 scan",
            method="GET",
            url=_join_url(config_base, "/tool/list"),
            headers=headers,
            timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 2),
        ),
        _http_call(
            name="Nexent list models before Task-2 agent update",
            method="GET",
            url=_join_url(config_base, "/model/list"),
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
    ]
    import_call, list_call, mcp_call, scan_call, tool_list_call, model_list_call = calls[:6]
    registered_service = _extract_registered_openapi_service(list_call.get("body_json"), TASK2_OPENAPI_SERVICE_NAME)
    mcp_registered_service = _extract_registered_openapi_service(mcp_call.get("body_json"), TASK2_OPENAPI_SERVICE_NAME)
    scanned_tools = _extract_payload_items(tool_list_call.get("body_json")) or _extract_payload_items(scan_call.get("body_json"))

    def find_tool(operation_id: str) -> dict[str, Any] | None:
        return next(
            (
                item for item in scanned_tools
                if item.get("origin_name") == operation_id
                or item.get("name") == operation_id
                or operation_id in str(item.get("name", ""))
                or operation_id in str(item.get("origin_name", ""))
                or (
                    TASK2_OPENAPI_SERVICE_NAME in str(item.get("class_name", ""))
                    and operation_id.replace("_medical_kg_agent", "") in str(item.get("name", ""))
                )
            ),
            None,
        )

    def extract_tool_id(tool: dict[str, Any] | None) -> int | None:
        if not isinstance(tool, dict):
            return None
        for key in ("tool_id", "id"):
            value = tool.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    build_tool = find_tool(TASK2_BUILD_OPERATION_ID)
    ask_tool = find_tool(TASK2_ASK_OPERATION_ID)
    build_tool_id = extract_tool_id(build_tool)
    ask_tool_id = extract_tool_id(ask_tool)
    enabled_tool_ids = [item for item in (build_tool_id, ask_tool_id) if item is not None]
    llm_model_id = _extract_nexent_model_id(model_list_call.get("body_json"), llm_model)
    agent_id: int | None = None
    task2_build_callable = "modelengine_task2_medical_kg_agent_build_medical_kg_agent"
    task2_ask_callable = "modelengine_task2_medical_kg_agent_ask_medical_kg_agent"
    task2_build_example = (
        "<code>modelengine_task2_medical_kg_agent_build_medical_kg_agent(input_path=\"data/raw/medical_demo.jsonl\",output_root=\".\",questions=[],kg_backend=\"rule\",use_llm=False,external_documents=\"data/external/converted/cblue/cmeie_documents.jsonl\",external_triples=\"data/external/converted/cblue/combined_triples.jsonl\",external_entities=\"data/external/converted/cblue/cmeee_entities.jsonl\")</code>"
    )
    if enabled_tool_ids:
        agent_update_call = _http_call(
            name="Nexent create/update Task-2 MedicalKGAgent",
            method="POST",
            url=_join_url(config_base, "/agent/update"),
            body={
                "name": "modelengine_task2_medical_kg_agent",
                "display_name": "ModelEngine 任务二医疗知识图谱智能体",
                "description": "基于 Nexent OpenAPI/MCP 工具运行任务二 MedicalKGAgent。",
                "business_description": (
                    "接收医疗文本或结构化医疗数据路径，自动编排实体识别、关系抽取、三元组生成与校验、"
                    "实体链接、图谱构建和知识图谱问答。"
                ),
                "author": "ModelEngineBySoldierBoy",
                "model_name": llm_model,
                "model_id": llm_model_id,
                "business_logic_model_name": llm_model,
                "business_logic_model_id": llm_model_id,
                "max_steps": 4,
                "provide_run_summary": True,
                "instructions": (
                    "你是比赛任务二医疗知识图谱生成和问答智能体。收到图谱构建请求后，第一步必须调用完整 MCP 工具名 "
                    f"`{task2_build_callable}`，不得调用短名 `{TASK2_BUILD_OPERATION_ID}`，不得只解释计划。"
                    "Nexent 执行器只会执行 `<code>...</code>` 中的合法 Python。"
                    "你的第一条输出必须只包含一行 `<code>...</code>`，在 `<code>` 前后不得输出任何解释、标题、列表或 Markdown。"
                    "为了通过 Python 预检，首步工具调用参数必须使用 ASCII-only 字符串；不要在首步调用里写中文问题。"
                    f"固定构建模板必须逐字使用：{task2_build_example}\n\n"
                    f"收到问答请求且已有图谱产物时，可以调用 `{task2_ask_callable}`，但不得编造三元组。"
                    "工具返回后必须基于返回的 kg_validation_report、kg_operator_profile、kg_quality_eval、trace、"
                    "artifact_paths 和 QA evidence 总结实体抽取、关系抽取、三元组校验、自动编排和图谱问答结果。"
                    "最终答案必须保留验收字段：status、documents、entities、triples、accepted_triples、"
                    "operator_count、operator_names、artifact_paths。"
                ),
                "duty_prompt": (
                    "你是比赛任务二 MedicalKGAgent，必须调用完整 MCP 工具名 "
                    f"{task2_build_callable} 构建医疗知识图谱，并基于工具结果回答。"
                ),
                "constraint_prompt": (
                    "不得绕过 Nexent 工具链生成虚假图谱结果；没有工具结果时必须说明阻塞原因。"
                    f"首个动作必须严格等于这一行：{task2_build_example}"
                    "不要在首个 `<code>` 前输出任何中文或英文解释。"
                    "不要使用 Markdown 代码块、JSON 字典、短工具名、自然语言工具声明、多行代码或中文参数。"
                    "工具返回 completed 后必须直接最终回答，不得再次进入代码动作。"
                    "不得宣称 NPU 优化。"
                ),
                "few_shots_prompt": task2_build_example,
                "enabled": True,
                "enabled_tool_ids": enabled_tool_ids,
            },
            headers=headers,
            timeout_seconds=max(timeout_seconds * 3, timeout_seconds + 3),
        )
        calls.append(agent_update_call)
        agent_payload = _unwrap_response_payload(agent_update_call.get("body_json"))
        if isinstance(agent_payload, dict):
            value = agent_payload.get("agent_id") or agent_payload.get("id")
            if isinstance(value, int):
                agent_id = value
            elif isinstance(value, str) and value.isdigit():
                agent_id = int(value)
    if agent_id is not None:
        agent_run_call = _http_call(
            name="Nexent run Task-2 MedicalKGAgent",
            method="POST",
            url=_join_url(urls["nexent_runtime_api"], "/agent/run"),
            body={
                "query": (
                    "Build Task-2 medical KG through Nexent MCP. FIRST RESPONSE MUST BE EXACTLY THIS ONE LINE, "
                    f"WITH NO TEXT BEFORE OR AFTER IT: {task2_build_example} "
                    "After the tool returns completed, answer in Chinese and include status, documents, entities, "
                    "triples, accepted_triples, operator_names, gold_quality_f1, artifact_paths. Do not claim NPU."
                ),
                "agent_id": agent_id,
                "conversation_id": None,
                "is_debug": True,
            },
            headers=headers,
            timeout_seconds=max(timeout_seconds * 10, timeout_seconds + 10),
            preview_chars=80_000,
        )
        calls.append(agent_run_call)
    else:
        agent_run_call = None

    mcp_tool_call = _run_nexent_mcp_task2_tool(timeout_seconds=timeout_seconds, output_root=output_root)
    calls.append(mcp_tool_call)
    imported = bool(import_call.get("ok"))
    listed = bool(registered_service)
    mcp_registered = bool(mcp_registered_service) or (
        bool(mcp_call.get("ok")) and TASK2_OPENAPI_SERVICE_NAME in json.dumps(mcp_call.get("body_json"), ensure_ascii=False)
    )
    tool_visible = bool(build_tool)
    agent_runtime_run_ok = False
    agent_runtime_blocker = ""
    if agent_run_call is not None:
        body_text = json.dumps(agent_run_call.get("body_json"), ensure_ascii=False)
        if not agent_run_call.get("body_json"):
            body_text = str(agent_run_call.get("body_preview") or "")
        decoded_body_text = _decode_sse_content(body_text)
        marker_text = body_text + "\n" + decoded_body_text
        agent_markers = _task2_kg_markers_from_payload(marker_text, output_root=output_root)
        agent_ready = _task2_kg_ready(agent_markers)
        agent_summary_markers = _task2_agent_run_summary_markers(marker_text)
        agent_summary_ready = _task2_agent_run_summary_ready(agent_summary_markers)
        agent_runtime_blocker = marker_text[:2000]
        hard_blocker_markers = ("Run Agent Error", "Missing credentials", "不可执行", "无法执行工具")
        agent_runtime_run_ok = bool(
            agent_run_call.get("ok")
            and (agent_ready or agent_summary_ready)
            and TASK2_BUILD_OPERATION_ID in marker_text
            and not any(marker in marker_text for marker in hard_blocker_markers)
        )
    else:
        agent_markers = {}
        agent_ready = False
        agent_summary_markers = {}
        agent_summary_ready = False
    mcp_markers = (mcp_tool_call.get("body_json") or {}).get("task2_kg_markers", {})
    mcp_ready = bool((mcp_tool_call.get("body_json") or {}).get("task2_kg_ready"))
    latest_markers = agent_markers if agent_ready else mcp_markers
    openapi_mcp_tool_chain_ready = imported and listed and mcp_registered and tool_visible and bool(mcp_tool_call.get("ok"))
    summary = {
        "service_name": TASK2_OPENAPI_SERVICE_NAME,
        "build_operation_id": TASK2_BUILD_OPERATION_ID,
        "ask_operation_id": TASK2_ASK_OPERATION_ID,
        "demo_api_url": demo_api_url,
        "nexent_server_url": nexent_server_url,
        "openapi_spec": str(spec_path.relative_to(output_root)),
        "registered_via_nexent_openapi_api": imported,
        "listed_by_nexent": listed,
        "registered_in_nexent_mcp": mcp_registered,
        "visible_in_nexent_tool_scan": tool_visible,
        "nexent_agent_build_tool_id": build_tool_id,
        "nexent_agent_ask_tool_id": ask_tool_id,
        "nexent_agent_id": agent_id,
        "nexent_agent_model": llm_model,
        "nexent_agent_model_id": llm_model_id,
        "nexent_runtime_run_attempted": agent_id is not None,
        "nexent_runtime_run_http_ok": bool(agent_run_call and agent_run_call.get("ok")),
        "nexent_runtime_run_ok": agent_runtime_run_ok,
        "nexent_runtime_run_blocker": "" if agent_runtime_run_ok else agent_runtime_blocker,
        "nexent_runtime_run_task2_ready": agent_ready,
        "nexent_runtime_run_task2_markers": agent_markers,
        "nexent_runtime_run_summary_ready": agent_summary_ready,
        "nexent_runtime_run_summary_markers": agent_summary_markers,
        "nexent_mcp_tool_call_ok": bool(mcp_tool_call.get("ok")),
        "nexent_mcp_tool_call_task2_ready": mcp_ready,
        "nexent_mcp_tool_call_task2_markers": mcp_markers,
        "nexent_openapi_mcp_tool_chain_ready": openapi_mcp_tool_chain_ready,
        "nexent_agent_runtime_autonomous_ready": agent_runtime_run_ok,
        "nexent_runtime_ready": openapi_mcp_tool_chain_ready,
        "task2_latest_kg_markers": latest_markers,
        "task2_latest_kg_ready": _task2_kg_ready(latest_markers),
        "registered_service": registered_service or {},
        "mcp_registered_service": mcp_registered_service or {},
        "scanned_build_tool": build_tool or {},
        "scanned_ask_tool": ask_tool or {},
    }
    evidence = {
        "created_at": _now(),
        "goal": "Register the Task-2 MedicalKGAgent as an official Nexent OpenAPI/MCP tool.",
        "summary": summary,
        "calls": calls,
        "artifact_paths": {
            "openapi_spec": str(spec_path),
            "trace": str(trace_path),
            "runtime_evidence": str(runtime_path),
        },
    }
    _write_jsonl(trace_path, calls)
    _write_json(runtime_path, evidence)
    return evidence


def _collect_task3_nexent_runtime_evidence(
    urls: dict[str, str],
    *,
    output_root: Path,
    integration_dir: Path,
    timeout_seconds: float,
    env_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    headers = _authorization_headers()
    config_base = urls["nexent_config_api"]
    mcp_management_api = urls["nexent_mcp_management_api"]
    demo_api_url = urls["demo_api_url"]
    nexent_server_url = urls["demo_api_nexent_url"]
    env_files = env_files or []
    llm_model = _env_value("LLM_MODEL", "gpt-5.4", env_files).strip() or "gpt-5.4"
    spec = build_task3_openapi_spec(nexent_server_url)
    spec_path = integration_dir / "task3_nexent_openapi.json"
    trace_path = integration_dir / "nexent_task3_tool_trace.jsonl"
    runtime_path = integration_dir / "nexent_task3_runtime_evidence.json"
    _write_json(spec_path, spec)

    registration_body = {
        "service_name": TASK3_OPENAPI_SERVICE_NAME,
        "server_url": nexent_server_url,
        "openapi_json": spec,
        "service_description": spec["info"]["description"],
        "force_update": True,
    }
    calls = [
        _http_call(
            name="Nexent import Task-3 AnalysisAgent OpenAPI service",
            method="POST",
            url=_join_url(config_base, "/tool/openapi_service"),
            body=registration_body,
            headers=headers,
            timeout_seconds=max(timeout_seconds * 4, timeout_seconds + 5),
        ),
        _http_call(
            name="Nexent list OpenAPI services after Task-3 import",
            method="GET",
            url=_join_url(config_base, "/tool/openapi_services"),
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
        _http_call(
            name="Nexent MCP list registered OpenAPI services after Task-3 import",
            method="GET",
            url=_join_url(mcp_management_api, "/tools/openapi_service?tenant_id=tenant_id"),
            timeout_seconds=timeout_seconds,
        ),
        _http_call(
            name="Nexent scan tools after Task-3 import",
            method="GET",
            url=_join_url(config_base, "/tool/scan_tool"),
            headers=headers,
            timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 2),
        ),
        _http_call(
            name="Nexent list tools after Task-3 scan",
            method="GET",
            url=_join_url(config_base, "/tool/list"),
            headers=headers,
            timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 2),
        ),
        _http_call(
            name="Nexent list models before Task-3 agent update",
            method="GET",
            url=_join_url(config_base, "/model/list"),
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
    ]
    import_call, list_call, mcp_call, scan_call, tool_list_call, model_list_call = calls[:6]
    registered_service = _extract_registered_openapi_service(list_call.get("body_json"), TASK3_OPENAPI_SERVICE_NAME)
    mcp_registered_service = _extract_registered_openapi_service(mcp_call.get("body_json"), TASK3_OPENAPI_SERVICE_NAME)
    scanned_tools = _extract_payload_items(tool_list_call.get("body_json")) or _extract_payload_items(scan_call.get("body_json"))

    def find_tool(operation_id: str) -> dict[str, Any] | None:
        return next(
            (
                item for item in scanned_tools
                if item.get("origin_name") == operation_id
                or item.get("name") == operation_id
                or operation_id in str(item.get("name", ""))
                or operation_id in str(item.get("origin_name", ""))
                or (
                    TASK3_OPENAPI_SERVICE_NAME in str(item.get("class_name", ""))
                    and operation_id.replace("_analysis_agent", "") in str(item.get("name", ""))
                )
            ),
            None,
        )

    def extract_tool_id(tool: dict[str, Any] | None) -> int | None:
        if not isinstance(tool, dict):
            return None
        for key in ("tool_id", "id"):
            value = tool.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    analyze_tool = find_tool(TASK3_ANALYZE_OPERATION_ID)
    insight_tool = find_tool(TASK3_INSIGHT_OPERATION_ID)
    config_tool = find_tool(TASK3_CONFIG_OPERATION_ID)
    analyze_tool_id = extract_tool_id(analyze_tool)
    insight_tool_id = extract_tool_id(insight_tool)
    config_tool_id = extract_tool_id(config_tool)
    enabled_tool_ids = [item for item in (analyze_tool_id, insight_tool_id, config_tool_id) if item is not None]
    llm_model_id = _extract_nexent_model_id(model_list_call.get("body_json"), llm_model)
    agent_id: int | None = None
    task3_analyze_callable = "modelengine_task3_analysis_agent_run_analysis_agent"
    task3_insight_callable = "modelengine_task3_analysis_agent_run_interactive_analysis"
    task3_config_callable = "modelengine_task3_analysis_agent_run_configured_analysis"
    task3_call_example = (
        "<code>modelengine_task3_analysis_agent_run_interactive_analysis("
        "output_root=\".\",question=\"统计每种疾病的并发症数量\""
        ")</code>"
    )
    if enabled_tool_ids:
        agent_update_call = _http_call(
            name="Nexent create/update Task-3 AnalysisAgent",
            method="POST",
            url=_join_url(config_base, "/agent/update"),
            body={
                "name": "modelengine_task3_analysis_agent",
                "display_name": "ModelEngine 任务三图谱驱动数据分析智能体",
                "description": "基于 Nexent OpenAPI/MCP 工具运行任务三 AnalysisAgent。",
                "business_description": (
                    "复用任务一处理数据和任务二医疗知识图谱，自动理解图谱结构，"
                    "面向自然语言分析需求执行 NL2SQL、统计分析、关联分析、图谱算法、BI 图表和图文洞察。"
                ),
                "author": "ModelEngineBySoldierBoy",
                "model_name": llm_model,
                "model_id": llm_model_id,
                "business_logic_model_name": llm_model,
                "business_logic_model_id": llm_model_id,
                "max_steps": 4,
                "provide_run_summary": True,
                "instructions": (
                    "你是比赛任务三数据分析智能体，必须基于 Nexent MCP 工具链调用 AnalysisAgent。"
                    "收到自然语言分析请求后，第一步必须调用完整 MCP 工具名 "
                    f"`{task3_insight_callable}`，不得调用短名 `{TASK3_INSIGHT_OPERATION_ID}`，不得只解释计划。"
                    "Nexent 执行器只会执行 `<code>...</code>` 中的合法 Python。"
                    "首条输出必须只包含一行 `<code>...</code>`，不要输出标题、解释、Markdown 代码块或 JSON 工具声明。"
                    f"固定自然语言分析模板：{task3_call_example}\n\n"
                    f"需要完整闭环验收时，可以调用 `{task3_analyze_callable}(output_root=\".\")`；"
                    f"需要配置化分析时，可以调用 `{task3_config_callable}`。"
                    "工具返回后必须基于返回的 query、row_count、nl2sql_plan、chart、graph_context、"
                    "insight_plan、insights、artifact_paths，以及完整分析产物中的 analysis_summary、"
                    "graph_understanding、visualization_manifest、task3_closed_loop_status 总结结果。"
                    "最终答案必须保留验收字段：status、question、query_id、row_count、sql_valid、"
                    "chart_type、nl2sql_accuracy、task3_closed_loop_ready、bi_chart_count、graph_visual_count、artifact_paths。"
                ),
                "duty_prompt": (
                    "你是比赛任务三 AnalysisAgent，必须调用完整 MCP 工具名 "
                    f"{task3_insight_callable} 执行自然语言图谱分析，并基于工具返回的 NL2SQL、图表、图谱上下文和洞察回答。"
                ),
                "constraint_prompt": (
                    "不得绕过 Nexent 工具链生成虚假分析结果；没有工具结果时必须说明阻塞原因。"
                    f"首个动作必须严格等于这一行：{task3_call_example}"
                    "不要使用 Markdown 代码块、JSON 字典、短工具名、自然语言工具声明或多行代码。"
                    "工具返回后必须直接最终回答，不得再次进入代码动作。"
                    "最终答案不得声称 NPU 优化，不得编造 NL2SQL 准确率；必须引用工具返回和产物路径。"
                ),
                "few_shots_prompt": task3_call_example,
                "enabled": True,
                "enabled_tool_ids": enabled_tool_ids,
            },
            headers=headers,
            timeout_seconds=max(timeout_seconds * 3, timeout_seconds + 3),
        )
        calls.append(agent_update_call)
        agent_payload = _unwrap_response_payload(agent_update_call.get("body_json"))
        if isinstance(agent_payload, dict):
            value = agent_payload.get("agent_id") or agent_payload.get("id")
            if isinstance(value, int):
                agent_id = value
            elif isinstance(value, str) and value.isdigit():
                agent_id = int(value)
    if agent_id is not None:
        agent_run_call = _http_call(
            name="Nexent run Task-3 AnalysisAgent",
            method="POST",
            url=_join_url(urls["nexent_runtime_api"], "/agent/run"),
            body={
                "query": (
                    "请通过 Nexent MCP 工具链完成任务三自然语言图谱分析。"
                    "第一步必须逐字输出下面这一行合法 Python 工具调用，不要换行、不要改参数："
                    f"{task3_call_example}"
                    "工具返回后用中文总结分析结果，必须包含 status、question、query_id、row_count、sql_valid、"
                    "chart_type、nl2sql_accuracy、task3_closed_loop_ready、bi_chart_count、graph_visual_count、artifact_paths。"
                    "不要编造结果，不要声称 NPU。"
                ),
                "agent_id": agent_id,
                "conversation_id": None,
                "is_debug": True,
            },
            headers=headers,
            timeout_seconds=max(timeout_seconds * 8, timeout_seconds + 10),
            preview_chars=80_000,
        )
        calls.append(agent_run_call)
    else:
        agent_run_call = None

    mcp_tool_call = _run_nexent_mcp_task3_tool(timeout_seconds=timeout_seconds, output_root=output_root)
    calls.append(mcp_tool_call)
    imported = bool(import_call.get("ok"))
    listed = bool(registered_service)
    mcp_registered = bool(mcp_registered_service) or (
        bool(mcp_call.get("ok")) and TASK3_OPENAPI_SERVICE_NAME in json.dumps(mcp_call.get("body_json"), ensure_ascii=False)
    )
    tool_visible = bool(analyze_tool and insight_tool)
    agent_runtime_run_ok = False
    agent_runtime_blocker = ""
    if agent_run_call is not None:
        body_text = json.dumps(agent_run_call.get("body_json"), ensure_ascii=False)
        if not agent_run_call.get("body_json"):
            body_text = str(agent_run_call.get("body_preview") or "")
        decoded_body_text = _decode_sse_content(body_text)
        marker_text = body_text + "\n" + decoded_body_text
        agent_analysis_markers = _task3_analysis_markers_from_payload(marker_text, output_root=output_root)
        agent_analysis_ready = _task3_analysis_ready(agent_analysis_markers)
        agent_interactive_ready = _task3_interactive_ready(agent_analysis_markers)
        agent_summary_markers = _task3_agent_run_summary_markers(marker_text)
        agent_summary_ready = _task3_agent_run_summary_ready(agent_summary_markers)
        agent_runtime_blocker = marker_text[:2000]
        hard_blocker_markers = ("Run Agent Error", "Missing credentials", "不可执行", "无法执行工具")
        agent_runtime_run_ok = bool(
            agent_run_call.get("ok")
            and TASK3_INSIGHT_OPERATION_ID in marker_text
            and (agent_interactive_ready or agent_summary_ready)
            and not any(marker in marker_text for marker in hard_blocker_markers)
        )
    else:
        agent_analysis_markers = {}
        agent_analysis_ready = False
        agent_interactive_ready = False
        agent_summary_markers = {}
        agent_summary_ready = False
    mcp_analysis_markers = (mcp_tool_call.get("body_json") or {}).get("task3_analysis_markers", {})
    mcp_analysis_ready = bool((mcp_tool_call.get("body_json") or {}).get("task3_analysis_ready"))
    mcp_interactive_markers = (mcp_tool_call.get("body_json") or {}).get("task3_interactive_markers", {})
    mcp_interactive_ready = bool((mcp_tool_call.get("body_json") or {}).get("task3_interactive_ready"))
    latest_markers = agent_analysis_markers if (agent_analysis_ready or agent_interactive_ready) else mcp_analysis_markers
    openapi_mcp_tool_chain_ready = imported and listed and mcp_registered and tool_visible and bool(mcp_tool_call.get("ok"))
    summary = {
        "service_name": TASK3_OPENAPI_SERVICE_NAME,
        "analyze_operation_id": TASK3_ANALYZE_OPERATION_ID,
        "insight_operation_id": TASK3_INSIGHT_OPERATION_ID,
        "config_operation_id": TASK3_CONFIG_OPERATION_ID,
        "demo_api_url": demo_api_url,
        "nexent_server_url": nexent_server_url,
        "openapi_spec": str(spec_path.relative_to(output_root)),
        "registered_via_nexent_openapi_api": imported,
        "listed_by_nexent": listed,
        "registered_in_nexent_mcp": mcp_registered,
        "visible_in_nexent_tool_scan": tool_visible,
        "nexent_agent_analyze_tool_id": analyze_tool_id,
        "nexent_agent_insight_tool_id": insight_tool_id,
        "nexent_agent_config_tool_id": config_tool_id,
        "nexent_agent_id": agent_id,
        "nexent_agent_model": llm_model,
        "nexent_agent_model_id": llm_model_id,
        "nexent_runtime_run_attempted": agent_id is not None,
        "nexent_runtime_run_http_ok": bool(agent_run_call and agent_run_call.get("ok")),
        "nexent_runtime_run_ok": agent_runtime_run_ok,
        "nexent_runtime_run_blocker": "" if agent_runtime_run_ok else agent_runtime_blocker,
        "nexent_runtime_run_task3_analysis_ready": agent_analysis_ready,
        "nexent_runtime_run_task3_interactive_ready": agent_interactive_ready,
        "nexent_runtime_run_task3_markers": agent_analysis_markers,
        "nexent_runtime_run_summary_ready": agent_summary_ready,
        "nexent_runtime_run_summary_markers": agent_summary_markers,
        "nexent_mcp_tool_call_ok": bool(mcp_tool_call.get("ok")),
        "nexent_mcp_tool_call_task3_analysis_ready": mcp_analysis_ready,
        "nexent_mcp_tool_call_task3_analysis_markers": mcp_analysis_markers,
        "nexent_mcp_tool_call_task3_interactive_ready": mcp_interactive_ready,
        "nexent_mcp_tool_call_task3_interactive_markers": mcp_interactive_markers,
        "nexent_openapi_mcp_tool_chain_ready": openapi_mcp_tool_chain_ready,
        "nexent_agent_runtime_autonomous_ready": agent_runtime_run_ok,
        "nexent_runtime_ready": openapi_mcp_tool_chain_ready,
        "task3_latest_analysis_markers": latest_markers,
        "task3_latest_analysis_ready": _task3_analysis_ready(latest_markers),
        "task3_latest_interactive_ready": agent_interactive_ready or mcp_interactive_ready,
        "registered_service": registered_service or {},
        "mcp_registered_service": mcp_registered_service or {},
        "scanned_analyze_tool": analyze_tool or {},
        "scanned_insight_tool": insight_tool or {},
        "scanned_config_tool": config_tool or {},
    }
    evidence = {
        "created_at": _now(),
        "goal": "Register the Task-3 AnalysisAgent as an official Nexent OpenAPI/MCP tool.",
        "summary": summary,
        "calls": calls,
        "artifact_paths": {
            "openapi_spec": str(spec_path),
            "trace": str(trace_path),
            "runtime_evidence": str(runtime_path),
        },
    }
    _write_jsonl(trace_path, calls)
    _write_json(runtime_path, evidence)
    return evidence


def _build_datamate_import_package(output_root: Path, integration_dir: Path) -> dict[str, Any]:
    triples = _read_jsonl(output_root / "graph" / "triples.jsonl")
    entities = _read_jsonl(output_root / "graph" / "entities.jsonl")
    processed_documents = _read_jsonl(output_root / "data" / "processed" / "documents.jsonl")
    graph_snapshot = _read_json(output_root / "graph" / "graph_snapshot.json")
    validation_report = _read_json(output_root / "graph" / "kg_validation_report.json")

    import_dir = integration_dir / "datamate_import"
    markdown_path = import_dir / "medical_kg_knowledge.md"
    jsonl_path = import_dir / "medical_kg_documents.jsonl"
    manifest_path = import_dir / "manifest.json"

    rows = _build_import_rows(triples)
    _write_jsonl(jsonl_path, rows)
    _write_text(markdown_path, _render_import_markdown(triples, entities, processed_documents, validation_report))

    manifest = {
        "name": "medical_kg_datamate_import",
        "created_at": _now(),
        "purpose": "Upload these files into a DataMate DOCUMENT knowledge base, then sync the knowledge base into Nexent and enable datamate_search.",
        "source_artifacts": {
            "processed_documents": "data/processed/documents.jsonl",
            "entities": "graph/entities.jsonl",
            "triples": "graph/triples.jsonl",
            "graph_snapshot": "graph/graph_snapshot.json",
            "validation_report": "graph/kg_validation_report.json",
        },
        "counts": {
            "documents": len(processed_documents),
            "entities": len(entities),
            "triples": len(triples),
            "graph_nodes": len(graph_snapshot.get("nodes", [])) if isinstance(graph_snapshot, dict) else 0,
            "graph_edges": len(graph_snapshot.get("edges", [])) if isinstance(graph_snapshot, dict) else 0,
            "import_rows": len(rows),
        },
        "files": {
            "markdown": str(markdown_path.relative_to(output_root)),
            "jsonl": str(jsonl_path.relative_to(output_root)),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "paths": {
            "markdown": str(markdown_path),
            "jsonl": str(jsonl_path),
            "manifest": str(manifest_path),
        },
    }


def _build_import_rows(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for triple in triples:
        rows.append(
            {
                "id": f"kg-{triple.get('triple_id', '')}",
                "title": f"{triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}",
                "content": (
                    f"{triple.get('subject', '')} 与 {triple.get('object', '')} 的关系是 "
                    f"{triple.get('predicate', '')}。证据：{triple.get('evidence', '')}"
                ),
                "metadata": {
                    "subject": triple.get("subject", ""),
                    "subject_type": triple.get("subject_type", ""),
                    "predicate": triple.get("predicate", ""),
                    "object": triple.get("object", ""),
                    "object_type": triple.get("object_type", ""),
                    "source_doc_id": triple.get("source_doc_id", ""),
                    "confidence": triple.get("confidence", 0),
                },
            }
        )
    return rows


def _render_import_markdown(
    triples: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    processed_documents: list[dict[str, Any]],
    validation_report: dict[str, Any],
) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for triple in triples:
        grouped[str(triple.get("subject", ""))].append(triple)

    lines: list[str] = [
        "# 医疗知识图谱 DataMate 导入包",
        "",
        "本文件由参赛仓库的任务一、任务二产物生成，可作为 DataMate DOCUMENT 知识库文件导入。",
        "",
        "## 图谱概览",
        "",
        f"- 文档数：{len(processed_documents)}",
        f"- 实体数：{len(entities)}",
        f"- 三元组数：{len(triples)}",
        f"- 校验通过三元组数：{validation_report.get('accepted_triples', len(triples))}",
        f"- 校验拒绝三元组数：{validation_report.get('rejected_triples', 0)}",
        "",
        "## 疾病知识条目",
    ]
    for disease, disease_triples in sorted(grouped.items()):
        if not disease:
            continue
        lines.append("")
        lines.append(f"### {disease}")
        for triple in disease_triples[:20]:
            evidence = str(triple.get("evidence", "")).strip()
            source_doc_id = str(triple.get("source_doc_id", "")).strip()
            lines.append(
                f"- `{triple.get('predicate', '')}` -> {triple.get('object', '')}"
                f"；来源：{source_doc_id}；证据：{evidence}"
            )
    return "\n".join(lines).strip() + "\n"


def _build_nexent_agent_config(
    output_root: Path,
    urls: dict[str, str],
    knowledge_base_ids: list[str],
    env_files: list[dict[str, str]],
) -> dict[str, Any]:
    model = _env_value("LLM_MODEL", "gpt-5.4", env_files)
    api_base = _env_value("OPENAI_API_BASE", "", env_files)
    datamate_server_url = _env_value(
        "NEXENT_TENANT_DATAMATE_URL",
        _container_accessible_url(urls["datamate_url"]),
        env_files,
    )
    return {
        "name": "medical_kg_datamate_agent",
        "description": "医疗知识图谱问答与分析 Agent，优先调用 DataMate 知识库检索，并回落到本仓库图谱产物。",
        "created_at": _now(),
        "model": {
            "provider": "openai-compatible",
            "model": model,
            "base_url": api_base,
            "api_key": "***REDACTED***" if _env_value("OPENAI_API_KEY", "", env_files) else "",
        },
        "tools": [
            {
                "name": "datamate_search",
                "class_name": "DataMateSearchTool",
                "server_url": datamate_server_url,
                "ui_url": urls["datamate_url"],
                "index_names": knowledge_base_ids or ["<fill DataMate DOCUMENT knowledge base id>"],
                "top_k": 5,
                "threshold": 0.2,
                "expected_backend_endpoint": "/api/knowledge-base/v2/retrieve",
            },
            {
                "name": TASK1_OPENAPI_SERVICE_NAME,
                "type": "nexent_openapi_mcp",
                "operation_id": TASK1_OPENAPI_OPERATION_ID,
                "server_url": _container_accessible_url(_env_value("MODELENGINE_DEMO_API_URL", DEFAULT_DEMO_API_URL, env_files)),
                "openapi_spec": "demo/integration/task1_nexent_openapi.json",
                "endpoints": [
                    "POST /api/datamate-process",
                ],
                "local_command": "python3 -m demo.app_server --host 127.0.0.1 --port 8088",
                "nexent_import_api": "/tool/openapi_service",
            },
            {
                "name": TASK2_OPENAPI_SERVICE_NAME,
                "type": "nexent_openapi_mcp",
                "operations": [TASK2_BUILD_OPERATION_ID, TASK2_ASK_OPERATION_ID],
                "server_url": _container_accessible_url(_env_value("MODELENGINE_DEMO_API_URL", DEFAULT_DEMO_API_URL, env_files)),
                "openapi_spec": "demo/integration/task2_nexent_openapi.json",
                "endpoints": [
                    "POST /api/medical-kg/build",
                    "POST /api/medical-kg/ask",
                ],
                "local_command": "python3 -m demo.app_server --host 127.0.0.1 --port 8088",
                "nexent_import_api": "/tool/openapi_service",
            },
            {
                "name": TASK3_OPENAPI_SERVICE_NAME,
                "type": "nexent_openapi_mcp",
                "operations": [
                    TASK3_ANALYZE_OPERATION_ID,
                    TASK3_INSIGHT_OPERATION_ID,
                    TASK3_CONFIG_OPERATION_ID,
                ],
                "server_url": _container_accessible_url(_env_value("MODELENGINE_DEMO_API_URL", DEFAULT_DEMO_API_URL, env_files)),
                "openapi_spec": "demo/integration/task3_nexent_openapi.json",
                "endpoints": [
                    "POST /api/analyze",
                    "POST /api/insight",
                    "POST /api/analyze-config",
                ],
                "local_command": "python3 -m demo.app_server --host 127.0.0.1 --port 8088",
                "nexent_import_api": "/tool/openapi_service",
            },
        ],
        "system_prompt": (
            "你是医疗知识图谱问答和分析智能体。回答必须优先使用 datamate_search 或本地 graph/triples.jsonl "
            "中的证据，输出疾病、症状、治疗、检查、科室和并发症时保留来源。不要新增没有证据的医学事实。"
        ),
        "data_sources": {
            "datamate_import_manifest": "demo/integration/datamate_import/manifest.json",
            "local_graph": "graph/graph_snapshot.json",
            "local_triples": "graph/triples.jsonl",
            "analysis_report": "demo/analysis_report.md",
        },
        "nexent_config_api": urls["nexent_config_api"],
        "export_note": "Use this file as a reviewable Agent configuration record; configure the same tool values in Nexent UI or API.",
    }


def _render_report(evidence: dict[str, Any], output_root: Path) -> str:
    lines = [
        "# DataMate/Nexent 集成证据报告",
        "",
        f"- 生成时间：{evidence['created_at']}",
        f"- DataMate URL：{evidence['urls']['datamate_url']}",
        f"- DataMate API：{evidence['urls']['datamate_api_base']}",
        f"- Nexent URL：{evidence['urls']['nexent_url']}",
        f"- Nexent Config API：{evidence['urls']['nexent_config_api']}",
        "",
        "## 服务探测",
        "",
        "| 服务 | 地址 | 状态码 | 可达 | 说明 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in evidence["service_probes"]:
        lines.append(
            f"| {item['name']} | `{item['url']}` | {item.get('status_code') or ''} | "
            f"{'是' if item.get('reachable') else '否'} | {item.get('error', '')} |"
        )

    lines.extend(["", "## DataMate API 调用", "", "| 调用 | 方法 | 地址 | 状态码 | 成功 | 错误 |", "| --- | --- | --- | ---: | --- | --- |"])
    for item in evidence.get("datamate_bootstrap_calls", []) + evidence["datamate_api_calls"]:
        lines.append(
            f"| {item['name']} | {item['method']} | `{item['url']}` | {item.get('status_code') or ''} | "
            f"{'是' if item.get('ok') else '否'} | {item.get('error', '')} |"
        )

    lines.extend(["", "## Nexent API 调用", "", "| 调用 | 方法 | 地址 | 状态码 | 成功 | 错误 |", "| --- | --- | --- | ---: | --- | --- |"])
    for item in evidence["nexent_api_calls"]:
        lines.append(
            f"| {item['name']} | {item['method']} | `{item['url']}` | {item.get('status_code') or ''} | "
            f"{'是' if item.get('ok') else '否'} | {item.get('error', '')} |"
        )

    lines.extend(["", "## datamate_search Trace", ""])
    for item in evidence["datamate_search_trace"]:
        lines.append(
            f"- 状态：{item['status']}；查询：{item.get('query', '')}；知识库：{item.get('knowledge_base_ids', [])}；"
            f"结果数：{item.get('result_count', 0)}；错误：{item.get('error') or item.get('reason', '')}"
        )

    task1_summary = evidence.get("task1_nexent_runtime", {}).get("summary", {})
    lines.extend(
        [
            "",
            "## 任务一 Nexent 运行链路",
            "",
            f"- OpenAPI 服务名：`{task1_summary.get('service_name', '')}`",
            f"- 操作 ID：`{task1_summary.get('operation_id', '')}`",
            f"- Demo API：`{task1_summary.get('demo_api_url', '')}`",
            f"- Nexent 容器访问地址：`{task1_summary.get('nexent_server_url', '')}`",
            f"- 已通过 Nexent OpenAPI API 注册：{'是' if task1_summary.get('registered_via_nexent_openapi_api') else '否'}",
            f"- 已出现在 Nexent OpenAPI 服务列表：{'是' if task1_summary.get('listed_by_nexent') else '否'}",
            f"- 已刷新到 Nexent MCP：{'是' if task1_summary.get('registered_in_nexent_mcp') else '否'}",
            f"- 已出现在 Nexent 工具扫描：{'是' if task1_summary.get('visible_in_nexent_tool_scan') else '否'}",
            f"- Nexent MCP 工具调用成功：{'是' if task1_summary.get('nexent_mcp_tool_call_ok') else '否'}",
            f"- OpenAPI/MCP 工具链就绪：{'是' if task1_summary.get('nexent_openapi_mcp_tool_chain_ready') else '否'}",
            f"- Nexent 自主 agent runtime 就绪：{'是' if task1_summary.get('nexent_agent_runtime_autonomous_ready') else '否'}",
            f"- Nexent agent_id：`{task1_summary.get('nexent_agent_id') or ''}`",
            f"- `/agent/run` 尝试：{'是' if task1_summary.get('nexent_runtime_run_attempted') else '否'}；成功：{'是' if task1_summary.get('nexent_runtime_run_ok') else '否'}",
        ]
    )
    if task1_summary.get("nexent_runtime_run_blocker"):
        lines.append("- `/agent/run` 阻塞原因：Nexent runtime 当前缺少真实 LLM 凭据，未完成自主规划执行。")

    task2_summary = evidence.get("task2_nexent_runtime", {}).get("summary", {})
    lines.extend(
        [
            "",
            "## 任务二 Nexent 运行链路",
            "",
            f"- OpenAPI 服务名：`{task2_summary.get('service_name', '')}`",
            f"- 构建操作 ID：`{task2_summary.get('build_operation_id', '')}`",
            f"- 问答操作 ID：`{task2_summary.get('ask_operation_id', '')}`",
            f"- Demo API：`{task2_summary.get('demo_api_url', '')}`",
            f"- Nexent 容器访问地址：`{task2_summary.get('nexent_server_url', '')}`",
            f"- 已通过 Nexent OpenAPI API 注册：{'是' if task2_summary.get('registered_via_nexent_openapi_api') else '否'}",
            f"- 已出现在 Nexent OpenAPI 服务列表：{'是' if task2_summary.get('listed_by_nexent') else '否'}",
            f"- 已刷新到 Nexent MCP：{'是' if task2_summary.get('registered_in_nexent_mcp') else '否'}",
            f"- 已出现在 Nexent 工具扫描：{'是' if task2_summary.get('visible_in_nexent_tool_scan') else '否'}",
            f"- Nexent MCP 工具调用成功：{'是' if task2_summary.get('nexent_mcp_tool_call_ok') else '否'}",
            f"- OpenAPI/MCP 工具链就绪：{'是' if task2_summary.get('nexent_openapi_mcp_tool_chain_ready') else '否'}",
            f"- Nexent 自主 agent runtime 就绪：{'是' if task2_summary.get('nexent_agent_runtime_autonomous_ready') else '否'}",
            f"- Nexent agent_id：`{task2_summary.get('nexent_agent_id') or ''}`",
            f"- `/agent/run` 尝试：{'是' if task2_summary.get('nexent_runtime_run_attempted') else '否'}；成功：{'是' if task2_summary.get('nexent_runtime_run_ok') else '否'}",
        ]
    )
    task2_markers = task2_summary.get("task2_latest_kg_markers", {})
    if task2_markers:
        lines.extend(
            [
                f"- KG 验收标志：documents={task2_markers.get('documents')}；entities={task2_markers.get('entities')}；"
                f"triples={task2_markers.get('triples')}；accepted_triples={task2_markers.get('accepted_triples')}",
                f"- 算子：`{', '.join(task2_markers.get('operator_names', []))}`",
            ]
        )

    task3_summary = evidence.get("task3_nexent_runtime", {}).get("summary", {})
    lines.extend(
        [
            "",
            "## 任务三 Nexent 运行链路",
            "",
            f"- OpenAPI 服务名：`{task3_summary.get('service_name', '')}`",
            f"- 完整分析操作 ID：`{task3_summary.get('analyze_operation_id', '')}`",
            f"- 自然语言分析操作 ID：`{task3_summary.get('insight_operation_id', '')}`",
            f"- Demo API：`{task3_summary.get('demo_api_url', '')}`",
            f"- Nexent 容器访问地址：`{task3_summary.get('nexent_server_url', '')}`",
            f"- 已通过 Nexent OpenAPI API 注册：{'是' if task3_summary.get('registered_via_nexent_openapi_api') else '否'}",
            f"- 已出现在 Nexent OpenAPI 服务列表：{'是' if task3_summary.get('listed_by_nexent') else '否'}",
            f"- 已刷新到 Nexent MCP：{'是' if task3_summary.get('registered_in_nexent_mcp') else '否'}",
            f"- 已出现在 Nexent 工具扫描：{'是' if task3_summary.get('visible_in_nexent_tool_scan') else '否'}",
            f"- Nexent MCP 工具调用成功：{'是' if task3_summary.get('nexent_mcp_tool_call_ok') else '否'}",
            f"- OpenAPI/MCP 工具链就绪：{'是' if task3_summary.get('nexent_openapi_mcp_tool_chain_ready') else '否'}",
            f"- Nexent 自主 agent runtime 就绪：{'是' if task3_summary.get('nexent_agent_runtime_autonomous_ready') else '否'}",
            f"- Nexent agent_id：`{task3_summary.get('nexent_agent_id') or ''}`",
            f"- `/agent/run` 尝试：{'是' if task3_summary.get('nexent_runtime_run_attempted') else '否'}；成功：{'是' if task3_summary.get('nexent_runtime_run_ok') else '否'}",
        ]
    )
    task3_markers = task3_summary.get("task3_latest_analysis_markers", {})
    if task3_markers:
        lines.extend(
            [
                f"- 分析验收标志：query_count={task3_markers.get('query_count')}；"
                f"nl2sql_accuracy={task3_markers.get('nl2sql_accuracy')}；"
                f"task3_closed_loop_ready={task3_markers.get('task3_closed_loop_ready')}",
                f"- 可视化：bi_chart_count={task3_markers.get('bi_chart_count')}；"
                f"graph_visual_count={task3_markers.get('graph_visual_count')}",
            ]
        )

    artifact_paths = evidence["artifact_paths"]
    lines.extend(
        [
            "",
            "## 证据文件",
            "",
            f"- 平台证据 JSON：`{Path(artifact_paths['platform_evidence']).relative_to(output_root)}`",
            f"- 服务/API trace：`{Path(artifact_paths['platform_probe_trace']).relative_to(output_root)}`",
            f"- datamate_search trace：`{Path(artifact_paths['datamate_search_trace']).relative_to(output_root)}`",
            f"- 任务一 OpenAPI 规范：`{Path(artifact_paths['task1_nexent_openapi']).relative_to(output_root)}`",
            f"- 任务一 Nexent 工具 trace：`{Path(artifact_paths['task1_nexent_tool_trace']).relative_to(output_root)}`",
            f"- 任务一 Nexent 运行证据：`{Path(artifact_paths['task1_nexent_runtime_evidence']).relative_to(output_root)}`",
            f"- 任务二 OpenAPI 规范：`{Path(artifact_paths['task2_nexent_openapi']).relative_to(output_root)}`",
            f"- 任务二 Nexent 工具 trace：`{Path(artifact_paths['task2_nexent_tool_trace']).relative_to(output_root)}`",
            f"- 任务二 Nexent 运行证据：`{Path(artifact_paths['task2_nexent_runtime_evidence']).relative_to(output_root)}`",
            f"- 任务三 OpenAPI 规范：`{Path(artifact_paths['task3_nexent_openapi']).relative_to(output_root)}`",
            f"- 任务三 Nexent 工具 trace：`{Path(artifact_paths['task3_nexent_tool_trace']).relative_to(output_root)}`",
            f"- 任务三 Nexent 运行证据：`{Path(artifact_paths['task3_nexent_runtime_evidence']).relative_to(output_root)}`",
            f"- Nexent Agent 配置导出：`{Path(artifact_paths['nexent_agent_config']).relative_to(output_root)}`",
            f"- DataMate 导入 Markdown：`{Path(artifact_paths['datamate_import_markdown']).relative_to(output_root)}`",
            f"- DataMate 导入 JSONL：`{Path(artifact_paths['datamate_import_jsonl']).relative_to(output_root)}`",
            "",
            "## 判定",
            "",
        ]
    )

    if evidence["summary"]["datamate_reachable"] and evidence["summary"]["nexent_reachable"]:
        lines.append("- DataMate 与 Nexent 服务当前均可达。")
    else:
        lines.append("- 当前运行环境未同时探测到 DataMate 与 Nexent 可达；启动两个 Docker 环境后重新运行采集命令即可更新本报告。")
    if evidence["summary"]["datamate_search_success"]:
        lines.append("- 已记录一次成功的 `datamate_search`/DataMate retrieve 调用。")
    else:
        lines.append("- 尚未记录成功的 `datamate_search` 调用；需要先在 DataMate 创建知识库并在命令中传入或同步知识库 ID。")
    if evidence["summary"].get("task1_nexent_openapi_mcp_tool_chain_ready"):
        lines.append("- 任务一 DataProcessingAgent 已通过 Nexent OpenAPI/MCP 工具体系注册，并已通过 MCP 真实调用触发 DataMate 处理任务。")
    else:
        lines.append("- 任务一 DataProcessingAgent 尚未完成 Nexent OpenAPI/MCP 注册；需要启动 Nexent Config API、MCP 管理服务和本项目 Demo API 后重新采集。")
    if not evidence["summary"].get("task1_nexent_agent_runtime_autonomous_ready"):
        lines.append("- Nexent `/agent/run` 自主智能体执行尚未跑通；当前阻塞点是真实 LLM 凭据未配置。")
    if evidence["summary"].get("task2_nexent_openapi_mcp_tool_chain_ready"):
        lines.append("- 任务二 MedicalKGAgent 已通过 Nexent OpenAPI/MCP 工具体系注册，并已通过 MCP 真实调用构建医疗知识图谱。")
    else:
        lines.append("- 任务二 MedicalKGAgent 尚未完成 Nexent OpenAPI/MCP 真实调用；需要启动 Nexent Config API、MCP 管理服务和本项目 Demo API 后重新采集。")
    if evidence["summary"].get("task3_nexent_openapi_mcp_tool_chain_ready"):
        lines.append("- 任务三 AnalysisAgent 已通过 Nexent OpenAPI/MCP 工具体系注册，并已通过 MCP 真实调用完成图谱驱动分析。")
    else:
        lines.append("- 任务三 AnalysisAgent 尚未完成 Nexent OpenAPI/MCP 真实调用；需要启动 Nexent Config API、MCP 管理服务和本项目 Demo API 后重新采集。")
    return "\n".join(lines).strip() + "\n"


def run_platform_integration_agent(
    output_root: Path,
    *,
    query: str = "高血压有哪些并发症？",
    knowledge_base_id: str | None = None,
    timeout_seconds: float = 1.5,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    project_env = _load_env_file(output_root / ".env")
    example_env = _load_env_file(output_root / ".env.example")
    env_files = [project_env, example_env]

    urls = {
        "datamate_url": _env_value("DATAMATE_URL", DEFAULT_DATAMATE_URL, env_files),
        "datamate_api_base": _env_value("DATAMATE_API_BASE", DEFAULT_DATAMATE_API_BASE, env_files),
        "datamate_runtime_api": _env_value("DATAMATE_RUNTIME_API", DEFAULT_DATAMATE_RUNTIME_API, env_files),
        "nexent_url": _env_value("NEXENT_URL", DEFAULT_NEXENT_URL, env_files),
        "nexent_config_api": _env_value("NEXENT_CONFIG_API", DEFAULT_NEXENT_CONFIG_API, env_files),
        "nexent_runtime_api": _env_value("NEXENT_RUNTIME_API", DEFAULT_NEXENT_RUNTIME_API, env_files),
        "nexent_mcp_management_api": _env_value("NEXENT_MCP_MANAGEMENT_API", DEFAULT_NEXENT_MCP_MANAGEMENT_API, env_files),
        "demo_api_url": _env_value("MODELENGINE_DEMO_API_URL", DEFAULT_DEMO_API_URL, env_files),
    }
    urls["demo_api_nexent_url"] = _env_value(
        "MODELENGINE_DEMO_API_NEXENT_URL",
        _container_accessible_url(urls["demo_api_url"]),
        env_files,
    )
    knowledge_base_id = knowledge_base_id or os.getenv("DATAMATE_KNOWLEDGE_BASE_ID") or project_env.get("DATAMATE_KNOWLEDGE_BASE_ID")

    integration_dir = output_root / "demo" / "integration"
    import_package = _build_datamate_import_package(output_root, integration_dir)
    bootstrap_api_calls: list[dict[str, Any]] = []
    bootstrap_state: dict[str, Any] = {
        "models": {},
        "dataset": {},
        "knowledge_base": {},
        "uploaded_files": [],
    }

    service_probes = _collect_service_probes(urls, timeout_seconds)

    datamate_bootstrap_calls, datamate_model_state = _ensure_datamate_models(
        urls,
        env_files,
        timeout_seconds=timeout_seconds,
    )
    bootstrap_api_calls.extend(datamate_bootstrap_calls)
    bootstrap_state["models"] = datamate_model_state

    dataset_name = "medical_kg_datamate_dataset"
    dataset_calls, dataset_state = _ensure_datamate_dataset(
        urls,
        dataset_name=dataset_name,
        description="Medical KG demo import dataset",
        dataset_type=DEFAULT_DATAMATE_DATASET_TYPE,
        timeout_seconds=timeout_seconds,
    )
    bootstrap_api_calls.extend(dataset_calls)
    bootstrap_state["dataset"] = dataset_state

    if dataset_state.get("id"):
        base_url = str(dataset_state.get("base_url") or urls["datamate_api_base"])
        dataset_file_calls: list[dict[str, Any]] = []
        list_calls, file_ids = _collect_dataset_file_ids(
            base_url=base_url,
            dataset_id=str(dataset_state["id"]),
            timeout_seconds=timeout_seconds,
        )
        dataset_file_calls.extend(list_calls)
        if not file_ids:
            for local_file in [
                Path(import_package["paths"]["markdown"]),
                Path(import_package["paths"]["jsonl"]),
            ]:
                dataset_file_calls.extend(
                    _upload_file_to_datamate_dataset(
                        base_url=base_url,
                        dataset_id=str(dataset_state["id"]),
                        file_path=local_file,
                        prefix="",
                        timeout_seconds=timeout_seconds,
                    )
                )
            list_calls, file_ids = _collect_dataset_file_ids(
                base_url=base_url,
                dataset_id=str(dataset_state["id"]),
                timeout_seconds=timeout_seconds,
            )
            dataset_file_calls.extend(list_calls)
        bootstrap_api_calls.extend(dataset_file_calls)
        bootstrap_state["uploaded_files"] = file_ids

        kb_state: dict[str, Any] = {}
        if datamate_model_state.get("embedding_model_id"):
            kb_name = "medical_kg_datamate_kb"
            kb_calls, kb_state = _create_datamate_knowledge_base(
                urls,
                knowledge_base_name=kb_name,
                description="Medical KG demo knowledge base",
                embedding_model_id=str(datamate_model_state["embedding_model_id"]),
                chat_model_id=str(datamate_model_state["chat_model_id"]) if datamate_model_state.get("chat_model_id") else None,
                timeout_seconds=timeout_seconds,
            )
            bootstrap_api_calls.extend(kb_calls)
        bootstrap_state["knowledge_base"] = kb_state

        attachable_file_ids = _datamate_kb_supported_files(file_ids)
        bootstrap_state["knowledge_base_attach_files"] = attachable_file_ids
        if kb_state.get("id") and attachable_file_ids:
            attach_calls = _attach_files_to_knowledge_base(
                base_url=str(kb_state.get("base_url") or base_url),
                knowledge_base_id=str(kb_state["id"]),
                dataset_file_ids=attachable_file_ids,
                timeout_seconds=timeout_seconds,
            )
            bootstrap_api_calls.extend(attach_calls)
            bootstrap_api_calls.extend(
                _wait_for_knowledge_base_files(
                    base_url=str(kb_state.get("base_url") or base_url),
                    knowledge_base_id=str(kb_state["id"]),
                    timeout_seconds=timeout_seconds,
                )
            )
            knowledge_base_id = str(kb_state["id"])
        elif kb_state.get("id"):
            knowledge_base_id = str(kb_state["id"])
    elif datamate_model_state.get("embedding_model_id") and knowledge_base_id:
        kb_base_url = urls["datamate_api_base"]
        for candidate_base in _base_url_variants(urls["datamate_api_base"]):
            detail_call = _http_call(
                name=f"DataMate knowledge-base detail for provided id via {candidate_base}",
                method="GET",
                url=_join_url(candidate_base, f"/api/knowledge-base/{knowledge_base_id}"),
                timeout_seconds=timeout_seconds,
            )
            bootstrap_api_calls.append(detail_call)
            if detail_call.get("ok"):
                kb_base_url = candidate_base
                break
        dataset_calls, dataset_state = _ensure_datamate_dataset(
            urls,
            dataset_name=dataset_name,
            description="Medical KG demo import dataset",
            dataset_type=DEFAULT_DATAMATE_DATASET_TYPE,
            timeout_seconds=timeout_seconds,
        )
        bootstrap_api_calls.extend(dataset_calls)
        bootstrap_state["dataset"] = dataset_state
        if dataset_state.get("id"):
            base_url = str(dataset_state.get("base_url") or urls["datamate_api_base"])
            list_calls, file_ids = _collect_dataset_file_ids(
                base_url=base_url,
                dataset_id=str(dataset_state["id"]),
                timeout_seconds=timeout_seconds,
            )
            bootstrap_api_calls.extend(list_calls)
            bootstrap_state["uploaded_files"] = file_ids
            attachable_file_ids = _datamate_kb_supported_files(file_ids)
            bootstrap_state["knowledge_base_attach_files"] = attachable_file_ids
            if attachable_file_ids:
                bootstrap_api_calls.extend(
                    _attach_files_to_knowledge_base(
                        base_url=kb_base_url,
                        knowledge_base_id=str(knowledge_base_id),
                        dataset_file_ids=attachable_file_ids,
                        timeout_seconds=timeout_seconds,
                    )
                )
                bootstrap_api_calls.extend(
                    _wait_for_knowledge_base_files(
                        base_url=kb_base_url,
                        knowledge_base_id=str(knowledge_base_id),
                        timeout_seconds=timeout_seconds,
                    )
                )

    datamate_api_calls, discovered_ids, datamate_search_trace = _collect_datamate_api_evidence(
        urls,
        query=query,
        knowledge_base_id=knowledge_base_id,
        timeout_seconds=timeout_seconds,
    )
    knowledge_base_ids = discovered_ids or ([knowledge_base_id] if knowledge_base_id else [])
    nexent_api_calls = _collect_nexent_api_evidence(
        urls,
        datamate_url=_container_accessible_url(urls["datamate_url"]),
        knowledge_base_ids=knowledge_base_ids,
        timeout_seconds=timeout_seconds,
    )
    task1_nexent_runtime = _collect_task1_nexent_runtime_evidence(
        urls,
        output_root=output_root,
        integration_dir=integration_dir,
        timeout_seconds=timeout_seconds,
        env_files=env_files,
    )
    task2_nexent_runtime = _collect_task2_nexent_runtime_evidence(
        urls,
        output_root=output_root,
        integration_dir=integration_dir,
        timeout_seconds=timeout_seconds,
        env_files=env_files,
    )
    task3_nexent_runtime = _collect_task3_nexent_runtime_evidence(
        urls,
        output_root=output_root,
        integration_dir=integration_dir,
        timeout_seconds=timeout_seconds,
        env_files=env_files,
    )
    nexent_agent_config = _build_nexent_agent_config(output_root, urls, knowledge_base_ids, env_files)

    platform_trace = (
        service_probes
        + bootstrap_api_calls
        + datamate_api_calls
        + nexent_api_calls
        + task1_nexent_runtime["calls"]
        + task2_nexent_runtime["calls"]
        + task3_nexent_runtime["calls"]
    )
    datamate_reachable = any(item["name"].startswith("DataMate") and item.get("reachable") for item in service_probes)
    nexent_reachable = any(item["name"].startswith("Nexent") and item.get("reachable") for item in service_probes)
    datamate_search_success = any(item.get("status") == "success" for item in datamate_search_trace)
    task1_nexent_summary = task1_nexent_runtime["summary"]
    task2_nexent_summary = task2_nexent_runtime["summary"]
    task3_nexent_summary = task3_nexent_runtime["summary"]

    artifact_paths = {
        "platform_evidence": str(integration_dir / "platform_evidence.json"),
        "platform_probe_trace": str(integration_dir / "platform_probe_trace.jsonl"),
        "datamate_search_trace": str(integration_dir / "datamate_search_trace.jsonl"),
        "task1_nexent_openapi": task1_nexent_runtime["artifact_paths"]["openapi_spec"],
        "task1_nexent_tool_trace": task1_nexent_runtime["artifact_paths"]["trace"],
        "task1_nexent_runtime_evidence": task1_nexent_runtime["artifact_paths"]["runtime_evidence"],
        "task2_nexent_openapi": task2_nexent_runtime["artifact_paths"]["openapi_spec"],
        "task2_nexent_tool_trace": task2_nexent_runtime["artifact_paths"]["trace"],
        "task2_nexent_runtime_evidence": task2_nexent_runtime["artifact_paths"]["runtime_evidence"],
        "task3_nexent_openapi": task3_nexent_runtime["artifact_paths"]["openapi_spec"],
        "task3_nexent_tool_trace": task3_nexent_runtime["artifact_paths"]["trace"],
        "task3_nexent_runtime_evidence": task3_nexent_runtime["artifact_paths"]["runtime_evidence"],
        "nexent_agent_config": str(integration_dir / "nexent_agent_config.json"),
        "integration_report": str(integration_dir / "integration_evidence_report.md"),
        "datamate_import_markdown": import_package["paths"]["markdown"],
        "datamate_import_jsonl": import_package["paths"]["jsonl"],
        "datamate_import_manifest": import_package["paths"]["manifest"],
    }
    evidence = {
        "created_at": _now(),
        "goal": "Collect auditable DataMate/Nexent integration evidence for the medical KG demo.",
        "urls": urls,
        "config_snapshot": _collect_config_snapshot(output_root, env_files),
        "service_probes": service_probes,
        "datamate_api_calls": datamate_api_calls,
        "datamate_bootstrap_calls": bootstrap_api_calls,
        "datamate_bootstrap_state": bootstrap_state,
        "nexent_api_calls": nexent_api_calls,
        "task1_nexent_runtime": task1_nexent_runtime,
        "task2_nexent_runtime": task2_nexent_runtime,
        "task3_nexent_runtime": task3_nexent_runtime,
        "datamate_search_trace": datamate_search_trace,
        "knowledge_base_ids": knowledge_base_ids,
        "datamate_import_package": import_package["manifest"],
        "nexent_agent_config_summary": {
            "name": nexent_agent_config["name"],
            "tools": [item["name"] for item in nexent_agent_config["tools"]],
            "model": nexent_agent_config["model"]["model"],
        },
        "summary": {
            "datamate_reachable": datamate_reachable,
            "nexent_reachable": nexent_reachable,
            "datamate_search_success": datamate_search_success,
            "task1_nexent_openapi_registered": task1_nexent_summary["registered_via_nexent_openapi_api"],
            "task1_nexent_tool_visible": task1_nexent_summary["visible_in_nexent_tool_scan"],
            "task1_nexent_mcp_registered": task1_nexent_summary["registered_in_nexent_mcp"],
            "task1_nexent_mcp_tool_call_ok": task1_nexent_summary["nexent_mcp_tool_call_ok"],
            "task1_nexent_openapi_mcp_tool_chain_ready": task1_nexent_summary["nexent_openapi_mcp_tool_chain_ready"],
            "task1_nexent_agent_runtime_autonomous_ready": task1_nexent_summary["nexent_agent_runtime_autonomous_ready"],
            "task1_nexent_runtime_ready": task1_nexent_summary["nexent_runtime_ready"],
            "task1_nexent_run_attempted": task1_nexent_summary["nexent_runtime_run_attempted"],
            "task1_nexent_run_ok": task1_nexent_summary["nexent_runtime_run_ok"],
            "task2_nexent_openapi_registered": task2_nexent_summary["registered_via_nexent_openapi_api"],
            "task2_nexent_tool_visible": task2_nexent_summary["visible_in_nexent_tool_scan"],
            "task2_nexent_mcp_registered": task2_nexent_summary["registered_in_nexent_mcp"],
            "task2_nexent_mcp_tool_call_ok": task2_nexent_summary["nexent_mcp_tool_call_ok"],
            "task2_nexent_openapi_mcp_tool_chain_ready": task2_nexent_summary["nexent_openapi_mcp_tool_chain_ready"],
            "task2_nexent_agent_runtime_autonomous_ready": task2_nexent_summary["nexent_agent_runtime_autonomous_ready"],
            "task2_nexent_runtime_ready": task2_nexent_summary["nexent_runtime_ready"],
            "task2_nexent_run_attempted": task2_nexent_summary["nexent_runtime_run_attempted"],
            "task2_nexent_run_ok": task2_nexent_summary["nexent_runtime_run_ok"],
            "task2_latest_kg_ready": task2_nexent_summary["task2_latest_kg_ready"],
            "task3_nexent_openapi_registered": task3_nexent_summary["registered_via_nexent_openapi_api"],
            "task3_nexent_tool_visible": task3_nexent_summary["visible_in_nexent_tool_scan"],
            "task3_nexent_mcp_registered": task3_nexent_summary["registered_in_nexent_mcp"],
            "task3_nexent_mcp_tool_call_ok": task3_nexent_summary["nexent_mcp_tool_call_ok"],
            "task3_nexent_openapi_mcp_tool_chain_ready": task3_nexent_summary["nexent_openapi_mcp_tool_chain_ready"],
            "task3_nexent_agent_runtime_autonomous_ready": task3_nexent_summary["nexent_agent_runtime_autonomous_ready"],
            "task3_nexent_runtime_ready": task3_nexent_summary["nexent_runtime_ready"],
            "task3_nexent_run_attempted": task3_nexent_summary["nexent_runtime_run_attempted"],
            "task3_nexent_run_ok": task3_nexent_summary["nexent_runtime_run_ok"],
            "task3_latest_analysis_ready": task3_nexent_summary["task3_latest_analysis_ready"],
            "task3_latest_interactive_ready": task3_nexent_summary["task3_latest_interactive_ready"],
            "service_probe_count": len(service_probes),
            "datamate_api_call_count": len(datamate_api_calls),
            "datamate_bootstrap_call_count": len(bootstrap_api_calls),
            "nexent_api_call_count": (
                len(nexent_api_calls)
                + len(task1_nexent_runtime["calls"])
                + len(task2_nexent_runtime["calls"])
                + len(task3_nexent_runtime["calls"])
            ),
        },
        "artifact_paths": artifact_paths,
    }

    _write_json(Path(artifact_paths["nexent_agent_config"]), nexent_agent_config)
    _write_jsonl(Path(artifact_paths["platform_probe_trace"]), platform_trace)
    _write_jsonl(Path(artifact_paths["datamate_search_trace"]), datamate_search_trace)
    _write_json(Path(artifact_paths["platform_evidence"]), evidence)
    _write_text(Path(artifact_paths["integration_report"]), _render_report(evidence, output_root))
    return evidence
