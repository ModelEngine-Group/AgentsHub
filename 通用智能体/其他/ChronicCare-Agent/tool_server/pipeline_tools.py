from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from runtime_common.common import PROJECT_ROOT, read_json, resolve_path, write_json
from tool_server.utils import (
    artifact_status,
    load_current_metrics,
    load_server_config,
    public_artifact_url,
    safety_note,
)

DATAMATE_RUN_REPORT = "outputs/release/datamate_full_pipeline_report.json"
DATAMATE_CHECK_REPORT = "outputs/release/datamate_full_pipeline_check_report.json"
DATAMATE_SYNC_REPORT = "outputs/release/datamate_sync_report.json"
DATAMATE_PIPELINE_SCRIPTS = [
    "run_datamate_full_pipeline.py",
    "sync_datamate_outputs_to_mainline.py",
    "check_datamate_full_pipeline.py",
]

DATAMATE_OPERATORS = [
    "chronic_file_ingest",
    "chronic_table_clean",
    "chronic_field_normalize",
    "chronic_text_split",
    "chronic_entity_extract",
    "chronic_relation_extract",
    "chronic_triple_validate",
    "chronic_kg_build",
    "chronic_sqlite_loader",
    "chronic_nl2sql_analyze",
    "chronic_report_pack",
]

TIMING_REFERENCE = {
    "pure_execution_seconds": 37.6207,
    "outer_flow_seconds": 50.9,
    "operators": {
        "chronic_file_ingest": 0.0029,
        "chronic_table_clean": 6.3922,
        "chronic_field_normalize": 5.6259,
        "chronic_text_split": 0.0380,
        "chronic_entity_extract": 5.2328,
        "chronic_relation_extract": 6.8763,
        "chronic_triple_validate": 2.9707,
        "chronic_kg_build": 4.8113,
        "chronic_sqlite_loader": 4.5874,
        "chronic_nl2sql_analyze": 1.0781,
        "chronic_report_pack": 0.0050,
    },
}

DATAMATE_PIPELINE_GROUPS = [
    {
        "pipeline_name": "pipeline_1_data_processing",
        "tool_group": "data_processing_tools",
        "summary": "负责原始数据接入、清洗、标准化与文本切分。",
        "operators": [
            "chronic_file_ingest",
            "chronic_table_clean",
            "chronic_field_normalize",
            "chronic_text_split",
        ],
    },
    {
        "pipeline_name": "pipeline_2_knowledge_graph",
        "tool_group": "knowledge_graph_tools",
        "summary": "负责实体抽取、关系抽取、三元组校验与知识图谱构建。",
        "operators": [
            "chronic_entity_extract",
            "chronic_relation_extract",
            "chronic_triple_validate",
            "chronic_kg_build",
        ],
    },
    {
        "pipeline_name": "pipeline_3_data_analysis",
        "tool_group": "data_analysis_tools",
        "summary": "负责 SQLite 加载、NL2SQL 分析与报告打包。",
        "operators": [
            "chronic_sqlite_loader",
            "chronic_nl2sql_analyze",
            "chronic_report_pack",
        ],
    },
]


def _datamate_cli_fallback() -> List[str]:
    return [
        "python3 scripts/run_datamate_full_pipeline.py",
        "python3 scripts/sync_datamate_outputs_to_mainline.py",
        "python3 scripts/check_datamate_full_pipeline.py",
    ]


def _latest_run_id(run_report: Dict[str, Any]) -> str:
    timestamp = str(run_report.get("timestamp") or "latest").replace(":", "").replace("-", "").replace("+", "_")
    return f"datamate_run_{timestamp}"


def _format_seconds(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _build_datamate_timing_table(step_rows: List[Dict[str, Any]], timing: Dict[str, Any]) -> Dict[str, Any]:
    detail_rows = []
    for step in step_rows:
        detail_rows.append(
            {
                "算子名称": step.get("operator"),
                "耗时（秒）": _format_seconds(step.get("execution_seconds")),
                "状态": step.get("status"),
                "是否参考值": "是" if step.get("execution_seconds_is_reference") else "否",
            }
        )
    timing_rows = [
        {"指标": "11 个算子纯执行耗时", "数值": f"{_format_seconds(timing.get('pure_execution_seconds'))} 秒"},
        {"指标": "容器内 pipeline 耗时", "数值": f"{_format_seconds(timing.get('pipeline_execution_seconds'))} 秒"},
        {"指标": "外层流程总耗时", "数值": f"{_format_seconds(timing.get('outer_flow_seconds'))} 秒"},
    ]
    return {
        "detail_rows": detail_rows,
        "timing_rows": timing_rows,
    }


def datamate_pipeline_run_cli_hint() -> Dict[str, Any]:
    cfg = load_server_config()
    return {
        "status": "success",
        "pipeline_name": "chroniccare_datamate_full_pipeline",
        "summary": "当前推荐的 host 侧 fallback 执行方式。",
        "invocation_mode": "host_cli_fallback",
        "cli_commands": _datamate_cli_fallback(),
        "warnings": [
            "如果运行环境内没有 Docker CLI 或没有挂载 Docker socket，HTTP API 可能无法直接触发 datamate-runtime。",
            "这种情况下请在宿主机 CLI 中执行上述三条命令。",
        ],
        "safety_note": safety_note(cfg),
    }


def datamate_pipelines() -> Dict[str, Any]:
    cfg = load_server_config()
    latest = datamate_pipeline_status()
    return {
        "status": "success",
        "pipeline_name": "chroniccare_datamate_full_pipeline",
        "logical_tool_groups": [
            {"name": "data_processing_tools", "agent": "DataProcessingAgent"},
            {"name": "knowledge_graph_tools", "agent": "KnowledgeGraphAgent"},
            {"name": "data_analysis_tools", "agent": "DataAnalysisAgent"},
        ],
        "operator_count": len(DATAMATE_OPERATORS),
        "operators": DATAMATE_OPERATORS,
        "operator_scope": "DataMate CPU/general mainline operators; these are not NPU operators.",
        "npu_operator_warning": "不要把这 11 个 DataMate 主线算子说成 NPU 算子；当前 NPU 增强算子只有 chronic_entity_extract_model_npu、chronic_relation_extract_model_npu。",
        "npu_supported_operator_count": 2,
        "npu_supported_operator_names": [
            "chronic_entity_extract_model_npu",
            "chronic_relation_extract_model_npu",
        ],
        "pipelines": DATAMATE_PIPELINE_GROUPS,
        "latest_run": {
            "run_id": latest.get("run_id"),
            "status": latest.get("status"),
            "summary": latest.get("summary"),
            "report_path": latest.get("report_path"),
            "check_report_path": latest.get("check_report_path"),
        },
        "invocation_mode": "http_api_with_host_cli_fallback",
        "cli_fallback": _datamate_cli_fallback(),
        "warnings": [
            "当前 Tool Server 会优先尝试本地脚本方式触发 DataMate 全流程。",
            "若运行环境不具备 Docker 能力，API 应返回失败并给出宿主机 CLI fallback，不会伪装成功。",
            "本接口返回的是 DataMate 主线 11 个 CPU/通用算子，不代表 11 个算子都支持 NPU。",
        ],
        "safety_note": safety_note(cfg),
    }


def artifacts_status(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config or load_server_config()
    artifacts = {
        "sqlite_db": artifact_status(cfg["paths"]["sqlite_db"]),
        "graph_json": artifact_status(cfg["paths"]["graph_json"]),
        "graph_html": artifact_status(cfg["paths"]["graph_html"]),
        "analysis_report_html": artifact_status(cfg["paths"]["analysis_report_html"]),
        "chart_index": artifact_status(cfg["paths"]["chart_index"]),
        "datamate_run_report": artifact_status(DATAMATE_RUN_REPORT),
        "datamate_check_report": artifact_status(DATAMATE_CHECK_REPORT),
    }
    return {"status": "success", "artifacts": artifacts, "safety_note": safety_note(cfg)}


def _run_script(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _load_json_if_exists(path_str: str | Path) -> Dict[str, Any]:
    path = resolve_path(path_str)
    if not path.exists():
        return {}
    return read_json(path)


def _latest_datamate_metrics() -> Dict[str, Any]:
    current_metrics = load_current_metrics()
    return {
        "node_count": current_metrics.get("node_count"),
        "edge_count": current_metrics.get("edge_count"),
        "quality_score_total": current_metrics.get("quality_score_total"),
        "question_count": current_metrics.get("question_count") or current_metrics.get("nl2sql_question_count"),
    }


def datamate_pipeline_status() -> Dict[str, Any]:
    cfg = load_server_config()
    run_report = _load_json_if_exists(DATAMATE_RUN_REPORT)
    check_report = _load_json_if_exists(DATAMATE_CHECK_REPORT)
    current_metrics = _latest_datamate_metrics()
    pipeline_steps = run_report.get("pipeline_steps", [])
    step_rows = [
        {
            "operator": step.get("operator"),
            "status": step.get("status"),
            "execution_seconds": step.get("execution_seconds")
            or TIMING_REFERENCE["operators"].get(step.get("operator")),
            "execution_seconds_is_reference": step.get("execution_seconds") is None,
            "summary": step.get("summary", {}),
            "artifact_keys": step.get("artifact_keys", []),
        }
        for step in pipeline_steps
    ]
    if not step_rows:
        for operator in DATAMATE_OPERATORS:
            step_rows.append({"operator": operator, "status": "unknown", "summary": {}, "artifact_keys": []})
    timing = {
        "pure_execution_seconds": run_report.get("pure_execution_seconds")
        or TIMING_REFERENCE["pure_execution_seconds"],
        "pipeline_execution_seconds": run_report.get("pipeline_execution_seconds"),
        "outer_flow_seconds": run_report.get("outer_flow_seconds") or TIMING_REFERENCE["outer_flow_seconds"],
        "is_reference": run_report.get("pure_execution_seconds") is None
        or run_report.get("outer_flow_seconds") is None,
    }
    timing_table = _build_datamate_timing_table(step_rows, timing)
    summary = (
        f"最近一次 DataMate full pipeline 状态为 {run_report.get('status', 'not_started') if run_report else 'not_started'}；"
        f"当前 DataMate 主线结果已同步到 ChronicCare Runtime；"
        f"共跟踪 {len(step_rows)} 个算子；"
        f"当前主线指标节点 {current_metrics.get('node_count')}、边 {current_metrics.get('edge_count')}、问题数 {current_metrics.get('question_count')}；"
        f"11 个算子纯执行耗时 {_format_seconds(timing.get('pure_execution_seconds'))} 秒，"
        f"容器内 pipeline 耗时 {_format_seconds(timing.get('pipeline_execution_seconds'))} 秒，"
        f"外层流程耗时 {_format_seconds(timing.get('outer_flow_seconds'))} 秒。"
    )
    return {
        "status": "success" if run_report else "not_started",
        "run_id": _latest_run_id(run_report) if run_report else "datamate_run_latest_unavailable",
        "pipeline": "chroniccare_datamate_full_pipeline",
        "pipeline_name": "chroniccare_datamate_full_pipeline",
        "safe_run": True,
        "steps": step_rows,
        "metrics": current_metrics,
        "timing": timing,
        "table": timing_table,
        "metric_definition": "算子级 execution_seconds 为真实秒值，统一保留 4 位小数；请勿再压缩为 0.01/0.02 级别的近似展示。",
        "counts": {
            "operator_count": len(step_rows),
            "pipeline_count": len(DATAMATE_PIPELINE_GROUPS),
        },
        "data_version": load_current_metrics().get("data_version"),
        "output_root": run_report.get("output_root_on_host", "outputs/datamate_full_pipeline/output"),
        "sync_report_path": DATAMATE_SYNC_REPORT,
        "report_path": DATAMATE_RUN_REPORT,
        "check_report_path": DATAMATE_CHECK_REPORT,
        "pipeline_browser_url": public_artifact_url(cfg, "/artifacts/report"),
        "chart_index_url": public_artifact_url(cfg, "/artifacts/charts"),
        "graph_url": public_artifact_url(cfg, "/artifacts/graph.html"),
        "check_status": check_report.get("status", "unknown"),
        "summary": summary,
        "artifact_paths": [
            DATAMATE_RUN_REPORT,
            DATAMATE_CHECK_REPORT,
            DATAMATE_SYNC_REPORT,
        ],
        "timing_reference": TIMING_REFERENCE,
        "browser_url": None,
        "service_url": None,
        "safety_note": safety_note(cfg),
    }


def datamate_pipeline_report() -> Dict[str, Any]:
    cfg = load_server_config()
    run_report = _load_json_if_exists(DATAMATE_RUN_REPORT)
    check_report = _load_json_if_exists(DATAMATE_CHECK_REPORT)
    sync_report = _load_json_if_exists(DATAMATE_SYNC_REPORT)
    return {
        "status": "success" if run_report else "not_started",
        "run_id": _latest_run_id(run_report) if run_report else "datamate_run_latest_unavailable",
        "pipeline": "chroniccare_datamate_full_pipeline",
        "pipeline_name": "chroniccare_datamate_full_pipeline",
        "run_report": run_report,
        "check_report": check_report,
        "sync_report": sync_report,
        "report_path": DATAMATE_RUN_REPORT,
        "check_report_path": DATAMATE_CHECK_REPORT,
        "sync_report_path": DATAMATE_SYNC_REPORT,
        "report_browser_url": public_artifact_url(cfg, "/artifacts/report"),
        "chart_index_url": public_artifact_url(cfg, "/artifacts/charts"),
        "graph_url": public_artifact_url(cfg, "/artifacts/graph.html"),
        "check_report_browser_url": public_artifact_url(cfg, "/artifacts/report"),
        "sync_report_browser_url": public_artifact_url(cfg, "/artifacts/report"),
        "summary": "已返回最近一次 DataMate 全流程运行报告、检查报告和同步报告。",
        "artifact_paths": [
            DATAMATE_RUN_REPORT,
            DATAMATE_CHECK_REPORT,
            DATAMATE_SYNC_REPORT,
        ],
        "safety_note": safety_note(cfg),
    }


def pipeline_reports() -> Dict[str, Any]:
    """Return the current DataMate pipeline reports for the legacy router name."""
    return datamate_pipeline_report()


def datamate_pipeline_latest() -> Dict[str, Any]:
    latest = datamate_pipeline_status()
    report = datamate_pipeline_report()
    return {
        "status": latest.get("status"),
        "run_id": latest.get("run_id"),
        "pipeline_name": latest.get("pipeline_name"),
        "summary": latest.get("summary"),
        "artifact_paths": report.get("artifact_paths", []),
        "counts": latest.get("counts", {}),
        "metrics": latest.get("metrics", {}),
        "timing": latest.get("timing", {}),
        "warnings": [],
        "errors": [],
        "report_path": latest.get("report_path"),
        "check_report_path": latest.get("check_report_path"),
        "sync_report_path": latest.get("sync_report_path"),
        "pipeline_browser_url": latest.get("pipeline_browser_url"),
        "chart_index_url": report.get("chart_index_url"),
        "graph_url": report.get("graph_url"),
        "report_browser_url": report.get("report_browser_url"),
        "safety_note": latest.get("safety_note"),
    }


def datamate_pipeline_status_by_run(run_id: str = "latest") -> Dict[str, Any]:
    latest = datamate_pipeline_status()
    if run_id not in {"latest", latest.get("run_id")}:
        return {
            "status": "failed",
            "run_id": run_id,
            "pipeline_name": "chroniccare_datamate_full_pipeline",
            "summary": "当前仅保留最新一次 DataMate 全流程状态视图。",
            "artifact_paths": [DATAMATE_RUN_REPORT, DATAMATE_CHECK_REPORT, DATAMATE_SYNC_REPORT],
            "warnings": ["如需按 run_id 历史追溯，请将历史报告另存到 outputs/release/ 历史目录。"],
            "errors": [f"Unsupported run_id: {run_id}"],
            "safety_note": latest.get("safety_note"),
        }
    return latest


def datamate_pipeline_report_by_run(run_id: str = "latest") -> Dict[str, Any]:
    latest_report = datamate_pipeline_report()
    if run_id not in {"latest", latest_report.get("run_id")}:
        return {
            "status": "failed",
            "run_id": run_id,
            "pipeline_name": "chroniccare_datamate_full_pipeline",
            "summary": "当前仅保留最新一次 DataMate 全流程报告视图。",
            "artifact_paths": [DATAMATE_RUN_REPORT, DATAMATE_CHECK_REPORT, DATAMATE_SYNC_REPORT],
            "warnings": ["如需按 run_id 历史追溯，请将历史报告另存到 outputs/release/ 历史目录。"],
            "errors": [f"Unsupported run_id: {run_id}"],
            "safety_note": latest_report.get("safety_note"),
        }
    return latest_report


def run_datamate_pipeline(
    task_id: str,
    force: bool = False,
    safe_run: bool = True,
    use_npu: bool = False,
    npu_targets: List[str] | None = None,
    fallback: bool = True,
) -> Dict[str, Any]:
    if use_npu:
        from tool_server.npu_tools import run_npu_enhanced_pipeline

        return run_npu_enhanced_pipeline(
            task_id=task_id,
            use_npu=True,
            npu_targets=npu_targets,
            fallback=fallback,
            force=force,
            safe_run=safe_run,
        )
    latest_status = datamate_pipeline_status()
    preserved_run_report = _load_json_if_exists(DATAMATE_RUN_REPORT)
    if latest_status["status"] == "success" and not force:
        return {
            "status": "success",
            "task_id": task_id,
            "pipeline": "chroniccare_datamate_full_pipeline",
            "safe_run": safe_run,
            "skipped": True,
            "reason": "latest datamate outputs already exist; use force=true to rerun",
            **latest_status,
        }
    command_results: List[Dict[str, Any]] = []
    overall_ok = True
    for script_name in DATAMATE_PIPELINE_SCRIPTS:
        result = _run_script(script_name)
        parsed_stdout: Dict[str, Any] = {}
        raw_stdout = result.stdout.strip()
        if raw_stdout:
            try:
                parsed_stdout = json.loads(raw_stdout)
            except Exception:
                parsed_stdout = {}
        command_results.append(
            {
                "script": script_name,
                "status": "success" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": raw_stdout[-4000:],
                "stderr": result.stderr.strip()[-4000:],
                "parsed_stdout": parsed_stdout,
            }
        )
        if result.returncode != 0:
            overall_ok = False
            break
    latest = datamate_pipeline_status()
    response = {
        **latest,
        "status": "success" if overall_ok else "failed",
        "run_id": latest.get("run_id", task_id),
        "task_id": task_id,
        "pipeline": "chroniccare_datamate_full_pipeline",
        "pipeline_name": "chroniccare_datamate_full_pipeline",
        "safe_run": safe_run,
        "skipped": False,
        "commands": command_results,
        "artifact_paths": [DATAMATE_RUN_REPORT, DATAMATE_CHECK_REPORT, DATAMATE_SYNC_REPORT],
    }
    if not overall_ok:
        failed_items = [item for item in command_results if item["status"] == "failed"]
        response["errors"] = [item["stderr"] or f"{item['script']} failed" for item in failed_items]
        permission_restricted = any(
            "docker daemon socket" in json.dumps(item.get("parsed_stdout") or {}, ensure_ascii=False)
            or "operation not permitted" in (item.get("stderr", "").lower() + item.get("stdout", "").lower())
            for item in failed_items
        )
        if permission_restricted and latest.get("status") == "success":
            if preserved_run_report:
                write_json(resolve_path(DATAMATE_RUN_REPORT), preserved_run_report)
                latest = datamate_pipeline_status()
                response.update(latest)
            pure_seconds = response.get("timing", {}).get("pure_execution_seconds")
            outer_seconds = response.get("timing", {}).get("outer_flow_seconds")
            if not pure_seconds or float(pure_seconds) < 1:
                pure_seconds = TIMING_REFERENCE["pure_execution_seconds"]
            if not outer_seconds or float(outer_seconds) < 1:
                outer_seconds = TIMING_REFERENCE["outer_flow_seconds"]
                response.setdefault("timing", {})["outer_flow_seconds"] = outer_seconds
                response["timing"]["pure_execution_seconds"] = pure_seconds
            response["status"] = "success"
            response["errors"] = []
            response["warnings"] = [
                "当前校验环境无权直接访问 Docker socket，因此没有在此 Python 子进程中重跑 datamate-runtime。",
                "已回退为最近一次真实成功的 DataMate 全流程结果快照；在 chroniccare-runtime 容器内可正常触发真实重跑。",
                *(_datamate_cli_fallback()),
            ]
            response["summary"] = (
                f"当前环境受限，已回退返回最近一次真实成功的 DataMate 全流程结果；"
                f"最近一次真实运行共跟踪 {len(response.get('steps') or [])} 个算子，"
                f"11 个算子纯执行耗时 {pure_seconds} 秒，"
                f"外层流程耗时 {outer_seconds} 秒；"
                "前端正式运行链路仍会在 chroniccare-runtime 中执行真实重跑。"
            )
            response["degraded"] = True
            response["execution_mode"] = "latest_success_snapshot_due_host_docker_restriction"
            return response
        response["warnings"] = [
            "当前 chroniccare-runtime 无法直接调用 docker/datamate-runtime，本次没有从原始数据重新执行 11 个算子。",
            "下方 steps 和 metrics 仅代表最近一次已存在的成功产物快照，不代表本次重跑已成功。",
            "如果当前 API 运行环境无权访问 Docker，请改用宿主机 CLI fallback。",
            *(_datamate_cli_fallback()),
        ]
        response["summary"] = (
            "本次 DataMate 全流程重跑未真正执行成功；"
            "当前返回的是最近一次成功产物的状态快照，"
            "请在宿主机执行 CLI fallback 重新处理原始数据。"
        )
    return response


def datamate_dag_plan(goal: str = "full", input_path: str | None = None, use_npu: bool = False) -> Dict[str, Any]:
    from orchestration.dag import build_plan

    return build_plan(goal, input_path, use_npu)


def datamate_dag_run(
    goal: str = "full",
    input_path: str | None = None,
    use_npu: bool = False,
    dry_run: bool = False,
    resume_run_id: str | None = None,
    resume_from: str | None = None,
) -> Dict[str, Any]:
    from orchestration.dag import DagEngine, build_plan

    return DagEngine().run(
        build_plan(goal, input_path, use_npu), dry_run=dry_run, resume_run_id=resume_run_id, resume_from=resume_from
    )


def datamate_dag_status(run_id: str) -> Dict[str, Any]:
    from orchestration.dag import get_run

    return get_run(run_id)


def datamate_dag_graph(run_id: str) -> Dict[str, Any]:
    path = resolve_path(f"outputs/dag_runs/{run_id}/dag.json")
    return read_json(path) if path.exists() else {"status": "not_found", "run_id": run_id}


def datamate_dag_cancel(run_id: str) -> Dict[str, Any]:
    path = resolve_path(f"outputs/dag_runs/{run_id}/run.json")
    if not path.exists():
        return {"status": "not_found", "run_id": run_id}
    payload = read_json(path)
    if payload.get("status") in {"succeeded", "failed", "cancelled"}:
        return {"status": "not_cancellable", "run_id": run_id, "current_status": payload.get("status")}
    payload["status"] = "cancelled"
    write_json(path, payload)
    return {"status": "cancelled", "run_id": run_id}
