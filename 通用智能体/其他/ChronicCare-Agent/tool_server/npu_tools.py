from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from runtime_common.common import PROJECT_ROOT, now_iso, read_json, relative_to_project, write_json
from runtime_common.npu_runtime import detect_npu_runtime, to_markdown_report
from tool_server.pipeline_tools import datamate_pipeline_status
from tool_server.utils import load_server_config, public_artifact_url, safety_note

NPU_EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
NPU_READINESS_REPORT = NPU_EVALUATION_DIR / "npu_readiness_report.json"
NPU_READINESS_MARKDOWN = NPU_EVALUATION_DIR / "npu_readiness_report.md"
NPU_BENCHMARK_REPORT = NPU_EVALUATION_DIR / "npu_operator_benchmark_report.json"
NPU_BENCHMARK_MARKDOWN = NPU_EVALUATION_DIR / "npu_operator_benchmark_report.md"
NPU_PIPELINE_REPORT = NPU_EVALUATION_DIR / "npu_pipeline_report.json"
NPU_PIPELINE_MARKDOWN = NPU_EVALUATION_DIR / "npu_pipeline_report.md"
DATAMATE_NPU_RUN_REPORT = PROJECT_ROOT / "outputs" / "release" / "datamate_npu_pipeline_report.json"
DATAMATE_NPU_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "datamate_full_pipeline_npu"
SUPPORTED_NPU_OPERATOR_NAMES = [
    "chronic_entity_extract_model_npu",
    "chronic_relation_extract_model_npu",
]
NPU_BENCHMARK_REPEAT_COUNT = 5


def _artifact_url(path: Path) -> str:
    cfg = load_server_config()
    return public_artifact_url(cfg, f"/artifacts/{path.name}")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _host_sidecar_path(path_value: Any, output_root: str | None) -> Any:
    if not isinstance(path_value, str) or not output_root:
        return path_value
    prefix = "/tmp/chroniccare_datamate_full_pipeline/output/"
    if path_value.startswith(prefix):
        return f"{output_root.rstrip('/')}/{path_value[len(prefix):]}"
    return path_value


def _round_div(numerator: Any, denominator: Any, digits: int) -> float | None:
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
        if denominator_f == 0:
            return None
        return round(numerator_f / denominator_f, digits)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _enrich_comparison_row(row: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(row)
    cpu_records = enriched.get("cpu_benchmark_records")
    npu_records = enriched.get("npu_record_count")
    cpu_sample_seconds = enriched.get("cpu_bge_sample_seconds") or enriched.get("cpu_bge_seconds")
    npu_sample_seconds = enriched.get("npu_bge_sample_seconds") or enriched.get("npu_same_sample_seconds")
    npu_full_seconds = enriched.get("npu_bge_full_seconds") or enriched.get("npu_bge_seconds")
    estimated_cpu_full_seconds = enriched.get("estimated_cpu_bge_full_seconds") or enriched.get("estimated_cpu_full_seconds")

    enriched.setdefault(
        "cpu_sample_throughput_records_per_second",
        _round_div(cpu_records, cpu_sample_seconds, 2),
    )
    enriched.setdefault(
        "cpu_avg_latency_ms_per_record",
        _round_div(float(cpu_sample_seconds) * 1000.0 if cpu_sample_seconds else None, cpu_records, 4),
    )
    enriched.setdefault(
        "cpu_estimated_full_throughput_records_per_second",
        _round_div(npu_records, estimated_cpu_full_seconds, 2),
    )
    enriched.setdefault(
        "cpu_estimated_avg_latency_ms_per_record",
        _round_div(float(estimated_cpu_full_seconds) * 1000.0 if estimated_cpu_full_seconds else None, npu_records, 4),
    )
    enriched.setdefault(
        "npu_sample_throughput_records_per_second",
        _round_div(cpu_records, npu_sample_seconds, 2),
    )
    enriched.setdefault(
        "npu_sample_avg_latency_ms_per_record",
        _round_div(float(npu_sample_seconds) * 1000.0 if npu_sample_seconds else None, cpu_records, 4),
    )
    enriched.setdefault(
        "npu_full_throughput_records_per_second",
        _round_div(npu_records, npu_full_seconds, 2),
    )
    enriched.setdefault(
        "npu_full_avg_latency_ms_per_record",
        _round_div(float(npu_full_seconds) * 1000.0 if npu_full_seconds else None, npu_records, 4),
    )
    enriched.setdefault("npu_avg_latency_ms_per_record", enriched.get("npu_full_avg_latency_ms_per_record"))
    enriched.setdefault("cpu_resource_utilization_percent", None)
    cpu_util = enriched.get("cpu_resource_utilization_percent")
    enriched.setdefault(
        "cpu_effective_cores",
        round(float(cpu_util) / 100.0, 4) if isinstance(cpu_util, (int, float)) else None,
    )
    enriched.setdefault("cpu_average_power_watt", None)
    enriched.setdefault("cpu_estimated_energy_wh", None)
    enriched.setdefault("cpu_resource_metrics_status", "not_collected")
    enriched.setdefault("resource_utilization_percent", None)
    enriched.setdefault("average_power_watt", None)
    enriched.setdefault("estimated_energy_wh", None)
    enriched.setdefault("resource_metrics_status", "not_collected")
    enriched.setdefault(
        "resource_metrics_note",
        "资源利用率/功耗/能耗需要 CPU 与 NPU 运行中采样；当前未采集时只展示吞吐量和平均单条延迟。",
    )
    return enriched


def _mean_numeric(rows: List[Dict[str, Any]], key: str, digits: int = 6) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
    return round(statistics.fmean(values), digits) if values else None


def _aggregate_comparison_rows(run_rows: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Aggregate independent runs without mixing operators or sample sizes."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rows in run_rows:
        for row in rows:
            operator = str(row.get("operator") or "")
            if operator:
                grouped.setdefault(operator, []).append(dict(row))
    averaged_keys = (
        "cpu_bge_sample_seconds", "npu_bge_sample_seconds", "npu_bge_full_seconds",
        "cpu_rule_seconds", "cpu_resource_utilization_percent",
        "npu_sample_average_power_watt", "npu_sample_max_power_watt",
        "npu_sample_estimated_energy_wh", "npu_sample_resource_utilization_percent",
        "npu_sample_max_resource_utilization_percent", "average_power_watt", "max_power_watt",
        "estimated_energy_wh", "resource_utilization_percent", "max_resource_utilization_percent",
        "average_hbm_used_mb", "max_hbm_used_mb",
    )
    derived_keys = (
        "cpu_sample_throughput_records_per_second", "cpu_avg_latency_ms_per_record",
        "npu_sample_throughput_records_per_second", "npu_sample_avg_latency_ms_per_record",
        "npu_full_throughput_records_per_second", "npu_full_avg_latency_ms_per_record",
        "npu_avg_latency_ms_per_record", "cpu_effective_cores", "sample_speedup", "same_sample_speedup",
    )
    aggregated: List[Dict[str, Any]] = []
    for operator in SUPPORTED_NPU_OPERATOR_NAMES:
        rows = grouped.get(operator, [])
        if not rows:
            continue
        result = dict(rows[-1])
        for key in averaged_keys:
            value = _mean_numeric(rows, key)
            if value is not None:
                result[key] = value
        for key in derived_keys:
            result.pop(key, None)
        speedup = _round_div(result.get("cpu_bge_sample_seconds"), result.get("npu_bge_sample_seconds"), 4)
        result.update({
            "sample_speedup": speedup,
            "same_sample_speedup": speedup,
            "benchmark_repeat_count": len(rows),
            "benchmark_sample_count": result.get("cpu_benchmark_records"),
            "cpu_bge_sample_seconds_runs": [row.get("cpu_bge_sample_seconds") for row in rows],
            "npu_bge_sample_seconds_runs": [row.get("npu_bge_sample_seconds") for row in rows],
            "sample_speedup_runs": [row.get("sample_speedup") for row in rows],
            "npu_bge_full_seconds_runs": [row.get("npu_bge_full_seconds") for row in rows],
            "timing_aggregation": f"arithmetic_mean_of_{len(rows)}_independent_runs",
        })
        aggregated.append(_enrich_comparison_row(result))
    return aggregated


def _npu_operator_comparison_rows(operator_steps: List[Dict[str, Any]], output_root: str | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in operator_steps:
        summary = step.get("summary") or {}
        model = summary.get("model_inference") or {}
        artifacts = step.get("artifact_paths") or {}
        sidecar_path = (
            artifacts.get("npu_entity_standardized")
            or artifacts.get("npu_relation_standardized")
            or artifacts.get("indicator_results")
        )
        cpu_sample_records = model.get("cpu_benchmark_record_count")
        npu_seconds_per_record = model.get("npu_seconds_per_record")
        npu_sample_seconds = model.get("npu_same_sample_total_model_seconds")
        same_sample_timing_source = model.get("same_sample_timing_source")
        cpu_sample_speedup = None
        try:
            if npu_sample_seconds is None and cpu_sample_records and npu_seconds_per_record:
                npu_sample_seconds = round(float(cpu_sample_records) * float(npu_seconds_per_record), 6)
                same_sample_timing_source = "normalized_from_full_run"
        except (TypeError, ValueError):
            npu_sample_seconds = None
        try:
            cpu_total_seconds = model.get("cpu_total_model_seconds") or model.get("cpu_embedding_seconds")
            if cpu_total_seconds and npu_sample_seconds:
                cpu_sample_speedup = round(float(cpu_total_seconds) / float(npu_sample_seconds), 4)
        except (TypeError, ValueError):
            cpu_sample_speedup = None
        npu_record_count = model.get("npu_record_count") or model.get("record_count")
        npu_full_seconds = model.get("npu_total_model_seconds") or model.get("npu_embedding_seconds")
        npu_sample_resource = model.get("npu_sample_resource_metrics") if isinstance(model.get("npu_sample_resource_metrics"), dict) else {}
        npu_full_resource = model.get("npu_full_resource_metrics") if isinstance(model.get("npu_full_resource_metrics"), dict) else {}
        if not npu_full_resource and isinstance(model.get("npu_resource_metrics"), dict):
            npu_full_resource = model.get("npu_resource_metrics")
        npu_resource = npu_full_resource
        row = {
                "operator": step.get("operator"),
                "status": step.get("status"),
                "backend": summary.get("backend"),
                "fallback_used": summary.get("fallback_used"),
                "cpu_rule_seconds": summary.get("cpu_rule_extraction_seconds"),
                "cpu_batch_size": model.get("cpu_batch_size"),
                "cpu_thread_count": model.get("cpu_thread_count"),
                "npu_batch_size": model.get("npu_batch_size") or model.get("batch_size"),
                "warmup_runs_per_device": model.get("warmup_runs_per_device"),
                "warmup_record_count": model.get("warmup_record_count"),
                "cpu_warmup_seconds": model.get("cpu_warmup_seconds"),
                "npu_warmup_seconds": model.get("npu_warmup_seconds"),
                "cpu_benchmark_records": model.get("cpu_benchmark_record_count"),
                "npu_record_count": npu_record_count,
                "cpu_bge_sample_seconds": model.get("cpu_total_model_seconds") or model.get("cpu_embedding_seconds"),
                "npu_bge_sample_seconds": npu_sample_seconds,
                "same_sample_timing_source": same_sample_timing_source,
                "sample_speedup": cpu_sample_speedup,
                "npu_bge_full_seconds": npu_full_seconds,
                "npu_sample_average_power_watt": npu_sample_resource.get("average_power_watt"),
                "npu_sample_max_power_watt": npu_sample_resource.get("max_power_watt"),
                "npu_sample_estimated_energy_wh": npu_sample_resource.get("estimated_energy_wh"),
                "npu_sample_resource_utilization_percent": npu_sample_resource.get("average_aicore_percent"),
                "npu_sample_max_resource_utilization_percent": npu_sample_resource.get("max_aicore_percent"),
                "npu_sample_resource_metrics_status": model.get("npu_sample_resource_metrics_status") or npu_sample_resource.get("status") or "not_collected",
                "npu_sample_resource_metrics_sample_count": npu_sample_resource.get("sample_count"),
                "npu_full_resource_metrics_status": model.get("npu_full_resource_metrics_status") or npu_full_resource.get("status") or "not_collected",
                "average_power_watt": npu_resource.get("average_power_watt"),
                "max_power_watt": npu_resource.get("max_power_watt"),
                "estimated_energy_wh": npu_resource.get("estimated_energy_wh"),
                "resource_utilization_percent": npu_resource.get("average_aicore_percent"),
                "max_resource_utilization_percent": npu_resource.get("max_aicore_percent"),
                "average_hbm_used_mb": npu_resource.get("average_hbm_used_mb"),
                "max_hbm_used_mb": npu_resource.get("max_hbm_used_mb"),
                "resource_metrics_status": model.get("npu_resource_metrics_status") or npu_resource.get("status") or "not_collected",
                "resource_metrics_sample_count": npu_resource.get("sample_count"),
                "resource_metrics_note": "NPU 2048 条与 NPU 全量分别启停 npu-smi 采样器；CPU 侧为进程 CPU time 估算，未采集整机功耗。",
                "cpu_resource_utilization_percent": model.get("cpu_compute_utilization_percent"),
                "cpu_average_power_watt": None,
                "cpu_estimated_energy_wh": None,
                "cpu_resource_metrics_status": model.get("cpu_resource_metrics_status") or "process_time_estimated",
                "cpu_resource_metrics_note": model.get("cpu_resource_metrics_note"),
                "estimated_cpu_bge_full_seconds": model.get("estimated_cpu_full_seconds"),
                "estimated_full_speedup": model.get("estimated_full_speedup") or model.get("model_speedup"),
                "cpu_bge_seconds": model.get("cpu_total_model_seconds") or model.get("cpu_embedding_seconds"),
                "npu_bge_seconds": model.get("npu_total_model_seconds") or model.get("npu_embedding_seconds"),
                "npu_same_sample_seconds": npu_sample_seconds,
                "same_sample_speedup": cpu_sample_speedup,
                "npu_similarity_seconds": model.get("npu_similarity_seconds"),
                "estimated_cpu_full_seconds": model.get("estimated_cpu_full_seconds"),
                "speedup": model.get("estimated_full_speedup") or model.get("model_speedup"),
                "sidecar_path": _host_sidecar_path(sidecar_path, output_root),
                "business_input": model.get("business_input"),
                "business_output": model.get("business_output"),
            }
        rows.append(_enrich_comparison_row(row))
    return rows


def _effective_npu_runtime(operator_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [step.get("summary") or {} for step in operator_steps]
    fallback_used = any(bool(summary.get("fallback_used")) for summary in summaries)
    npu_available = any(bool(summary.get("npu_available")) for summary in summaries)
    npu_execution_used = any(bool(summary.get("npu_execution_used")) for summary in summaries)
    return {
        "backend": (
            "datamate_npu"
            if npu_execution_used
            else "cpu_compat_npu_ready"
            if npu_available
            else "cpu_fallback"
        ),
        "npu_available": npu_available,
        "npu_execution_used": npu_execution_used,
        "fallback_required": fallback_used if operator_steps else True,
    }


def _run_datamate_npu_pipeline_script(fallback: bool = True) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_datamate_full_pipeline.py"),
        "--use-npu",
        "--host-output-root",
        str(DATAMATE_NPU_OUTPUT_ROOT),
        "--report-path",
        str(DATAMATE_NPU_RUN_REPORT),
        "--allow-metric-mismatch",
    ]
    if not fallback:
        command.append("--no-npu-fallback")
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    parsed: Dict[str, Any] = {}
    if completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout.splitlines()[-1])
        except Exception:
            parsed = {}
    if DATAMATE_NPU_RUN_REPORT.exists():
        parsed = _load_if_exists(DATAMATE_NPU_RUN_REPORT) or parsed
    return {
        "status": "success" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "report": parsed,
        "report_path": relative_to_project(DATAMATE_NPU_RUN_REPORT),
        "output_root": relative_to_project(DATAMATE_NPU_OUTPUT_ROOT),
    }


def _benchmark_markdown(report: Dict[str, Any]) -> str:
    repeat_count = int(report.get("benchmark_repeat_count", 1) or 1)
    lines = [
        "# ChronicCare NPU Operator Benchmark",
        "",
        f"- status: {report.get('status')}",
        f"- runtime_backend: {report.get('runtime', {}).get('backend')}",
        f"- npu_available: {report.get('runtime', {}).get('npu_available')}",
        f"- fallback_used: {report.get('fallback_used')}",
        f"- generated_at: {report.get('timestamp')}",
        f"- benchmark_repeat_count: {report.get('benchmark_repeat_count', 1)}",
        f"- npu_physical_device_id: {report.get('benchmark_npu_physical_device_id', 'not_recorded')}",
        "",
    ]
    for row in report.get("npu_comparison_rows", []):
        row = _enrich_comparison_row(row)
        lines.extend(
            [
                f"## {row.get('operator')}",
                "",
                f"| 指标 | CPU（2048 条，{repeat_count}轮均值） | NPU（2048 条，{repeat_count}轮均值） | NPU（全量，{repeat_count}轮均值） |",
                "| --- | ---: | ---: | ---: |",
                f"| 处理量 | {row.get('cpu_benchmark_records')} 条 | {row.get('cpu_benchmark_records')} 条 | {row.get('npu_record_count')} 条 |",
                f"| 执行配置 | {row.get('cpu_thread_count')} 线程 / batch {row.get('cpu_batch_size')} | batch {row.get('npu_batch_size')} | batch {row.get('npu_batch_size')} |",
                "| 正式测量前预热 | 1 次 | 1 次 | 沿用已预热状态 |",
                f"| BGE 实测耗时 | {row.get('cpu_bge_sample_seconds')} 秒 | {row.get('npu_bge_sample_seconds')} 秒 | {row.get('npu_bge_full_seconds')} 秒 |",
                f"| 吞吐量 | {row.get('cpu_sample_throughput_records_per_second')} 条/秒 | {row.get('npu_sample_throughput_records_per_second')} 条/秒 | {row.get('npu_full_throughput_records_per_second')} 条/秒 |",
                f"| 平均单条延迟 | {row.get('cpu_avg_latency_ms_per_record')} ms | {row.get('npu_sample_avg_latency_ms_per_record')} ms | {row.get('npu_full_avg_latency_ms_per_record')} ms |",
                f"| 2048 条同样本加速比 | 1.00x | {row.get('sample_speedup')}x | 不适用 |",
                f"| 重复测量次数 | {row.get('benchmark_repeat_count', 1)} | {row.get('benchmark_repeat_count', 1)} | {row.get('benchmark_repeat_count', 1)} |",
                f"| 规则召回耗时 | {row.get('cpu_rule_seconds')} 秒 | 不适用 | 不适用 |",
                f"| 资源利用率 | 约 {row.get('cpu_effective_cores')} 核等效 | AICore {row.get('npu_sample_resource_utilization_percent') if row.get('npu_sample_resource_utilization_percent') is not None else '未采集'} | AICore {row.get('resource_utilization_percent') if row.get('resource_utilization_percent') is not None else '未采集'} |",
                f"| 平均功耗 | {row.get('cpu_average_power_watt') or '未采集'} | {row.get('npu_sample_average_power_watt') if row.get('npu_sample_average_power_watt') is not None else '未采集'} W | {row.get('average_power_watt') if row.get('average_power_watt') is not None else '未采集'} W |",
                f"| 估算能耗 | {row.get('cpu_estimated_energy_wh') or '未采集'} | {row.get('npu_sample_estimated_energy_wh') if row.get('npu_sample_estimated_energy_wh') is not None else '未采集'} Wh | {row.get('estimated_energy_wh') if row.get('estimated_energy_wh') is not None else '未采集'} Wh |",
                "",
                f"### {repeat_count}轮原始耗时",
                "",
                "| 轮次 | CPU 2048 条（秒） | NPU 2048 条（秒） | 同样本加速比 | NPU 全量（秒） |",
                "| ---: | ---: | ---: | ---: | ---: |",
                *[
                    f"| {index + 1} | {cpu} | {npu} | {speedup} | {full} |"
                    for index, (cpu, npu, speedup, full) in enumerate(zip(
                        row.get("cpu_bge_sample_seconds_runs", []),
                        row.get("npu_bge_sample_seconds_runs", []),
                        row.get("sample_speedup_runs", []),
                        row.get("npu_bge_full_seconds_runs", []),
                    ))
                ],
                "",
            ]
        )
    lines.extend(
        [
            "",
            f"说明：每轮均在正式测量前分别完成 CPU 与 NPU 预热；CPU（2048 条）、NPU（2048 条）和 NPU（全量）分别真实计时，共独立执行{repeat_count}轮。表格采用{repeat_count}轮耗时的算术平均值，并以 CPU 平均耗时除以 NPU 平均耗时计算同样本加速比。NPU 使用 batch 1024，2048 条与全量分别启停独立的 npu-smi 采样器。",
            "",
            str(report.get("safety_note") or ""),
        ]
    )
    return "\n".join(lines).strip() + "\n"


def npu_supported_operators() -> Dict[str, Any]:
    cfg = load_server_config()
    supported_operators = [
        {
            "operator": "chronic_entity_extract_model_npu",
            "cpu_operator": "chronic_entity_extract",
            "acceleration_scope": "真实 BGE embedding 生成、实体类型语义标准化、候选过滤打分；保留 CPU 规则实体产物以保证下游一致。",
            "datamate_cpu_path": "/opt/runtime/datamate/ops/mapper/chronic_entity_extract/process.py",
            "datamate_npu_path": "/opt/runtime/datamate/ops/mapper/chronic_entity_extract_model_npu/process.py",
            "fallback_policy": "NPU runtime 不可用时明确失败或按调用参数回退；输出 CPU/NPU 同模型耗时对照。",
        },
        {
            "operator": "chronic_relation_extract_model_npu",
            "cpu_operator": "chronic_relation_extract",
            "acceleration_scope": "真实 BGE embedding 生成、关系类型语义重排、候选过滤打分；保留 CPU 规则关系产物以保证下游一致。",
            "datamate_cpu_path": "/opt/runtime/datamate/ops/mapper/chronic_relation_extract/process.py",
            "datamate_npu_path": "/opt/runtime/datamate/ops/mapper/chronic_relation_extract_model_npu/process.py",
            "fallback_policy": "NPU runtime 不可用时明确失败或按调用参数回退；输出 CPU/NPU 同模型耗时对照。",
        },
    ]
    return {
        "status": "success",
        "supported_operator_count": len(supported_operators),
        "supported_operators": supported_operators,
        "recommended_targets": SUPPORTED_NPU_OPERATOR_NAMES,
        "not_npu_operators_note": "DataMate 主线 11 个 chronic_* 算子是 CPU/通用处理链路，不等于 11 个 NPU 算子；当前 NPU 增强只支持本接口返回的 2 个 BGE 模型增强算子。",
        "safety_note": safety_note(cfg),
    }


def npu_readiness() -> Dict[str, Any]:
    cfg = load_server_config()
    status = detect_npu_runtime()
    payload = {
        "status": "success",
        "timestamp": now_iso(),
        "runtime": status,
        "supported_operators": npu_supported_operators()["supported_operators"],
        "readiness_summary": status.get("reason") or "NPU runtime check completed.",
        "report_path": relative_to_project(NPU_READINESS_REPORT),
        "markdown_report_path": relative_to_project(NPU_READINESS_MARKDOWN),
        "report_url": _artifact_url(NPU_READINESS_REPORT),
        "markdown_report_url": _artifact_url(NPU_READINESS_MARKDOWN),
        "safety_note": safety_note(cfg),
    }
    write_json(NPU_READINESS_REPORT, payload)
    _write_markdown(NPU_READINESS_MARKDOWN, to_markdown_report("ChronicCare NPU Readiness", status))
    return payload


def run_npu_operator_benchmark(use_npu: bool = True, fallback: bool = True) -> Dict[str, Any]:
    cfg = load_server_config()
    datamate_runs = [
        _run_datamate_npu_pipeline_script(fallback=fallback)
        for _ in range(NPU_BENCHMARK_REPEAT_COUNT)
    ] if use_npu else [{"status": "skipped", "report": {}}]
    datamate_npu_run = datamate_runs[-1]
    npu_report = datamate_npu_run.get("report") or {}
    operator_results = [
        step
        for step in npu_report.get("pipeline_steps", [])
        if str(step.get("operator", "")) in SUPPORTED_NPU_OPERATOR_NAMES
    ]
    fallback_used = any(bool((item.get("summary") or {}).get("fallback_used")) for item in operator_results)
    runtime = _effective_npu_runtime(operator_results)
    run_comparison_rows: List[List[Dict[str, Any]]] = []
    benchmark_runs: List[Dict[str, Any]] = []
    errors: List[str] = []
    for run_index, datamate_run in enumerate(datamate_runs, start=1):
        current_report = datamate_run.get("report") or {}
        current_results = [
            step for step in current_report.get("pipeline_steps", [])
            if str(step.get("operator", "")) in SUPPORTED_NPU_OPERATOR_NAMES
        ]
        current_rows = _npu_operator_comparison_rows(current_results, datamate_run.get("output_root"))
        run_comparison_rows.append(current_rows)
        current_errors = list(current_report.get("errors", []) if isinstance(current_report, dict) else [])
        errors.extend(current_errors)
        benchmark_runs.append({
            "run_index": run_index,
            "status": datamate_run.get("status"),
            "timestamp": current_report.get("timestamp"),
            "fallback_used": any(bool((item.get("summary") or {}).get("fallback_used")) for item in current_results),
            "comparison_rows": current_rows,
            "errors": current_errors,
        })
    comparison_rows = _aggregate_comparison_rows(run_comparison_rows)
    report = {
        "status": "success" if all(run.get("status") == "success" for run in datamate_runs) and operator_results else "failed",
        "timestamp": now_iso(),
        "runtime": runtime,
        "use_npu": use_npu,
        "fallback_enabled": fallback,
        "fallback_used": fallback_used,
        "benchmark_repeat_count": NPU_BENCHMARK_REPEAT_COUNT,
        "benchmark_sample_count": 2048,
        "benchmark_npu_physical_device_id": int(os.getenv("CHRONICCARE_NPU_DEVICE_ID", "0") or 0),
        "benchmark_runs": benchmark_runs,
        "operator_results": operator_results,
        "npu_comparison_rows": comparison_rows,
        "summary": {
            "operator_count": len(operator_results),
            "npu_operator_comparison": comparison_rows,
            "speedup_claim": {
                row.get("operator"): row.get("sample_speedup") for row in comparison_rows
            },
            "speedup_explanation": f"同一批 2048 条样本独立执行{NPU_BENCHMARK_REPEAT_COUNT}轮；表中 CPU/NPU 耗时均为算术平均值，加速比=CPU 平均耗时/NPU 平均耗时。",
        },
        "errors": errors,
        "report_path": relative_to_project(NPU_BENCHMARK_REPORT),
        "markdown_report_path": relative_to_project(NPU_BENCHMARK_MARKDOWN),
        "report_url": _artifact_url(NPU_BENCHMARK_REPORT),
        "markdown_report_url": _artifact_url(NPU_BENCHMARK_MARKDOWN),
        "safety_note": safety_note(cfg),
    }
    write_json(NPU_BENCHMARK_REPORT, report)
    _write_markdown(NPU_BENCHMARK_MARKDOWN, _benchmark_markdown(report))
    return report


def npu_benchmark_report() -> Dict[str, Any]:
    if not NPU_BENCHMARK_REPORT.exists():
        return {
            "status": "no_cached_report",
            "message": "No cached NPU benchmark report exists. Call /npu/benchmark to run a new benchmark.",
            "report_path": relative_to_project(NPU_BENCHMARK_REPORT),
            "operator_results": [],
            "npu_comparison_rows": [],
            "fallback_used": None,
        }
    report = _load_if_exists(NPU_BENCHMARK_REPORT)
    rows = [_enrich_comparison_row(row) for row in report.get("npu_comparison_rows", []) if isinstance(row, dict)]
    if rows:
        report["npu_comparison_rows"] = rows
        if isinstance(report.get("summary"), dict):
            report["summary"]["npu_operator_comparison"] = rows
    return report


def _pipeline_markdown(report: Dict[str, Any]) -> str:
    benchmark = report.get("npu_benchmark") or {}
    lines = [
        "# ChronicCare DataMate NPU Enhanced Pipeline",
        "",
        f"- status: {report.get('status')}",
        f"- task_id: {report.get('task_id')}",
        f"- use_npu: {report.get('use_npu')}",
        f"- fallback_used: {benchmark.get('fallback_used')}",
        f"- base_pipeline_status: {report.get('base_pipeline', {}).get('status')}",
        f"- generated_at: {report.get('timestamp')}",
        "",
        "NPU 增强范围覆盖实体候选 BGE 标准化、关系候选 BGE 重排/过滤；NL2SQL 仍使用 CPU/通用主线算子。",
        "",
        str(report.get("safety_note") or ""),
    ]
    return "\n".join(lines).strip() + "\n"


def run_npu_enhanced_pipeline(
    task_id: str,
    *,
    use_npu: bool = True,
    npu_targets: List[str] | None = None,
    fallback: bool = True,
    force: bool = False,
    safe_run: bool = True,
) -> Dict[str, Any]:
    cfg = load_server_config()
    timer = time.perf_counter()
    targets = npu_targets or SUPPORTED_NPU_OPERATOR_NAMES
    cached_report = _load_if_exists(NPU_PIPELINE_REPORT)
    cached_rows = cached_report.get("npu_comparison_rows", []) if isinstance(cached_report, dict) else []
    if not force and cached_report.get("status") == "success" and cached_rows:
        cached_report = dict(cached_report)
        cached_report.update({
            "task_id": task_id,
            "skipped": True,
            "reason": "returned latest completed real NPU full-pipeline report; set force=true only for an explicit forced rerun",
            "cache_source": relative_to_project(NPU_PIPELINE_REPORT),
        })
        return cached_report
    base_pipeline = datamate_pipeline_status()
    if use_npu:
        benchmark = run_npu_operator_benchmark(use_npu=True, fallback=fallback)
        npu_report = _load_if_exists(DATAMATE_NPU_RUN_REPORT)
        datamate_npu_run = {
            "status": benchmark.get("status"),
            "report": npu_report,
            "report_path": relative_to_project(DATAMATE_NPU_RUN_REPORT),
            "output_root": relative_to_project(DATAMATE_NPU_OUTPUT_ROOT),
        }
    else:
        benchmark = {
            "status": "skipped",
            "runtime": {},
            "fallback_used": True,
            "operator_results": [],
            "npu_comparison_rows": [],
            "errors": [],
        }
        npu_report = {}
        datamate_npu_run = {"status": "skipped", "report": {}}
    npu_operator_steps = benchmark.get("operator_results", [])
    comparison_rows = benchmark.get("npu_comparison_rows", [])
    cpu_seconds = (base_pipeline.get("timing") or {}).get("pure_execution_seconds")
    npu_seconds = npu_report.get("pure_execution_seconds")
    report = {
        "status": "success" if base_pipeline.get("status") == "success" and datamate_npu_run.get("status") == "success" else "failed",
        "timestamp": now_iso(),
        "task_id": task_id,
        "pipeline": "chroniccare_datamate_full_pipeline_npu_enhanced",
        "use_npu": use_npu,
        "npu_targets": targets,
        "fallback_enabled": fallback,
        "safe_run": safe_run,
        "force": force,
        "duration_seconds": round(time.perf_counter() - timer, 4),
        "base_pipeline": {
            "status": base_pipeline.get("status"),
            "run_id": base_pipeline.get("run_id"),
            "skipped": base_pipeline.get("skipped"),
            "timing": base_pipeline.get("timing"),
            "source": "latest_status_snapshot",
            "note": f"NPU run requests execute {NPU_BENCHMARK_REPEAT_COUNT} independent benchmark rounds; each round compares the same 2,048 CPU/NPU samples, and formal timings use arithmetic means.",
            "report_path": base_pipeline.get("report_path"),
        },
        "datamate_npu_pipeline": {
            "status": datamate_npu_run.get("status"),
            "returncode": datamate_npu_run.get("returncode"),
            "report_path": datamate_npu_run.get("report_path"),
            "output_root": datamate_npu_run.get("output_root"),
            "timing": {
                "pure_execution_seconds": npu_seconds,
                "pipeline_execution_seconds": npu_report.get("pipeline_execution_seconds"),
                "outer_flow_seconds": npu_report.get("outer_flow_seconds"),
            },
            "operator_steps": npu_operator_steps,
            "stderr": datamate_npu_run.get("stderr"),
        },
        "comparison": {
            "cpu_pure_execution_seconds": cpu_seconds,
            "npu_pure_execution_seconds": npu_seconds,
            "speedup": round(float(cpu_seconds) / float(npu_seconds), 4) if cpu_seconds and npu_seconds and not benchmark.get("fallback_used") else None,
            "speedup_explanation": "fallback_used=true 时不声明 NPU 加速比。",
        },
        "npu_benchmark": benchmark,
        "npu_comparison_rows": comparison_rows,
        "recommended_frontend_table_columns": [
            "指标",
            "CPU",
            "NPU",
            "对比说明",
        ],
        "recommended_frontend_table_layout": "vertical_metric_comparison_per_operator",
        "npu_enhancement_summary": {
            "operator_count": len(npu_operator_steps),
            "comparison_rows": comparison_rows,
            "report_path": datamate_npu_run.get("report_path"),
            "output_root": datamate_npu_run.get("output_root"),
        },
        "report_path": relative_to_project(NPU_PIPELINE_REPORT),
        "markdown_report_path": relative_to_project(NPU_PIPELINE_MARKDOWN),
        "report_url": _artifact_url(NPU_PIPELINE_REPORT),
        "markdown_report_url": _artifact_url(NPU_PIPELINE_MARKDOWN),
        "warnings": [
            "NPU 增强分支不会修改 DataMate CPU 算子代码或覆盖主线产物。",
            "若 runtime 不可用或未配置模型服务，本次运行会明确标记为 CPU fallback。",
        ],
        "errors": [*base_pipeline.get("errors", []), *benchmark.get("errors", [])],
        "safety_note": safety_note(cfg),
    }
    write_json(NPU_PIPELINE_REPORT, report)
    _write_markdown(NPU_PIPELINE_MARKDOWN, _pipeline_markdown(report))
    benchmark_report = {
        "status": benchmark.get("status"),
        "timestamp": report.get("timestamp"),
        "runtime": benchmark.get("runtime"),
        "use_npu": use_npu,
        "fallback_enabled": fallback,
        "fallback_used": benchmark.get("fallback_used"),
        "benchmark_repeat_count": benchmark.get("benchmark_repeat_count"),
        "benchmark_sample_count": benchmark.get("benchmark_sample_count"),
        "benchmark_npu_physical_device_id": benchmark.get("benchmark_npu_physical_device_id"),
        "benchmark_runs": benchmark.get("benchmark_runs", []),
        "operator_results": benchmark.get("operator_results", []),
        "npu_comparison_rows": comparison_rows,
        "summary": {
            "operator_count": len(npu_operator_steps),
            "npu_operator_comparison": comparison_rows,
            "pipeline_duration_seconds": report.get("duration_seconds"),
            "speedup_explanation": f"同一批 2,048 条样本独立执行{NPU_BENCHMARK_REPEAT_COUNT}轮；CPU/NPU耗时均取算术平均值，加速比=CPU平均耗时/NPU平均耗时；NPU全量也按相同轮数独立计时取平均值。",
            "metrics_explanation": "吞吐量=全量记录数/NPU BGE 全量耗时；平均单条延迟=NPU BGE 全量耗时/全量记录数。资源利用率/功耗/能耗需要运行中 npu-smi 采样，本报告未采集时不展示数值。",
        },
        "errors": benchmark.get("errors", []),
        "source_pipeline_report_path": relative_to_project(NPU_PIPELINE_REPORT),
        "report_path": relative_to_project(NPU_BENCHMARK_REPORT),
        "markdown_report_path": relative_to_project(NPU_BENCHMARK_MARKDOWN),
        "report_url": _artifact_url(NPU_BENCHMARK_REPORT),
        "markdown_report_url": _artifact_url(NPU_BENCHMARK_MARKDOWN),
        "safety_note": safety_note(cfg),
    }
    write_json(NPU_BENCHMARK_REPORT, benchmark_report)
    _write_markdown(NPU_BENCHMARK_MARKDOWN, _benchmark_markdown(benchmark_report))
    return report
