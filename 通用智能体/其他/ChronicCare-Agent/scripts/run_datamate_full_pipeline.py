from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from datamate_full_pipeline_common import (
    CONFIG_ROOT,
    CONTAINER_NAME,
    CONTAINER_WORK_ROOT,
    DATAMATE_OUTPUT_ROOT,
    OFFICIAL_METRICS,
    PROJECT_ROOT,
    SAFETY_NOTE,
    ensure_directory,
    existing_status,
    extract_pipeline_metrics,
    materialized_output_root,
    now_iso,
    now_stamp,
    relative_to_project,
    required_pipeline_paths,
    run_command,
    validate_official_metrics,
    write_json,
)

REPORT_PATH = PROJECT_ROOT / "outputs" / "release" / "datamate_full_pipeline_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ChronicCare full mapper pipeline inside datamate-runtime.")
    parser.add_argument("--container", default=CONTAINER_NAME)
    parser.add_argument("--host-output-root", default=str(DATAMATE_OUTPUT_ROOT))
    parser.add_argument("--container-work-root", default=str(CONTAINER_WORK_ROOT))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--allow-metric-mismatch", action="store_true")
    parser.add_argument("--use-npu", action="store_true", help="Use DataMate NPU-enhanced operators for selected pipeline stages.")
    parser.add_argument("--npu-targets", default="chronic_entity_extract_model_npu,chronic_relation_extract_model_npu")
    parser.add_argument("--no-npu-fallback", action="store_true")
    return parser.parse_args()


def checked_run(args: List[str], *, input_text: str | None = None) -> Dict[str, Any]:
    result = run_command(args, input_text=input_text)
    return {
        "args": args,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "ok": result.returncode == 0,
    }


def datamate_ascend_shell_prefix() -> str:
    configured_root = os.getenv("CHRONICCARE_CANN_ROOT", "").strip()
    npu_device_id = os.getenv("CHRONICCARE_NPU_DEVICE_ID", "").strip()
    quoted_root = json.dumps(configured_root)
    quoted_device_id = json.dumps(npu_device_id)
    return (
        f"CANN_ROOT={quoted_root}; "
        f"NPU_DEVICE_ID={quoted_device_id}; "
        "if [ -z \"$CANN_ROOT\" ]; then "
        "ENV_FILE=$(find / -type f -path '*/cann-*/set_env.sh' 2>/dev/null | head -n 1); "
        "CANN_ROOT=$(dirname \"$ENV_FILE\"); fi; "
        "test -n \"$CANN_ROOT\" && test -f \"$CANN_ROOT/set_env.sh\"; "
        "source \"$CANN_ROOT/set_env.sh\" >/dev/null 2>&1; "
        "export ASCEND_HOME_PATH=\"$CANN_ROOT\"; "
        "export ASCEND_TOOLKIT_HOME=\"$CANN_ROOT\"; "
        "export ASCEND_OPP_PATH=\"$CANN_ROOT/opp\"; "
        "if [ -n \"$NPU_DEVICE_ID\" ]; then "
        "export ASCEND_RT_VISIBLE_DEVICES=\"$NPU_DEVICE_ID\"; "
        "export ASCEND_VISIBLE_DEVICES=\"$NPU_DEVICE_ID\"; "
        "export CHRONICCARE_NPU_PHYSICAL_DEVICE_ID=\"$NPU_DEVICE_ID\"; fi; "
        "export OPENBLAS_NUM_THREADS=\"64\"; "
        "export OMP_NUM_THREADS=\"64\"; "
        "export MKL_NUM_THREADS=\"64\"; "
        "export NUMEXPR_NUM_THREADS=\"64\"; "
        "export PYTHONPATH=\"$CANN_ROOT/python/site-packages:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe:/opt/runtime/datamate/:$PYTHONPATH\"; "
        "export LD_LIBRARY_PATH=\"$CANN_ROOT/lib64:$CANN_ROOT/lib64/plugin/opskernel:$CANN_ROOT/lib64/plugin/nnengine:"
        "$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:"
        "$CANN_ROOT/tools/aml/lib64:$CANN_ROOT/tools/aml/lib64/plugin:"
        "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH\"; "
    )


def container_python(container: str, code: str) -> Dict[str, Any]:
    return checked_run(["docker", "exec", "-i", container, "sh", "-lc", datamate_ascend_shell_prefix() + "python -"], input_text=code)


def ensure_container_workspace(container: str, container_work_root: str) -> Dict[str, Any]:
    code = f"""
from pathlib import Path
import shutil

root = Path({container_work_root!r})
if root.exists():
    shutil.rmtree(root)
(root / "input").mkdir(parents=True, exist_ok=True)
(root / "output").mkdir(parents=True, exist_ok=True)
print(root.as_posix())
"""
    return container_python(container, code)


def copy_inputs_to_container(container: str, container_work_root: str) -> List[Dict[str, Any]]:
    input_root = Path(container_work_root) / "input"
    return [
        checked_run(["docker", "cp", str(PROJECT_ROOT / "data" / "raw" / "."), f"{container}:{(input_root / 'raw').as_posix()}"]),
        checked_run(["docker", "cp", str(CONFIG_ROOT / "."), f"{container}:{(input_root / 'configs').as_posix()}"]),
    ]


def build_container_pipeline_code(container_work_root: str, use_npu: bool = False, npu_fallback: bool = True, npu_targets: List[str] | None = None) -> str:
    input_root = Path(container_work_root) / "input"
    output_root = Path(container_work_root) / "output"
    questions_path = input_root / "configs" / "nl2sql_questions.json"
    current_metrics_path = input_root / "configs" / "current_metrics.json"
    targets = set(npu_targets or [])
    use_entity_npu = use_npu and "chronic_entity_extract_model_npu" in targets
    use_relation_npu = use_npu and "chronic_relation_extract_model_npu" in targets
    entity_import = (
        "from datamate.ops.mapper.chronic_entity_extract_model_npu.process import chronic_entity_extract_model_npu as chronic_entity_extract"
        if use_entity_npu
        else "from datamate.ops.mapper.chronic_entity_extract.process import chronic_entity_extract"
    )
    relation_import = (
        "from datamate.ops.mapper.chronic_relation_extract_model_npu.process import chronic_relation_extract_model_npu as chronic_relation_extract"
        if use_relation_npu
        else "from datamate.ops.mapper.chronic_relation_extract.process import chronic_relation_extract"
    )
    nl2sql_import = "from datamate.ops.mapper.chronic_nl2sql_analyze.process import chronic_nl2sql_analyze"
    entity_step_name = "chronic_entity_extract_model_npu" if use_entity_npu else "chronic_entity_extract"
    relation_step_name = "chronic_relation_extract_model_npu" if use_relation_npu else "chronic_relation_extract"
    common_npu_params = {
        "use_npu": True,
        "fallback": npu_fallback,
        "embedding_model_path": "/models/MedCleanStd/bge-small-zh-v1.5",
        "npu_max_records": 0,
        "cpu_benchmark_records": 2048,
        "model_batch_size": 1024,
        "cpu_model_batch_size": 64,
        "npu_model_batch_size": 1024,
        "model_max_length": 64,
    } if use_npu else {}
    entity_step_params = common_npu_params if use_entity_npu else {}
    relation_step_params = common_npu_params if use_relation_npu else {}
    return f"""
import json
import time
from pathlib import Path

from datamate.ops.mapper.chronic_file_ingest.process import chronic_file_ingest
from datamate.ops.mapper.chronic_table_clean.process import chronic_table_clean
from datamate.ops.mapper.chronic_field_normalize.process import chronic_field_normalize
from datamate.ops.mapper.chronic_text_split.process import chronic_text_split
{entity_import}
{relation_import}
from datamate.ops.mapper.chronic_triple_validate.process import chronic_triple_validate
from datamate.ops.mapper.chronic_kg_build.process import chronic_kg_build
from datamate.ops.mapper.chronic_sqlite_loader.process import chronic_sqlite_loader
{nl2sql_import}
from datamate.ops.mapper.chronic_report_pack.process import chronic_report_pack

steps = [
    ("chronic_file_ingest", chronic_file_ingest(), {{}}),
    ("chronic_table_clean", chronic_table_clean(), {{}}),
    ("chronic_field_normalize", chronic_field_normalize(), {{}}),
    ("chronic_text_split", chronic_text_split(), {{}}),
    ({entity_step_name!r}, chronic_entity_extract(), {entity_step_params!r}),
    ({relation_step_name!r}, chronic_relation_extract(), {relation_step_params!r}),
    ("chronic_triple_validate", chronic_triple_validate(), {{}}),
    ("chronic_kg_build", chronic_kg_build(), {{"current_metrics_path": {current_metrics_path.as_posix()!r}}}),
    ("chronic_sqlite_loader", chronic_sqlite_loader(), {{}}),
    ("chronic_nl2sql_analyze", chronic_nl2sql_analyze(), {{"analysis_questions_path": {questions_path.as_posix()!r}}}),
    ("chronic_report_pack", chronic_report_pack(), {{}}),
]

sample = {{
    "filePath": {str(input_root / 'raw')!r},
    "export_path": {output_root.as_posix()!r},
}}
history = []
pipeline_timer = time.perf_counter()
for name, operator, params in steps:
    step_timer = time.perf_counter()
    sample = operator.execute(sample, params)
    history.append({{
        "operator": name,
        "status": sample.get("status"),
        "execution_seconds": round(time.perf_counter() - step_timer, 4),
        "summary": sample.get("summary"),
        "artifact_paths": sample.get("artifact_paths") or {{}},
        "artifact_keys": sorted((sample.get("artifact_paths") or {{}}).keys()),
        "filePath": sample.get("filePath"),
    }})

report_path = Path({(output_root / 'pipeline_runtime_result.json').as_posix()!r})
report_path.write_text(json.dumps({{
    "status": "success",
    "use_npu": {use_npu!r},
    "pure_execution_seconds": round(sum(float(item.get("execution_seconds", 0) or 0) for item in history), 4),
    "pipeline_execution_seconds": round(time.perf_counter() - pipeline_timer, 4),
    "pipeline_steps": history,
    "final_sample": sample,
}}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({{
    "status": "success",
    "use_npu": {use_npu!r},
    "pure_execution_seconds": round(sum(float(item.get("execution_seconds", 0) or 0) for item in history), 4),
    "pipeline_execution_seconds": round(time.perf_counter() - pipeline_timer, 4),
    "pipeline_steps": history,
    "final_sample": sample,
}}, ensure_ascii=False))
"""


def run_pipeline_in_container(container: str, container_work_root: str, use_npu: bool = False, npu_fallback: bool = True, npu_targets: List[str] | None = None) -> Dict[str, Any]:
    result = container_python(container, build_container_pipeline_code(container_work_root, use_npu=use_npu, npu_fallback=npu_fallback, npu_targets=npu_targets))
    payload: Dict[str, Any] = {}
    if result["ok"] and result["stdout"]:
        try:
            payload = json.loads(result["stdout"].splitlines()[-1])
        except json.JSONDecodeError:
            payload = {}
    result["payload"] = payload
    return result


def copy_outputs_from_container(container: str, container_work_root: str, host_output_root: Path) -> Dict[str, Any]:
    if host_output_root.exists():
        archived = host_output_root.parent / f"{host_output_root.name}_previous_{now_stamp()}"
        if archived.exists():
            shutil.rmtree(archived)
        shutil.move(str(host_output_root), str(archived))
    ensure_directory(host_output_root)
    return checked_run(["docker", "cp", f"{container}:{(Path(container_work_root) / 'output' / '.').as_posix()}", str(host_output_root)])


def expected_pipeline_paths(host_output_root: Path, *, use_npu: bool) -> Dict[str, Path]:
    paths = required_pipeline_paths(host_output_root)
    return paths


def run_npu_extension(args: argparse.Namespace) -> Dict[str, Any]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tool_server.npu_tools import run_npu_operator_benchmark

    targets = [item.strip() for item in str(args.npu_targets or "").split(",") if item.strip()]
    report = run_npu_operator_benchmark(use_npu=True, fallback=not args.no_npu_fallback)
    return {
        "status": report.get("status"),
        "requested_targets": targets,
        "fallback_enabled": not args.no_npu_fallback,
        "fallback_used": report.get("fallback_used"),
        "runtime": report.get("runtime"),
        "report_path": report.get("report_path"),
        "markdown_report_path": report.get("markdown_report_path"),
        "operator_results": report.get("operator_results", []),
        "errors": report.get("errors", []),
    }


def main() -> None:
    args = parse_args()
    overall_timer = time.perf_counter()
    host_output_root = Path(args.host_output_root).resolve()
    report_path = Path(args.report_path).resolve()
    ensure_directory(report_path.parent)

    warnings: List[str] = []
    errors: List[str] = []

    prep = ensure_container_workspace(args.container, args.container_work_root)
    if not prep["ok"]:
        errors.append(f"failed to prepare container workspace: {prep['stderr'] or prep['stdout']}")

    copy_results = copy_inputs_to_container(args.container, args.container_work_root) if not errors else []
    for item in copy_results:
        if not item["ok"]:
            errors.append(f"docker cp failed: {item['stderr'] or item['stdout']}")

    pipeline_run = run_pipeline_in_container(
        args.container,
        args.container_work_root,
        use_npu=args.use_npu,
        npu_fallback=not args.no_npu_fallback,
        npu_targets=[item.strip() for item in str(args.npu_targets or "").split(",") if item.strip()],
    ) if not errors else {"payload": {}}
    if not errors and not pipeline_run["ok"]:
        errors.append(f"container pipeline execution failed: {pipeline_run['stderr'] or pipeline_run['stdout']}")

    copy_output_result = copy_outputs_from_container(args.container, args.container_work_root, host_output_root) if not errors else {}
    if copy_output_result and not copy_output_result["ok"]:
        errors.append(f"failed to copy outputs back to host: {copy_output_result['stderr'] or copy_output_result['stdout']}")

    actual_output_root = materialized_output_root(host_output_root) if host_output_root.exists() else host_output_root
    actual_paths = expected_pipeline_paths(host_output_root, use_npu=args.use_npu) if host_output_root.exists() else {}
    output_exists = existing_status(actual_paths) if actual_paths else {}
    if actual_paths:
        missing = [name for name, exists in output_exists.items() if not exists]
        if missing:
            errors.append(f"missing pipeline outputs: {', '.join(missing)}")

    metrics = extract_pipeline_metrics(host_output_root) if host_output_root.exists() else {
        "node_count": 0,
        "edge_count": 0,
        "quality_score_total": 0,
        "question_count": 0,
    }
    metric_errors = validate_official_metrics(metrics)
    if metric_errors:
        if args.allow_metric_mismatch:
            warnings.extend(metric_errors)
        else:
            errors.extend(metric_errors)

    npu_enhancement = None
    if args.use_npu:
        npu_steps = [
            step
            for step in (pipeline_run.get("payload") or {}).get("pipeline_steps", [])
            if str(step.get("operator", "")).endswith("_npu")
        ]
        npu_enhancement = {
            "status": "success" if npu_steps else "failed",
            "source": "datamate-runtime:/opt/runtime/datamate/ops/mapper",
            "operator_steps": npu_steps,
            "fallback_enabled": not args.no_npu_fallback,
            "fallback_used": any((step.get("summary") or {}).get("fallback_used") for step in npu_steps),
        }
        if not npu_steps:
            errors.append("NPU mode requested but no DataMate NPU operator steps were recorded.")

    report = {
        "status": "success" if not errors else "failed",
        "timestamp": now_iso(),
        "container": args.container,
        "use_npu": args.use_npu,
        "npu_targets": [item.strip() for item in str(args.npu_targets or "").split(",") if item.strip()],
        "pipeline_steps": (pipeline_run.get("payload") or {}).get("pipeline_steps", []),
        "pure_execution_seconds": (pipeline_run.get("payload") or {}).get("pure_execution_seconds"),
        "pipeline_execution_seconds": (pipeline_run.get("payload") or {}).get("pipeline_execution_seconds"),
        "outer_flow_seconds": round(time.perf_counter() - overall_timer, 4),
        "output_root_in_container": str(Path(args.container_work_root) / "output"),
        "output_root_on_host": relative_to_project(actual_output_root),
        "required_outputs": {name: relative_to_project(path) for name, path in actual_paths.items()},
        "required_output_exists": output_exists,
        "metrics": metrics,
        "validation_mode": "semantic_consistency",
        "npu_enhancement": npu_enhancement,
        "reference_metrics": OFFICIAL_METRICS,
        "copy_inputs": copy_results,
        "container_workspace_prepare": prep,
        "container_pipeline_run": {
            "ok": pipeline_run.get("ok", False),
            "stdout_preview": (pipeline_run.get("stdout") or "")[:2000],
            "stderr": pipeline_run.get("stderr", ""),
        },
        "copy_outputs": copy_output_result,
        "host_output_files": sorted(relative_to_project(path) for path in actual_output_root.rglob("*") if path.is_file()) if actual_output_root.exists() else [],
        "warnings": warnings,
        "errors": errors,
        "safety_note": SAFETY_NOTE,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
