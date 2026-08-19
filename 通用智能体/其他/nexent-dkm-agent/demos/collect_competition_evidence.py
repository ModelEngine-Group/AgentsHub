"""Collect reviewer-facing logs, specs, reports, and result figures.

The collector is intentionally read-only for external services: it probes
Nexent/DataMate availability, runs local demos in dry-run mode, and writes all
evidence into one ignored output folder for handoff or screenshots.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.figure_export import export_all_defense_figures
from src.common.integration import build_integration_report, probe_datamate, probe_nexent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect competition evidence into one folder.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "competition_evidence"),
        help="Root folder for evidence bundles.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional bundle suffix. Defaults to current local timestamp.",
    )
    parser.add_argument(
        "--datamate-url",
        default="http://localhost:18000",
        help="DataMate backend base URL. Use 'none' to skip probing and pass-through.",
    )
    parser.add_argument(
        "--nexent-url",
        default="http://localhost:3000",
        help="Nexent web URL. Use 'none' to skip probing.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI. Use 'none' to skip probing.",
    )
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument(
        "--neo4j-password-file",
        default=None,
        help="Read the Neo4j password from an ignored local file.",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP probe timeout in seconds.")
    parser.add_argument("--command-timeout", type=int, default=180, help="Per-command timeout in seconds.")
    parser.add_argument("--include-pytest", action="store_true", help="Also run full pytest and save the log.")
    parser.add_argument("--include-ruff", action="store_true", help="Also run ruff and save the log.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.neo4j_password = _resolve_optional_password(
        args.neo4j_password,
        args.neo4j_password_file,
    )
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle_dir = Path(args.output_dir) / stamp
    dirs = _make_dirs(bundle_dir)

    _write_json(dirs["probes"] / "datamate_probe.json", _probe_datamate(args.datamate_url, args.timeout))
    _write_json(dirs["probes"] / "nexent_probe.json", probe_nexent(args.nexent_url, timeout=args.timeout))
    _write_json(
        dirs["probes"] / "integration_report.json",
        build_integration_report(datamate_url=args.datamate_url, nexent_url=args.nexent_url, timeout=args.timeout),
    )
    _write_json(
        dirs["probes"] / "neo4j_probe.json",
        _probe_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password),
    )

    _collect_nexent_specs(dirs["specs"], args.datamate_url)
    command_results = _run_evidence_commands(args, dirs)
    _write_json(bundle_dir / "command_results.json", command_results)

    artifacts = _collect_artifacts(dirs)
    _write_json(bundle_dir / "artifact_manifest.json", artifacts)

    _collect_npu_evidence(dirs["logs"])
    figures = export_all_defense_figures(
        output_dir=dirs["figures"],
        task1_quality_report=dirs["generated"] / "benchmarks" / "task1_data_quality.json",
        kg_graph_file=dirs["generated"] / "task2" / "medical_kg.json",
        task3_report_file=dirs["generated"] / "task3" / "task3_analysis_report.json",
        oov_extraction_report=dirs["generated"] / "benchmarks" / "task2_oov_extraction_quality.json",
        nl2sql_report=dirs["generated"] / "benchmarks" / "task3_nl2sql_report.json",
        planner_llm_report=dirs["generated"] / "benchmarks" / "planner_llm_evidence.json",
        pipeline_latency_report=dirs["generated"] / "benchmarks" / "task2_pipeline_latency.json",
        task2_tensor_report=ROOT / "benchmarks" / "reports" / "task2_relation_tensor_ascend_910b2c_xlarge.json",
        task3_tensor_report=ROOT / "benchmarks" / "reports" / "task3_graph_tensor_ascend_910b2c_xlarge.json",
        npu_utilization_reports={
            "task2_xlarge": ROOT / "benchmarks" / "reports" / "task2_relation_tensor_ascend_910b2c_xlarge.json",
            "task3_50k": ROOT / "benchmarks" / "reports" / "task3_graph_tensor_ascend_910b2c_xlarge.json",
        },
    )
    _write_json(bundle_dir / "figure_manifest.json", figures)

    _write_readme(bundle_dir, command_results, artifacts, figures)
    print(json.dumps({"status": "completed", "bundle_dir": str(bundle_dir)}, ensure_ascii=False, indent=2))
    return 0 if all(item["returncode"] == 0 for item in command_results) else 1


def _make_dirs(bundle_dir: Path) -> dict[str, Path]:
    dirs = {
        "logs": bundle_dir / "logs",
        "specs": bundle_dir / "nexent_specs",
        "probes": bundle_dir / "integration_probes",
        "generated": bundle_dir / "generated_outputs",
        "artifacts": bundle_dir / "artifacts",
        "figures": bundle_dir / "figures",
        "screenshots": bundle_dir / "screenshots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _probe_datamate(base_url: str, timeout: float) -> dict[str, Any]:
    return probe_datamate(base_url, timeout=timeout)


def _probe_neo4j(
    uri: str,
    user: str,
    password: str | None,
) -> dict[str, Any]:
    if uri.lower() == "none":
        return {"status": "skipped", "reason": "neo4j-uri=none"}
    if not password:
        return {
            "status": "credentials_missing",
            "uri": uri,
            "message": "Provide --neo4j-password-file for a live Neo4j probe.",
        }
    try:
        from src.operators.kg_ops.neo4j_store import check_neo4j_connection, neo4j_to_graph
    except ImportError as exc:
        return {"status": "driver_unavailable", "uri": uri, "message": str(exc)}
    connection = check_neo4j_connection(uri, user, password)
    if connection.get("status") != "connected":
        return {"status": "unavailable", "uri": uri, "connection": connection}
    graph = neo4j_to_graph(uri, user, password)
    return {
        "status": "available",
        "uri": uri,
        "connection": connection,
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
        "browser_url": "http://localhost:7474",
    }


def _probe_url(name: str, url: str, timeout: float) -> dict[str, Any]:
    if url.lower() == "none":
        return {"status": "skipped", "reason": f"{name}-url=none"}
    try:
        request = Request(url, headers={"Accept": "text/html,application/json"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read(800).decode("utf-8", errors="replace")
            return {
                "status": "available",
                "url": url,
                "final_url": response.url,
                "http_status": response.status,
                "body_snippet": body,
            }
    except HTTPError as exc:
        body = exc.read(800).decode("utf-8", errors="replace")
        return {
            "status": "http_error",
            "url": url,
            "http_status": exc.code,
            "message": str(exc),
            "body_snippet": body,
        }
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        return {"status": "unavailable", "url": url, "message": str(exc)}


def _collect_npu_evidence(log_dir: Path) -> None:
    """Snapshot Ascend NPU info and utilization via ``npu-smi`` (best-effort).

    The snapshot is purely diagnostic and degrades gracefully when ``npu-smi``
    is unavailable (e.g. when collecting evidence off the NPU server), so the
    bundle never fabricates NPU activity.
    """

    snapshots = [
        ("npu_smi_info.log", ["npu-smi", "info"]),
        ("npu_smi_version.log", ["npu-smi", "info", "-t", "board", "-i", "0"]),
    ]
    for filename, command in snapshots:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            body = completed.stdout if completed.returncode == 0 else (
                completed.stderr or "npu-smi unavailable on this host"
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            body = f"npu-smi unavailable on this host: {exc}"
        (log_dir / filename).write_text(body, encoding="utf-8")


def _collect_nexent_specs(spec_dir: Path, datamate_url: str) -> None:
    specs = [
        ("task1_agent_spec.json", [sys.executable, "demos/task1_nexent_spec.py", "--model-name", "main_model"]),
        ("task2_agent_spec.json", [sys.executable, "demos/task2_nexent_spec.py", "--model-name", "main_model"]),
        ("task3_agent_spec.json", [sys.executable, "demos/task3_nexent_spec.py", "--model-name", "main_model"]),
        ("dkm_suite_agent_spec.json", [sys.executable, "demos/dkm_nexent_spec.py", "--model-name", "main_model"]),
    ]
    if datamate_url.lower() != "none":
        specs[0][1].extend(["--datamate-url", datamate_url])
    for filename, command in specs:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_subprocess_env(),
            check=False,
        )
        target = spec_dir / filename
        target.write_text(completed.stdout if completed.returncode == 0 else completed.stderr, encoding="utf-8")


def _run_evidence_commands(args: argparse.Namespace, dirs: dict[str, Path]) -> list[dict[str, Any]]:
    datamate_url = args.datamate_url if args.datamate_url.lower() != "none" else "none"
    generated = dirs["generated"]
    commands: list[tuple[str, list[str]]] = [
        (
            "task1_demo",
            [
                sys.executable,
                "demos/task1_demo.py",
                "--output-dir",
                str(generated / "task1"),
                "--datamate-url",
                datamate_url,
                "--datamate-mode",
                "dry_run",
            ],
        ),
        (
            "task2_demo",
            [sys.executable, "demos/task2_demo.py", "--output-dir", str(generated / "task2")],
        ),
        (
            "task3_demo",
            [
                sys.executable,
                "demos/task3_demo.py",
                "--graph-file",
                str(generated / "task2" / "medical_kg.json"),
                "--output-dir",
                str(generated / "task3"),
            ],
        ),
        (
            "end_to_end_demo",
            [
                sys.executable,
                "demos/end_to_end_demo.py",
                "--output-root",
                str(generated / "end_to_end"),
                "--datamate-url",
                datamate_url,
                "--datamate-mode",
                "dry_run",
                "--datamate-timeout",
                str(args.timeout),
            ],
        ),
        (
            "dkm_orchestrator_plan_demo",
            [
                sys.executable,
                "demos/dkm_orchestrator_demo.py",
                "--plan-only",
                "--request",
                "请清洗医疗文本，构建知识图谱并生成分析洞察",
            ],
        ),
        (
            "dkm_orchestrator_execute_evidence_demo",
            [
                sys.executable,
                "demos/dkm_orchestrator_execute_evidence_demo.py",
                "--output-dir",
                str(generated / "dkm_orchestrator"),
                "--output",
                str(generated / "benchmarks" / "dkm_orchestrator_execute_evidence.json"),
                "--datamate-url",
                "none",
            ],
        ),
        (
            "task1_datamate_hybrid_evidence_demo",
            [
                sys.executable,
                "demos/task1_datamate_hybrid_evidence_demo.py",
                "--output",
                str(generated / "benchmarks" / "task1_datamate_hybrid_evidence.json"),
            ],
        ),
        (
            "planner_comparison_demo",
            [
                sys.executable,
                "demos/planner_comparison_demo.py",
                "--output",
                str(generated / "benchmarks" / "planner_comparison.json"),
            ],
        ),
        (
            "planner_llm_evidence_demo",
            [
                sys.executable,
                "demos/planner_llm_evidence_demo.py",
                "--llm",
                "--output",
                str(generated / "benchmarks" / "planner_llm_evidence.json"),
            ],
        ),
        (
            "dkm_nexent_toolchain_demo",
            [
                sys.executable,
                "demos/dkm_nexent_toolchain_demo.py",
                "--output-dir",
                str(generated / "nexent_toolchain"),
                "--datamate-url",
                "none",
            ],
        ),
        (
            "task1_data_quality_benchmark",
            [
                sys.executable,
                "benchmarks/task1_data_quality_benchmark.py",
                "--report",
                str(generated / "benchmarks" / "task1_data_quality.json"),
            ],
        ),
        (
            "task3_nl2sql_benchmark",
            [
                sys.executable,
                "benchmarks/task3_nl2sql_benchmark.py",
                "--report",
                str(generated / "benchmarks" / "task3_nl2sql_report.json"),
            ],
        ),
        (
            "task2_extraction_quality_benchmark",
            [
                sys.executable,
                "benchmarks/task2_extraction_quality_benchmark.py",
                "--report",
                str(generated / "benchmarks" / "task2_kg_extraction_quality.json"),
            ],
        ),
        (
            "task2_relation_quality_benchmark",
            [
                sys.executable,
                "benchmarks/task2_relation_quality_benchmark.py",
                "--backend",
                "rule",
                "--report",
                str(generated / "benchmarks" / "task2_relation_quality.json"),
            ],
        ),
        (
            "task2_oov_extraction_benchmark",
            [
                sys.executable,
                "benchmarks/task2_oov_extraction_benchmark.py",
                "--report",
                str(generated / "benchmarks" / "task2_oov_extraction_quality.json"),
            ],
        ),
        (
            "task2_pipeline_latency_benchmark",
            [
                sys.executable,
                "benchmarks/task2_pipeline_latency_benchmark.py",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "--report",
                str(generated / "benchmarks" / "task2_pipeline_latency.json"),
            ],
        ),
        (
            "service_reachability_probe",
            [
                sys.executable,
                "benchmarks/service_reachability_probe.py",
                "--host-label",
                "local_evidence_host",
                "--neo4j-uri",
                args.neo4j_uri,
                "--datamate-url",
                datamate_url,
                "--nexent-url",
                args.nexent_url if args.nexent_url.lower() != "none" else "none",
                "--report",
                str(generated / "benchmarks" / "service_reachability.json"),
            ],
        ),
    ]
    if args.neo4j_uri.lower() != "none" and args.neo4j_password:
        commands.append(
            (
                "task2_neo4j_live_smoke",
                [
                    sys.executable,
                    "demos/task2_neo4j_live_smoke.py",
                    "--uri",
                    args.neo4j_uri,
                    "--user",
                    args.neo4j_user,
                    "--password-stdin",
                    "--skip-pipeline",
                    "--report",
                    str(generated / "benchmarks" / "task2_neo4j_live_smoke.json"),
                ],
            )
        )
    if args.include_ruff:
        commands.append(("ruff", [sys.executable, "-m", "ruff", "check", "."]))
    if args.include_pytest:
        commands.append(("pytest", [sys.executable, "-m", "pytest", "-q"]))

    results = []
    for name, command in commands:
        if name == "task2_neo4j_live_smoke":
            results.append(
                _run_command(
                    name,
                    command,
                    dirs["logs"],
                    args.command_timeout,
                    stdin_text=f"{args.neo4j_password}\n",
                    sensitive_values=(args.neo4j_password,),
                )
            )
        else:
            results.append(
                _run_command(name, command, dirs["logs"], args.command_timeout)
            )
    return results


def _resolve_optional_password(
    password: str | None,
    password_file: str | None,
) -> str | None:
    if password and password_file:
        raise ValueError(
            "Provide either --neo4j-password or --neo4j-password-file, not both."
        )
    if password_file:
        resolved = Path(password_file).read_text(encoding="utf-8").strip()
        if not resolved:
            raise ValueError("Neo4j password file must not be empty.")
        return resolved
    return password


def _run_command(
    name: str,
    command: list[str],
    log_dir: Path,
    timeout: int,
    *,
    stdin_text: str | None = None,
    sensitive_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_subprocess_env(),
            input=stdin_text,
            timeout=timeout,
            check=False,
        )
        duration = round(time.perf_counter() - started, 3)
        result = {
            "name": name,
            "command": _redact_command(command, sensitive_values),
            "returncode": completed.returncode,
            "duration_sec": duration,
            "log_path": str(log_dir / f"{name}.log"),
        }
        _write_command_log(
            log_dir / f"{name}.log",
            result,
            completed.stdout,
            completed.stderr,
            sensitive_values=sensitive_values,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        duration = round(time.perf_counter() - started, 3)
        result = {
            "name": name,
            "command": _redact_command(command, sensitive_values),
            "returncode": 124,
            "duration_sec": duration,
            "log_path": str(log_dir / f"{name}.log"),
            "message": f"Timed out after {timeout} seconds.",
        }
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr)
        _write_command_log(
            log_dir / f"{name}.log",
            result,
            stdout,
            stderr,
            sensitive_values=sensitive_values,
        )
        return result


def _write_command_log(
    path: Path,
    result: dict[str, Any],
    stdout: str,
    stderr: str,
    *,
    sensitive_values: tuple[str, ...] = (),
) -> None:
    path.write_text(
        "COMMAND\n"
        + " ".join(result["command"])
        + "\n\nRESULT\n"
        + json.dumps({k: v for k, v in result.items() if k != "command"}, ensure_ascii=False, indent=2)
        + "\n\nSTDOUT\n"
        + _redact_text(stdout, sensitive_values)
        + "\n\nSTDERR\n"
        + _redact_text(stderr, sensitive_values),
        encoding="utf-8",
    )


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _redact_command(command: list[str], sensitive_values: tuple[str, ...]) -> list[str]:
    return [_redact_text(part, sensitive_values) for part in command]


def _redact_text(text: str, sensitive_values: tuple[str, ...]) -> str:
    redacted = text
    for value in sensitive_values:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _collect_artifacts(dirs: dict[str, Path]) -> list[dict[str, str]]:
    generated = dirs["generated"]
    artifact_dir = dirs["artifacts"]
    candidates = [
        generated / "task1" / "task1_patients_cleaned.csv",
        generated / "task2" / "medical_kg.json",
        generated / "task3" / "task3_analysis_report.json",
        generated / "task3" / "task3_insight_report.md",
        generated / "task3" / "task3_insight_report.html",
        generated / "task3" / "task3_analysis_dashboard.html",
        generated / "task3" / "task3_interactive_dashboard.html",
        generated / "end_to_end" / "task3" / "task3_insight_report.html",
        generated / "end_to_end" / "task3" / "task3_analysis_dashboard.html",
        generated / "benchmarks" / "task1_data_quality.json",
        generated / "benchmarks" / "task3_nl2sql_report.json",
        generated / "benchmarks" / "task2_kg_extraction_quality.json",
        generated / "benchmarks" / "task2_oov_extraction_quality.json",
        generated / "benchmarks" / "task2_pipeline_latency.json",
        generated / "benchmarks" / "task2_relation_quality.json",
        generated / "benchmarks" / "planner_comparison.json",
        generated / "benchmarks" / "planner_llm_evidence.json",
        generated / "nexent_toolchain" / "nexent_toolchain_evidence.json",
        generated / "benchmarks" / "dkm_orchestrator_execute_evidence.json",
        generated / "benchmarks" / "task1_datamate_hybrid_evidence.json",
        generated / "benchmarks" / "planner_llm_evidence.json",
        generated / "benchmarks" / "task2_neo4j_live_smoke.json",
        ROOT / "benchmarks" / "reports" / "task1_data_quality.json",
        ROOT / "benchmarks" / "reports" / "task2_neo4j_live_smoke.json",
        ROOT / "benchmarks" / "reports" / "task2_kg_extraction_quality.json",
        ROOT / "benchmarks" / "reports" / "task2_relation_quality_rule.json",
        ROOT / "benchmarks" / "reports" / "task2_relation_quality_ascend_910b2c_npu.json",
        ROOT / "benchmarks" / "reports" / "task2_relation_tensor_ascend_910b2c_xlarge.json",
        ROOT / "benchmarks" / "reports" / "task3_graph_tensor_ascend_910b2c_xlarge.json",
        ROOT / "benchmarks" / "reports" / "service_reachability_ascend_910b2c.json",
        ROOT / "benchmarks" / "reports" / "ascend_910b2c_experiment_summary.md",
    ]
    copied = []
    for source in candidates:
        if not source.exists():
            continue
        target = artifact_dir / source.name
        if target.exists():
            target = artifact_dir / f"{source.parent.name}_{source.name}"
        shutil.copy2(source, target)
        copied.append({"source": str(source), "target": str(target)})
    return copied


def _write_readme(
    bundle_dir: Path,
    command_results: list[dict[str, Any]],
    artifacts: list[dict[str, str]],
    figures: list[dict[str, str]],
) -> None:
    passed = sum(1 for item in command_results if item["returncode"] == 0)
    lines = [
        "# Competition Evidence Bundle",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Commands passed: {passed}/{len(command_results)}",
        f"- Copied artifacts: {len(artifacts)}",
        f"- SVG result figures: {len(figures)}",
        "",
        "## Folders",
        "",
        "- `logs/`: stdout/stderr and exit code for each demo or benchmark, plus `npu_smi_*.log` NPU snapshots.",
        "- `nexent_specs/`: task agent specs ready for Nexent registration review.",
        "- `integration_probes/`: Nexent/DataMate/Neo4j probe results, including unavailable reasons.",
        "- `screenshots/`: optional答辩截图（Neo4j Browser 等），需手动放入。",
        "- `artifacts/`: copied HTML/Markdown/JSON outputs for handoff (含关系级质量、NPU 超大负载能效报告).",
        "- `figures/`: architecture/workflow diagrams, task-1 quality, task-2 KG, task-3 SVG charts, "
        "and NPU mode-speedup / utilization figures.",
        "",
        "## Safety",
        "",
        "The collector uses DataMate dry-run mode and does not submit cleaning templates or tasks.",
        "Neo4j evidence collection uses --skip-pipeline and only performs connection/read/query checks.",
    ]
    (bundle_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
