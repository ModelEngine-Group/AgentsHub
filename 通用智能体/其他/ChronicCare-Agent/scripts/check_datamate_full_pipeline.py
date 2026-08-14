from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from datamate_full_pipeline_common import (  # noqa: E402
    DATAMATE_OUTPUT_ROOT,
    OFFICIAL_METRICS,
    PROJECT_ROOT,
    SAFETY_NOTE,
    extract_pipeline_metrics,
    load_json,
    now_iso,
    relative_to_project,
    required_pipeline_paths,
    run_command,
    validate_official_metrics,
    write_json,
)
from release_common import load_question_count  # noqa: E402
from runtime_http_common import request_text, safe_request_json  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "outputs" / "release" / "datamate_full_pipeline_check_report.json"
RUN_REPORT_PATH = PROJECT_ROOT / "outputs" / "release" / "datamate_full_pipeline_report.json"
SYNC_REPORT_PATH = PROJECT_ROOT / "outputs" / "release" / "datamate_sync_report.json"
CURRENT_METRICS_PATH = PROJECT_ROOT / "configs" / "current_metrics.json"
GRAPH_SUMMARY_PATH = PROJECT_ROOT / "data" / "graph" / "graph_summary.json"
SQLITE_PATH = PROJECT_ROOT / "data" / "sqlite" / "chroniccare.db"


def nexent_mcp_probe() -> Dict[str, Any]:
    mcp_url = os.getenv("CHRONICCARE_NEXENT_MCP_URL", "http://host.docker.internal:18188")
    code = (
        "import urllib.request;"
        "opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));"
        f"print(opener.open({mcp_url!r}, timeout=10).read().decode()[:300])"
    )
    result = run_command(["docker", "exec", "nexent-runtime", "python", "-c", code])
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    errors: List[str] = []
    warnings: List[str] = []
    expected_nl2sql_question_count = load_question_count()

    output_root = DATAMATE_OUTPUT_ROOT
    required = required_pipeline_paths(output_root) if output_root.exists() else {}
    required_exists = {name: path.exists() for name, path in required.items()}

    for label, path in [
        ("outputs/datamate_full_pipeline", output_root),
        ("outputs/release/datamate_full_pipeline_report.json", RUN_REPORT_PATH),
        ("outputs/release/datamate_sync_report.json", SYNC_REPORT_PATH),
        ("data/graph/graph_summary.json", GRAPH_SUMMARY_PATH),
        ("data/sqlite/chroniccare.db", SQLITE_PATH),
        ("configs/current_metrics.json", CURRENT_METRICS_PATH),
    ]:
        if not path.exists():
            errors.append(f"missing required path: {label}")

    if required and not all(required_exists.values()):
        missing = [name for name, ok in required_exists.items() if not ok]
        errors.append(f"missing required DataMate output artifacts: {', '.join(missing)}")

    metrics = load_json(CURRENT_METRICS_PATH) if CURRENT_METRICS_PATH.exists() else {}
    errors.extend(validate_official_metrics(metrics))

    kg_probe = safe_request_json("http://127.0.0.1:18088/kg/summary")
    examples_probe = safe_request_json("http://127.0.0.1:18088/analysis/open-sql/examples")
    mcp_probe = safe_request_json("http://127.0.0.1:18188")
    trace_probe = safe_request_json("http://127.0.0.1:18188/trace/summary")

    try:
        _, streamlit_status, streamlit_latency = request_text("http://127.0.0.1:18501", method="HEAD")
        streamlit_probe = {"ok": streamlit_status < 400, "status_code": streamlit_status, "latency_ms": streamlit_latency}
    except Exception as exc:
        streamlit_probe = {"ok": False, "error": str(exc)}

    if not kg_probe.get("ok"):
        errors.append(f"kg summary probe failed: {kg_probe.get('error')}")
    else:
        payload = kg_probe["json"]
        quality_total = (payload.get("quality_score") or {}).get("total", payload.get("quality_score_total"))
        if payload.get("node_count") != metrics.get("node_count"):
            errors.append(f"tool_server node_count mismatch: {payload.get('node_count')}")
        if payload.get("edge_count") != metrics.get("edge_count"):
            errors.append(f"tool_server edge_count mismatch: {payload.get('edge_count')}")
        if quality_total != metrics.get("quality_score_total"):
            errors.append(f"tool_server quality_score_total mismatch: {quality_total}")

    if not examples_probe.get("ok"):
        errors.append(f"open SQL examples probe failed: {examples_probe.get('error')}")
    else:
        example_count = int((examples_probe["json"] or {}).get("example_count", 0) or 0)
        if example_count <= 0:
            errors.append("open SQL examples probe returned no examples")

    if int(metrics.get("question_count", 0) or 0) != expected_nl2sql_question_count:
        errors.append(
            f"nl2sql question count mismatch: pipeline={metrics.get('question_count')} expected={expected_nl2sql_question_count}"
        )

    if not mcp_probe.get("ok"):
        errors.append(f"MCP adapter probe failed: {mcp_probe.get('error')}")
    if not trace_probe.get("ok"):
        errors.append(f"trace summary probe failed: {trace_probe.get('error')}")
    if not streamlit_probe.get("ok"):
        errors.append(f"streamlit probe failed: {streamlit_probe.get('error', streamlit_probe)}")

    nexent_probe = nexent_mcp_probe()
    if not nexent_probe["ok"]:
        errors.append(f"nexent runtime cannot access MCP adapter: {nexent_probe['stderr'] or nexent_probe['stdout']}")

    pipeline_metrics = extract_pipeline_metrics(output_root) if output_root.exists() else {}
    report = {
        "status": "success" if not errors else "failed",
        "timestamp": now_iso(),
        "output_root": relative_to_project(output_root),
        "required_output_exists": required_exists,
        "run_report_exists": RUN_REPORT_PATH.exists(),
        "sync_report_exists": SYNC_REPORT_PATH.exists(),
        "graph_summary_exists": GRAPH_SUMMARY_PATH.exists(),
        "sqlite_exists": SQLITE_PATH.exists(),
        "current_metrics_exists": CURRENT_METRICS_PATH.exists(),
        "current_metrics": metrics,
        "pipeline_metrics": pipeline_metrics,
        "validation_mode": "semantic_consistency",
        "reference_metrics": OFFICIAL_METRICS,
        "tool_server_probes": {
            "kg_summary": kg_probe,
            "open_sql_examples": examples_probe,
            "mcp_root": mcp_probe,
            "trace_summary": trace_probe,
            "streamlit_head": streamlit_probe,
        },
        "nexent_probe": nexent_probe,
        "warnings": warnings,
        "errors": errors,
        "safety_note": SAFETY_NOTE,
    }
    write_json(REPORT_PATH, report)
    if errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print("[OK] DataMate full pipeline integration check passed.")


if __name__ == "__main__":
    main()
