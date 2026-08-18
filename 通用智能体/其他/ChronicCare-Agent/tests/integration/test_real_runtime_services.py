from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mcp_adapter import server
from mcp_adapter.chroniccare_http import ChronicCareClient
from orchestration.dag.datamate_runner import RealDataMateRunner
from tool_server import app as tool_app

pytestmark = pytest.mark.real_integration


def _real_integration_enabled() -> bool:
    return os.getenv("CHRONICCARE_REAL_INTEGRATION", "").strip() == "1"


@pytest.fixture(autouse=True)
def require_real_integration() -> None:
    if not _real_integration_enabled():
        pytest.skip("set CHRONICCARE_REAL_INTEGRATION=1 to run deployed-service tests")


@pytest.fixture
def live_base_url() -> str:
    return os.getenv("CHRONICCARE_REAL_BASE_URL", "http://127.0.0.1:18088").rstrip("/")


def test_live_tool_server_read_only_surfaces(live_base_url: str) -> None:
    client = ChronicCareClient(live_base_url, timeout=15, conversation_id="real-integration-read-only")

    health = client.get("/health")
    tools = client.get("/tools")
    data = client.get("/system/data-summary")
    schema = client.get("/analysis/open-sql/schema")
    graph = client.get("/kg/summary")
    readiness = client.get("/npu/readiness")
    operators = client.get("/npu/supported-operators")
    pipeline = client.get("/datamate/pipeline/status")

    assert health["status"] == "ok"
    assert tools["tool_count"] == len(tools["tools"])
    assert data["patient_count"] == 2000
    assert schema["status"] == "success"
    assert graph["node_count"] > 0 and graph["edge_count"] > 0
    assert readiness["status"] == "success"
    assert operators["status"] == "success"
    assert isinstance(pipeline, dict) and pipeline


def test_live_mcp_aggregation_and_json_rpc(live_base_url: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHRONICCARE_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("CHRONICCARE_TRACE_FILE", str(tmp_path / "traces" / "calls.jsonl"))
    monkeypatch.setenv("CHRONICCARE_TRACE_SUMMARY_FILE", str(tmp_path / "traces" / "summary.json"))
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: {
            "tool_server_url": live_base_url,
            "host": "127.0.0.1",
            "port": 18188,
            "transport": "streamable-http",
            "sdk_available": True,
        },
    )

    for tool_name in (
        "chroniccare_health_check",
        "chroniccare_data_summary",
        "chroniccare_datamate_pipelines",
        "chroniccare_datamate_pipeline_status",
        "chroniccare_datamate_pipeline_latest",
        "chroniccare_datamate_pipeline_report",
        "chroniccare_npu_readiness",
        "chroniccare_npu_supported_operators",
        "chroniccare_npu_operator_benchmark",
        "chroniccare_kg_summary",
        "chroniccare_open_sql_schema",
        "chroniccare_open_sql_examples",
        "chroniccare_report_summary",
    ):
        result = server.execute_tool(tool_name, {}, base_url=live_base_url)
        assert result["tool"] == tool_name
        assert isinstance(result["data"], dict)

    app_client = TestClient(server.create_app())
    invoke = app_client.post("/invoke", json={"name": "chroniccare_health_check", "arguments": {}})
    rpc = app_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "real-health",
            "method": "tools/call",
            "params": {"name": "chroniccare_health_check", "arguments": {}},
        },
    )
    assert invoke.status_code == 200
    assert invoke.json()["status"] == "success"
    assert rpc.status_code == 200
    assert rpc.json()["result"]["isError"] is False


def test_real_datamate_file_ingest_isolated(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "patient_profile.csv").write_text(
        "patient_id,age,gender,disease_tags,risk_level,bmi,smoking,drinking\nT0001,61,F,hypertension,high,25.1,no,no\n",
        encoding="utf-8",
    )
    run_id = f"pytest-real-ingest-{os.getpid()}-{int(time.time())}"
    runner = RealDataMateRunner(input_path=str(raw), use_npu=False)
    try:
        result = runner(
            "chronic_file_ingest",
            {
                "run_id": run_id,
                "profile_input_hash": hashlib.sha256(str(raw).encode()).hexdigest(),
                "timeout_seconds": 120,
            },
        )
        assert result["status"] == "success"
        assert result["execution_mode"] == "real_datamate_operator"
        assert result["artifacts"]["manifest"]["exists"] is True
    finally:
        subprocess.run(
            ["docker", "exec", "datamate-runtime", "rm", "-rf", f"/tmp/chroniccare_real_dag/{run_id}"],
            check=False,
            capture_output=True,
        )


def test_real_ascend_npu_tensor_smoke() -> None:
    cann_root = os.getenv("CHRONICCARE_REAL_CANN_ROOT", "").strip()
    code = (
        "import json,torch,torch_npu;"
        "x=torch.tensor([1.0,2.0],device='npu:0');"
        "print('__NPU_SMOKE__'+json.dumps({"
        "'available':bool(torch.npu.is_available()),"
        "'device_count':int(torch.npu.device_count()),"
        "'sum':float(x.sum().cpu())}))"
    )
    discover = (
        f"CANN_ROOT={json.dumps(cann_root)}; "
        "if [ -z \"$CANN_ROOT\" ]; then "
        "ENV_FILE=$(find / -type f -path '*/cann-*/set_env.sh' 2>/dev/null | head -n 1); "
        "CANN_ROOT=$(dirname \"$ENV_FILE\"); fi; "
        "test -n \"$CANN_ROOT\" && test -f \"$CANN_ROOT/set_env.sh\"; "
    )
    shell = (
        discover
        + "source \"$CANN_ROOT/set_env.sh\" >/dev/null 2>&1; "
        + "export LD_LIBRARY_PATH=\"$CANN_ROOT/lib64:$CANN_ROOT/lib64/plugin/opskernel:"
        "$CANN_ROOT/lib64/plugin/nnengine:$CANN_ROOT/tools/aml/lib64:"
        "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:"
        "/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH\"; "
        + f"python3 -c {json.dumps(code)}"
    )
    completed = subprocess.run(
        ["docker", "exec", "datamate-runtime", "bash", "-lc", shell],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    marker = next(
        (
            line.removeprefix("__NPU_SMOKE__")
            for line in completed.stdout.splitlines()
            if line.startswith("__NPU_SMOKE__")
        ),
        None,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert marker is not None
    payload = json.loads(marker)
    assert payload["available"] is True
    assert payload["device_count"] >= 1
    assert payload["sum"] == 3.0


def test_host_npu_smi_is_readable() -> None:
    completed = subprocess.run(
        ["npu-smi", "info"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0
    assert "910B" in completed.stdout


def test_local_tool_server_real_read_only_routes() -> None:
    client = TestClient(tool_app.app)
    routes = (
        "/health",
        "/tools",
        "/artifacts/status",
        "/artifacts/summary",
        "/system/data-summary",
        "/analysis/open-sql/eval",
        "/analysis/open-sql/schema",
        "/analysis/open-sql/examples",
        "/npu/readiness",
        "/npu/supported-operators",
        "/datamate/pipeline/status",
        "/datamate/pipeline/report",
        "/reports/summary",
        "/charts/list",
    )
    payloads = {}
    for route in routes:
        response = client.get(route)
        assert response.status_code == 200, (route, response.text[:500])
        payloads[route] = response.json()
    assert payloads["/system/data-summary"]["patient_count"] == 2000
    assert payloads["/analysis/open-sql/schema"]["status"] == "success"
    assert payloads["/npu/readiness"]["status"] == "success"
    assert payloads["/tools"]["tool_count"] == len(payloads["/tools"]["tools"])
