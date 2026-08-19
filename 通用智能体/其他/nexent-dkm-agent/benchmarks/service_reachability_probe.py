"""Protocol-level reachability probe for Neo4j / DataMate / Nexent.

This verification is deliberately strict and honest:

- **Neo4j** is checked with a real Bolt handshake via
  ``check_neo4j_connection`` (not a bare TCP port check).
- **DataMate** is checked through health plus its database-backed operator,
  template, and task APIs via ``probe_datamate``.
- **Nexent** is checked with an HTTP GET, but a generic HTTP response is not
  treated as proof of Nexent: the ``Server`` header and response body are
  fingerprinted so an unrelated server on the same port (e.g. the Jupyter /
  Tornado IDE backend on :3000) is reported as ``not_nexent`` instead of a
  false ``available``.

Usage:
    python benchmarks/service_reachability_probe.py \\
        --report benchmarks/reports/service_reachability_ascend_910b2c.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.integration import probe_datamate, probe_nexent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol-level service reachability probe.")
    parser.add_argument("--host-label", default="ascend_910b2c_npu_server", help="Label for the host being probed.")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument(
        "--neo4j-password-file",
        default=None,
        help="Read the Neo4j password from an ignored local file.",
    )
    parser.add_argument("--datamate-url", default="http://localhost:18000")
    parser.add_argument("--nexent-url", default="http://localhost:3000")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


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
        from src.operators.kg_ops.neo4j_store import check_neo4j_connection
    except ImportError as exc:
        return {"status": "driver_unavailable", "uri": uri, "message": str(exc)}
    return check_neo4j_connection(uri, user, password)


def _probe_nexent_strict(url: str, timeout: float) -> dict[str, Any]:
    return probe_nexent(url, timeout=timeout)


def _collect_environment_facts() -> dict[str, Any]:
    """Gather runtime and environment facts relevant to the three services.

    Returns a dict with:
    - is_container, environment_type
    - java_installed, node_installed, docker_installed
    - python_version, torch_npu_version (if importable)
    - ram_gb, disk_gb, npu_model (if npu-smi available)
    """

    import shutil
    import platform

    facts: dict[str, Any] = {}

    # Container detection
    facts["is_container"] = Path("/.dockerenv").exists()
    pid1 = Path("/proc/1/comm").read_text(encoding="utf-8").strip() if Path("/proc/1/comm").exists() else ""
    facts["pid1"] = pid1
    if facts["is_container"] or pid1 in ("entrypoint.sh", "docker-init", "containerd"):
        facts["environment_type"] = "npu_compute_container"
    else:
        facts["environment_type"] = "bare_metal_or_vm"

    # Runtime existence
    facts["java_installed"] = shutil.which("java") is not None
    facts["node_installed"] = shutil.which("node") is not None
    facts["docker_installed"] = shutil.which("docker") is not None
    facts["podman_installed"] = shutil.which("podman") is not None

    # Python & torch_npu
    facts["python_version"] = platform.python_version()
    try:
        import torch
        facts["torch_version"] = torch.__version__
        if hasattr(torch, "npu") and torch.npu.is_available():
            facts["torch_npu_available"] = True
            facts["torch_npu_version"] = getattr(torch, "npu_version", "unknown")
        else:
            facts["torch_npu_available"] = False
    except Exception:
        facts["torch_npu_available"] = False

    # NPU info via npu-smi (best-effort)
    try:
        import subprocess
        completed = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=60
        )
        if completed.returncode != 0:
            facts["npu_model"] = "unknown"
        elif "910B2C" in completed.stdout:
            facts["npu_model"] = "Ascend 910B2C"
        else:
            facts["npu_model"] = "Ascend (model not parsed)"
    except Exception:
        facts["npu_model"] = "unknown"

    # RAM and disk
    try:
        import subprocess
        out = subprocess.run(["free", "-g"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                facts["ram_gb"] = int(parts[1])
                break
    except Exception:
        facts["ram_gb"] = "unknown"
    try:
        import subprocess
        out = subprocess.run(["df", "-BG", "/"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.startswith("/"):
                parts = line.split()
                facts["disk_gb"] = int(parts[1].rstrip("G"))
                break
    except Exception:
        facts["disk_gb"] = "unknown"

    return facts


def _build_runtime_analysis(env_facts: dict[str, Any]) -> dict[str, str]:
    available_components = [f"Python {env_facts.get('python_version', 'unknown')}"]
    if env_facts.get("torch_npu_available"):
        available_components.append("torch_npu")
    npu_model = env_facts.get("npu_model")
    if npu_model and npu_model != "unknown":
        available_components.append(str(npu_model))
    if env_facts.get("docker_installed"):
        available_components.append("Docker")
    elif env_facts.get("podman_installed"):
        available_components.append("Podman")

    has_direct_runtime = env_facts.get("java_installed") or env_facts.get("node_installed")
    has_container_runtime = env_facts.get("docker_installed") or env_facts.get("podman_installed")
    if has_direct_runtime:
        conclusion = "存在部分本地运行时，服务状态需以协议探测为准"
    elif has_container_runtime:
        conclusion = "存在容器运行能力，服务状态需以协议探测为准"
    else:
        conclusion = "缺少 Java/Node.js/容器运行时，三服务无法在本节点直接启动"

    return {
        "neo4j_requires": "Java 11+ (JVM) 或容器运行时",
        "datamate_requires": "Java (Spring Boot) 或容器运行时",
        "nexent_requires": "Node.js 或容器运行时",
        "node_has": " + ".join(available_components),
        "conclusion": conclusion,
    }


def main() -> int:
    args = parse_args()
    neo4j_password = args.neo4j_password
    if args.neo4j_password_file:
        if neo4j_password:
            raise ValueError(
                "Provide either --neo4j-password or "
                "--neo4j-password-file, not both."
            )
        neo4j_password = Path(args.neo4j_password_file).read_text(
            encoding="utf-8"
        ).strip()
        if not neo4j_password:
            raise ValueError("Neo4j password file must not be empty.")
    neo4j = _probe_neo4j(args.neo4j_uri, args.neo4j_user, neo4j_password)
    datamate = probe_datamate(args.datamate_url, timeout=args.timeout)
    nexent = _probe_nexent_strict(args.nexent_url, timeout=args.timeout)

    summary = {
        "neo4j": neo4j.get("status"),
        "datamate": datamate.get("status"),
        "nexent": nexent.get("status"),
    }
    reachable = {
        "neo4j": neo4j.get("status") == "connected",
        "datamate": datamate.get("status") == "available",
        "nexent": nexent.get("status") == "available",
    }

    # Environment facts: runtime dependencies for the three services.
    env_facts = _collect_environment_facts()
    runtime_analysis = _build_runtime_analysis(env_facts)

    result = {
        "collected_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "host": args.host_label,
        "environment_type": env_facts.get("environment_type", "unknown"),
        "environment_facts": env_facts,
        "runtime_dependency_analysis": runtime_analysis,
        "method": (
            "protocol-level：Neo4j 走 Bolt 握手(check_neo4j_connection)；"
            "DataMate 走 health 与核心业务 API(probe_datamate)；"
            "Nexent 经 HTTP GET 并用 Server 头+响应体指纹识别 Nexent 或排除非 Nexent 服务；"
            "系统级记录 Java/Node.js/容器运行时可用性"
        ),
        "endpoints": {
            "neo4j": args.neo4j_uri,
            "datamate": args.datamate_url,
            "nexent": args.nexent_url,
        },
        "results": {"neo4j": neo4j, "datamate": datamate, "nexent": nexent},
        "summary": summary,
        "reachable": reachable,
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
