from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "process_medical_data",
    "build_medical_knowledge_graph",
    "ask_medical_knowledge_graph",
    "analyze_medical_graph",
    "query_medical_graph_analysis",
    "run_data_knowledge_insight_pipeline",
}
AGENT_NAME = "medinsight_data_knowledge_insight_agent"


def request_json(url: str, method: str = "GET", payload: Any = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc
    return json.loads(body) if body else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Register MedInsight MCP and Agent in Nexent")
    parser.add_argument("--config-api", default="http://127.0.0.1:5010")
    parser.add_argument("--mcp-url", default="http://host.docker.internal:18100/sse")
    args = parser.parse_args()
    base = args.config_api.rstrip("/")

    connection = request_json(
        f"{base}/mcp/test-connection",
        "POST",
        {"server_url": args.mcp_url},
    )
    if not connection.get("success"):
        raise RuntimeError(f"Nexent cannot initialize the MCP service: {connection}")

    mcp_list = request_json(f"{base}/mcp/list").get("remote_mcp_server_list", [])
    service = next(
        (item for item in mcp_list if item.get("remote_mcp_server_name") == "medinsight"),
        None,
    )
    if service is None:
        request_json(
            f"{base}/mcp/add",
            "POST",
            {
                "name": "medinsight",
                "server_url": args.mcp_url,
                "description": "MedInsight 医疗数据知识洞察智能体 MCP",
                "source": "local",
                "tags": ["medical", "data-processing", "knowledge-graph", "insight"],
                "enabled": True,
            },
        )

    request_json(f"{base}/tool/scan_tool")
    tools = request_json(f"{base}/tool/list")
    discovered = {
        item.get("name") for item in tools if item.get("usage") == "medinsight"
    }
    missing = EXPECTED_TOOLS - discovered
    if missing:
        raise RuntimeError(f"Nexent did not persist MCP tools: {sorted(missing)}")

    agents = request_json(f"{base}/agent/list")
    existing = next((item for item in agents if item.get("name") == AGENT_NAME), None)
    if existing is None:
        agent = json.loads((ROOT / "agent.json").read_text(encoding="utf-8"))
        request_json(
            f"{base}/agent/import",
            "POST",
            {"agent_info": agent, "force_import": False},
        )
        agents = request_json(f"{base}/agent/list")
        existing = next((item for item in agents if item.get("name") == AGENT_NAME), None)

    result = {
        "status": "ok",
        "mcp_url": args.mcp_url,
        "tool_count": len(EXPECTED_TOOLS),
        "tools": sorted(EXPECTED_TOOLS),
        "agent": existing,
        "next_step": "Select an available tool-calling model in Nexent before chat-based routing.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
