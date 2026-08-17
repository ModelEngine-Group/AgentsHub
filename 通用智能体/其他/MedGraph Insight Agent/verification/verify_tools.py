from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
MCP_ROOT = AGENT_ROOT / "mcps" / "medgraph-insight-agent"
sys.path.insert(0, str(MCP_ROOT / "src"))

from medgraph_agent.integrations.mcp_server import TOOLS, call_tool  # noqa: E402


def payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def main() -> None:
    source = AGENT_ROOT / "knowledge" / "medical_cases.jsonl"
    assert source.is_file()
    assert len(TOOLS) == 6

    with tempfile.TemporaryDirectory(prefix="medgraph-agent-") as temp_dir:
        os.environ["MEDGRAPH_SOURCE"] = str(source)
        os.environ["MEDGRAPH_OUTPUT_DIR"] = temp_dir

        run = payload(call_tool("run_pipeline", {"task": "构建医疗知识图谱", "source": str(source), "output_dir": temp_dir}))
        assert run["status"] == "succeeded"
        graph = payload(call_tool("query_graph", {"output_dir": temp_dir}))
        assert graph["stats"]["relation_count"] > 0
        answer = payload(call_tool("answer_medical_question", {"question": "高血压有哪些症状？", "output_dir": temp_dir}))
        assert answer["evidence"]
        analysis = payload(call_tool("run_analysis", {"question": "统计关系类型分布", "output_dir": temp_dir}))
        assert analysis["rows"]
        benchmark = payload(call_tool("get_benchmark", {"source": str(source), "repeat": 1}))
        cpu = next(item for item in benchmark["results"] if item["backend"] == "cpu")
        assert cpu["available"] is True
        quality = payload(call_tool("get_quality_report", {"output_dir": temp_dir}))
        assert quality["passed"] is True

    print("verified_tools=6")


if __name__ == "__main__":
    main()
