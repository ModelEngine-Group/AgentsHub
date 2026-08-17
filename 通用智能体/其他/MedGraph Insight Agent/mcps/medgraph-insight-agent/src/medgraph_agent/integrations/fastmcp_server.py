from __future__ import annotations

import argparse
import json
from typing import Any

from fastmcp import FastMCP

from medgraph_agent.integrations.mcp_server import call_tool


mcp = FastMCP(
    name="MedGraph Insight Agent",
    instructions=(
        "Offline medical data processing and knowledge-graph analysis tools. "
        "Outputs are for research and data analysis only and must not be treated as medical advice."
    ),
)


def _result(name: str, arguments: dict[str, Any]) -> Any:
    response = call_tool(name, arguments)
    return json.loads(response["content"][0]["text"])


@mcp.tool
def run_pipeline(task: str = "构建医疗数据知识图谱", source: str = "", output_dir: str = "") -> Any:
    """Clean medical records and build the evidence-linked knowledge graph."""
    return _result("run_pipeline", {"task": task, "source": source or None, "output_dir": output_dir or None})


@mcp.tool
def query_graph(output_dir: str = "") -> Any:
    """Return the latest graph and its entity and relation statistics."""
    return _result("query_graph", {"output_dir": output_dir or None})


@mcp.tool
def answer_medical_question(question: str, output_dir: str = "") -> Any:
    """Answer a medical knowledge question using graph evidence; not for diagnosis."""
    return _result("answer_medical_question", {"question": question, "output_dir": output_dir or None})


@mcp.tool
def run_analysis(question: str, output_dir: str = "") -> Any:
    """Run NL2SQL-style graph analysis and return chart-ready rows."""
    return _result("run_analysis", {"question": question, "output_dir": output_dir or None})


@mcp.tool
def get_benchmark(source: str = "", repeat: int = 20) -> Any:
    """Report measured CPU performance and CUDA or Ascend NPU availability."""
    return _result("get_benchmark", {"source": source or None, "repeat": repeat})


@mcp.tool
def get_quality_report(output_dir: str = "") -> Any:
    """Audit schema validity, evidence coverage and graph integrity."""
    return _result("get_quality_report", {"output_dir": output_dir or None})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MedGraph Insight MCP service")
    parser.add_argument("--transport", choices=("stdio", "sse", "http"), default="sse")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
