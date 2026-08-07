from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from medgraph_agent.cli import default_output_dir, default_source, ensure_graph
from medgraph_agent.core.analytics import GraphAnalyzer
from medgraph_agent.core.benchmark import run_benchmarks
from medgraph_agent.core.pipeline import PipelineRunner
from medgraph_agent.core.quality import audit_graph
from medgraph_agent.core.qa import answer_question
from medgraph_agent.core.storage import load_graph_json
from medgraph_agent.core.models import to_dict


TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_pipeline",
        "description": "Run the full medical data processing and knowledge graph pipeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "source": {"type": "string"},
                "output_dir": {"type": "string"},
            },
        },
    },
    {
        "name": "query_graph",
        "description": "Return the latest graph and graph statistics.",
        "inputSchema": {"type": "object", "properties": {"output_dir": {"type": "string"}}},
    },
    {
        "name": "answer_medical_question",
        "description": "Answer a medical question using graph evidence.",
        "inputSchema": {
            "type": "object",
            "required": ["question"],
            "properties": {"question": {"type": "string"}, "output_dir": {"type": "string"}},
        },
    },
    {
        "name": "run_analysis",
        "description": "Run NL2SQL-style graph analysis and return chart data.",
        "inputSchema": {
            "type": "object",
            "required": ["question"],
            "properties": {"question": {"type": "string"}, "output_dir": {"type": "string"}},
        },
    },
    {
        "name": "get_benchmark",
        "description": "Run or return CPU/CUDA/NPU benchmark and availability information.",
        "inputSchema": {
            "type": "object",
            "properties": {"source": {"type": "string"}, "repeat": {"type": "integer"}},
        },
    },
    {
        "name": "get_quality_report",
        "description": "Audit graph integrity, schema validity, evidence coverage, alias normalization, and known false triples.",
        "inputSchema": {"type": "object", "properties": {"output_dir": {"type": "string"}}},
    },
]


def _content(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(to_dict(payload), ensure_ascii=False, indent=2)}]}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(arguments.get("output_dir") or default_output_dir())
    if name == "run_pipeline":
        source = arguments.get("source") or str(default_source())
        task = arguments.get("task") or "构建医疗数据处理、知识图谱问答和图谱分析闭环"
        return _content(PipelineRunner(output_dir).run(task, source))
    if name == "query_graph":
        graph = load_graph_json(ensure_graph(output_dir, default_source()))
        return _content({"graph": graph, "stats": graph.stats()})
    if name == "answer_medical_question":
        graph = load_graph_json(ensure_graph(output_dir, default_source()))
        return _content(answer_question(arguments["question"], graph))
    if name == "run_analysis":
        ensure_graph(output_dir, default_source())
        return _content(GraphAnalyzer(output_dir / "medgraph.db").analyze(arguments["question"]))
    if name == "get_benchmark":
        return _content(run_benchmarks(arguments.get("source") or default_source(), repeat=int(arguments.get("repeat", 20))))
    if name == "get_quality_report":
        graph = load_graph_json(ensure_graph(output_dir, default_source()))
        return _content(audit_graph(graph))
    raise ValueError(f"unknown tool: {name}")


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    msg_id = message.get("id")
    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "medgraph-insight-agent", "version": "0.1.0"},
                },
            }
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
        if method == "tools/call":
            params = message.get("params", {})
            result = call_tool(params["name"], params.get("arguments", {}))
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"method not found: {method}"}}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle(json.loads(line))
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
