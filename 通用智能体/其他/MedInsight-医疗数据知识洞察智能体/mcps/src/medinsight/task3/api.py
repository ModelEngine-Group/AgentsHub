from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from medinsight.task3.agent import GraphInsightAgent
from medinsight.task3.service import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_GRAPH_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TASK1_REPORT_CANDIDATES,
    DEFAULT_TASK,
    evaluate_nl2sql,
    run_analysis,
    run_configured_analysis,
    run_query,
)


DEFAULT_TASK1_REPORT_PATH = str(DEFAULT_TASK1_REPORT_CANDIDATES[0])


TASK3_OPERATION_IDS = [
    "medinsight_analyze_medical_graph_insights",
    "medinsight_query_medical_graph_analysis",
    "medinsight_run_configured_graph_analysis",
    "medinsight_visualize_medical_graph_insights",
    "medinsight_evaluate_nl2sql_accuracy",
]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if not length:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


class MedInsightTask3APIHandler(BaseHTTPRequestHandler):
    server_version = "MedInsightTask3API/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            json_response(self, 200, {"status": "ok", "service": "medinsight-task3-api"})
            return
        if parsed.path in {"/openapi.json", "/api/nexent/openapi.json", "/api/nexent/task3/openapi.json"}:
            params = parse_qs(parsed.query)
            server_url = params.get("server_url", ["http://host.docker.internal:18082"])[0]
            json_response(self, 200, build_task3_nexent_openapi_spec(server_url))
            return
        if parsed.path in {"/api/nexent/tools", "/api/nexent/task3/tools"}:
            json_response(self, 200, {"status": "ok", "tools": build_task3_tool_catalog()})
            return
        json_response(self, 404, {"status": "error", "message": f"unknown endpoint: {parsed.path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path in {"/api/task3/analyze", "/api/nexent/task3/analyze"}:
                json_response(self, 200, run_task3_analysis_tool(payload))
                return
            if parsed.path in {"/api/task3/query", "/api/nexent/task3/query"}:
                json_response(self, 200, run_task3_query_tool(payload))
                return
            if parsed.path in {"/api/task3/config-analyze", "/api/nexent/task3/config-analyze"}:
                json_response(self, 200, run_task3_configured_analysis_tool(payload))
                return
            if parsed.path in {"/api/task3/visualize", "/api/nexent/task3/visualize"}:
                result = run_task3_analysis_tool(payload)
                json_response(self, 200, {"status": "ok", "visualization": result["result"]["visualization_path"], "result": result["result"]})
                return
            if parsed.path in {"/api/task3/nl2sql-eval", "/api/nexent/task3/nl2sql-eval"}:
                analysis = run_task3_analysis_tool(payload)["result"]
                json_response(self, 200, {"status": "ok", "result": analysis["nl2sql_evaluation"]})
                return
            json_response(self, 404, {"status": "error", "message": f"unknown endpoint: {parsed.path}"})
        except Exception as exc:
            json_response(self, 500, {"status": "error", "message": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(host: str = "127.0.0.1", port: int = 18082) -> None:
    server = ThreadingHTTPServer((host, port), MedInsightTask3APIHandler)
    print(f"MedInsight Task3 API running at http://{host}:{port}")
    print("Nexent OpenAPI: /api/nexent/task3/openapi.json")
    server.serve_forever()


def build_task3_tool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": "medical_graph_insight_agent",
            "operation_id": "medinsight_analyze_medical_graph_insights",
            "description": "只读复用任务一产物和任务二知识图谱，生成图谱 schema、SQLite 分析库、洞察指标、NL2SQL 评测和 BI 可视化。",
            "nexent_endpoint": "/api/nexent/task3/analyze",
        },
        {
            "name": "medical_graph_nl2sql_query_agent",
            "operation_id": "medinsight_query_medical_graph_analysis",
            "description": "将自然语言分析需求转成 SQL，执行 SQL 校验，在图谱分析库上查询，并返回证据路径和子图摘要。",
            "nexent_endpoint": "/api/nexent/task3/query",
        },
        {
            "name": "medical_graph_configured_analysis_agent",
            "operation_id": "medinsight_run_configured_graph_analysis",
            "description": "按 JSON 配置文件批量触发图谱分析任务，生成 SQL 结果、指标报告和 BI 看板。",
            "nexent_endpoint": "/api/nexent/task3/config-analyze",
        },
        {
            "name": "medical_graph_visualization_builder",
            "operation_id": "medinsight_visualize_medical_graph_insights",
            "description": "生成图谱驱动 BI HTML 看板，包含 Agent 计划、指标卡、分布图、知识缺口和证据路径样例。",
            "nexent_endpoint": "/api/nexent/task3/visualize",
        },
        {
            "name": "medical_graph_nl2sql_evaluator",
            "operation_id": "medinsight_evaluate_nl2sql_accuracy",
            "description": "评测 Graph-aware NL2SQL 在内置图谱分析问题集上的准确率，目标不低于 85%。",
            "nexent_endpoint": "/api/nexent/task3/nl2sql-eval",
        },
    ]


def run_task3_analysis_tool(payload: dict[str, Any]) -> dict[str, Any]:
    graph_path = payload.get("graph_path") or None
    output_dir = payload.get("output_dir") or str(DEFAULT_OUTPUT_DIR)
    task = payload.get("task") or DEFAULT_TASK
    task1_report_path = payload.get("task1_report_path") or None
    result = GraphInsightAgent().run_analysis(
        graph_path=graph_path,
        output_dir=output_dir,
        task=task,
        task1_report_path=task1_report_path,
    )
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_analyze_medical_graph_insights",
        "result": result,
    }


def run_task3_query_tool(payload: dict[str, Any]) -> dict[str, Any]:
    graph_path = payload.get("graph_path") or None
    output_dir = payload.get("output_dir") or str(DEFAULT_OUTPUT_DIR)
    question = str(payload.get("question") or "统计实体分布")
    task1_report_path = payload.get("task1_report_path") or None
    result = GraphInsightAgent().ask(
        question=question,
        graph_path=graph_path,
        output_dir=output_dir,
        task1_report_path=task1_report_path,
    )
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_query_medical_graph_analysis",
        "result": result,
    }


def run_task3_configured_analysis_tool(payload: dict[str, Any]) -> dict[str, Any]:
    graph_path = payload.get("graph_path") or None
    output_dir = payload.get("output_dir") or str(DEFAULT_OUTPUT_DIR)
    task1_report_path = payload.get("task1_report_path") or None
    config = payload.get("config") or payload.get("analysis_config") or None
    config_path = payload.get("config_path") or None
    result = run_configured_analysis(
        config=config or config_path,
        graph_path=graph_path,
        output_dir=output_dir,
        task1_report_path=task1_report_path,
    )
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_run_configured_graph_analysis",
        "result": result,
    }


def build_task3_nexent_openapi_spec(server_url: str = "http://host.docker.internal:18082") -> dict[str, Any]:
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "MedInsight Nexent Graph Insight Tools",
            "version": "0.1.0",
            "description": "任务三数据分析智能体的 Nexent 工具契约，可通过 Nexent Outer API Tools 导入并调用。",
        },
        "servers": [{"url": server_url.rstrip("/")}],
        "paths": {
            "/api/nexent/task3/analyze": {
                "post": {
                    "operationId": "medinsight_analyze_medical_graph_insights",
                    "summary": "Analyze Task2 medical knowledge graph",
                    "description": "只读复用任务一产物和任务二知识图谱，生成图谱 schema、SQLite 分析库、洞察指标、NL2SQL 评测和 BI 可视化。",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task3AnalyzeRequest"}}},
                    },
                    "responses": {"200": {"description": "Task3 graph insight analysis result."}},
                }
            },
            "/api/nexent/task3/query": {
                "post": {
                    "operationId": "medinsight_query_medical_graph_analysis",
                    "summary": "Query graph analytics with NL2SQL",
                    "description": "将自然语言数据分析需求映射为 SQL，执行 SQL 校验，并在任务三图谱分析库上返回统计结果、证据路径和子图摘要。",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task3QueryRequest"}}},
                    },
                    "responses": {"200": {"description": "NL2SQL query result."}},
                }
            },
            "/api/nexent/task3/config-analyze": {
                "post": {
                    "operationId": "medinsight_run_configured_graph_analysis",
                    "summary": "Run configured graph analysis tasks",
                    "description": "读取 JSON 配置中的分析任务列表，批量触发图谱指标、SQL 查询和 BI 可视化。",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task3ConfigAnalyzeRequest"}}},
                    },
                    "responses": {"200": {"description": "Configured graph analysis result."}},
                }
            },
            "/api/nexent/task3/visualize": {
                "post": {
                    "operationId": "medinsight_visualize_medical_graph_insights",
                    "summary": "Build graph-driven BI visualization",
                    "description": "生成图谱驱动 BI HTML 看板。",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task3AnalyzeRequest"}}},
                    },
                    "responses": {"200": {"description": "Visualization result."}},
                }
            },
            "/api/nexent/task3/nl2sql-eval": {
                "post": {
                    "operationId": "medinsight_evaluate_nl2sql_accuracy",
                    "summary": "Evaluate NL2SQL accuracy",
                    "description": "评测任务三 Graph-aware NL2SQL 在内置分析问题集上的准确率。",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task3AnalyzeRequest"}}},
                    },
                    "responses": {"200": {"description": "NL2SQL evaluation result."}},
                }
            },
        },
        "components": {
            "schemas": {
                "Task3AnalyzeRequest": {
                    "type": "object",
                    "required": ["graph_path"],
                    "properties": {
                        "graph_path": {
                            "type": "string",
                            "description": "任务二生成的知识图谱 JSON 路径。",
                            "default": str(DEFAULT_GRAPH_PATH),
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "任务三分析产物输出目录。",
                            "default": str(DEFAULT_OUTPUT_DIR),
                        },
                        "task": {
                            "type": "string",
                            "description": "自然语言分析任务。",
                            "default": DEFAULT_TASK,
                        },
                        "task1_report_path": {
                            "type": "string",
                            "description": "可选任务一数据处理报告路径，用于任务三报告展示数据处理复用情况。",
                            "default": DEFAULT_TASK1_REPORT_PATH,
                        },
                    },
                },
                "Task3QueryRequest": {
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "自然语言分析问题。",
                            "default": "统计实体分布",
                        },
                        "graph_path": {
                            "type": "string",
                            "description": "任务二生成的知识图谱 JSON 路径。",
                            "default": str(DEFAULT_GRAPH_PATH),
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "任务三分析产物输出目录。",
                            "default": str(DEFAULT_OUTPUT_DIR),
                        },
                        "task1_report_path": {
                            "type": "string",
                            "description": "可选任务一数据处理报告路径，用于任务三报告展示数据处理复用情况。",
                            "default": DEFAULT_TASK1_REPORT_PATH,
                        },
                    },
                },
                "Task3ConfigAnalyzeRequest": {
                    "type": "object",
                    "required": ["config_path"],
                    "properties": {
                        "config_path": {
                            "type": "string",
                            "description": "配置化分析任务 JSON 文件路径。",
                            "default": str(DEFAULT_CONFIG_PATH),
                        },
                        "graph_path": {
                            "type": "string",
                            "description": "可选任务二知识图谱路径，会覆盖配置文件中的 graph_path。",
                            "default": str(DEFAULT_GRAPH_PATH),
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "任务三分析产物输出目录。",
                            "default": str(DEFAULT_OUTPUT_DIR),
                        },
                        "task1_report_path": {
                            "type": "string",
                            "description": "可选任务一数据处理报告路径。",
                            "default": DEFAULT_TASK1_REPORT_PATH,
                        },
                    },
                },
            }
        },
    }
    return normalize_object_required(spec)


def normalize_object_required(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("type") == "object" and value.get("required") is None:
            value["required"] = []
        for item in value.values():
            normalize_object_required(item)
    elif isinstance(value, list):
        for item in value:
            normalize_object_required(item)
    return value
