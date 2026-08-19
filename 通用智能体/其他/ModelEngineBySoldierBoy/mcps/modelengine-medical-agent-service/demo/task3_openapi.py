"""OpenAPI schema for the Task-3 graph-driven analysis agent API."""

from __future__ import annotations

from typing import Any


TASK3_OPENAPI_SERVICE_NAME = "modelengine_task3_analysis_agent"
TASK3_ANALYZE_OPERATION_ID = "run_analysis_agent"
TASK3_INSIGHT_OPERATION_ID = "run_interactive_analysis"
TASK3_CONFIG_OPERATION_ID = "run_configured_analysis"


def build_task3_openapi_spec(server_url: str = "http://127.0.0.1:8088") -> dict[str, Any]:
    """Build the Nexent-importable OpenAPI contract for Task-3."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "ModelEngine Task-3 AnalysisAgent",
            "version": "1.0.0",
            "description": (
                "Competition Task-3 graph-driven data analysis agent. It reuses Task-1 "
                "processed data and Task-2 medical knowledge graph artifacts, understands "
                "graph schema and relations, executes NL2SQL/statistical/graph analyses, "
                "and returns BI charts, graph visualizations, and narrative insights."
            ),
        },
        "servers": [{"url": server_url.rstrip("/")}],
        "paths": {
            "/api/analyze": {
                "post": {
                    "operationId": TASK3_ANALYZE_OPERATION_ID,
                    "summary": "Run the full Task-3 graph-driven analysis workflow",
                    "description": (
                        "Execute the full AnalysisAgent chain from existing Task-1/Task-2 "
                        "artifacts to graph understanding, NL2SQL evaluation, BI chart "
                        "generation, graph visualization, and closed-loop evidence."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Task3AnalyzeRequest"},
                                "examples": {"default": {"value": {"output_root": "."}}},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Task-3 analysis completed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Task3AnalyzeResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Missing upstream Task-1/Task-2 artifacts",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/api/insight": {
                "post": {
                    "operationId": TASK3_INSIGHT_OPERATION_ID,
                    "summary": "Run natural-language graph analysis and insight generation",
                    "description": (
                        "Map a natural-language analysis request to an NL2SQL/graph query, "
                        "execute it over the generated medical graph, and return rows, chart "
                        "payload, graph context, explanation, trace, and follow-up analysis."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Task3InsightRequest"},
                                "examples": {
                                    "complication_counts": {
                                        "value": {
                                            "output_root": ".",
                                            "question": "统计每种疾病的并发症数量",
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Natural-language insight completed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Task3InsightResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid question or missing upstream artifacts",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/api/analyze-config": {
                "post": {
                    "operationId": TASK3_CONFIG_OPERATION_ID,
                    "summary": "Run a configured Task-3 analysis plan",
                    "description": (
                        "Execute a validated configured analysis plan over Task-1/Task-2 "
                        "artifacts, including selected SQL analyses and graph-analysis tools."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Task3ConfiguredRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Configured analysis completed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Task3ConfiguredResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid configuration or SQL",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "Task3AnalyzeRequest": {
                    "type": "object",
                    "properties": {
                        "output_root": {
                            "type": "string",
                            "default": ".",
                            "description": "Output root containing Task-1 processed data and Task-2 graph artifacts.",
                        }
                    },
                },
                "Task3InsightRequest": {
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural-language analysis question over the generated graph.",
                        },
                        "output_root": {"type": "string", "default": "."},
                    },
                },
                "Task3ConfiguredRequest": {
                    "type": "object",
                    "properties": {
                        "output_root": {"type": "string", "default": "."},
                        "config": {"type": "object", "additionalProperties": True},
                        "config_path": {"type": "string"},
                    },
                },
                "Task3AnalyzeResponse": {
                    "type": "object",
                    "required": [
                        "status",
                        "analysis_summary",
                        "analysis_planning",
                        "nl2sql_eval",
                        "task3_closed_loop_ready",
                        "artifact_paths",
                    ],
                    "properties": {
                        "status": {"type": "string", "enum": ["ok"]},
                        "analysis_summary": {"type": "object", "additionalProperties": True},
                        "analysis_planning": {"type": "object", "additionalProperties": True},
                        "query_count": {"type": "integer"},
                        "subgraph_count": {"type": "integer"},
                        "nl2sql_eval": {"type": "object", "additionalProperties": True},
                        "task3_lineage_ready": {"type": "boolean"},
                        "task3_closed_loop_ready": {"type": "boolean"},
                        "graph_understanding": {"type": "object", "additionalProperties": True},
                        "visualization_manifest": {"type": "object", "additionalProperties": True},
                        "artifact_paths": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                },
                "Task3InsightResponse": {
                    "type": "object",
                    "required": ["status", "question", "query", "row_count", "chart", "nl2sql_plan", "trace"],
                    "properties": {
                        "status": {"type": "string", "enum": ["ok"]},
                        "question": {"type": "string"},
                        "query": {"type": "object", "additionalProperties": True},
                        "rows": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "row_count": {"type": "integer"},
                        "chart": {"type": "object", "additionalProperties": True},
                        "subgraph": {"type": "object", "additionalProperties": True},
                        "graph_context": {"type": "object", "additionalProperties": True},
                        "nl2sql_plan": {"type": "object", "additionalProperties": True},
                        "insight_plan": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "insights": {"type": "object", "additionalProperties": True},
                        "recommended_followups": {"type": "array", "items": {"type": "string"}},
                        "explanation": {"type": "string"},
                        "trace": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "artifact_paths": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                },
                "Task3ConfiguredResponse": {
                    "type": "object",
                    "required": ["status", "analysis_id", "summary", "queries", "charts", "trace"],
                    "properties": {
                        "status": {"type": "string", "enum": ["ok"]},
                        "analysis_id": {"type": "string"},
                        "summary": {"type": "object", "additionalProperties": True},
                        "queries": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "charts": {"type": "object", "additionalProperties": True},
                        "graph_tool_plan": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "graph_tool_result": {"type": "object", "additionalProperties": True},
                        "artifact_paths": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "required": ["error"],
                    "properties": {"error": {"type": "string"}},
                },
            }
        },
    }
