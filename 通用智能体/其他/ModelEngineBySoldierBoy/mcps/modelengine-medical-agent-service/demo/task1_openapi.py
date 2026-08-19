"""OpenAPI schema for the Task-1 data processing agent API."""

from __future__ import annotations

from typing import Any


TASK1_OPENAPI_SERVICE_NAME = "modelengine_task1_data_processing_agent"
TASK1_OPENAPI_OPERATION_ID = "run_datamate_processing_agent"


def build_task1_openapi_spec(server_url: str = "http://127.0.0.1:8088") -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "ModelEngine Task-1 DataProcessingAgent",
            "version": "1.0.0",
            "description": (
                "Competition Task-1 data processing agent. It performs task understanding, "
                "planning, DAG execution, quality checks, recovery, and artifact generation "
                "for structured and unstructured medical data."
            ),
        },
        "servers": [{"url": server_url.rstrip("/")}],
        "paths": {
            "/api/datamate-process": {
                "post": {
                    "operationId": TASK1_OPENAPI_OPERATION_ID,
                    "summary": "Run the Task-1 DataMate-backed data processing agent",
                    "description": (
                        "Execute the Task-1 agent chain through DataMate data-management "
                        "and cleaning-task APIs, then return planning, status, result, and traces."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Task1ProcessRequest"},
                                "examples": {
                                    "official_medical_demo": {
                                        "value": {
                                            "input_path": "data/raw/medical_demo.jsonl",
                                            "input_paths": ["data/raw/medical_demo_structured.csv", "data/raw/medical_demo_notes.txt"],
                                            "output_root": ".",
                                            "task": "基于DataMate完成医疗文本数据接入、清洗、转换和质量处理",
                                            "wait_seconds": 60,
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Task-1 DataMate processing completed or reported a blocker",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Task1ProcessResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid input path, task, or DataMate execution failure",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "Task1ProcessRequest": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Input JSONL, JSON, CSV, TXT, MD file path, or a directory containing supported files.",
                        },
                        "input_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of input files/directories. When provided, all files are ingested into one DataMate source dataset.",
                        },
                        "output_root": {
                            "type": "string",
                            "default": ".",
                            "description": "Output root for generated processing artifacts.",
                        },
                        "task": {
                            "type": "string",
                            "description": "Natural language processing goal for task understanding and planning.",
                        },
                        "datamate_api_base": {
                            "type": "string",
                            "default": "http://localhost:8080",
                            "description": "DataMate API base URL reachable from this API service.",
                        },
                        "wait_seconds": {
                            "type": "number",
                            "default": 60,
                            "description": "Maximum seconds to wait for the DataMate cleaning task.",
                        },
                    },
                },
                "Task1ProcessResponse": {
                    "type": "object",
                    "required": ["status", "summary", "plan", "trace", "api_calls", "output_paths"],
                    "properties": {
                        "status": {"type": "string", "enum": ["completed", "blocked"]},
                        "summary": {"type": "object", "additionalProperties": True},
                        "plan": {"type": "object", "additionalProperties": True},
                        "trace": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "api_calls": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "cleaning_task": {"type": "object", "additionalProperties": True},
                        "result": {"type": "object", "additionalProperties": True},
                        "output_paths": {"type": "object", "additionalProperties": {"type": "string"}},
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
