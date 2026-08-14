"""OpenAPI schema for the Task-2 medical KG agent API."""

from __future__ import annotations

from typing import Any


TASK2_OPENAPI_SERVICE_NAME = "modelengine_task2_medical_kg_agent"
TASK2_BUILD_OPERATION_ID = "build_medical_kg_agent"
TASK2_ASK_OPERATION_ID = "ask_medical_kg_agent"


def build_task2_openapi_spec(server_url: str = "http://127.0.0.1:8088") -> dict[str, Any]:
    """Build the Nexent-importable OpenAPI contract for Task-2."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "ModelEngine Task-2 MedicalKGAgent",
            "version": "1.0.0",
            "description": (
                "Competition Task-2 medical knowledge graph generation and QA agent. "
                "It orchestrates reusable entity extraction, relation extraction, "
                "triple normalization, validation, entity linking, graph generation, "
                "and KG-backed medical QA operators through Nexent OpenAPI/MCP."
            ),
        },
        "servers": [{"url": server_url.rstrip("/")}],
        "paths": {
            "/api/medical-kg/build": {
                "post": {
                    "operationId": TASK2_BUILD_OPERATION_ID,
                    "summary": "Build a medical knowledge graph through the Task-2 agent",
                    "description": (
                        "Run the Task-2 MedicalKGAgent from raw medical text or tables to "
                        "a validated medical knowledge graph. The response includes operator "
                        "dispatch, status tracking, validation, quality evaluation, QA traces, "
                        "and artifact locations."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MedicalKGBuildRequest"},
                                "examples": {
                                    "official_medical_kg": {
                                        "value": {
                                            "input_path": "data/raw/medical_demo.jsonl",
                                            "output_root": ".",
                                            "questions": ["高血压有哪些并发症？"],
                                            "kg_backend": "rule",
                                            "use_llm": False,
                                            "external_documents": "data/external/cblue_cmeie_documents.jsonl",
                                            "external_triples": "data/external/cblue_cmeie_triples.jsonl",
                                            "external_entities": "data/external/cblue_cmeie_entities.jsonl",
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Medical KG build completed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MedicalKGBuildResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid input or missing required files",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/api/medical-kg/ask": {
                "post": {
                    "operationId": TASK2_ASK_OPERATION_ID,
                    "summary": "Answer a medical question with KG and DataMate evidence",
                    "description": (
                        "Answer a medical-domain question from the generated knowledge graph, "
                        "optionally fusing live DataMate retrieval evidence when a knowledge "
                        "base is configured."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MedicalKGAskRequest"},
                                "examples": {
                                    "hypertension_complications": {
                                        "value": {
                                            "output_root": ".",
                                            "question": "高血压有哪些并发症？",
                                            "use_hybrid": True,
                                            "enable_live_datamate": True,
                                            "datamate_timeout_seconds": 1.5,
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "KG-backed answer with evidence and reasoning trace",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MedicalKGAskResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid question or missing KG artifacts",
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
                "MedicalKGBuildRequest": {
                    "type": "object",
                    "required": ["input_path"],
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Medical JSONL, JSON, CSV, TXT, MD file path, or a supported input file.",
                        },
                        "output_root": {
                            "type": "string",
                            "default": ".",
                            "description": "Output root for graph artifacts and QA traces.",
                        },
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Medical QA questions used to verify the generated graph.",
                        },
                        "kg_backend": {
                            "type": "string",
                            "enum": ["rule", "llm"],
                            "description": "Knowledge extraction backend. LLM mode requires configured credentials.",
                        },
                        "use_llm": {
                            "type": "boolean",
                            "description": "Whether to request the LLM extraction/answer polishing path.",
                        },
                        "external_documents": {
                            "type": "string",
                            "description": "Optional official/medical external documents JSONL to merge into KG construction.",
                        },
                        "external_triples": {
                            "type": "string",
                            "description": "Optional gold/external triples JSONL for merge and quality evaluation.",
                        },
                        "external_entities": {
                            "type": "string",
                            "description": "Optional external entities JSONL for entity linking.",
                        },
                    },
                },
                "MedicalKGBuildResponse": {
                    "type": "object",
                    "required": [
                        "status",
                        "documents",
                        "entities",
                        "triples",
                        "kg_validation_report",
                        "kg_operator_profile",
                        "artifact_paths",
                    ],
                    "properties": {
                        "status": {"type": "string", "enum": ["completed"]},
                        "agent": {"type": "string"},
                        "pipeline": {"type": "array", "items": {"type": "string"}},
                        "documents": {"type": "integer"},
                        "entities": {"type": "integer"},
                        "triples": {"type": "integer"},
                        "qa_count": {"type": "integer"},
                        "kg_validation_report": {"type": "object", "additionalProperties": True},
                        "kg_quality_eval": {"type": "object", "additionalProperties": True},
                        "kg_backend_probe": {"type": "object", "additionalProperties": True},
                        "kg_operator_profile": {"type": "object", "additionalProperties": True},
                        "kg_backend": {"type": "object", "additionalProperties": True},
                        "external_summary": {"type": "object", "additionalProperties": True},
                        "artifact_paths": {"type": "object", "additionalProperties": {"type": "string"}},
                        "trace": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    },
                },
                "MedicalKGAskRequest": {
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": {"type": "string"},
                        "output_root": {"type": "string", "default": "."},
                        "use_llm": {"type": "boolean"},
                        "use_hybrid": {"type": "boolean", "default": True},
                        "knowledge_base_id": {"type": "string"},
                        "enable_live_datamate": {"type": "boolean", "default": True},
                        "datamate_timeout_seconds": {"type": "number", "default": 1.5},
                    },
                },
                "MedicalKGAskResponse": {
                    "type": "object",
                    "required": ["question", "answer", "supporting_triples"],
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "supporting_triples": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                        },
                        "hybrid_qa": {"type": "object", "additionalProperties": True},
                        "reasoning_paths": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                        },
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
