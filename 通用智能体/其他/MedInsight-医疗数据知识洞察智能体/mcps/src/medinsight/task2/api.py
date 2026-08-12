from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from medinsight.task2.operators import (
    KnowledgeGraphQAOperator,
    MedicalConflictDetector,
    MedicalEntityExtractor,
    MedicalGraphEnricher,
    MedicalKnowledgeGraphBuilder,
    MedicalRelationExtractor,
    MedicalTripleBuilder,
    MedicalTripleRepairer,
    MedicalTripleValidator,
)
from medinsight.task2.operators.qa import answer_question
from medinsight.task2.service import (
    DEFAULT_TASK2_INPUT,
    DEFAULT_TASK2_OUTPUT,
    DEFAULT_TASK2_TASK,
    DEFAULT_TASK2_TEXT_OUTPUT,
    DEFAULT_TASK2_TEXT_TASK,
    run_task2_kg_build,
    run_task2_text_kg_build,
)


NEXENT_TASK2_TOOLS = {
    MedicalEntityExtractor.spec.name: MedicalEntityExtractor,
    MedicalRelationExtractor.spec.name: MedicalRelationExtractor,
    MedicalGraphEnricher.spec.name: MedicalGraphEnricher,
    MedicalTripleBuilder.spec.name: MedicalTripleBuilder,
    MedicalTripleValidator.spec.name: MedicalTripleValidator,
    MedicalTripleRepairer.spec.name: MedicalTripleRepairer,
    MedicalConflictDetector.spec.name: MedicalConflictDetector,
    MedicalKnowledgeGraphBuilder.spec.name: MedicalKnowledgeGraphBuilder,
    KnowledgeGraphQAOperator.spec.name: KnowledgeGraphQAOperator,
}


TASK2_OPERATION_IDS = [
    "medinsight_plan_medical_kg_flow",
    "medinsight_run_medical_kg_qa_agent",
    "medinsight_answer_medical_kg_question",
    "medinsight_generate_medical_kg_from_text",
    "medinsight_medical_entity_extractor",
    "medinsight_medical_relation_extractor",
    "medinsight_medical_graph_enricher",
    "medinsight_medical_triple_builder",
    "medinsight_medical_triple_validator",
    "medinsight_medical_triple_repairer",
    "medinsight_medical_conflict_detector",
    "medinsight_medical_knowledge_graph_builder",
    "medinsight_kg_qa_agent",
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


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class MedInsightTask2APIHandler(BaseHTTPRequestHandler):
    server_version = "MedInsightTask2API/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            json_response(self, 200, {"status": "ok", "service": "medinsight-task2-api"})
            return
        if parsed.path in {"/openapi.json", "/api/nexent/openapi.json", "/api/nexent/task2/openapi.json"}:
            params = parse_qs(parsed.query)
            server_url = params.get("server_url", ["http://host.docker.internal:18081"])[0]
            json_response(self, 200, build_task2_nexent_openapi_spec(server_url))
            return
        if parsed.path in {"/api/nexent/tools", "/api/nexent/task2/tools"}:
            json_response(self, 200, {"status": "ok", "tools": build_task2_nexent_tool_catalog()})
            return
        if parsed.path == "/api/task2/report":
            params = parse_qs(parsed.query)
            report_path = Path(params.get("path", [str(DEFAULT_TASK2_OUTPUT / "kg_build_report.json")])[0])
            if not report_path.exists():
                json_response(self, 404, {"status": "error", "message": f"report not found: {report_path}"})
                return
            json_response(self, 200, json.loads(report_path.read_text(encoding="utf-8")))
            return
        json_response(self, 404, {"status": "error", "message": f"unknown endpoint: {parsed.path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path in {"/api/task2/run", "/api/nexent/task2/run"}:
                json_response(self, 200, run_nexent_task2_kg_agent_tool(payload))
                return
            if parsed.path in {"/api/task2/run_text", "/api/nexent/task2/run_text"}:
                json_response(self, 200, run_nexent_task2_text_kg_tool(payload))
                return
            if parsed.path == "/api/nexent/task2/plan":
                json_response(self, 200, build_nexent_kg_plan(payload))
                return
            if parsed.path in {"/api/task2/qa", "/api/nexent/task2/qa"}:
                json_response(self, 200, run_nexent_task2_qa_tool(payload))
                return
            if parsed.path.startswith("/api/nexent/task2/tools/"):
                tool_name = parsed.path.rsplit("/", 1)[-1]
                json_response(self, 200, run_nexent_task2_operator_tool(tool_name, payload))
                return
            json_response(self, 404, {"status": "error", "message": f"unknown endpoint: {parsed.path}"})
        except Exception as exc:
            json_response(self, 500, {"status": "error", "message": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(host: str = "127.0.0.1", port: int = 18081) -> None:
    server = ThreadingHTTPServer((host, port), MedInsightTask2APIHandler)
    print(f"MedInsight Task2 API running at http://{host}:{port}")
    print("Nexent OpenAPI: /api/nexent/task2/openapi.json")
    server.serve_forever()


def build_task2_nexent_tool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": operator_class.spec.description,
            "input_schema": operator_class.spec.input_schema,
            "output_schema": operator_class.spec.output_schema,
            "nexent_endpoint": f"/api/nexent/task2/tools/{name}",
        }
        for name, operator_class in NEXENT_TASK2_TOOLS.items()
    ]


def run_nexent_task2_operator_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    operator_class = NEXENT_TASK2_TOOLS.get(tool_name)
    if not operator_class:
        return {"status": "error", "message": f"unknown task2 tool: {tool_name}"}
    result = operator_class().run(payload)
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": f"medinsight_{tool_name}",
        "result": result,
    }


def build_nexent_kg_plan(payload: dict[str, Any]) -> dict[str, Any]:
    task = str(payload.get("task") or DEFAULT_TASK2_TASK)
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool_contract",
        "task": task,
        "task_understanding": {
            "canonical_task": "医疗知识图谱生成与问答",
            "required_capabilities": ["实体抽取", "关系抽取", "术语标准化", "医学编码增强", "三元组生成", "三元组校验", "自动修复", "冲突检测", "图谱问答"],
        },
        "recommended_execution": [
            {"step": 1, "tool": "medical_entity_extractor", "reason": "抽取疾病、症状、药物、检查等实体。"},
            {"step": 2, "tool": "medical_relation_extractor", "reason": "抽取诊断、治疗、并发等医学关系。"},
            {"step": 3, "tool": "medical_graph_enricher", "reason": "融合 CHIP-CDN 术语归一化、ICD-10 编码、手术词表和 CMedCausal 因果关系，提升图谱覆盖。"},
            {"step": 4, "tool": "medical_triple_builder", "reason": "生成标准三元组并去重。"},
            {"step": 5, "tool": "medical_triple_validator", "reason": "按 CMeIE schema 与增强 schema 校验三元组。"},
            {"step": 6, "tool": "medical_triple_repairer", "reason": "对可安全修复的 schema 类型错误进行自动纠正。"},
            {"step": 7, "tool": "medical_conflict_detector", "reason": "检测重复三元组和实体类型冲突。"},
            {"step": 8, "tool": "medical_knowledge_graph_builder", "reason": "构建可复用知识图谱。"},
            {"step": 9, "tool": "kg_qa_agent", "reason": "基于图谱回答用户问题并返回证据。"},
        ],
        "on_demand_text_kg": {
            "tool": "medinsight_generate_medical_kg_from_text",
            "reason": "用户可输入新的医疗文本，智能体现场抽取实体和关系并生成可问答知识图谱。",
        },
        "nexent_integration": {
            "agent_name": "medinsight_medical_kg_qa_agent",
            "planner_tool": "medinsight_plan_medical_kg_flow",
            "entry_tool": "medinsight_run_medical_kg_qa_agent",
            "operator_tools": [f"medinsight_{name}" for name in NEXENT_TASK2_TOOLS],
            "role": "Nexent is the primary agent orchestration layer; MedInsight task2 tools are imported as Outer API Tools.",
        },
    }


def run_nexent_task2_kg_agent_tool(payload: dict[str, Any]) -> dict[str, Any]:
    input_dir = Path(payload.get("input_dir", DEFAULT_TASK2_INPUT))
    output_dir = Path(payload.get("output_dir", DEFAULT_TASK2_OUTPUT))
    result = run_task2_kg_build(
        input_dir=input_dir,
        output_dir=output_dir,
        shared_output_dir=Path(payload.get("shared_output_dir", "data/shared/knowledge_graph")),
        task=str(payload.get("task") or DEFAULT_TASK2_TASK),
        cmeee_split=str(payload.get("cmeee_split") or "train"),
        cmeie_split=str(payload.get("cmeie_split") or "train"),
        cmeee_limit=int(payload.get("cmeee_limit", 2000) or 0),
        cmeie_limit=int(payload.get("cmeie_limit", 2000) or 0),
        chip_cdn_split=str(payload.get("chip_cdn_split") or "train"),
        cmedcausal_split=str(payload.get("cmedcausal_split") or "train"),
        chip_cdn_limit=int(payload.get("chip_cdn_limit", 0) or 0),
        cmedcausal_limit=int(payload.get("cmedcausal_limit", 0) or 0),
        icd10_limit=int(payload.get("icd10_limit", 0) or 0),
        surgery_lexicon_limit=int(payload.get("surgery_lexicon_limit", 0) or 0),
        enrichment_enabled=parse_bool(payload.get("enrichment_enabled"), True),
    )
    enrich_task2_report_with_nexent_context(result)
    qa_answer = None
    question = str(payload.get("question") or "").strip()
    if question:
        qa_answer = answer_question(load_graph(Path(result["shared_graph_path"])), question)
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_run_medical_kg_qa_agent",
        "agent": "medinsight_medical_kg_qa_agent",
        "qa_answer": qa_answer,
        "result": result,
}


def run_nexent_task2_text_kg_tool(payload: dict[str, Any]) -> dict[str, Any]:
    text = str(payload.get("text") or "")
    result = run_task2_text_kg_build(
        text=text,
        output_dir=Path(payload.get("output_dir", DEFAULT_TASK2_TEXT_OUTPUT)),
        shared_output_dir=Path(payload.get("shared_output_dir", "data/shared/knowledge_graph")),
        task=str(payload.get("task") or DEFAULT_TASK2_TEXT_TASK),
        doc_id=str(payload.get("doc_id") or "custom-medical-text-000001"),
        question=str(payload.get("question") or "").strip() or None,
    )
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_generate_medical_kg_from_text",
        "agent": "medinsight_medical_kg_qa_agent",
        "result": result,
    }


def run_nexent_task2_qa_tool(payload: dict[str, Any]) -> dict[str, Any]:
    graph_path = Path(payload.get("graph_path", "data/shared/knowledge_graph/task2_user_medical_kg_library.json"))
    question = str(payload.get("question") or "")
    graph = payload.get("graph") if isinstance(payload.get("graph"), dict) else load_graph(graph_path)
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_answer_medical_kg_question",
        "answer": answer_question(graph, question),
    }


def load_graph(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"knowledge graph not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_task2_report_with_nexent_context(result: dict[str, Any]) -> None:
    report_path = Path(result["report_path"])
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["nexent_integration"] = {
        "mode": "outer_api_tool",
        "agent_name": "medinsight_medical_kg_qa_agent",
        "planner_tool": "medinsight_plan_medical_kg_flow",
        "entry_tool": "medinsight_run_medical_kg_qa_agent",
        "qa_tool": "medinsight_answer_medical_kg_question",
        "operator_tools": [f"medinsight_{name}" for name in NEXENT_TASK2_TOOLS],
        "role": "Nexent is the primary agent orchestration layer; Task2 operators are callable tools.",
    }
    report_path.write_text(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")


def build_task2_nexent_openapi_spec(server_url: str = "http://host.docker.internal:18081") -> dict[str, Any]:
    paths: dict[str, Any] = {
        "/api/nexent/task2/plan": {
            "post": {
                "operationId": "medinsight_plan_medical_kg_flow",
                "summary": "Plan a medical knowledge graph build flow",
                "description": "由 Nexent 调用的任务二规划工具：理解医疗知识图谱构建或问答需求，并返回实体抽取、关系抽取、术语/编码增强、三元组校验、图谱构建和问答工具链。",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task2PlanRequest"}}},
                },
                "responses": {"200": {"description": "Task2 Nexent plan."}},
            }
        },
        "/api/nexent/task2/run": {
            "post": {
                "operationId": "medinsight_run_medical_kg_qa_agent",
                "summary": "Run the MedInsight task-two medical KG and QA agent",
                "description": "由 Nexent 作为主编排层触发任务二智能体，自动完成实体抽取、关系抽取、术语/编码增强、三元组生成、Schema 校验、自动修复、冲突检测、图谱构建，并可基于图谱回答问题。",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task2RunRequest"}}},
                },
                "responses": {"200": {"description": "Task2 KG build result."}},
            }
        },
        "/api/nexent/task2/run_text": {
            "post": {
                "operationId": "medinsight_generate_medical_kg_from_text",
                "summary": "Generate a medical knowledge graph from Task1-processed medical text",
                "description": "由 Nexent 调用的现场图谱生成工具：接收任务一处理后的医疗文本记录，自动抽取实体和关系，完成三元组生成、Schema 校验、自动修复、冲突检测，并构建可问答知识图谱。",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task2TextKGRequest"}}},
                },
                "responses": {"200": {"description": "Task2 custom text KG build result."}},
            }
        },
        "/api/nexent/task2/qa": {
            "post": {
                "operationId": "medinsight_answer_medical_kg_question",
                "summary": "Answer a question from the generated medical knowledge graph",
                "description": "由 Nexent 调用的图谱问答工具：读取任务二生成的图谱并返回答案、图谱事实和证据文本。",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Task2QARequest"}}},
                },
                "responses": {"200": {"description": "Graph-grounded answer."}},
            }
        },
    }
    for name, operator_class in NEXENT_TASK2_TOOLS.items():
        paths[f"/api/nexent/task2/tools/{name}"] = {
            "post": {
                "operationId": f"medinsight_{name}",
                "summary": operator_class.spec.description,
                "description": f"Nexent 可调用的任务二算子：{operator_class.spec.description}",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
                "responses": {"200": {"description": f"{name} execution result."}},
            }
        }
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "MedInsight Nexent Medical KG QA Tools",
            "version": "0.1.0",
            "description": "任务二医疗知识图谱生成与问答智能体的 Nexent 工具契约，可通过 Nexent Outer API Tools 导入并调用。",
        },
        "servers": [{"url": server_url.rstrip("/")}],
        "paths": paths,
        "components": {
            "schemas": {
                "Task2PlanRequest": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "default": DEFAULT_TASK2_TASK},
                        "input_dir": {"type": "string", "default": str(DEFAULT_TASK2_INPUT)},
                    },
                },
                "Task2RunRequest": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "default": DEFAULT_TASK2_TASK},
                        "input_dir": {"type": "string", "default": str(DEFAULT_TASK2_INPUT)},
                        "output_dir": {"type": "string", "default": str(DEFAULT_TASK2_OUTPUT)},
                        "shared_output_dir": {"type": "string", "default": "data/shared/knowledge_graph"},
                        "cmeee_split": {"type": "string", "default": "train", "description": "Supports train, dev, test, train+dev, or all."},
                        "cmeie_split": {"type": "string", "default": "train", "description": "Supports train, dev, test, train+dev, or all."},
                        "cmeee_limit": {"type": "integer", "default": 2000},
                        "cmeie_limit": {"type": "integer", "default": 2000},
                        "chip_cdn_split": {"type": "string", "default": "train", "description": "Supports train, dev, test, train+dev, or all."},
                        "cmedcausal_split": {"type": "string", "default": "train", "description": "Supports train, dev, test, train+dev, or all."},
                        "chip_cdn_limit": {"type": "integer", "default": 0, "description": "0 means all CHIP-CDN records."},
                        "cmedcausal_limit": {"type": "integer", "default": 0, "description": "0 means all CMedCausal records."},
                        "icd10_limit": {"type": "integer", "default": 0, "description": "0 means all ICD-10 lexicon rows."},
                        "surgery_lexicon_limit": {"type": "integer", "default": 0, "description": "0 means all surgery lexicon rows."},
                        "enrichment_enabled": {"type": "boolean", "default": True},
                        "question": {"type": "string", "description": "Optional graph QA question after KG build."},
                    },
                },
                "Task2TextKGRequest": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "default": DEFAULT_TASK2_TEXT_TASK},
                        "text": {
                            "type": "string",
                            "default": "肺炎主要表现为发热、咳嗽、咳痰和胸痛，重症患者可出现呼吸困难。检查可进行血常规、胸片和肺部CT，细菌性肺炎可使用抗生素治疗。",
                        },
                        "doc_id": {"type": "string", "default": "custom-medical-text-000001"},
                        "output_dir": {"type": "string", "default": str(DEFAULT_TASK2_TEXT_OUTPUT)},
                        "shared_output_dir": {"type": "string", "default": "data/shared/knowledge_graph"},
                        "question": {"type": "string", "default": "肺炎有什么症状？"},
                    },
                    "required": ["text"],
                },
                "Task2QARequest": {
                    "type": "object",
                    "properties": {
                        "graph_path": {
                            "type": "string",
                            "default": "data/shared/knowledge_graph/task2_user_medical_kg_library.json",
                        },
                        "question": {"type": "string", "default": "溶血性贫血有什么症状？"},
                    },
                    "required": ["question"],
                },
            }
        },
    }
