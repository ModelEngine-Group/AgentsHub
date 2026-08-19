"""REST API server for task 2 medical knowledge graph agent."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.operators.kg_ops import query_graph_neighbors
from src.operators.kg_ops.multi_hop_qa import (
    answer_with_evidence_chain,
    build_evidence_chain,
    multi_hop_query,
)
from src.common.llm_config import load_llm_config
from src.common.path_security import safe_path_string
from src.pipelines.task2_kg_pipeline import run_task2_pipeline

app = FastAPI(
    title="Task 2 Medical Knowledge Graph Agent API",
    description="REST interface for medical KG generation, QA, multi-hop reasoning, and graph querying.",
    version="2.0.0",
)

_tasks: dict[str, dict[str, Any]] = {}


class ProcessRequest(BaseModel):
    input_path: str | None = None
    output_dir: str | None = None
    question: str | None = None
    task_request: str | None = None
    llm_config_path: str | None = None
    local_model_path: str | None = None


class QueryRequest(BaseModel):
    task_id: str
    entity: str
    relation: str | None = None
    direction: str = Field(default="out", pattern="^(out|in|both)$")
    limit: int = Field(default=20, ge=1, le=100)


class MultiHopRequest(BaseModel):
    task_id: str
    start_entity: str
    target_entity: str | None = None
    max_hops: int = Field(default=3, ge=1, le=5)
    max_paths: int = Field(default=5, ge=1, le=20)


class EvidenceQARequest(BaseModel):
    task_id: str
    question: str
    max_hops: int = Field(default=2, ge=1, le=4)


class TaskResponse(BaseModel):
    task_id: str
    status: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "task2_medical_kg_agent", "version": "2.0.0"}


@app.get("/api/task2/operators")
def list_operators() -> dict[str, Any]:
    operators = [
        "extract_medical_entities", "extract_relations", "validate_triples",
        "build_medical_graph", "answer_graph_question", "build_kg_quality_report",
        "find_graph_entities", "query_graph_neighbors",
        "extract_entities_with_llm", "extract_relations_with_llm",
        "multi_hop_query", "build_evidence_chain", "answer_with_evidence_chain",
    ]
    return {"operators": operators, "count": len(operators)}


@app.post("/api/task2/process", response_model=TaskResponse)
def submit_task(req: ProcessRequest) -> TaskResponse:
    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {"status": "pending", "result": None}
    try:
        input_path = safe_path_string(req.input_path, label="input_path")
        output_dir = safe_path_string(req.output_dir, label="output_dir")
        llm_config_path = safe_path_string(req.llm_config_path, label="llm_config_path")
        local_model_path = safe_path_string(req.local_model_path, label="local_model_path")
    except ValueError as exc:
        _tasks[task_id] = {"status": "error", "error": str(exc)}
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    llm_config = load_llm_config(llm_config_path)
    if req.llm_config_path and llm_config is None:
        _tasks[task_id] = {
            "status": "error",
            "error": "LLM config is missing or incomplete.",
        }
        raise HTTPException(status_code=400, detail="LLM config is missing or incomplete.")
    try:
        result = run_task2_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            question=req.question,
            task_request=req.task_request,
            llm_config=llm_config,
            local_model_path=local_model_path,
        )
        _tasks[task_id] = {"status": result.status, "result": asdict(result)}
    except Exception as exc:
        _tasks[task_id] = {"status": "error", "error": str(exc)}
    return TaskResponse(task_id=task_id, status=_tasks[task_id]["status"])


@app.get("/api/task2/status/{task_id}")
def get_status(task_id: str) -> dict[str, Any]:
    entry = _require_task(task_id)
    return {"task_id": task_id, "status": entry["status"]}


@app.get("/api/task2/report/{task_id}")
def get_report(task_id: str) -> dict[str, Any]:
    entry = _require_task(task_id)
    if entry["status"] == "pending":
        raise HTTPException(status_code=202, detail="Task still running")
    return entry.get("result", {"status": entry["status"], "error": entry.get("error")})


@app.post("/api/task2/query")
def query_task_graph(req: QueryRequest) -> dict[str, Any]:
    entry = _require_task(req.task_id)
    if entry["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Task is not completed: {entry['status']}")
    graph = _load_graph(entry)
    return query_graph_neighbors(
        entity=req.entity, graph=graph,
        relation=req.relation, direction=req.direction, limit=req.limit,
    )


@app.post("/api/task2/multi-hop")
def multi_hop_query_endpoint(req: MultiHopRequest) -> dict[str, Any]:
    entry = _require_task(req.task_id)
    if entry["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Task is not completed: {entry['status']}")
    graph = _load_graph(entry)
    return multi_hop_query(
        graph=graph,
        start_entity=req.start_entity,
        target_entity=req.target_entity,
        max_hops=req.max_hops,
        max_paths=req.max_paths,
    )


@app.post("/api/task2/evidence-qa")
def evidence_qa_endpoint(req: EvidenceQARequest) -> dict[str, Any]:
    entry = _require_task(req.task_id)
    if entry["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Task is not completed: {entry['status']}")
    graph = _load_graph(entry)
    return answer_with_evidence_chain(
        question=req.question,
        graph=graph,
    )


@app.post("/api/task2/evidence-chain")
def evidence_chain_endpoint(req: EvidenceQARequest) -> dict[str, Any]:
    entry = _require_task(req.task_id)
    if entry["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Task is not completed: {entry['status']}")
    graph = _load_graph(entry)
    return build_evidence_chain(
        graph=graph,
        question=req.question,
        max_hops=req.max_hops,
    )


def serve(host: str = "127.0.0.1", port: int = 8002) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def _require_task(task_id):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]


def _load_graph(entry):
    result = entry.get("result") or {}
    graph_artifact = result.get("artifacts", {}).get("graph", {})
    output_path = graph_artifact.get("output_path")
    if not output_path:
        raise HTTPException(status_code=404, detail="Graph artifact not found")
    path = Path(output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph artifact missing: {output_path}")
    return json.loads(path.read_text(encoding="utf-8"))
