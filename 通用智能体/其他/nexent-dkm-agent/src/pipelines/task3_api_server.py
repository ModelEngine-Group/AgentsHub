"""REST API server for task 3 graph-driven analysis agent."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common.path_security import safe_path_string
from src.operators.analysis_ops import (
    REGISTERED_ANALYSIS_OPERATORS,
    build_graph_sqlite,
    compute_centrality,
    compute_shortest_paths,
    detect_communities,
    load_graph,
    translate_question_to_sql_with_llm,
)
from src.pipelines.task3_insight_pipeline import run_task3_pipeline

app = FastAPI(
    title="Task 3 Graph Analysis Agent API",
    description="REST interface for graph-driven statistics, NL2SQL, graph analytics, and visualization.",
    version="2.0.0",
)

_tasks: dict[str, dict[str, Any]] = {}
_llm_config: dict[str, Any] | None = None


def set_llm_config(config: dict[str, Any] | None) -> None:
    """Set the LLM configuration for NL2SQL enhancement."""
    global _llm_config
    _llm_config = config


class ProcessRequest(BaseModel):
    graph_file: str | None = None
    output_dir: str | None = None
    question: str | None = None
    task_request: str | None = None


class SqlRequest(BaseModel):
    task_id: str
    question: str = Field(default="哪些疾病关联最多症状？")


class Nl2SqlRequest(BaseModel):
    question: str = Field(default="哪些疾病关联最多症状？")
    graph_file: str | None = None


class PathRequest(BaseModel):
    task_id: str
    start_entity: str
    end_entity: str | None = None
    max_depth: int = Field(default=4, ge=1, le=10)


class TaskResponse(BaseModel):
    task_id: str
    status: str


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "task3_graph_analysis_agent",
        "version": "2.0.0",
        "llm_enabled": _llm_config is not None,
    }


@app.get("/api/task3/operators")
def list_operators() -> dict[str, Any]:
    return {"operators": REGISTERED_ANALYSIS_OPERATORS, "count": len(REGISTERED_ANALYSIS_OPERATORS)}


@app.post("/api/task3/process", response_model=TaskResponse)
def submit_task(req: ProcessRequest) -> TaskResponse:
    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {"status": "pending", "result": None}
    try:
        graph_file = safe_path_string(req.graph_file, label="graph_file")
        output_dir = safe_path_string(req.output_dir, label="output_dir")
    except ValueError as exc:
        _tasks[task_id] = {"status": "error", "error": str(exc)}
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = run_task3_pipeline(
            graph_file=graph_file,
            output_dir=output_dir,
            question=req.question,
            task_request=req.task_request,
            llm_config=_llm_config,
        )
        _tasks[task_id] = {"status": result.status, "result": asdict(result)}
    except Exception as exc:
        _tasks[task_id] = {"status": "error", "error": str(exc)}
    return TaskResponse(task_id=task_id, status=_tasks[task_id]["status"])


@app.get("/api/task3/status/{task_id}")
def get_status(task_id: str) -> dict[str, Any]:
    entry = _require_task(task_id)
    return {"task_id": task_id, "status": entry["status"]}


@app.get("/api/task3/report/{task_id}")
def get_report(task_id: str) -> dict[str, Any]:
    entry = _require_task(task_id)
    if entry["status"] == "pending":
        raise HTTPException(status_code=202, detail="Task still running")
    return entry.get("result", {"status": entry["status"], "error": entry.get("error")})


@app.post("/api/task3/sql")
def run_sql_query(req: SqlRequest) -> dict[str, Any]:
    entry = _require_task(req.task_id)
    if entry["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Task is not completed: {entry['status']}")
    graph = _load_graph(entry)
    conn = build_graph_sqlite(graph)
    result = translate_question_to_sql_with_llm(
        req.question, conn, llm_config=_llm_config,
    )
    return result


@app.post("/api/nl2sql")
def run_nl2sql_query(req: Nl2SqlRequest) -> dict[str, Any]:
    """Stateless NL2SQL endpoint for interactive dashboard queries."""

    from src.operators.analysis_ops.llm_nl2sql import translate_question_with_fallbacks
    from src.operators.analysis_ops.nl2sql import execute_read_only_sql

    graph = _load_graph_file_relaxed(req.graph_file)
    conn = build_graph_sqlite(graph)
    translation = translate_question_with_fallbacks(
        req.question,
        conn,
        llm_config=_llm_config,
    )
    rows = execute_read_only_sql(conn, translation["sql"], max_limit=20)["rows"]
    return {
        "status": translation.get("status", "completed"),
        "intent": translation.get("intent"),
        "translator": translation.get("translator", "template"),
        "sql": translation.get("sql", ""),
        "rows": rows,
    }


@app.post("/api/task3/centrality")
def get_centrality(req: ProcessRequest) -> dict[str, Any]:
    """Compute centrality analysis for a graph."""
    return compute_centrality(_load_graph_file(req.graph_file))


@app.post("/api/task3/paths")
def find_paths(req: PathRequest) -> dict[str, Any]:
    """Find shortest paths between entities in a graph."""
    entry = _require_task(req.task_id)
    if entry["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Task is not completed: {entry['status']}")
    graph = _load_graph(entry)
    return compute_shortest_paths(graph, req.start_entity, req.end_entity, req.max_depth)


@app.post("/api/task3/communities")
def find_communities(req: ProcessRequest) -> dict[str, Any]:
    """Detect communities in a graph."""
    return detect_communities(_load_graph_file(req.graph_file))


def serve(host: str = "127.0.0.1", port: int = 8003) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def _require_task(task_id: str) -> dict[str, Any]:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]


def _load_graph(entry: dict[str, Any]) -> dict[str, Any]:
    result = entry.get("result") or {}
    input_path = result.get("artifacts", {}).get("input", {}).get("path")
    if not input_path:
        raise HTTPException(status_code=404, detail="Graph input path not found")
    path = Path(input_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph file missing: {input_path}")
    return load_graph(path)


def _load_graph_file(graph_file: str | None) -> dict[str, Any]:
    if not graph_file:
        raise HTTPException(status_code=400, detail="graph_file is required")
    try:
        path = safe_path_string(graph_file, label="graph_file")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = Path(path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph file not found: {graph_file}")
    return load_graph(path)


def _load_graph_file_relaxed(graph_file: str | None) -> dict[str, Any]:
    """Load a graph for stateless NL2SQL queries, tolerating missing ``statistics``.

    Interactive dashboard queries only need ``nodes`` and ``edges``; the full
    task-3 graph artifact also carries a ``statistics`` block that callers in a
    query-only context may not have on hand. This loader validates the
    node/edge contract and back-fills an empty ``statistics`` block so the
    downstream SQLite builder never sees a KeyError.
    """

    if not graph_file:
        raise HTTPException(status_code=400, detail="graph_file is required")
    try:
        path = safe_path_string(graph_file, label="graph_file")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = Path(path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph file not found: {graph_file}")
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict):
        raise HTTPException(status_code=400, detail="Graph file must contain a JSON object")
    if "statistics" not in graph:
        graph["statistics"] = {"entity_type_counts": {}, "relation_type_counts": {}}
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise HTTPException(status_code=400, detail="Graph nodes and edges must be lists")
    for node in graph["nodes"]:
        if not {"id", "name", "type"}.issubset(node):
            raise HTTPException(status_code=400, detail=f"Invalid graph node: {node}")
    for edge in graph["edges"]:
        if not {"source", "target", "predicate"}.issubset(edge):
            raise HTTPException(status_code=400, detail=f"Invalid graph edge: {edge}")
    return graph
