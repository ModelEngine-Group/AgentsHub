from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from medgraph_agent.cli import default_source, ensure_graph
from medgraph_agent.core.analytics import GraphAnalyzer
from medgraph_agent.core.benchmark import run_benchmarks
from medgraph_agent.core.pipeline import PipelineRunner
from medgraph_agent.core.quality import audit_graph
from medgraph_agent.core.qa import answer_question
from medgraph_agent.core.storage import load_graph_json, read_json, write_json
from medgraph_agent.core.models import to_dict


def output_dir() -> Path:
    return Path(os.environ.get("MEDGRAPH_OUTPUT_DIR", "outputs/latest"))


def source_path() -> Path:
    return Path(os.environ.get("MEDGRAPH_SOURCE", str(default_source())))


def create_app() -> FastAPI:
    app = FastAPI(title="MedGraph Insight Agent", version="0.1.0")
    static_dir = Path(__file__).resolve().parents[1] / "web/static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "output_dir": str(output_dir()), "source": str(source_path())}

    @app.post("/api/pipelines/run")
    def run_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
        task = payload.get("task") or "构建医疗数据处理、知识图谱问答和图谱分析闭环"
        source = payload.get("source") or str(source_path())
        run = PipelineRunner(output_dir()).run(task, source)
        if run.status != "succeeded":
            raise HTTPException(status_code=500, detail=run.error)
        return to_dict(run)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        path = output_dir() / "run.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no run available")
        run = read_json(path)
        if run.get("id") != run_id and run_id != "latest":
            raise HTTPException(status_code=404, detail="run not found in latest artifact")
        return run

    @app.get("/api/graph")
    def get_graph() -> dict[str, Any]:
        graph_path = ensure_graph(output_dir(), source_path())
        graph = load_graph_json(graph_path)
        return {"graph": to_dict(graph), "stats": graph.stats()}

    @app.post("/api/qa")
    def qa(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        graph = load_graph_json(ensure_graph(output_dir(), source_path()))
        return to_dict(answer_question(question, graph))

    @app.post("/api/analyze")
    def analyze(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        ensure_graph(output_dir(), source_path())
        return to_dict(GraphAnalyzer(output_dir() / "medgraph.db").analyze(question))

    @app.get("/api/benchmarks/latest")
    def benchmark_latest() -> dict[str, Any]:
        path = output_dir() / "benchmark.json"
        if path.exists():
            return read_json(path)
        result = run_benchmarks(source_path(), repeat=20)
        write_json(path, result)
        return result

    @app.get("/api/quality")
    def quality() -> dict[str, Any]:
        graph = load_graph_json(ensure_graph(output_dir(), source_path()))
        result = audit_graph(graph)
        write_json(output_dir() / "quality_report.json", result)
        return result

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "medgraph_agent.api.server:app",
        host=os.environ.get("MEDGRAPH_HOST", "127.0.0.1"),
        port=int(os.environ.get("MEDGRAPH_PORT", "8080")),
        reload=False,
    )


if __name__ == "__main__":
    main()
