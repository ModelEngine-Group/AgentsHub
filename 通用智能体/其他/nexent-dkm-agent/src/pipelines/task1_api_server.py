"""REST API server for task 1 data processing agent."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.data_processing_agent.planner import REGISTERED_OPERATORS
from src.common.llm_config import load_llm_config
from src.common.path_security import safe_path_string
from src.operators.data_ops.datamate_client import DataMateClient
from src.pipelines.task1_data_pipeline import run_task1_pipeline

app = FastAPI(
    title="Task 1 Data Processing Agent API",
    description="REST interface for the Nexent-based data processing agent.",
    version="1.0.0",
)

_executor = ThreadPoolExecutor(max_workers=4)
_tasks: dict[str, dict[str, Any]] = {}
_configured_datamate_url: str | None = "http://localhost:18000"
_datamate_write_enabled = False


class ProcessRequest(BaseModel):
    task: str | None = None
    input_path: str | None = None
    output_dir: str | None = None
    datamate_url: str = "http://localhost:18000"
    datamate_mode: Literal["dry_run", "submit", "auto"] = "dry_run"
    src_dataset_id: str | None = None
    src_dataset_name: str | None = None
    dest_dataset_name: str | None = None
    use_llm: bool = False
    env_file: str | None = None
    llm_config_path: str | None = None
    local_model_path: str | None = None


class TaskResponse(BaseModel):
    task_id: str
    status: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "task1_data_processing_agent"}


@app.get("/api/task1/operators")
def list_operators() -> dict[str, Any]:
    return {
        "operators": sorted(REGISTERED_OPERATORS),
        "count": len(REGISTERED_OPERATORS),
    }


@app.post("/api/task1/process", response_model=TaskResponse)
def submit_task(req: ProcessRequest) -> TaskResponse:
    try:
        datamate_url = _resolve_api_datamate_url(req.datamate_url)
        input_path = safe_path_string(req.input_path, label="input_path")
        output_dir = safe_path_string(req.output_dir, label="output_dir")
        local_model_path = safe_path_string(req.local_model_path, label="local_model_path")
        llm_config_path = safe_path_string(req.llm_config_path, label="llm_config_path")
        env_file = safe_path_string(req.env_file, label="env_file")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.datamate_mode == "submit" and not _datamate_write_enabled:
        raise HTTPException(
            status_code=403,
            detail=(
                "DataMate submit is disabled. Enable it explicitly at API "
                "server startup."
            ),
        )

    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {"status": "pending", "result": None}

    llm_config = None
    if req.use_llm:
        config_path = llm_config_path or env_file
        llm_config = load_llm_config(config_path)
        if config_path and llm_config is None:
            _tasks[task_id] = {
                "status": "error",
                "error": "LLM config is missing or incomplete.",
            }
            raise HTTPException(status_code=400, detail="LLM config is missing or incomplete.")

    def _run():
        try:
            result = run_task1_pipeline(
                task_request=req.task,
                input_path=input_path,
                output_dir=output_dir,
                datamate_base_url=datamate_url,
                datamate_src_dataset_id=req.src_dataset_id,
                datamate_src_dataset_name=req.src_dataset_name,
                datamate_dest_dataset_name=req.dest_dataset_name,
                datamate_mode=req.datamate_mode,
                llm_config=llm_config,
                local_model_path=local_model_path,
            )
            _tasks[task_id] = {
                "status": result.status,
                "result": asdict(result),
            }
        except Exception as exc:
            _tasks[task_id] = {"status": "error", "error": str(exc)}

    _executor.submit(_run)
    return TaskResponse(task_id=task_id, status="pending")


@app.get("/api/task1/status/{task_id}")
def get_status(task_id: str) -> dict[str, Any]:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    entry = _tasks[task_id]
    return {"task_id": task_id, "status": entry["status"]}


@app.get("/api/task1/report/{task_id}")
def get_report(task_id: str) -> dict[str, Any]:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    entry = _tasks[task_id]
    if entry["status"] == "pending":
        raise HTTPException(status_code=202, detail="Task still running")
    return entry.get("result", {"status": entry["status"], "error": entry.get("error")})


def configure_datamate_access(
    *,
    base_url: str | None = "http://localhost:18000",
    allow_write: bool = False,
) -> None:
    """Configure the only DataMate endpoint reachable through this API."""

    global _configured_datamate_url, _datamate_write_enabled
    if base_url is None or str(base_url).lower() == "none":
        _configured_datamate_url = None
    else:
        _configured_datamate_url = DataMateClient(str(base_url)).base_url
    _datamate_write_enabled = bool(allow_write)


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    datamate_url: str | None = "http://localhost:18000",
    allow_datamate_write: bool = False,
) -> None:
    """Start the API server."""
    import uvicorn

    configure_datamate_access(
        base_url=datamate_url,
        allow_write=allow_datamate_write,
    )
    uvicorn.run(app, host=host, port=port)


def _resolve_api_datamate_url(requested_url: str) -> str | None:
    if requested_url.lower() == "none":
        return None
    normalized = DataMateClient(requested_url).base_url
    if normalized != _configured_datamate_url:
        raise ValueError(
            "datamate_url must match the server-configured DataMate URL."
        )
    return normalized
