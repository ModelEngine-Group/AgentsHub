"""Cross-task DKM orchestrator for planning and executing multi-stage workflows."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agents.analysis_agent.agent import GraphAnalysisAgent
from src.agents.data_processing_agent.agent import DataProcessingAgent
from src.agents.kg_agent.agent import MedicalKGAgent
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dkm_orchestrated"

_TASK1_TRIGGERS = ("清洗", "去重", "缺失", "csv", "etl", "数据处理", "clean", "dedup")
_TASK2_TRIGGERS = ("建图", "知识图谱", "抽取", "实体", "关系", "kg", "graph", "问答")
_TASK3_TRIGGERS = ("分析", "洞察", "可视化", "nl2sql", "统计", "dashboard", "bi", "关联", "趋势")


def plan_dkm_workflow(
    request: str,
    question: str | None = None,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan which DKM stages to run from a natural-language request."""

    if llm_config:
        try:
            return _llm_plan_dkm_workflow(request, question=question, llm_config=llm_config)
        except Exception:
            logger.warning("LLM DKM planning failed; falling back to rules.", exc_info=True)

    normalized = (request or "").lower()
    wants_task1 = any(trigger in normalized or trigger in request for trigger in _TASK1_TRIGGERS)
    wants_task2 = any(trigger in normalized or trigger in request for trigger in _TASK2_TRIGGERS)
    wants_task3 = any(trigger in normalized or trigger in request for trigger in _TASK3_TRIGGERS)
    has_existing_graph = any(
        phrase in request
        for phrase in ("已有图谱", "已有图", "existing graph", "existing kg")
    )
    only_task1 = request.startswith("只") and wants_task1

    if has_existing_graph and wants_task3:
        stages = [_task3_stage(request, question)]
    elif only_task1 or (wants_task1 and not wants_task2 and not wants_task3):
        stages = [_task1_stage("清洗并导出结构化数据")]
    elif wants_task1 and wants_task2 and wants_task3:
        stages = _default_full_pipeline_stages(question)
    elif wants_task1 and wants_task2:
        stages = _default_full_pipeline_stages(question)[:2]
    elif wants_task2 and not wants_task1:
        stages = [_task2_stage(question)]
    elif wants_task3 and not wants_task1 and not wants_task2:
        stages = [_task3_stage(request, question)]
    else:
        stages = _default_full_pipeline_stages(question)

    return {
        "original_request": request,
        "planner_mode": "rule",
        "stages": stages,
        "rationale": [
            "Detected requested DKM stages from task keywords.",
            "Stages execute in task1 -> task2 -> task3 order when selected.",
        ],
        "confidence": 0.85,
    }


class DKMOrchestrator:
    """Execute a planned DKM workflow and track stage artifacts."""

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        local_model_path: str | None = None,
        datamate_base_url: str | None = "http://localhost:18000",
        datamate_mode: str = "dry_run",
    ) -> None:
        self._llm_config = llm_config
        self._local_model_path = local_model_path
        self._datamate_base_url = datamate_base_url
        self._datamate_mode = datamate_mode

    def run(
        self,
        request: str,
        output_root: str | Path | None = None,
        question: str | None = None,
        text_input: str | Path | None = None,
        graph_file: str | Path | None = None,
    ) -> dict[str, Any]:
        plan = plan_dkm_workflow(request, question=question, llm_config=self._llm_config)
        root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
        root.mkdir(parents=True, exist_ok=True)

        stages: list[dict[str, Any]] = []
        cleaned_text: str | None = None
        graph_path: str | None = str(graph_file) if graph_file else None

        for stage in plan["stages"]:
            task_name = stage["task"]
            if task_name == "task1":
                payload = run_task1_stage(
                    task_request=stage.get("task_request", request),
                    input_path=text_input,
                    output_dir=root / "task1",
                    llm_config=self._llm_config,
                    local_model_path=self._local_model_path,
                    datamate_base_url=self._datamate_base_url,
                    datamate_mode=self._datamate_mode,
                )
                cleaned_text = payload.get("artifacts", {}).get("processing", {}).get("output_path")
                if not cleaned_text:
                    cleaned_text = payload.get("artifacts", {}).get("cleaning", {}).get("output_path")
            elif task_name == "task2":
                input_path = cleaned_text or text_input
                payload = run_task2_stage(
                    input_path=input_path,
                    output_dir=root / "task2",
                    question=stage.get("question", question),
                    task_request=stage.get("task_request"),
                    llm_config=self._llm_config,
                    local_model_path=self._local_model_path,
                )
                graph_path = payload.get("artifacts", {}).get("graph", {}).get("output_path")
            elif task_name == "task3":
                payload = run_task3_stage(
                    graph_file=graph_path or graph_file,
                    output_dir=root / "task3",
                    question=stage.get("question", question),
                    task_request=stage.get("task_request", request),
                    llm_config=self._llm_config,
                    local_model_path=self._local_model_path,
                )
            else:
                raise ValueError(f"Unsupported DKM stage: {task_name}")

            stage_record = {
                "task": task_name,
                "status": payload.get("status"),
                "message": payload.get("message"),
            }
            stages.append(stage_record)
            if payload.get("status") not in {"completed", "completed_with_warnings"}:
                return {
                    "status": "failed",
                    "plan": plan,
                    "stages": stages,
                    "failed_stage": task_name,
                    "message": payload.get("message"),
                    "artifacts": payload.get("artifacts", {}),
                }

        return {
            "status": "completed",
            "plan": plan,
            "stages": stages,
            "output_root": str(root),
        }


def run_task1_stage(**kwargs: Any) -> dict[str, Any]:
    result = DataProcessingAgent(
        llm_config=kwargs.get("llm_config"),
        local_model_path=kwargs.get("local_model_path"),
    ).run(
        task_request=kwargs.get("task_request"),
        input_path=kwargs.get("input_path"),
        output_dir=kwargs.get("output_dir"),
        datamate_base_url=kwargs.get("datamate_base_url"),
        datamate_mode=kwargs.get("datamate_mode", "dry_run"),
    )
    return asdict(result)


def run_task2_stage(**kwargs: Any) -> dict[str, Any]:
    result = MedicalKGAgent(
        llm_config=kwargs.get("llm_config"),
        local_model_path=kwargs.get("local_model_path"),
    ).run(
        input_path=kwargs.get("input_path"),
        output_dir=kwargs.get("output_dir"),
        question=kwargs.get("question"),
        task_request=kwargs.get("task_request"),
    )
    return asdict(result)


def run_task3_stage(**kwargs: Any) -> dict[str, Any]:
    result = GraphAnalysisAgent(
        llm_config=kwargs.get("llm_config"),
        local_model_path=kwargs.get("local_model_path"),
    ).run(
        graph_file=kwargs.get("graph_file"),
        output_dir=kwargs.get("output_dir"),
        question=kwargs.get("question"),
        task_request=kwargs.get("task_request"),
    )
    return asdict(result)


def _default_full_pipeline_stages(question: str | None) -> list[dict[str, Any]]:
    return [
        _task1_stage("清洗并标准化医疗文本数据"),
        _task2_stage(question),
        _task3_stage("分析图谱统计、关联与可视化洞察", question),
    ]


def _task1_stage(task_request: str) -> dict[str, Any]:
    return {"task": "task1", "task_request": task_request}


def _task2_stage(question: str | None) -> dict[str, Any]:
    return {
        "task": "task2",
        "task_request": "构建医疗知识图谱",
        "question": question or "高血压有哪些症状和用药？",
    }


def _task3_stage(task_request: str, question: str | None) -> dict[str, Any]:
    return {
        "task": "task3",
        "task_request": task_request,
        "question": question or "哪些疾病关联最多症状？",
    }


def _llm_plan_dkm_workflow(
    request: str,
    question: str | None,
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    from src.agents.data_processing_agent.llm_orchestrator import request_plan

    llm_result = request_plan(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        model_name=llm_config.get("model_name", "glm-5.1"),
        task_request=(
            f"Plan a DKM workflow for: {request}. "
            "Return operators from: task1_data_processing, task2_medical_kg, task3_graph_analysis."
        ),
        available_operators=["task1_data_processing", "task2_medical_kg", "task3_graph_analysis"],
        timeout=llm_config.get("timeout", 30.0),
        llm_config=llm_config,
    )
    operator_to_stage = {
        "task1_data_processing": _task1_stage(request),
        "task2_medical_kg": _task2_stage(question),
        "task3_graph_analysis": _task3_stage(request, question),
    }
    stages = [
        operator_to_stage[op]
        for op in operator_to_stage
        if op in set(llm_result.get("operators", []))
    ]
    if not stages:
        raise ValueError("LLM returned no valid DKM stages.")
    return {
        "original_request": request,
        "planner_mode": "llm",
        "stages": stages,
        "rationale": llm_result.get("rationale", ["LLM-generated DKM workflow."]),
        "confidence": min(llm_result.get("confidence", 0.8), 0.99),
    }
