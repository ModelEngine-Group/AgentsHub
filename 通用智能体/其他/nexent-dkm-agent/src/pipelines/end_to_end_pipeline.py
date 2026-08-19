"""End-to-end data -> knowledge -> insight pipeline.

Chains the three task agents into one closed loop, reusing each stage's
output as the next stage's input:

1. Task 1 (DataProcessingAgent): clean raw medical text.
2. Task 2 (MedicalKGAgent): build a medical knowledge graph from the
   cleaned text produced by Task 1.
3. Task 3 (GraphAnalysisAgent): analyze the Task 2 graph and produce
   BI/insight artifacts.

This demonstrates the "数据 -> 知识 -> 洞察" reuse required by task 3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agents.analysis_agent.agent import GraphAnalysisAgent
from src.agents.data_processing_agent.agent import DEFAULT_SAMPLE_TEXT, DataProcessingAgent
from src.agents.kg_agent.agent import MedicalKGAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "end_to_end"
DEFAULT_QUESTION = "高血压有哪些症状和用药？"


def run_end_to_end_pipeline(
    text_input: str | Path | None = None,
    output_root: str | Path | None = None,
    question: str | None = DEFAULT_QUESTION,
    analysis_request: str | None = "分析图谱核心枢纽、社区结构并生成可视化",
    llm_config: dict[str, Any] | None = None,
    datamate_base_url: str | None = None,
    datamate_timeout: float = 3.0,
    datamate_mode: str = "auto",
) -> dict[str, Any]:
    """Run the Task1 -> Task2 -> Task3 closed loop and return a combined report."""

    source = Path(text_input) if text_input else DEFAULT_SAMPLE_TEXT
    root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    stages: list[dict[str, Any]] = []

    # Stage 1: clean raw medical text (Task 1).
    task1 = DataProcessingAgent(llm_config=llm_config).run(
        task_request="清洗并标准化医疗文本数据",
        input_path=source,
        output_dir=root / "task1",
        datamate_base_url=datamate_base_url,
        datamate_timeout=datamate_timeout,
        datamate_mode=datamate_mode,
    )
    cleaned_text = task1.artifacts.get("processing", {}).get("output_path")
    if not cleaned_text:
        cleaned_text = task1.artifacts.get("cleaning", {}).get("output_path")
    stages.append({"task": task1.task, "status": task1.status})
    if task1.status not in {"completed", "completed_with_warnings"} or not cleaned_text:
        return _failure("task1", task1, stages)

    # Stage 2: build the medical KG from Task 1's cleaned text (Task 2).
    task2 = MedicalKGAgent(llm_config=llm_config).run(
        input_path=cleaned_text,
        output_dir=root / "task2",
        question=question,
    )
    graph_path = task2.artifacts.get("graph", {}).get("output_path")
    stages.append({"task": task2.task, "status": task2.status})
    if task2.status != "completed" or not graph_path:
        return _failure("task2", task2, stages)

    # Stage 3: analyze the Task 2 graph (Task 3).
    task3 = GraphAnalysisAgent(llm_config=llm_config).run(
        graph_file=graph_path,
        output_dir=root / "task3",
        question=question,
        task_request=analysis_request,
    )
    stages.append({"task": task3.task, "status": task3.status})
    if task3.status != "completed":
        return _failure("task3", task3, stages)

    return {
        "status": "completed",
        "stages": stages,
        "data_flow": {
            "raw_text": str(source),
            "cleaned_text": cleaned_text,
            "knowledge_graph": graph_path,
            "insight_report": task3.artifacts.get("insight_report", {}),
        },
        "task1": {
            "status": task1.status,
            "output_format": task1.artifacts.get("input", {}).get("format"),
            "quality_report": task1.artifacts.get("quality_report", {}),
        },
        "task2": {
            "status": task2.status,
            "graph": task2.artifacts.get("graph", {}),
            "qa": task2.artifacts.get("qa", {}),
        },
        "task3": {
            "status": task3.status,
            "statistics": task3.artifacts.get("statistics", {}),
            "nl2sql": task3.artifacts.get("nl2sql", {}),
            "plan_execution": task3.artifacts.get("plan_execution", {}),
        },
    }


def _failure(stage: str, result, stages: list[dict[str, Any]]) -> dict[str, Any]:
    logger.warning("End-to-end pipeline failed at %s: %s", stage, result.message)
    return {
        "status": "failed",
        "failed_stage": stage,
        "message": result.message,
        "stages": stages,
    }
