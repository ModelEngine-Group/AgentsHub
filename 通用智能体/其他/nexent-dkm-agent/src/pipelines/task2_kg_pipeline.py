"""Task 2 medical knowledge graph pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.kg_agent import MedicalKGAgent
from src.common.results import PipelineResult


def run_task2_pipeline(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    question: str | None = None,
    task_request: str | None = None,
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
    neo4j_config: dict[str, str] | None = None,
    relation_backend: str = "rule",
    graph_file: str | Path | None = None,
) -> PipelineResult:
    """Run the task 2 medical KG pipeline."""

    agent = MedicalKGAgent(
        llm_config=llm_config,
        local_model_path=local_model_path,
        neo4j_config=neo4j_config,
        relation_backend=relation_backend,
    )
    return agent.run(
        input_path=input_path,
        output_dir=output_dir,
        question=question,
        task_request=task_request,
        graph_file=graph_file,
    )
