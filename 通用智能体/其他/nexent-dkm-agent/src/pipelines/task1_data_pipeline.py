"""Task 1 data processing pipeline."""

from pathlib import Path
from typing import Any

from src.agents.data_processing_agent import DataProcessingAgent
from src.common.results import PipelineResult


def run_task1_pipeline(
    task_request: str | None = None,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    datamate_base_url: str | None = "http://localhost:18000",
    datamate_timeout: float = 3.0,
    datamate_src_dataset_id: str | None = None,
    datamate_src_dataset_name: str | None = None,
    datamate_dest_dataset_name: str | None = None,
    datamate_mode: str = "dry_run",
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
    transforms: list[dict[str, Any]] | None = None,
) -> PipelineResult:
    """Run the task 1 data processing pipeline."""

    agent = DataProcessingAgent(
        llm_config=llm_config,
        local_model_path=local_model_path,
    )
    return agent.run(
        task_request=task_request,
        input_path=input_path,
        output_dir=output_dir,
        datamate_base_url=datamate_base_url,
        datamate_timeout=datamate_timeout,
        datamate_src_dataset_id=datamate_src_dataset_id,
        datamate_src_dataset_name=datamate_src_dataset_name,
        datamate_dest_dataset_name=datamate_dest_dataset_name,
        datamate_mode=datamate_mode,
        transforms=transforms,
    )
