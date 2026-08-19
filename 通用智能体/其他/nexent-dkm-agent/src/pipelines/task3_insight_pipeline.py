"""Task 3 insight generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.agents.analysis_agent import GraphAnalysisAgent
from src.common.results import PipelineResult

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "task3_analysis.yaml"


def load_task3_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load task 3 YAML config. Returns empty dict when file is missing."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_task3_pipeline(
    graph_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    question: str | None = None,
    task_request: str | None = None,
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
    config_path: str | Path | None = None,
) -> PipelineResult:
    """Run graph-driven analysis over a task-2 KG artifact.

    When ``config_path`` is provided (or the default config exists), values
    from the config are used as defaults that can be overridden by explicit
    arguments.
    """

    cfg = load_task3_config(config_path)

    # Config provides defaults; explicit arguments take precedence
    effective_graph_file = graph_file or cfg.get("input", {}).get("graph_file")
    effective_question = question or cfg.get("nl2sql", {}).get("default_question")

    return GraphAnalysisAgent(
        llm_config=llm_config,
        local_model_path=local_model_path,
    ).run(
        graph_file=effective_graph_file,
        output_dir=output_dir,
        question=effective_question,
        task_request=task_request,
    )
