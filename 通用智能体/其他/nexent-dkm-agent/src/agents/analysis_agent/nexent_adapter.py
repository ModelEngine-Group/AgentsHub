"""Optional Nexent-facing adapter for task 3 graph analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.pipelines.task3_insight_pipeline import run_task3_pipeline

TOOL_CLASS_NAME = "GraphAnalysisAgentTool"
TOOL_NAME = "task3_graph_analysis"


class GraphAnalysisAgentTool:
    """Nexent/smolagents-style tool wrapper around task-3 analysis."""

    name = TOOL_NAME
    description = (
        "Analyze a medical knowledge graph with LLM-enhanced NL2SQL, "
        "graph analytics (centrality, paths, communities), statistics, "
        "association analysis, trends, and interactive visualization."
    )
    inputs = {
        "graph_file": {
            "type": "string",
            "description": "Task-2 graph JSON path. If missing, bootstraps the sample graph.",
            "nullable": True,
        },
        "question": {
            "type": "string",
            "description": "Natural-language analysis question for NL2SQL.",
            "nullable": True,
        },
        "task_request": {
            "type": "string",
            "description": "Free-form analysis request (e.g., '统计分析', '关联分析').",
            "nullable": True,
        },
        "llm_config": {
            "type": "object",
            "description": "LLM config with base_url, api_key, model_name for NL2SQL enhancement.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, output_dir: str | None = None, llm_config: dict[str, Any] | None = None) -> None:
        self.output_dir = output_dir
        self.llm_config = llm_config

    def forward(
        self,
        graph_file: str | None = None,
        question: str | None = None,
        task_request: str | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> str:
        result = run_task3_pipeline(
            graph_file=graph_file,
            output_dir=self.output_dir,
            question=question,
            task_request=task_request,
            llm_config=llm_config or self.llm_config,
        )
        return json.dumps(asdict(result), ensure_ascii=False)


def build_nexent_tool_spec(output_dir: str | None = None) -> dict[str, Any]:
    """Return a Nexent ToolConfig-compatible dictionary."""

    return {
        "class_name": TOOL_CLASS_NAME,
        "name": TOOL_NAME,
        "description": GraphAnalysisAgentTool.description,
        "inputs": json.dumps(GraphAnalysisAgentTool.inputs, ensure_ascii=False),
        "output_type": GraphAnalysisAgentTool.output_type,
        "params": {"output_dir": output_dir},
        "source": "local",
        "metadata": {
            "task": "task3",
            "framework": "nexent",
            "analysis_output": "outputs/task3/task3_analysis_report.json",
        },
    }


def build_nexent_agent_spec(
    model_name: str = "main_model",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a Nexent AgentConfig-compatible dictionary."""

    return {
        "name": "task3_graph_analysis_agent",
        "description": (
            "Nexent-compatible task-3 agent for graph-driven statistics, "
            "association analysis, trends, NL2SQL, and BI visualizations."
        ),
        "tools": [build_nexent_tool_spec(output_dir=output_dir)],
        "max_steps": 8,
        "model_name": model_name,
        "provide_run_summary": True,
        "instructions": (
            "Use task3_graph_analysis after a task-2 graph is available. "
            "Answer with graph-backed metrics, SQL evidence, and visualization specs."
        ),
    }


def maybe_create_nexent_agent_config(
    model_name: str = "main_model",
    output_dir: str | None = None,
) -> Any:
    """Create real Nexent Pydantic config objects when the SDK is importable."""

    agent_spec = build_nexent_agent_spec(model_name=model_name, output_dir=output_dir)
    try:
        from nexent.core.agents.agent_model import AgentConfig, ToolConfig
    except ImportError:
        return agent_spec
    tool_configs = [ToolConfig(**tool) for tool in agent_spec["tools"]]
    return AgentConfig(**{**agent_spec, "tools": tool_configs})
