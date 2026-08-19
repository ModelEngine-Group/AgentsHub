"""Optional Nexent-facing adapter for the task-1 data processing agent."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.pipelines.task1_data_pipeline import run_task1_pipeline

TOOL_CLASS_NAME = "DataProcessingAgentTool"
TOOL_NAME = "task1_data_processing"


class DataProcessingAgentTool:
    """Nexent/smolagents-style tool wrapper around the task-1 pipeline."""

    name = TOOL_NAME
    description = (
        "Understand, plan, execute, and track a CSV data-cleaning workflow. "
        "It profiles structured data, runs deterministic local cleaning, and "
        "prepares DataMate cleaning template/task payloads."
    )
    description_zh = (
        "理解并编排CSV数据清洗任务，完成画像、去重、缺失值填补、类型规范化、"
        "状态追踪，并生成DataMate清洗模板和任务payload。"
    )
    inputs = {
        "task_request": {
            "type": "string",
            "description": "Free-form data-processing request.",
            "description_zh": "自然语言数据处理需求。",
            "nullable": True,
        },
        "input_path": {
            "type": "string",
            "description": "Input CSV path. Uses the bundled sample if omitted.",
            "description_zh": "输入CSV路径；为空时使用内置样例。",
            "nullable": True,
        },
        "datamate_mode": {
            "type": "string",
            "description": "dry_run prepares payloads; submit posts to DataMate.",
            "description_zh": "dry_run只生成payload；submit会提交到DataMate。",
            "default": "dry_run",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        datamate_base_url: str | None = "http://localhost:18000",
        output_dir: str | None = None,
    ) -> None:
        self.datamate_base_url = datamate_base_url
        self.output_dir = output_dir

    def forward(
        self,
        task_request: str | None = None,
        input_path: str | None = None,
        datamate_mode: str = "dry_run",
    ) -> str:
        result = run_task1_pipeline(
            task_request=task_request,
            input_path=input_path,
            output_dir=self.output_dir,
            datamate_base_url=self.datamate_base_url,
            datamate_mode=datamate_mode,
        )
        return json.dumps(asdict(result), ensure_ascii=False)


def build_nexent_tool_spec(
    datamate_base_url: str | None = "http://localhost:18000",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a Nexent ToolConfig-compatible dictionary."""

    return {
        "class_name": TOOL_CLASS_NAME,
        "name": TOOL_NAME,
        "description": DataProcessingAgentTool.description,
        "inputs": json.dumps(DataProcessingAgentTool.inputs, ensure_ascii=False),
        "output_type": DataProcessingAgentTool.output_type,
        "params": {
            "datamate_base_url": datamate_base_url,
            "output_dir": output_dir,
        },
        "source": "local",
        "metadata": {
            "task": "task1",
            "framework": "nexent",
            "default_datamate_mode": "dry_run",
        },
    }


def build_nexent_agent_spec(
    model_name: str = "main_model",
    datamate_base_url: str | None = "http://localhost:18000",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a Nexent AgentConfig-compatible dictionary."""

    return {
        "name": "task1_data_processing_agent",
        "description": (
            "Nexent-compatible task-1 agent for structured CSV data cleaning, "
            "operator planning, DataMate payload preparation, and run-state tracking."
        ),
        "tools": [
            build_nexent_tool_spec(
                datamate_base_url=datamate_base_url,
                output_dir=output_dir,
            )
        ],
        "max_steps": 6,
        "model_name": model_name,
        "provide_run_summary": True,
        "instructions": (
            "Understand the user's data-processing request first, then call the "
            "task1_data_processing tool. Keep datamate_mode=dry_run unless the "
            "user explicitly requests a DataMate submission."
        ),
    }


def maybe_create_nexent_agent_config(
    model_name: str = "main_model",
    datamate_base_url: str | None = "http://localhost:18000",
    output_dir: str | None = None,
) -> Any:
    """Create real Nexent Pydantic config objects when the SDK is importable.

    The project remains runnable without the Nexent SDK installed; callers can
    always use the dictionary returned by ``build_nexent_agent_spec``.
    """

    agent_spec = build_nexent_agent_spec(
        model_name=model_name,
        datamate_base_url=datamate_base_url,
        output_dir=output_dir,
    )
    try:
        from nexent.core.agents.agent_model import AgentConfig, ToolConfig
    except ImportError:
        return agent_spec

    tool_configs = [ToolConfig(**tool) for tool in agent_spec["tools"]]
    return AgentConfig(**{**agent_spec, "tools": tool_configs})
