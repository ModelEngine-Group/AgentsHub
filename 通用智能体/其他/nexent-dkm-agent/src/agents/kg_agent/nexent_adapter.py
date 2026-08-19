"""Optional Nexent-facing adapter for the task-2 KG agent."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.pipelines.task2_kg_pipeline import run_task2_pipeline

TOOL_CLASS_NAME = "MedicalKGAgentTool"
TOOL_NAME = "task2_medical_kg"


class MedicalKGAgentTool:
    """Nexent/smolagents-style tool wrapper around the task-2 pipeline."""

    name = TOOL_NAME
    description = (
        "Extract medical entities and relations from text, build a JSON "
        "knowledge graph, validate triples, and answer graph-backed questions."
    )
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Medical text input path. Uses bundled sample if omitted.",
            "nullable": True,
        },
        "question": {
            "type": "string",
            "description": "Question to answer from the generated graph.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir

    def forward(
        self,
        input_path: str | None = None,
        question: str | None = None,
    ) -> str:
        result = run_task2_pipeline(
            input_path=input_path,
            output_dir=self.output_dir,
            question=question,
        )
        return json.dumps(asdict(result), ensure_ascii=False)


def build_nexent_tool_spec(output_dir: str | None = None) -> dict[str, Any]:
    """Return a Nexent ToolConfig-compatible dictionary."""

    return {
        "class_name": TOOL_CLASS_NAME,
        "name": TOOL_NAME,
        "description": MedicalKGAgentTool.description,
        "inputs": json.dumps(MedicalKGAgentTool.inputs, ensure_ascii=False),
        "output_type": MedicalKGAgentTool.output_type,
        "params": {"output_dir": output_dir},
        "source": "local",
        "metadata": {
            "task": "task2",
            "framework": "nexent",
            "graph_output": "outputs/task2/medical_kg.json",
        },
    }


def build_nexent_agent_spec(
    model_name: str = "main_model",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a Nexent AgentConfig-compatible dictionary."""

    return {
        "name": "task2_medical_kg_agent",
        "description": (
            "Nexent-compatible task-2 agent for medical entity extraction, "
            "relation extraction, KG construction, triple validation, and QA."
        ),
        "tools": [build_nexent_tool_spec(output_dir=output_dir)],
        "max_steps": 6,
        "model_name": model_name,
        "provide_run_summary": True,
        "instructions": (
            "Use the task2_medical_kg tool to build a medical knowledge graph "
            "before answering disease, symptom, drug, examination, or treatment questions."
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
