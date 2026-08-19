"""Unified Nexent agent spec for the full data-knowledge-insight workflow."""

from __future__ import annotations

from typing import Any

from src.agents.analysis_agent.nexent_adapter import build_nexent_tool_spec as build_task3_tool_spec
from src.agents.data_processing_agent.nexent_adapter import build_nexent_tool_spec as build_task1_tool_spec
from src.agents.kg_agent.nexent_adapter import build_nexent_tool_spec as build_task2_tool_spec

SUITE_AGENT_NAME = "dkm_end_to_end_agent"


def build_dkm_nexent_suite_spec(
    model_name: str = "main_model",
    datamate_base_url: str | None = "http://localhost:18000",
    output_root: str | None = None,
) -> dict[str, Any]:
    """Return a Nexent AgentConfig with all three task tools registered."""

    task1_output = f"{output_root}/task1" if output_root else None
    task2_output = f"{output_root}/task2" if output_root else None
    task3_output = f"{output_root}/task3" if output_root else None
    tools = [
        build_task1_tool_spec(datamate_base_url=datamate_base_url, output_dir=task1_output),
        build_task2_tool_spec(output_dir=task2_output),
        build_task3_tool_spec(output_dir=task3_output),
    ]
    return {
        "name": SUITE_AGENT_NAME,
        "description": (
            "Nexent-compatible DKM suite agent that chains data processing, "
            "medical knowledge graph generation, and graph-driven analysis."
        ),
        "tools": tools,
        "max_steps": 12,
        "model_name": model_name,
        "provide_run_summary": True,
        "instructions": (
            "You orchestrate the medical DKM workflow. "
            "1) Call task1_data_processing to clean structured or text inputs and "
            "prepare DataMate payloads (keep datamate_mode=dry_run unless submit is requested). "
            "2) Call task2_medical_kg on the cleaned text to build medical_kg.json. "
            "3) Call task3_graph_analysis on the graph artifact for statistics, NL2SQL, "
            "and BI/insight visualizations. "
            "Reuse each stage output as the next stage input."
        ),
        "metadata": {
            "framework": "nexent",
            "workflow": "data_knowledge_insight",
            "default_datamate_mode": "dry_run",
        },
    }
