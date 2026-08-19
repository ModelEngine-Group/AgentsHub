"""Offline smoke tests for Nexent-compatible tool wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.agents.analysis_agent.nexent_adapter import GraphAnalysisAgentTool
from src.agents.data_processing_agent.nexent_adapter import DataProcessingAgentTool
from src.agents.kg_agent.nexent_adapter import MedicalKGAgentTool


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "nexent_smoke"


def test_task1_nexent_tool_forward_runs_offline(output_dir):
    tool = DataProcessingAgentTool(datamate_base_url=None, output_dir=str(output_dir))
    payload = json.loads(
        tool.forward(
            task_request="请清洗患者CSV，删除重复记录，填补空值",
            input_path=str(ROOT / "data" / "samples" / "task1_patients.csv"),
            datamate_mode="dry_run",
        )
    )
    assert payload["status"] in {"completed", "completed_with_warnings"}
    assert payload["artifacts"]["cleaning"]["output_path"]


def test_task2_nexent_tool_forward_runs_offline(output_dir):
    tool = MedicalKGAgentTool(output_dir=str(output_dir))
    payload = json.loads(
        tool.forward(
            input_path=str(ROOT / "data" / "samples" / "task2_medical_notes.txt"),
            question="高血压有哪些症状？",
        )
    )
    assert payload["status"] == "completed"
    assert payload["artifacts"]["graph"]["output_path"]


def test_task3_nexent_tool_forward_runs_offline(output_dir):
    task2 = MedicalKGAgentTool(output_dir=str(output_dir / "task2"))
    graph_payload = json.loads(
        task2.forward(
            input_path=str(ROOT / "data" / "samples" / "task2_medical_notes.txt"),
        )
    )
    graph_file = graph_payload["artifacts"]["graph"]["output_path"]

    tool = GraphAnalysisAgentTool(output_dir=str(output_dir / "task3"))
    payload = json.loads(
        tool.forward(
            graph_file=graph_file,
            question="哪些疾病关联最多症状？",
        )
    )
    assert payload["status"] == "completed"
    assert payload["artifacts"]["export"]["output_path"]
