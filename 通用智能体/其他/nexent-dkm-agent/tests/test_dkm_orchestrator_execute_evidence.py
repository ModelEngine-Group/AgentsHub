"""Tests for DKM orchestrator execution evidence collection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demos.dkm_orchestrator_execute_evidence_demo import collect_orchestrator_execution_evidence
from src.operators.data_ops.datamate_client import summarize_hybrid_execution_plan

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TEXT = ROOT / "data" / "samples" / "task1_medical_notes.txt"


def test_summarize_hybrid_execution_plan_splits_local_and_datamate():
    summary = summarize_hybrid_execution_plan(
        ["fill_missing_values", "drop_duplicate_rows", "normalize_column_types"]
    )

    assert summary["hybrid"] is True
    assert summary["local_preprocessing"] == ["fill_missing_values"]
    assert "drop_duplicate_rows" in summary["datamate_template_operators"]


def test_collect_orchestrator_execution_evidence_offline(tmp_path: Path):
    if not SAMPLE_TEXT.is_file():
        pytest.skip("sample text missing")

    evidence = collect_orchestrator_execution_evidence(
        request="请清洗医疗文本，构建知识图谱并生成分析洞察",
        question="哪些疾病关联最多症状？",
        text_input=SAMPLE_TEXT,
        output_dir=tmp_path / "orchestrated",
        datamate_url="none",
    )

    assert evidence["status"] == "completed"
    assert evidence["stage_order"] == ["task1", "task2", "task3"]
    assert evidence["artifacts"]["task2_graph"]["node_count"] > 0
    graph_path = Path(evidence["artifacts"]["task2_graph"]["path"])
    assert graph_path.is_file()


def test_orchestrator_execute_evidence_demo_cli(tmp_path: Path, monkeypatch):
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "dkm_orchestrator_execute_evidence_demo.py",
            "--input",
            str(SAMPLE_TEXT),
            "--output-dir",
            str(tmp_path / "run"),
            "--output",
            str(output),
        ],
    )
    from demos.dkm_orchestrator_execute_evidence_demo import main

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "dkm_orchestrator_run"
