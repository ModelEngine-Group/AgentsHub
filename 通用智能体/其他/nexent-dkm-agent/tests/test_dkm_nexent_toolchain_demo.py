"""Tests for offline Nexent toolchain simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_nexent_toolchain_demo_runs_offline(tmp_path):
    from demos.dkm_nexent_toolchain_demo import run_nexent_toolchain

    result = run_nexent_toolchain(
        text_input=ROOT / "data" / "samples" / "task1_medical_notes.txt",
        question="哪些疾病关联最多症状？",
        output_dir=tmp_path,
        datamate_url=None,
        request="请清洗医疗文本，构建知识图谱并生成分析洞察",
    )

    assert result["status"] == "completed"
    assert result["tool_count"] == 3
    assert len(result["steps"]) == 3
    assert result["steps"][0]["tool"] == "task1_data_processing"
    assert result["steps"][1]["tool"] == "task2_medical_kg"
    assert result["steps"][2]["tool"] == "task3_graph_analysis"
    assert result["planner"]["rule_plan"]["stages"]


def test_nexent_toolchain_demo_cli_writes_evidence(tmp_path, monkeypatch):
    from demos import dkm_nexent_toolchain_demo as demo

    monkeypatch.setattr(
        demo,
        "parse_args",
        lambda: argparse.Namespace(
            text_input=str(ROOT / "data" / "samples" / "task1_medical_notes.txt"),
            question="哪些疾病关联最多症状？",
            output_dir=str(tmp_path),
            datamate_url="none",
            request="请清洗医疗文本，构建知识图谱并生成分析洞察",
        ),
    )

    assert demo.main() == 0
    evidence = json.loads((tmp_path / "nexent_toolchain_evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "completed"
