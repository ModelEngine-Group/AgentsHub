"""Tests for LLM planner evidence collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_collect_planner_llm_evidence_uses_snapshot_when_llm_missing(tmp_path):
    from src.agents.planner_comparison import collect_planner_llm_evidence

    snapshot = {
        "metadata": {"llm_model": "demo-model"},
        "llm_enhanced": {
            "task1": {
                "task": "task1",
                "rule_plan": {"planner_mode": "rule", "operators": ["load_csv"]},
                "hybrid_plan": {"planner_mode": "llm", "operators": ["load_csv", "drop_column"]},
                "hybrid_mode": "llm",
            },
            "task2": {
                "task": "task2",
                "rule_plan": {"planner_mode": "rule", "operators": ["extract_medical_entities"]},
                "hybrid_plan": {"planner_mode": "llm", "operators": ["extract_medical_entities"]},
                "hybrid_mode": "llm",
            },
            "task3": {
                "task": "task3",
                "rule_plan": {"planner_mode": "rule", "operators": ["load_graph", "execute_sql"]},
                "hybrid_plan": {"planner_mode": "llm", "operators": ["load_graph", "execute_sql"]},
                "hybrid_mode": "llm",
            },
            "orchestrator": {
                "rule_plan": {"planner_mode": "rule", "stages": [{"task": "task1"}]},
                "enhanced_plan": {"planner_mode": "llm", "stages": [{"task": "task1"}]},
                "enhanced_mode": "llm",
            },
        },
    }
    snapshot_path = tmp_path / "planner_llm_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    report = collect_planner_llm_evidence(llm_config=None, snapshot_path=snapshot_path)

    assert report["collection_mode"] == "rule_plus_recorded_snapshot"
    assert report["snapshot_used"] is True
    assert report["diff_summary"]
    task1 = next(item for item in report["diff_summary"] if item["task"] == "task1")
    assert task1["operators_match"] is False


def test_collect_planner_llm_evidence_live_llm(monkeypatch):
    from src.agents import planner_comparison as module

    def fake_compare_all_planners(**kwargs):
        if kwargs.get("llm_config"):
            return {
                "task1": {
                    "task": "task1",
                    "rule_plan": {"operators": ["a"]},
                    "hybrid_plan": {"operators": ["a", "b"], "planner_mode": "llm"},
                    "hybrid_mode": "llm",
                },
                "task2": {
                    "task": "task2",
                    "rule_plan": {"operators": ["x"]},
                    "hybrid_plan": {"operators": ["x"], "planner_mode": "llm"},
                    "hybrid_mode": "llm",
                },
                "task3": {
                    "task": "task3",
                    "rule_plan": {"operators": ["p"]},
                    "hybrid_plan": {"operators": ["p"], "planner_mode": "llm"},
                    "hybrid_mode": "llm",
                },
                "orchestrator": {
                    "rule_plan": {"stages": [{"task": "task1"}]},
                    "enhanced_plan": {"stages": [{"task": "task1"}]},
                    "enhanced_mode": "llm",
                },
            }
        return {
            "task1": {
                "task": "task1",
                "rule_plan": {"operators": ["a"], "planner_mode": "rule"},
                "hybrid_plan": {"operators": ["a"], "planner_mode": "rule"},
                "hybrid_mode": "rule",
            },
            "task2": {
                "task": "task2",
                "rule_plan": {"operators": ["x"], "planner_mode": "rule"},
                "hybrid_plan": {"operators": ["x"], "planner_mode": "rule"},
                "hybrid_mode": "rule",
            },
            "task3": {
                "task": "task3",
                "rule_plan": {"operators": ["p"], "planner_mode": "rule"},
                "hybrid_plan": {"operators": ["p"], "planner_mode": "rule"},
                "hybrid_mode": "rule",
            },
            "orchestrator": {"rule_plan": {"stages": [{"task": "task1"}]}},
        }

    monkeypatch.setattr(module, "compare_all_planners", fake_compare_all_planners)

    report = module.collect_planner_llm_evidence(
        llm_config={"base_url": "https://example.test/v1", "api_key": "k", "model_name": "demo"},
    )

    assert report["collection_mode"] == "live_llm"
    assert report["llm_available"] is True
    assert report["diff_summary"][0]["operators_match"] is False


def test_planner_llm_evidence_demo_writes_output(tmp_path, monkeypatch):
    from demos import planner_llm_evidence_demo as demo

    monkeypatch.setattr(
        demo,
        "parse_args",
        lambda: argparse.Namespace(
            llm=False,
            llm_config=None,
            local_model=None,
            snapshot=str(ROOT / "benchmarks" / "data" / "planner_llm_snapshot.json"),
            output=str(tmp_path / "planner_llm_evidence.json"),
        ),
    )

    assert demo.main() == 0
    payload = json.loads((tmp_path / "planner_llm_evidence.json").read_text(encoding="utf-8"))
    assert payload["rule_only"]
    assert payload["llm_enhanced"]
