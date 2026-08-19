"""Tests for competition figure export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.figure_export import (
    export_all_defense_figures,
    export_architecture_diagram,
    export_kg_overview_figure,
    export_nl2sql_accuracy_figure,
    export_npu_mode_speedup_figure,
    export_npu_utilization_figure,
    export_oov_extraction_figure,
    export_task1_quality_figure,
    export_task3_figures_from_report,
)


def test_export_architecture_diagram_writes_svg(tmp_path: Path):
    target = tmp_path / "architecture.svg"
    result = export_architecture_diagram(target)
    assert result["status"] == "completed"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "DataMate" in text
    assert "Nexent" in text


def test_export_task1_quality_figure_from_benchmark(tmp_path: Path):
    report = {
        "quality_metrics": {
            "quality_score_before": 0.8,
            "quality_score_after": 1.0,
            "duplicate_rows_before": 1,
            "duplicate_rows_after": 0,
            "missing_values_before": 3,
            "missing_values_after": 0,
        }
    }
    report_path = tmp_path / "task1_data_quality.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "task1_quality.svg"
    result = export_task1_quality_figure(report_path, target)
    assert result["status"] == "completed"
    assert "0.8" in target.read_text(encoding="utf-8")


def test_export_kg_overview_figure_from_graph(tmp_path: Path):
    graph = {
        "nodes": [
            {"id": "d1", "name": "高血压", "type": "Disease"},
            {"id": "s1", "name": "头痛", "type": "Symptom"},
        ],
        "edges": [{"source": "d1", "target": "s1", "type": "has_symptom"}],
    }
    graph_path = tmp_path / "medical_kg.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    target = tmp_path / "task2_kg_overview.svg"
    result = export_kg_overview_figure(graph_path, target)
    assert result["status"] == "completed"
    text = target.read_text(encoding="utf-8")
    assert "高血压" in text


def test_export_task3_figures_from_report(tmp_path: Path):
    report = {
        "visualizations": {
            "charts": {
                "entity_distribution": {
                    "type": "bar",
                    "title": "Entity type distribution",
                    "data": [{"category": "Disease", "value": 3}],
                }
            }
        }
    }
    report_path = tmp_path / "task3_analysis_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    figures = export_task3_figures_from_report(report_path, tmp_path / "figures")
    assert len(figures) == 1
    assert figures[0]["chart"] == "entity_distribution"


def test_export_all_defense_figures_bundle(tmp_path: Path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "task1_data_quality.json").write_text(
        json.dumps(
            {
                "quality_metrics": {
                    "quality_score_before": 0.8,
                    "quality_score_after": 1.0,
                    "duplicate_rows_before": 1,
                    "duplicate_rows_after": 0,
                    "missing_values_before": 3,
                    "missing_values_after": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    graph = {
        "nodes": [{"id": "d1", "name": "高血压", "type": "Disease"}],
        "edges": [],
    }
    graph_path = tmp_path / "medical_kg.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    task3_report = {
        "visualizations": {
            "charts": {
                "entity_distribution": {
                    "type": "bar",
                    "title": "Entity type distribution",
                    "data": [{"category": "Disease", "value": 1}],
                }
            }
        }
    }
    task3_path = tmp_path / "task3_analysis_report.json"
    task3_path.write_text(json.dumps(task3_report), encoding="utf-8")

    manifest = export_all_defense_figures(
        output_dir=tmp_path / "figures",
        task1_quality_report=report_dir / "task1_data_quality.json",
        kg_graph_file=graph_path,
        task3_report_file=task3_path,
    )
    names = {item["name"] for item in manifest}
    assert "architecture_diagram" in names
    assert "dkm_workflow" in names
    assert "task1_quality_improvement" in names
    assert "task2_kg_overview" in names


def test_export_oov_extraction_figure(tmp_path: Path):
    report = {
        "comparison": {"closed_overall_f1": 1.0, "oov_overall_f1": 0.0, "oov_recall": 0.0},
        "closed_corpus": {"overall": {"recall": 1.0}},
        "oov_corpus": {"overall": {"f1": 0.0}},
    }
    report_path = tmp_path / "task2_oov_extraction_quality.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "oov.svg"
    result = export_oov_extraction_figure(report_path, target)
    assert result["status"] == "completed"
    assert "<svg" in target.read_text(encoding="utf-8")


def test_export_nl2sql_accuracy_figure(tmp_path: Path):
    report = {
        "accuracy": 1.0,
        "intent_classification": {"accuracy": 1.0},
        "execution": {"accuracy": 1.0},
        "holdout_generalization": {"accuracy": 1.0},
    }
    report_path = tmp_path / "task3_nl2sql_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "nl2sql.svg"
    result = export_nl2sql_accuracy_figure(report_path, target)
    assert result["status"] == "completed"
    assert "100" in target.read_text(encoding="utf-8")


def test_export_planner_operator_figure(tmp_path: Path):
    from src.common.figure_export import export_planner_operator_figure

    report = {
        "diff_summary": [
            {"task": "task1", "rule_operator_count": 7, "enhanced_operator_count": 7},
            {"task": "task3", "rule_operator_count": 11, "enhanced_operator_count": 9},
        ]
    }
    report_path = tmp_path / "planner_llm_evidence.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "planner.svg"
    result = export_planner_operator_figure(report_path, target)
    assert result["status"] == "completed"
    assert "<svg" in target.read_text(encoding="utf-8")


def test_export_task2_pipeline_latency_figure(tmp_path: Path):
    from src.common.figure_export import export_task2_pipeline_latency_figure

    report = {
        "rule_backend": {"latency_ms_avg": 12.5},
        "tensor_cpu_backend": {"latency_ms_avg": 8.2},
    }
    report_path = tmp_path / "task2_pipeline_latency.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "latency.svg"
    result = export_task2_pipeline_latency_figure(report_path, target)
    assert result["status"] == "completed"
    assert "12.5" in target.read_text(encoding="utf-8")


def test_export_npu_mode_speedup_figure_from_report(tmp_path: Path):
    report = {
        "mode_benchmarks": [
            {"name": "baseline_full_logits", "speedup_vs_cpu": 0.77},
            {"name": "cached_topk_labels", "speedup_vs_cpu": 79.89},
        ]
    }
    report_path = tmp_path / "task2_xlarge.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "npu_task2.svg"
    result = export_npu_mode_speedup_figure(report_path, target, "npu_task2", "Task2 NPU")
    assert result["status"] == "completed"
    text = target.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "cached_topk_labels" in text
    assert "cpu_baseline" in text


def test_export_npu_mode_speedup_skips_missing_report(tmp_path: Path):
    result = export_npu_mode_speedup_figure(
        tmp_path / "missing.json", tmp_path / "x.svg", "npu", "title"
    )
    assert result["status"] == "skipped"


def test_export_npu_utilization_figure_from_reports(tmp_path: Path):
    report = {
        "npu_utilization": {
            "available": True,
            "npu_utilization_pct": {"min": 0, "avg": 25.2, "max": 100.0},
            "power_w": {"min": 89.4, "avg": 95.5, "max": 99.1},
        }
    }
    report_path = tmp_path / "task3_xlarge.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = tmp_path / "npu_util.svg"
    result = export_npu_utilization_figure({"task3_50k": report_path}, target)
    assert result["status"] == "completed"
    assert "<svg" in target.read_text(encoding="utf-8")


def test_export_npu_utilization_figure_skips_when_unavailable(tmp_path: Path):
    report = {"npu_utilization": {"available": False}}
    report_path = tmp_path / "no_util.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = export_npu_utilization_figure(
        {"w": report_path}, tmp_path / "x.svg"
    )
    assert result["status"] == "skipped"
