"""Tests for the Task1 -> Task2 -> Task3 closed-loop pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.results import PipelineResult
from src.pipelines.end_to_end_pipeline import run_end_to_end_pipeline

SAMPLE_TEXT = (
    "记录 1:\n患者张三，头晕、头痛，既往有高血压病史，长期服用氨氯地平，"
    "建议血常规检查，继续服用阿司匹林。\n"
    "---\n"
    "记录 2:\n患者李四，口渴、多尿，既往有糖尿病病史，长期服用二甲双胍，"
    "建议尿常规、血糖监测。\n"
)


def test_end_to_end_closed_loop(tmp_path):
    raw = tmp_path / "notes.txt"
    raw.write_text(SAMPLE_TEXT, encoding="utf-8")

    report = run_end_to_end_pipeline(
        text_input=raw,
        output_root=tmp_path / "e2e",
        question="高血压有哪些症状和用药？",
    )

    assert report["status"] == "completed"
    assert [s["task"] for s in report["stages"]] == [
        "task1_data_processing_agent",
        "task2_kg_agent",
        "task3_analysis_agent",
    ]


def test_end_to_end_reuses_stage_outputs(tmp_path):
    raw = tmp_path / "notes.txt"
    raw.write_text(SAMPLE_TEXT, encoding="utf-8")

    report = run_end_to_end_pipeline(
        text_input=raw,
        output_root=tmp_path / "e2e",
    )

    flow = report["data_flow"]
    # Each stage consumes the previous stage's output artifact.
    assert Path(flow["raw_text"]) == raw
    assert Path(flow["cleaned_text"]).exists()
    assert Path(flow["knowledge_graph"]).exists()
    assert flow["cleaned_text"].endswith("_cleaned.txt")

    # Task 3 produced real analysis over the reused graph.
    assert report["task2"]["graph"]["node_count"] > 0
    assert report["task3"]["statistics"]["graph_size"]["node_count"] > 0
    assert report["task3"]["plan_execution"]["extended_analytics"] is True


def test_end_to_end_reports_failed_stage(tmp_path):
    """A failure in the first stage should short-circuit and be reported."""
    report = run_end_to_end_pipeline(
        text_input=tmp_path / "does_not_exist.txt",
        output_root=tmp_path / "e2e",
    )

    assert report["status"] == "failed"
    assert report["failed_stage"] == "task1"
    assert len(report["stages"]) == 1


def test_end_to_end_passes_llm_and_datamate_options(monkeypatch, tmp_path):
    """The closed loop should forward shared runtime options to stage agents."""
    import src.pipelines.end_to_end_pipeline as pipeline

    captured = {}
    llm_config = {
        "api_key": "sk-test",
        "base_url": "https://example.test/v1",
        "model_name": "glm-5.1",
    }

    class FakeTask1:
        task_name = "task1_data_processing_agent"

        def __init__(self, llm_config=None):
            captured["task1_llm_config"] = llm_config

        def run(self, **kwargs):
            captured["task1_kwargs"] = kwargs
            cleaned = tmp_path / "cleaned.txt"
            cleaned.write_text("cleaned medical text", encoding="utf-8")
            return PipelineResult(
                task=self.task_name,
                status="completed",
                message="ok",
                artifacts={
                    "processing": {"output_path": str(cleaned)},
                    "input": {"format": "text"},
                    "quality_report": {},
                },
            )

    class FakeTask2:
        task_name = "task2_kg_agent"

        def __init__(self, llm_config=None):
            captured["task2_llm_config"] = llm_config

        def run(self, **kwargs):
            captured["task2_kwargs"] = kwargs
            graph = tmp_path / "graph.json"
            graph.write_text("{}", encoding="utf-8")
            return PipelineResult(
                task=self.task_name,
                status="completed",
                message="ok",
                artifacts={
                    "graph": {"output_path": str(graph), "node_count": 1},
                    "qa": {},
                },
            )

    class FakeTask3:
        task_name = "task3_analysis_agent"

        def __init__(self, llm_config=None):
            captured["task3_llm_config"] = llm_config

        def run(self, **kwargs):
            captured["task3_kwargs"] = kwargs
            return PipelineResult(
                task=self.task_name,
                status="completed",
                message="ok",
                artifacts={
                    "insight_report": {},
                    "statistics": {},
                    "nl2sql": {},
                    "plan_execution": {},
                },
            )

    monkeypatch.setattr(pipeline, "DataProcessingAgent", FakeTask1)
    monkeypatch.setattr(pipeline, "MedicalKGAgent", FakeTask2)
    monkeypatch.setattr(pipeline, "GraphAnalysisAgent", FakeTask3)

    report = pipeline.run_end_to_end_pipeline(
        text_input=tmp_path / "raw.txt",
        output_root=tmp_path / "e2e",
        llm_config=llm_config,
        datamate_base_url="http://localhost:18000",
        datamate_mode="dry_run",
        datamate_timeout=7.5,
    )

    assert report["status"] == "completed"
    assert captured["task1_llm_config"] is llm_config
    assert captured["task2_llm_config"] is llm_config
    assert captured["task3_llm_config"] is llm_config
    assert captured["task1_kwargs"]["datamate_base_url"] == "http://localhost:18000"
    assert captured["task1_kwargs"]["datamate_mode"] == "dry_run"
    assert captured["task1_kwargs"]["datamate_timeout"] == 7.5


def test_end_to_end_demo_cli_rejects_incomplete_llm_config(tmp_path):
    import os
    import subprocess

    bad_config = tmp_path / "llm_config.env"
    bad_config.write_text("OPENAI_BASE_URL=https://example.test/v1\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "demos/end_to_end_demo.py",
            "--llm-config",
            str(bad_config),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    assert proc.returncode == 2
    assert "LLM config is missing or incomplete" in proc.stdout


def test_end_to_end_falls_back_to_cleaning_output_path_for_csv(monkeypatch, tmp_path):
    """When Task 1 outputs CSV (cleaning.output_path), the pipeline should fallback correctly."""
    import src.pipelines.end_to_end_pipeline as pipeline

    captured = {}

    class FakeTask1CSV:
        task_name = "task1_data_processing_agent"

        def __init__(self, llm_config=None):
            captured["task1_llm_config"] = llm_config

        def run(self, **kwargs):
            captured["task1_kwargs"] = kwargs
            cleaned = tmp_path / "cleaned.txt"
            cleaned.write_text("cleaned medical text", encoding="utf-8")
            # CSV path returns cleaning.output_path, NOT processing.output_path
            return PipelineResult(
                task=self.task_name,
                status="completed",
                message="ok",
                artifacts={
                    "cleaning": {"output_path": str(cleaned)},
                    "processing": {},  # Empty for CSV path
                    "input": {"format": "csv"},
                    "quality_report": {},
                },
            )

    class FakeTask2:
        task_name = "task2_kg_agent"

        def __init__(self, llm_config=None):
            captured["task2_llm_config"] = llm_config

        def run(self, **kwargs):
            captured["task2_kwargs"] = kwargs
            graph = tmp_path / "graph.json"
            graph.write_text("{}", encoding="utf-8")
            return PipelineResult(
                task=self.task_name,
                status="completed",
                message="ok",
                artifacts={
                    "graph": {"output_path": str(graph), "node_count": 1},
                    "qa": {},
                },
            )

    class FakeTask3:
        task_name = "task3_analysis_agent"

        def __init__(self, llm_config=None):
            captured["task3_llm_config"] = llm_config

        def run(self, **kwargs):
            captured["task3_kwargs"] = kwargs
            return PipelineResult(
                task=self.task_name,
                status="completed",
                message="ok",
                artifacts={
                    "insight_report": {},
                    "statistics": {},
                    "nl2sql": {},
                    "plan_execution": {},
                },
            )

    monkeypatch.setattr(pipeline, "DataProcessingAgent", FakeTask1CSV)
    monkeypatch.setattr(pipeline, "MedicalKGAgent", FakeTask2)
    monkeypatch.setattr(pipeline, "GraphAnalysisAgent", FakeTask3)

    report = pipeline.run_end_to_end_pipeline(
        text_input=tmp_path / "raw.csv",
        output_root=tmp_path / "e2e",
    )

    assert report["status"] == "completed"
    # Verify that cleaned_text was extracted from cleaning.output_path fallback
    assert Path(report["data_flow"]["cleaned_text"]).exists()
    assert Path(report["data_flow"]["cleaned_text"]).name == "cleaned.txt"
