from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tool_server import pipeline_tools
from visualization import chart_render


def _config() -> dict:
    return {
        "paths": {
            "sqlite_db": "db.sqlite",
            "graph_json": "graph.json",
            "graph_html": "graph.html",
            "analysis_report_html": "report.html",
            "chart_index": "charts.html",
        }
    }


def _patch_pipeline_runtime(monkeypatch: pytest.MonkeyPatch, reports: dict[str, dict] | None = None) -> None:
    reports = reports or {}
    monkeypatch.setattr(pipeline_tools, "load_server_config", _config)
    monkeypatch.setattr(pipeline_tools, "safety_note", lambda _cfg: "安全")
    monkeypatch.setattr(pipeline_tools, "public_artifact_url", lambda _cfg, path: f"http://public{path}")
    monkeypatch.setattr(
        pipeline_tools,
        "load_current_metrics",
        lambda: {
            "node_count": 10,
            "edge_count": 20,
            "quality_score_total": 90,
            "question_count": 240,
            "data_version": "synthetic_chroniccare",
        },
    )
    monkeypatch.setattr(pipeline_tools, "_load_json_if_exists", lambda path: reports.get(str(path), {}))


def test_pipeline_helpers_and_artifact_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_tools, "load_server_config", _config)
    monkeypatch.setattr(pipeline_tools, "safety_note", lambda _cfg: "安全")
    monkeypatch.setattr(pipeline_tools, "artifact_status", lambda path: {"path": path, "exists": True})
    assert pipeline_tools._format_seconds("1.23456", 2) == "1.23"
    assert pipeline_tools._format_seconds("bad") == "N/A"
    table = pipeline_tools._build_datamate_timing_table(
        [{"operator": "op", "execution_seconds": 1.2, "status": "success", "execution_seconds_is_reference": True}],
        {"pure_execution_seconds": 1, "pipeline_execution_seconds": 2, "outer_flow_seconds": 3},
    )
    assert table["detail_rows"][0]["是否参考值"] == "是"
    assert len(table["timing_rows"]) == 3
    hint = pipeline_tools.datamate_pipeline_run_cli_hint()
    assert hint["status"] == "success" and len(hint["cli_commands"]) == 3
    artifacts = pipeline_tools.artifacts_status()
    assert artifacts["status"] == "success"
    assert all(item["exists"] for item in artifacts["artifacts"].values())


def test_pipeline_status_without_and_with_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline_runtime(monkeypatch)
    empty = pipeline_tools.datamate_pipeline_status()
    assert empty["status"] == "not_started"
    assert len(empty["steps"]) == len(pipeline_tools.DATAMATE_OPERATORS)
    assert all(step["status"] == "unknown" for step in empty["steps"])

    reports = {
        pipeline_tools.DATAMATE_RUN_REPORT: {
            "status": "success",
            "timestamp": "2026-07-28T10:00:00+08:00",
            "pipeline_steps": [
                {
                    "operator": "chronic_file_ingest",
                    "status": "success",
                    "execution_seconds": 0,
                    "summary": {"rows": 1},
                    "artifact_keys": ["input"],
                },
                {"operator": "chronic_table_clean", "status": "success"},
            ],
            "pure_execution_seconds": 4,
            "pipeline_execution_seconds": 5,
            "outer_flow_seconds": 6,
            "output_root_on_host": "/output",
        },
        pipeline_tools.DATAMATE_CHECK_REPORT: {"status": "success"},
    }
    _patch_pipeline_runtime(monkeypatch, reports)
    result = pipeline_tools.datamate_pipeline_status()
    assert result["status"] == "success"
    assert result["run_id"].startswith("datamate_run_20260728T100000")
    assert (
        result["steps"][0]["execution_seconds"] == pipeline_tools.TIMING_REFERENCE["operators"]["chronic_file_ingest"]
    )
    assert result["steps"][1]["execution_seconds_is_reference"] is True
    assert result["timing"]["is_reference"] is False
    assert result["check_status"] == "success"


def test_pipeline_reports_latest_and_run_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = {
        pipeline_tools.DATAMATE_RUN_REPORT: {"status": "success", "timestamp": "now"},
        pipeline_tools.DATAMATE_CHECK_REPORT: {"status": "success"},
        pipeline_tools.DATAMATE_SYNC_REPORT: {"status": "success"},
    }
    _patch_pipeline_runtime(monkeypatch, reports)
    report = pipeline_tools.datamate_pipeline_report()
    assert report["status"] == "success"
    assert pipeline_tools.pipeline_reports() == report
    latest = pipeline_tools.datamate_pipeline_latest()
    assert latest["status"] == "success"
    assert latest["metrics"]["node_count"] == 10
    assert pipeline_tools.datamate_pipeline_status_by_run("latest")["status"] == "success"
    assert pipeline_tools.datamate_pipeline_report_by_run(report["run_id"])["status"] == "success"
    bad_status = pipeline_tools.datamate_pipeline_status_by_run("old")
    bad_report = pipeline_tools.datamate_pipeline_report_by_run("old")
    assert bad_status["status"] == bad_report["status"] == "failed"
    assert "Unsupported run_id" in bad_status["errors"][0]


def test_pipeline_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline_runtime(monkeypatch)
    result = pipeline_tools.datamate_pipelines()
    assert result["operator_count"] == 11
    assert result["npu_supported_operator_count"] == 2
    assert len(result["pipelines"]) == 3


def test_run_pipeline_skip_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_tools,
        "datamate_pipeline_status",
        lambda: {"status": "success", "run_id": "latest", "steps": [], "timing": {}},
    )
    monkeypatch.setattr(pipeline_tools, "_load_json_if_exists", lambda _path: {})
    skipped = pipeline_tools.run_datamate_pipeline("task")
    assert skipped["skipped"] is True

    statuses = iter(
        [
            {"status": "not_started"},
            {"status": "success", "run_id": "new", "steps": [], "timing": {}},
        ]
    )
    monkeypatch.setattr(pipeline_tools, "datamate_pipeline_status", lambda: next(statuses))
    monkeypatch.setattr(
        pipeline_tools,
        "_run_script",
        lambda name: subprocess.CompletedProcess([name], 0, stdout='{"status":"success"}', stderr=""),
    )
    success = pipeline_tools.run_datamate_pipeline("task", force=True)
    assert success["status"] == "success"
    assert len(success["commands"]) == 3
    assert success["commands"][0]["parsed_stdout"]["status"] == "success"

    statuses = iter(
        [
            {"status": "not_started"},
            {"status": "not_started", "steps": [], "timing": {}},
        ]
    )
    monkeypatch.setattr(pipeline_tools, "datamate_pipeline_status", lambda: next(statuses))
    monkeypatch.setattr(
        pipeline_tools,
        "_run_script",
        lambda name: subprocess.CompletedProcess([name], 2, stdout="not-json", stderr="boom"),
    )
    failed = pipeline_tools.run_datamate_pipeline("task", force=True)
    assert failed["status"] == "failed"
    assert failed["errors"] == ["boom"]
    assert failed["commands"][0]["parsed_stdout"] == {}
    assert len(failed["commands"]) == 1


def test_run_pipeline_npu_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_server import npu_tools

    monkeypatch.setattr(npu_tools, "run_npu_enhanced_pipeline", lambda **kwargs: kwargs)
    result = pipeline_tools.run_datamate_pipeline(
        "npu-task", force=True, safe_run=False, use_npu=True, npu_targets=["entity"], fallback=False
    )
    assert result["task_id"] == "npu-task"
    assert result["use_npu"] is True
    assert result["npu_targets"] == ["entity"]


def test_dag_graph_and_cancel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    graph = tmp_path / "dag.json"
    run = tmp_path / "run.json"

    def resolve(path: str) -> Path:
        return graph if path.endswith("dag.json") else run

    monkeypatch.setattr(pipeline_tools, "resolve_path", resolve)
    assert pipeline_tools.datamate_dag_graph("missing") == {"status": "not_found", "run_id": "missing"}
    graph.write_text('{"status":"ready"}', encoding="utf-8")
    assert pipeline_tools.datamate_dag_graph("run")["status"] == "ready"
    assert pipeline_tools.datamate_dag_cancel("missing")["status"] == "not_found"
    run.write_text('{"status":"succeeded"}', encoding="utf-8")
    assert pipeline_tools.datamate_dag_cancel("run")["status"] == "not_cancellable"
    run.write_text('{"status":"running"}', encoding="utf-8")
    result = pipeline_tools.datamate_dag_cancel("run")
    assert result["status"] == "cancelled"
    assert json.loads(run.read_text(encoding="utf-8"))["status"] == "cancelled"


class _FakeFigure:
    def __init__(self) -> None:
        self.layout: dict = {}

    def update_layout(self, **kwargs) -> None:
        self.layout.update(kwargs)

    def to_html(self, **_kwargs) -> str:
        return "<html>plotly</html>"


class _FakePlotly:
    def bar(self, *_args, **_kwargs) -> _FakeFigure:
        return _FakeFigure()

    def line(self, *_args, **_kwargs) -> _FakeFigure:
        return _FakeFigure()


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ([], ("value", "value")),
        (["only"], ("only", "only")),
        (["drug_category", "patient_count"], ("drug_category", "patient_count")),
        (["entity_type", "node_count"], ("entity_type", "node_count")),
        (["relation_type", "edge_count"], ("relation_type", "edge_count")),
        (["month", "count", "extra"], ("month", "count")),
        (["x", "y"], ("x", "y")),
    ],
)
def test_guess_xy_branches(columns: list[str], expected: tuple[str, str]) -> None:
    assert chart_render._guess_xy({"table": {"columns": columns}}) == expected


def test_chart_render_plotly_and_fallback_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {"chart_defaults": {"template": "white", "width": 800, "height": 500}}
    monkeypatch.setattr(chart_render, "load_plotly", lambda: _FakePlotly())
    for chart_type in ("bar", "line"):
        output = tmp_path / f"{chart_type}.html"
        result = chart_render.render_indicator_chart(
            {
                "chart_type": chart_type,
                "question": "趋势",
                "insight": "说明",
                "table": {"columns": ["month", "count"], "rows": [{"month": "7月", "count": 2}]},
            },
            output,
            config,
            "安全",
        )
        assert result == {"plotly_available": True, "fallback_used": False}
        assert "plotly" in output.read_text(encoding="utf-8")

    monkeypatch.setattr(chart_render, "load_plotly", lambda: None)
    for chart_type, columns, rows in (
        ("table", ["a"], [{"a": 1}]),
        ("bar", ["a"], [{"a": 1}]),
        ("line", ["a", "b"], []),
        ("unknown", ["a"], [{"a": 1}]),
    ):
        output = tmp_path / f"{chart_type}-{len(rows)}.html"
        chart_render.render_indicator_chart(
            {
                "chart_type": chart_type,
                "question": "测试",
                "insight": "说明",
                "table": {"columns": columns, "rows": rows},
            },
            output,
            config,
            "安全",
        )
        assert output.exists()


def test_graph_summary_and_quality_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {"chart_defaults": {"template": "white", "width": 800, "height": 500}}
    output = tmp_path / "summary.html"
    monkeypatch.setattr(chart_render, "load_plotly", lambda: _FakePlotly())
    result = chart_render.render_graph_summary_chart(
        "图谱", [{"type": "A", "count": 2}], "type", "count", output, config, "安全"
    )
    assert result["fallback_used"] is False
    monkeypatch.setattr(chart_render, "load_plotly", lambda: None)
    result = chart_render.render_graph_summary_chart(
        "图谱", [{"type": "A", "count": 2}], "type", "count", output, config, "安全"
    )
    assert result["fallback_used"] is True
    quality = tmp_path / "quality.html"
    chart_render.render_quality_score("质量", {"完整性": 95}, quality, "安全")
    assert "95" in quality.read_text(encoding="utf-8")
