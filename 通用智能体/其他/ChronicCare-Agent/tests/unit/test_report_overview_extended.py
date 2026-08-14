from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_server import report_tools


def _patch_report_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    cfg = {
        "paths": {
            "analysis_report_html": "report.html",
            "analysis_report_md": "report.md",
            "chart_index": "charts.html",
            "graph_html": "graph.html",
            "demo_manifest": "manifest.json",
        }
    }
    monkeypatch.setattr(report_tools, "load_server_config", lambda: cfg)
    monkeypatch.setattr(report_tools, "load_current_metrics", lambda: {"data_version": "synthetic_chroniccare"})
    monkeypatch.setattr(report_tools, "safety_note", lambda _cfg: "仅用于辅助分析")
    monkeypatch.setattr(report_tools, "public_artifact_url", lambda _cfg, path: f"http://public{path}")
    monkeypatch.setattr(report_tools, "service_artifact_url", lambda _cfg, path: f"http://service{path}")
    monkeypatch.setattr(report_tools, "artifact_route_path", lambda path: f"/route{path}")
    monkeypatch.setattr(report_tools, "latest_subgraph_public_url", lambda _cfg: "http://public/latest")
    monkeypatch.setattr(report_tools, "latest_subgraph_service_url", lambda _cfg: "http://service/latest")
    monkeypatch.setattr(report_tools, "relative_to_project", lambda path: path.relative_to(tmp_path).as_posix())
    monkeypatch.setattr(report_tools, "resolve_path", lambda value: tmp_path / value)
    return cfg


def test_graph_items_latest_report_and_overviews(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_report_runtime(monkeypatch, tmp_path)
    graph_dir = tmp_path / "outputs/runtime_generated/graph_driven_analysis"
    chart_dir = tmp_path / "outputs/runtime_generated/charts"
    graph_dir.mkdir(parents=True)
    chart_dir.mkdir(parents=True)
    static = graph_dir / "analysis_high_salt_bp_abnormal_rate_chart.html"
    static.write_text("static", encoding="utf-8")
    latest_json = graph_dir / "analysis_latest.json"
    latest_json.write_text(
        json.dumps(
            {
                "analysis_id": "analysis_latest",
                "title": "最新分析",
                "graph_url": "http://graph/latest",
                "cohort_table_url": "http://table/latest",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (graph_dir / "analysis_latest_chart.html").write_text("chart", encoding="utf-8")
    (chart_dir / "NLQ001.html").write_text("legacy", encoding="utf-8")

    items = report_tools._graph_driven_items({})
    assert len(items) == 1
    assert items[0]["title"] == "高盐饮食与血压异常"
    latest = report_tools._latest_graph_driven_report({})
    assert latest is not None
    assert latest["analysis_id"] == "analysis_latest"
    assert latest["chart_url"].endswith("analysis_latest_chart")

    gallery = report_tools.chart_gallery_html()
    markdown = report_tools.report_overview_markdown()
    html = report_tools.report_overview_html()
    summary = report_tools.reports_summary()
    chart_list = report_tools.charts_list()
    assert "高盐饮食与血压异常" in gallery
    assert "已隐藏 1 个" in gallery
    assert "最新分析" in markdown
    assert "最新分析图表页" in markdown
    assert "最新分析专属子图" in markdown
    assert "最新分析全量患者列表" in markdown
    assert "最新图谱驱动分析" in html
    assert summary["latest_subgraph_url"] == "http://graph/latest"
    assert len(summary["entry_guide"]) == 4
    assert chart_list["public_chart_count"] == 1
    assert chart_list["hidden_legacy_chart_count"] == 1


def test_report_overviews_empty_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_report_runtime(monkeypatch, tmp_path)
    (tmp_path / "outputs/runtime_generated/charts").mkdir(parents=True)
    gallery = report_tools.chart_gallery_html()
    markdown = report_tools.report_overview_markdown()
    html = report_tools.report_overview_html()
    summary = report_tools.reports_summary()
    charts = report_tools.charts_list()
    assert "当前还没有可公开展示" in gallery
    assert "当前暂无公开图表" in markdown
    assert "当前暂无公开图表" in html
    assert summary["latest_graph_driven_analysis"] is None
    assert summary["latest_subgraph_url"] == "http://public/latest"
    assert charts["public_chart_count"] == 0
