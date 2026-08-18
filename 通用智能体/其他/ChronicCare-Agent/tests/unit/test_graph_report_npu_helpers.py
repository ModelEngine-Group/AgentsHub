from __future__ import annotations

import json
import os
import runpy
import urllib.error
from pathlib import Path

import networkx as nx
import pytest

from analysis.open_sql import llm_sql_candidate
from kg import graph_visualize
from tool_server import app as tool_app
from tool_server import npu_tools, report_tools


def test_acceptance_freshness_scans_integration_config_and_deploy_sources(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(project_root / "scripts" / "final_competition_acceptance.py"))
    validation_input_mtime = namespace["validation_input_mtime"]
    validation_input_mtime.__globals__["ROOT"] = tmp_path

    source = tmp_path / "tool_server" / "app.py"
    integration = tmp_path / "integrations" / "datamate" / "metadata.yml"
    config = tmp_path / "configs" / "tool_server_config.yaml"
    dockerfile = tmp_path / "deploy" / "Dockerfile.runtime"
    for path in (source, integration, config, dockerfile):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    os.utime(source, (1, 1))
    os.utime(integration, (2, 2))
    os.utime(config, (3, 3))
    os.utime(dockerfile, (4, 4))

    latest, latest_mtime = validation_input_mtime()
    assert latest == dockerfile
    assert latest_mtime == 4


def _sample_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("Patient::P0001", type="Patient", display_name="患者一")
    graph.add_node("Disease::hypertension", type="Disease", display_name="高血压")
    graph.add_node("Indicator::hba1c", type="Indicator", display_name="糖化血红蛋白")
    graph.add_node("Drug::metformin", type="Drug", display_name="二甲双胍")
    graph.add_edge("Patient::P0001", "Disease::hypertension", relation="patient_has_disease")
    graph.add_edge("Disease::hypertension", "Indicator::hba1c", relation_type="associated_indicator")
    return graph


def test_graph_visualization_selects_focus_and_renders_html(tmp_path: Path) -> None:
    graph = _sample_graph()
    selected = graph_visualize.choose_subgraph_nodes(graph, max_nodes=3)
    assert "Disease::hypertension" in selected
    assert len(selected) == 3
    output = tmp_path / "graph.html"
    rendered, node_count, edge_count = graph_visualize.render_graph_html(
        graph, output, total_node_count=197_404, total_edge_count=396_928
    )
    content = output.read_text(encoding="utf-8")
    assert rendered is True
    assert node_count == 4
    assert edge_count == 2
    assert "197,404" in content
    assert "高血压" in content
    assert "patient_has_disease" in content
    assert "associated_indicator" in content


def test_graph_visualization_ranked_fallback_and_labels() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edges_from([("a", "b"), ("a", "c"), ("a", "d")])
    selected = graph_visualize.choose_subgraph_nodes(graph, max_nodes=2)
    assert "a" in selected and len(selected) == 2
    assert graph_visualize._label_for("Patient::P0099", {}) == "患者 P0099"
    assert graph_visualize._label_for("Disease::very_long_name", {}) == "very_long_name"
    assert graph_visualize._label_for("x", {"name": "<危险>"}) == "<危险>"
    rows = graph_visualize._iter_table_rows([("<script>", 2)])
    assert "&lt;script&gt;" in rows and "<script>" not in rows


def test_graph_visualization_fallback_html_escapes_content(tmp_path: Path) -> None:
    output = tmp_path / "fallback.html"
    graph_visualize.build_fallback_html(
        output,
        summary={"node_count": 10, "edge_count": 20},
        entity_type_count={"Disease": 2},
        relation_type_count={"related_to": 3},
        top_degree_nodes=[{"id": "Disease::hypertension", "type": "Disease", "label": "<高血压>"}],
        note="<仅作测试>",
    )
    content = output.read_text(encoding="utf-8")
    assert "&lt;高血压&gt;" in content
    assert "&lt;仅作测试&gt;" in content
    assert "当前展示边</div><div class='v'>0" in content


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(10, 4, 2.5), ("10", "2", 5.0), (1, 0, None), ("bad", 2, None), (None, 2, None)],
)
def test_npu_round_div(numerator, denominator, expected) -> None:
    assert npu_tools._round_div(numerator, denominator, 3) == expected


def test_npu_enrich_comparison_row_calculates_independent_metrics() -> None:
    row = npu_tools._enrich_comparison_row(
        {
            "cpu_benchmark_records": 2048,
            "npu_record_count": 4096,
            "cpu_bge_sample_seconds": 2.0,
            "npu_bge_sample_seconds": 0.5,
            "npu_bge_full_seconds": 1.5,
            "estimated_cpu_bge_full_seconds": 4.0,
            "cpu_resource_utilization_percent": 6400,
        }
    )
    assert row["cpu_sample_throughput_records_per_second"] == 1024.0
    assert row["npu_sample_throughput_records_per_second"] == 4096.0
    assert row["npu_full_throughput_records_per_second"] == 2730.67
    assert row["cpu_avg_latency_ms_per_record"] == 0.9766
    assert row["cpu_effective_cores"] == 64.0
    assert row["resource_metrics_status"] == "not_collected"


def test_npu_aggregate_comparison_rows_keeps_raw_runs_and_recomputes_speedup() -> None:
    rows = npu_tools._aggregate_comparison_rows([
        [{"operator": "chronic_entity_extract_model_npu", "cpu_benchmark_records": 2048,
          "npu_record_count": 4096, "cpu_bge_sample_seconds": 3.0,
          "npu_bge_sample_seconds": 0.3, "npu_bge_full_seconds": 1.0, "sample_speedup": 10.0}],
        [{"operator": "chronic_entity_extract_model_npu", "cpu_benchmark_records": 2048,
          "npu_record_count": 4096, "cpu_bge_sample_seconds": 6.0,
          "npu_bge_sample_seconds": 0.5, "npu_bge_full_seconds": 1.2, "sample_speedup": 12.0}],
        [{"operator": "chronic_entity_extract_model_npu", "cpu_benchmark_records": 2048,
          "npu_record_count": 4096, "cpu_bge_sample_seconds": 3.0,
          "npu_bge_sample_seconds": 0.4, "npu_bge_full_seconds": 1.1, "sample_speedup": 7.5}],
    ])
    row = rows[0]
    assert row["benchmark_repeat_count"] == 3
    assert row["cpu_bge_sample_seconds"] == 4.0
    assert row["npu_bge_sample_seconds"] == 0.4
    assert row["sample_speedup"] == 10.0
    assert row["cpu_bge_sample_seconds_runs"] == [3.0, 6.0, 3.0]


def test_npu_comparison_rows_normalizes_sidecar_and_resources() -> None:
    rows = npu_tools._npu_operator_comparison_rows(
        [
            {
                "operator": "entity",
                "status": "success",
                "artifact_paths": {
                    "npu_entity_standardized": "/tmp/chroniccare_datamate_full_pipeline/output/entity.jsonl"
                },
                "summary": {
                    "backend": "npu",
                    "fallback_used": False,
                    "model_inference": {
                        "cpu_benchmark_record_count": 100,
                        "cpu_total_model_seconds": 2.0,
                        "npu_seconds_per_record": 0.005,
                        "npu_record_count": 1000,
                        "npu_total_model_seconds": 3.0,
                        "cpu_compute_utilization_percent": 800,
                        "npu_resource_metrics": {
                            "status": "collected",
                            "average_power_watt": 120,
                            "estimated_energy_wh": 0.1,
                            "average_aicore_percent": 50,
                        },
                    },
                },
            }
        ],
        output_root="/host/output",
    )
    row = rows[0]
    assert row["npu_bge_sample_seconds"] == 0.5
    assert row["sample_speedup"] == 4.0
    assert row["sidecar_path"] == "/host/output/entity.jsonl"
    assert row["average_power_watt"] == 120
    assert row["cpu_effective_cores"] == 8.0


@pytest.mark.parametrize(
    ("steps", "backend", "fallback"),
    [
        ([], "cpu_fallback", True),
        ([{"summary": {"npu_available": True}}], "cpu_compat_npu_ready", False),
        ([{"summary": {"npu_available": True, "npu_execution_used": True}}], "datamate_npu", False),
        ([{"summary": {"fallback_used": True}}], "cpu_fallback", True),
    ],
)
def test_effective_npu_runtime(steps, backend: str, fallback: bool) -> None:
    result = npu_tools._effective_npu_runtime(steps)
    assert result["backend"] == backend
    assert result["fallback_required"] is fallback


def test_npu_markdown_builders_render_repeated_measurement_columns() -> None:
    markdown = npu_tools._benchmark_markdown(
        {
            "status": "success",
            "runtime": {"backend": "npu", "npu_available": True},
            "fallback_used": False,
            "timestamp": "now",
            "benchmark_repeat_count": 5,
            "npu_comparison_rows": [
                {
                    "operator": "entity",
                    "cpu_benchmark_records": 2048,
                    "npu_record_count": 10000,
                    "cpu_bge_sample_seconds": 2.0,
                    "npu_bge_sample_seconds": 0.5,
                    "npu_bge_full_seconds": 2.5,
                    "sample_speedup": 4.0,
                    "npu_batch_size": 1024,
                }
            ],
            "safety_note": "安全",
        }
    )
    assert "CPU（2048 条，5轮均值）" in markdown
    assert "NPU（全量）" in markdown
    assert "batch 1024" in markdown
    assert "安全" in markdown


def test_npu_supported_operators_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(npu_tools, "load_server_config", lambda: {})
    monkeypatch.setattr(npu_tools, "safety_note", lambda cfg: "安全")
    payload = npu_tools.npu_supported_operators()
    assert payload["supported_operator_count"] == 2
    assert payload["recommended_targets"] == npu_tools.SUPPORTED_NPU_OPERATOR_NAMES
    assert all("_npu" in item["operator"] for item in payload["supported_operators"])


def test_report_html_helpers_handle_empty_and_dual_links() -> None:
    assert report_tools._entry_button("x", "") == ""
    assert report_tools._entry_meta("x", "") == ""
    primary = report_tools._entry_button("打开", "/a", primary=True)
    assert "linear-gradient" in primary and "href='/a'" in primary
    card = report_tools._artifact_entry_card(
        title="报告", description="说明", browser_href="/browser", service_href="/service"
    )
    assert "浏览器入口" in card and "服务入口" in card
    same = report_tools._entry_table_row("名称", "说明", "/same", "/same")
    assert "同浏览器入口" in same


def test_report_dynamic_items_choose_latest_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    graph_dir = runtime / "graph_driven_analysis"
    chart_dir = runtime / "charts"
    graph_dir.mkdir(parents=True)
    chart_dir.mkdir(parents=True)
    for name in (
        "analysis_future_followup_chart_bundle_7d.html",
        "analysis_future_followup_chart_bundle_7d_chart.html",
    ):
        (graph_dir / name).write_text("x", encoding="utf-8")
    (chart_dir / "line_followup_trend_7d.svg").write_text("line", encoding="utf-8")
    (chart_dir / "pie_risk_distribution_7d.svg").write_text("pie", encoding="utf-8")

    def resolve(value: str) -> Path:
        if value.startswith("outputs/runtime_generated/"):
            return runtime / value.removeprefix("outputs/runtime_generated/")
        return tmp_path / "missing" / value

    monkeypatch.setattr(report_tools, "resolve_path", resolve)
    monkeypatch.setattr(report_tools, "relative_to_project", lambda path: str(path))
    monkeypatch.setattr(report_tools, "public_artifact_url", lambda cfg, path: f"http://public{path}")
    monkeypatch.setattr(report_tools, "service_artifact_url", lambda cfg, path: f"http://service{path}")
    items = report_tools._dynamic_followup_items({})
    assert len(items) == 3
    assert {item["kind"] for item in items} == {"html", "image"}
    assert all("7" in item["title"] for item in items)


@pytest.mark.parametrize(
    ("payload", "keys", "expected"),
    [
        ({"question": " q "}, ("question",), "q"),
        ({"query": 123}, ("query",), "123"),
        ({}, ("question",), ""),
        (None, ("question",), ""),
    ],
)
def test_tool_app_payload_value(payload, keys, expected) -> None:
    assert tool_app._request_payload_value(payload, *keys) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [({"days": 7}, 7), ({"days": "30"}, 30), ({"days": "bad"}, None), ({}, None)],
)
def test_tool_app_payload_int(payload, expected) -> None:
    assert tool_app._request_payload_int(payload, "days") == expected


def test_tool_app_alias_media_and_legacy_question_helpers() -> None:
    assert tool_app._guess_media_type(Path("x.html")) == "text/html"
    assert tool_app._guess_media_type(Path("x.svg")) == "image/svg+xml"
    assert tool_app._guess_media_type(Path("x.bin")) == "application/octet-stream"
    assert tool_app._normalize_subgraph_id("../高血压") == "高血压"
    assert tool_app._legacy_subgraph_query_from_id("hypertension") == "高血压的知识图谱子图"
    assert "高盐饮食" in tool_app._legacy_subgraph_query_from_id("high_salt_hypertension")
    assert tool_app._legacy_subgraph_query_from_id("unknown_english") is None
    assert tool_app._legacy_subgraph_query_from_id("糖尿病") == "糖尿病的知识图谱子图"
    assert tool_app._no_cache_headers()["Cache-Control"].startswith("no-store")


def _fake_urlopen(content: str, usage: dict | None = None):
    data = {"choices": [{"message": {"content": content}}], "usage": usage or {"total_tokens": 5}}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(data).encode("utf-8")

    return Response()


def test_llm_candidate_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_SQL_LLM_ENABLED", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = llm_sql_candidate.generate_llm_sql_candidate("患者数", {}, {})
    assert result["status"] == "skipped"


def test_llm_candidate_success_unsupported_empty_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_SQL_LLM_ENABLED", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(llm_sql_candidate, "_ensure_proxy_env", lambda: None)
    catalog = {"tables": {"patient_profile": {"fields": [{"name": "patient_id"}]}}}

    monkeypatch.setattr(
        llm_sql_candidate.urllib.request,
        "urlopen",
        lambda request, timeout: _fake_urlopen(
            '```json\n{"sql":"SELECT COUNT(*) FROM patient_profile","confidence":0.9,"reason":"统计"}\n```'
        ),
    )
    success = llm_sql_candidate.generate_llm_sql_candidate("患者数", {"tables": ["patient_profile"]}, catalog)
    assert success["status"] == "success"
    assert success["temperature"] == 0
    assert success["prompt_version"] == llm_sql_candidate.PROMPT_VERSION

    monkeypatch.setattr(
        llm_sql_candidate.urllib.request,
        "urlopen",
        lambda request, timeout: _fake_urlopen('{"sql":"UNSUPPORTED","confidence":0,"reason":"不支持"}'),
    )
    assert llm_sql_candidate.generate_llm_sql_candidate("未知", {}, catalog)["status"] == "unsupported"

    monkeypatch.setattr(
        llm_sql_candidate.urllib.request, "urlopen", lambda request, timeout: _fake_urlopen('{"sql":"","confidence":0}')
    )
    assert llm_sql_candidate.generate_llm_sql_candidate("空", {}, catalog)["reason"] == "llm_empty_sql"

    def fail(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(llm_sql_candidate.urllib.request, "urlopen", fail)
    failed = llm_sql_candidate.generate_llm_sql_candidate("失败", {}, catalog)
    assert failed["status"] == "failed"
    assert "offline" in failed["reason"]
