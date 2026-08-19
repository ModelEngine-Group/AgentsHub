"""Tests for Nexent/DataMate integration helpers and DKM orchestrator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.data_processing_agent.planner import HybridPlanner, plan_data_task
from src.agents.dkm_nexent_suite import build_dkm_nexent_suite_spec
from src.agents.dkm_orchestrator import DKMOrchestrator, plan_dkm_workflow
from src.common.integration import (
    build_integration_report,
    probe_datamate,
    probe_nexent,
    summarize_graph_for_planning,
)
from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner, plan_analysis_task
from scripts.datamate_readiness import evaluate_readiness

ROOT = Path(__file__).resolve().parents[1]


def test_probe_datamate_skips_when_url_is_none():
    report = probe_datamate("none")
    assert report["status"] == "skipped"


def test_probe_datamate_accepts_successful_core_apis_when_health_route_fails(
    monkeypatch,
):
    class FakeDataMateClient:
        def __init__(self, base_url, timeout):
            self.base_url = base_url

        def health(self):
            return {"status": "unavailable", "message": "HTTP Error 500"}

        def list_operators(self, size):
            return {"code": "0", "data": {"totalElements": 210}}

        def list_cleaning_templates(self, page, size):
            return {"code": "0", "data": {"totalElements": 7}}

        def list_cleaning_tasks(self, page, size):
            return {"code": "0", "data": {"totalElements": 3}}

    monkeypatch.setattr(
        "src.common.integration.DataMateClient",
        FakeDataMateClient,
    )

    report = probe_datamate("http://localhost:8080")

    assert report["status"] == "available"
    assert report["readiness_basis"] == "core_api_probes"
    assert report["health"]["status"] == "unavailable"


def test_probe_datamate_does_not_treat_health_only_as_fully_available(
    monkeypatch,
):
    class FakeDataMateClient:
        def __init__(self, base_url, timeout):
            self.base_url = base_url

        def health(self):
            return {"status": "healthy"}

        def list_operators(self, size):
            return {"status": "unavailable"}

        def list_cleaning_templates(self, page, size):
            return {"status": "unavailable"}

        def list_cleaning_tasks(self, page, size):
            return {"status": "unavailable"}

    monkeypatch.setattr(
        "src.common.integration.DataMateClient",
        FakeDataMateClient,
    )

    report = probe_datamate("http://localhost:8080")

    assert report["status"] == "partial"
    assert report["readiness_basis"] == "health_endpoint_only"


def test_datamate_readiness_requires_all_core_business_probes():
    readiness = evaluate_readiness(
        {
            "status": "partial",
            "readiness_basis": "health_endpoint_only",
            "successful_core_probes": 0,
            "core_probe_count": 3,
            "health": {"status": "healthy"},
        }
    )

    assert readiness["ready"] is False
    assert readiness["status"] == "partial"


def test_datamate_readiness_accepts_complete_core_business_probes():
    readiness = evaluate_readiness(
        {
            "status": "available",
            "readiness_basis": "core_api_probes",
            "successful_core_probes": 3,
            "core_probe_count": 3,
            "health": {"status": "unavailable"},
        }
    )

    assert readiness["ready"] is True
    assert readiness["successful_core_probes"] == 3


def test_probe_nexent_skips_when_url_is_none():
    report = probe_nexent("none")
    assert report["status"] == "skipped"


def test_probe_nexent_rejects_jupyter_tornado_false_positive(monkeypatch):
    class FakeResponse:
        status = 200
        headers = {"Server": "TornadoServer/6.5"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return b"<html><title>Jupyter Server</title></html>"[:size]

    monkeypatch.setattr("src.common.integration.urlopen", lambda *args, **kwargs: FakeResponse())

    report = probe_nexent("http://localhost:3000")

    assert report["status"] == "not_nexent"
    assert report["detected_service"] == "Jupyter/Tornado"


def test_probe_nexent_does_not_accept_unknown_http_service(monkeypatch):
    class FakeResponse:
        status = 200
        headers = {"Server": "nginx"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return b"<html><title>Unrelated App</title></html>"[:size]

    monkeypatch.setattr("src.common.integration.urlopen", lambda *args, **kwargs: FakeResponse())

    report = probe_nexent("http://localhost:3000")

    assert report["status"] == "unknown_http_service"


def test_probe_nexent_accepts_nextjs_nexent_web_via_api_fingerprint(monkeypatch):
    """The real Nexent v2.x web front-end is a Next.js app whose HTML body
    does not contain the word 'nexent'. The probe must confirm it by hitting
    a Nexent-specific JSON API endpoint."""

    call_count = {"n": 0}

    class HtmlResponse:
        status = 200
        headers = {"Server": "", "X-Powered-By": "Next.js"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return b'<!DOCTYPE html><html><script src="/_next/static/chunks/main.js">'[:size]

    class ApiResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return b'{"message":"success","data":[]}'[:size]

    def fake_urlopen(request, timeout=None):
        call_count["n"] += 1
        url = str(request.full_url) if hasattr(request, "full_url") else str(request)
        if "/api/tool/openapi_services" in url:
            return ApiResponse()
        return HtmlResponse()

    monkeypatch.setattr("src.common.integration.urlopen", fake_urlopen)

    report = probe_nexent("http://localhost:3000")
    assert report["status"] == "available"
    assert report["detected_service"] == "nexent"


def test_probe_nexent_accepts_nexent_fingerprint(monkeypatch):
    class FakeResponse:
        status = 200
        headers = {"Server": "nginx"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return b"<html><title>Nexent</title></html>"[:size]

    monkeypatch.setattr("src.common.integration.urlopen", lambda *args, **kwargs: FakeResponse())

    report = probe_nexent("http://localhost:3000")

    assert report["status"] == "available"
    assert report["detected_service"] == "nexent"


def test_build_integration_report_marks_stack_ready_when_both_skipped():
    report = build_integration_report(datamate_url="none", nexent_url="none")
    assert report["stack_status"] == "offline"
    assert "datamate" in report
    assert "nexent" in report


def test_build_integration_report_does_not_treat_not_nexent_as_ready(monkeypatch):
    monkeypatch.setattr(
        "src.common.integration.probe_datamate",
        lambda *args, **kwargs: {"status": "available"},
    )
    monkeypatch.setattr(
        "src.common.integration.probe_nexent",
        lambda *args, **kwargs: {"status": "not_nexent"},
    )

    report = build_integration_report()

    assert report["stack_status"] == "partial"


def test_build_integration_report_marks_skipped_service_as_partial(monkeypatch):
    monkeypatch.setattr(
        "src.common.integration.probe_datamate",
        lambda *args, **kwargs: {"status": "skipped"},
    )
    monkeypatch.setattr(
        "src.common.integration.probe_nexent",
        lambda *args, **kwargs: {"status": "available"},
    )

    report = build_integration_report()

    assert report["stack_status"] == "partial"


def test_environment_probe_does_not_claim_npu_model_when_npu_smi_fails(monkeypatch):
    from benchmarks.service_reachability_probe import _collect_environment_facts

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="failed")

    monkeypatch.setattr("subprocess.run", fake_run)

    facts = _collect_environment_facts()

    assert facts["npu_model"] == "unknown"


def test_runtime_analysis_respects_container_runtime_without_npu():
    from benchmarks.service_reachability_probe import _build_runtime_analysis

    analysis = _build_runtime_analysis(
        {
            "python_version": "3.12",
            "java_installed": False,
            "node_installed": False,
            "docker_installed": True,
            "podman_installed": False,
            "torch_npu_available": False,
            "npu_model": "unknown",
        }
    )

    assert analysis["conclusion"] == "存在容器运行能力，服务状态需以协议探测为准"
    assert "torch_npu" not in analysis["node_has"]
    assert "CANN" not in analysis["node_has"]


def test_dkm_suite_spec_exports_three_tools():
    spec = build_dkm_nexent_suite_spec(model_name="main_model")
    tool_names = {tool["name"] for tool in spec["tools"]}
    assert tool_names == {
        "task1_data_processing",
        "task2_medical_kg",
        "task3_graph_analysis",
    }
    assert "数据" in spec["instructions"] or "data" in spec["instructions"].lower()


def test_plan_dkm_workflow_detects_full_pipeline_request():
    plan = plan_dkm_workflow("请清洗医疗文本，构建知识图谱并生成分析洞察报告")
    assert [stage["task"] for stage in plan["stages"]] == ["task1", "task2", "task3"]
    assert plan["planner_mode"] == "rule"


def test_plan_dkm_workflow_detects_single_task_requests():
    task1_plan = plan_dkm_workflow("只清洗患者CSV并导出")
    assert [stage["task"] for stage in task1_plan["stages"]] == ["task1"]

    task3_plan = plan_dkm_workflow("基于已有图谱做NL2SQL分析和可视化")
    assert [stage["task"] for stage in task3_plan["stages"]] == ["task3"]


def test_llm_dkm_workflow_normalizes_stage_order_and_duplicates(monkeypatch):
    monkeypatch.setattr(
        "src.agents.data_processing_agent.llm_orchestrator.request_plan",
        lambda **kwargs: {
            "operators": [
                "task3_graph_analysis",
                "task1_data_processing",
                "task2_medical_kg",
                "task2_medical_kg",
            ],
            "confidence": 0.9,
        },
    )

    plan = plan_dkm_workflow(
        "清洗医疗文本，构建知识图谱并生成分析报告",
        llm_config={
            "base_url": "https://example.test/v1",
            "api_key": "test",
            "model_name": "test-model",
        },
    )

    assert plan["planner_mode"] == "llm"
    assert [stage["task"] for stage in plan["stages"]] == ["task1", "task2", "task3"]


def test_planner_comparison_reports_rule_and_hybrid_task1_plans():
    from src.agents.planner_comparison import compare_task1_planners

    report = compare_task1_planners(
        "请清洗患者CSV，删除重复记录，填补空值，并统一字段类型后导出",
        input_path=ROOT / "data" / "samples" / "task1_patients.csv",
    )

    assert report["task"] == "task1"
    assert report["rule_plan"]["operators"]
    assert report["hybrid_plan"]["operators"]
    assert report["hybrid_mode"] == "rule"


def test_planner_comparison_reports_dkm_orchestrator_stages():
    from src.agents.planner_comparison import compare_dkm_orchestrator_planners

    report = compare_dkm_orchestrator_planners(
        "请清洗医疗文本，构建知识图谱并生成分析洞察",
        question="哪些疾病关联最多症状？",
    )

    assert report["rule_plan"]["stages"]
    assert [stage["task"] for stage in report["rule_plan"]["stages"]] == [
        "task1",
        "task2",
        "task3",
    ]


def test_dkm_orchestrator_runs_partial_pipeline(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_task1(**kwargs):
        calls.append("task1")
        return {"status": "completed", "artifacts": {"processing": {"output_path": str(tmp_path / "clean.txt")}}}

    def fake_task2(**kwargs):
        calls.append("task2")
        return {"status": "completed", "artifacts": {"graph": {"output_path": str(tmp_path / "kg.json")}}}

    def fake_task3(**kwargs):
        calls.append("task3")
        return {"status": "completed", "artifacts": {}}

    monkeypatch.setattr("src.agents.dkm_orchestrator.run_task1_stage", fake_task1)
    monkeypatch.setattr("src.agents.dkm_orchestrator.run_task2_stage", fake_task2)
    monkeypatch.setattr("src.agents.dkm_orchestrator.run_task3_stage", fake_task3)

    orchestrator = DKMOrchestrator()
    result = orchestrator.run(
        "清洗文本并建图",
        output_root=tmp_path / "orchestrated",
    )
    assert calls == ["task1", "task2"]
    assert result["status"] == "completed"
    assert len(result["stages"]) == 2


def test_dkm_orchestrator_failure_includes_message(monkeypatch, tmp_path):
    """Orchestrator failure response should propagate the stage's message."""

    def fake_task1(**kwargs):
        return {
            "status": "failed",
            "message": "Input file not found",
            "artifacts": {},
        }

    monkeypatch.setattr("src.agents.dkm_orchestrator.run_task1_stage", fake_task1)

    orchestrator = DKMOrchestrator()
    result = orchestrator.run(
        "清洗文本并建图",
        output_root=tmp_path / "orchestrated",
    )

    assert result["status"] == "failed"
    assert result["failed_stage"] == "task1"
    assert result["message"] == "Input file not found"


def test_dkm_orchestrator_executes_full_pipeline_offline(tmp_path):
    """Real offline execution through task1 -> task2 -> task3 (no mocks)."""

    sample = ROOT / "data" / "samples" / "task1_medical_notes.txt"
    if not sample.is_file():
        pytest.skip("sample text missing")

    orchestrator = DKMOrchestrator(datamate_base_url=None, datamate_mode="dry_run")
    result = orchestrator.run(
        "请清洗医疗文本，构建知识图谱并生成分析洞察",
        output_root=tmp_path / "orchestrated",
        text_input=sample,
        question="哪些疾病关联最多症状？",
    )

    assert result["status"] == "completed"
    assert [stage["task"] for stage in result["stages"]] == ["task1", "task2", "task3"]
    assert (tmp_path / "orchestrated" / "task2" / "medical_kg.json").is_file()


def test_plan_data_task_enriched_with_datamate_catalog():
    profile = {
        "file_name": "patients.csv",
        "duplicate_rows": 1,
        "missing_cells": {"age": 1},
        "columns": [{"name": "age", "inferred_type": "integer"}],
    }
    plan = plan_data_task("清洗CSV并去重", data_profile=profile)
    enriched = HybridPlanner(datamate_operators=["DuplicateFilesFilter", "text_type_normalizer"]).enrich_plan(
        plan,
        datamate_catalog={
            "operator_count": 42,
            "sample_operator_ids": ["DuplicateFilesFilter", "UnicodeSpaceCleaner"],
            "candidate_mappings": {
                "drop_duplicate_rows": {
                    "selected_operator_ids": ["DuplicateFilesFilter"],
                    "support_level": "datamate",
                }
            },
        },
    )
    assert enriched.datamate_integration["status"] == "mapped"
    assert "DuplicateFilesFilter" in enriched.datamate_integration["selected_operator_ids"]
    assert any("DataMate" in item for item in enriched.rationale)


def test_graph_aware_analysis_planning_adds_graph_analytics_for_large_graph():
    graph_summary = summarize_graph_for_planning(
        {
            "nodes": [{"id": f"n{i}", "type": "Disease"} for i in range(30)],
            "edges": [{"source": f"n{i}", "target": f"n{i+1}", "type": "HAS_SYMPTOM"} for i in range(29)],
        }
    )
    plan = plan_analysis_task(
        "全面分析图谱统计与可视化洞察",
        question="哪些疾病最多？",
        graph_summary=graph_summary,
    )
    assert "graph_analytics" in plan["intent_keywords"]
    assert plan["graph_context"]["node_count"] == 30


def test_graph_summary_counts_task2_predicate_edges():
    summary = summarize_graph_for_planning(
        {
            "nodes": [
                {"id": "Disease:高血压", "type": "Disease"},
                {"id": "Symptom:头晕", "type": "Symptom"},
            ],
            "edges": [
                {
                    "source": "Disease:高血压",
                    "target": "Symptom:头晕",
                    "predicate": "has_symptom",
                }
            ],
        }
    )

    assert summary["relation_counts"] == {"has_symptom": 1}


def test_graph_aware_planning_skips_extended_analytics_for_question_only():
    graph_summary = summarize_graph_for_planning(
        {
            "nodes": [{"id": f"n{i}", "type": "Disease"} for i in range(30)],
            "edges": [{"source": f"n{i}", "target": f"n{i+1}", "type": "HAS_SYMPTOM"} for i in range(29)],
        }
    )
    plan = plan_analysis_task(question="哪些疾病关联最多症状？", graph_summary=graph_summary)
    assert "graph_analytics" not in plan["intent_keywords"]


def test_analysis_hybrid_planner_accepts_graph_summary():
    planner = AnalysisHybridPlanner()
    plan = planner.plan(
        "分析图谱统计和关联",
        question="高血压有哪些症状？",
        graph_summary={"node_count": 12, "edge_count": 18, "disease_count": 4},
    )
    assert plan["planner_mode"] == "rule"
    assert "statistics" in plan["intent_keywords"]


def test_fetch_datamate_catalog_hints_without_service(monkeypatch):
    from src.agents.data_processing_agent.agent import fetch_datamate_catalog_hints

    def fake_inspect(*args, **kwargs):
        return {"status": "unavailable", "operators": {"status": "skipped"}}

    monkeypatch.setattr(
        "src.agents.data_processing_agent.agent.inspect_datamate",
        fake_inspect,
    )
    hints = fetch_datamate_catalog_hints("http://localhost:18000")
    assert hints["status"] == "unavailable"
