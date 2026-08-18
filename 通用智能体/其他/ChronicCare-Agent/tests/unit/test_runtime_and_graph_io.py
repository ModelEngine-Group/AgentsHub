from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kg import graph_io
from runtime_common import analysis_context, cohort_context
from runtime_common.analysis_context import AnalysisContext, attach_analysis_context
from runtime_common.common import (
    Timer,
    build_result,
    count_missing_values,
    ensure_directory,
    payload_with_defaults,
    read_json,
    read_jsonl,
    relative_to_project,
    resolve_path,
    write_json,
    write_jsonl,
)


def test_common_json_and_jsonl_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "value.json"
    jsonl_path = tmp_path / "nested" / "rows.jsonl"
    write_json(json_path, {"中文": "值", "count": 2})
    count = write_jsonl(jsonl_path, [{"id": 1}, {"id": 2, "value": None}])
    assert read_json(json_path) == {"中文": "值", "count": 2}
    assert count == 2
    assert list(read_jsonl(jsonl_path)) == [{"id": 1}, {"id": 2, "value": None}]
    assert count_missing_values(read_jsonl(jsonl_path)) == 1


def test_common_paths_and_payload_helpers(tmp_path: Path) -> None:
    nested = ensure_directory(tmp_path / "a" / "b")
    assert nested.is_dir()
    assert resolve_path("configs").is_absolute()
    assert relative_to_project(tmp_path).startswith("/")
    result = build_result(
        task_id="t1",
        operator="clean",
        status="success",
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        metrics={"rows": 2},
    )
    payload = payload_with_defaults(
        task_id="t1",
        input_path=tmp_path / "in",
        export_path=tmp_path / "out",
        params={"overwrite": False},
    )
    assert result["metrics"] == {"rows": 2}
    assert result["errors"] == []
    assert payload["params"] == {"encoding": "utf-8", "overwrite": False}
    assert Timer().elapsed() >= 0


def test_analysis_context_window_cohort_and_attachment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analysis_context,
        "_read_current_metrics",
        lambda: {
            "data_version": "synthetic_chroniccare",
            "metric_definition_version": "2.0",
        },
    )
    monkeypatch.setattr(analysis_context, "_file_hash", lambda path: "a" * 64)
    context = AnalysisContext.current(as_of_date="2026-07-28")
    window = context.with_window(7).with_cohort("c1", {"disease": "hypertension"})
    payload = attach_analysis_context({}, window)
    assert context.data_version == "synthetic_chroniccare"
    assert context.sqlite_version == f"sha256:{'a' * 64}"
    assert window.window_start == "2026-07-28"
    assert window.window_end == "2026-08-03"
    assert window.cohort_definition == {"disease": "hypertension"}
    assert payload["analysis_context"]["cohort_id"] == "c1"
    assert payload["as_of_date"] == "2026-07-28"


def test_analysis_context_clamps_window_and_accepts_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis_context, "_read_current_metrics", lambda: {})
    monkeypatch.setattr(analysis_context, "_file_hash", lambda path: None)
    context = AnalysisContext.from_mapping(
        {
            "as_of_date": "2026-01-01",
            "timezone": "Asia/Shanghai",
            "data_version": "custom",
            "ignored": "value",
        }
    )
    assert context.data_version == "custom"
    assert context.sqlite_version == "unavailable"
    assert context.with_window(0).window_end == "2026-01-01"
    assert context.with_window(999).window_end == "2027-01-01"


def _redirect_context_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cohort_context,
        "_candidate_paths",
        lambda path_str: [tmp_path / Path(path_str).name],
    )


def test_legacy_cohort_state_keeps_last_twenty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_context_files(monkeypatch, tmp_path)
    cohort_context.save_last_cohort({})
    for index in range(22):
        cohort_context.save_last_cohort({"cohort_id": f"c{index}"})
    state = cohort_context.load_conversation_state()
    assert cohort_context.load_last_cohort()["cohort_id"] == "c21"
    assert len(state["history"]) == 20
    assert state["history"][0]["cohort_id"] == "c2"


def test_conversation_context_save_load_resolve_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_context_files(monkeypatch, tmp_path)
    first = cohort_context.save_conversation_cohort(
        "conversation/unsafe",
        {
            "cohort_id": "first",
            "cohort_label": "高血压",
            "patient_ids": ["must-not-persist"],
            "data_version": "v1",
        },
    )
    second = cohort_context.save_conversation_cohort(
        "conversation/unsafe",
        {"cohort_id": "second", "data_version": "v1"},
    )
    assert "patient_ids" not in first
    assert second["cohort_id"] == "second"
    latest = cohort_context.resolve_cohort_reference(
        "这些患者有多少？",
        "conversation/unsafe",
        current_data_version="v1",
    )
    previous = cohort_context.resolve_cohort_reference(
        "前一个群体有多少？",
        "conversation/unsafe",
        current_data_version="v1",
    )
    assert latest["cohort"]["cohort_id"] == "second"
    assert latest["resolution"] == "latest"
    assert previous["cohort"]["cohort_id"] == "first"
    assert previous["resolution"] == "previous"

    state_path = tmp_path / "conversationunsafe.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["cohorts"][-1]["expires_at"] = (datetime.now().astimezone() - timedelta(seconds=1)).isoformat()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    valid = cohort_context.load_conversation_cohorts(
        "conversation/unsafe",
        current_data_version="v1",
    )
    assert [item["cohort_id"] for item in valid] == ["first"]


def test_cohort_reference_clarification_and_context_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_context_files(monkeypatch, tmp_path)
    assert cohort_context.resolve_cohort_reference("高血压人数", "c")["status"] == "no_reference"
    unresolved = cohort_context.resolve_cohort_reference("这些患者人数", "c")
    assert unresolved["status"] == "needs_clarification"
    token = cohort_context.bind_conversation_context("c")
    try:
        assert cohort_context.get_current_conversation_id() == "c"
        monkeypatch.setattr(cohort_context, "active_data_version", lambda: None)
        active = cohort_context.resolve_active_cohort("普通问题")
        assert active["context_mode"] == "conversation_isolated"
        assert active["conversation_id"] == "c"
    finally:
        cohort_context.reset_conversation_context(token)
    assert cohort_context.get_current_conversation_id() is None


def test_graph_json_and_adjacency_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "graph" / "graph.json"
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {"source": "a", "target": "b", "relation": "related"},
            {"source": "b", "target": "a", "relation": "reverse"},
        ],
    }
    graph_io.save_graph_json(path, data)
    loaded = graph_io.load_graph_json(path)
    outgoing, incoming = graph_io.adjacency_indexes(loaded["edges"])
    assert loaded == data
    assert outgoing["a"][0]["target"] == "b"
    assert incoming["a"][0]["source"] == "b"


def test_graph_optional_dependencies_report_clear_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(graph_io, "nx", None)
    monkeypatch.setattr(graph_io, "pd", None)
    with pytest.raises(RuntimeError, match="networkx"):
        graph_io.build_multidigraph([], [])
    with pytest.raises(RuntimeError, match="pandas"):
        graph_io.write_nodes_csv(tmp_path / "nodes.csv", [])
    with pytest.raises(RuntimeError, match="pandas"):
        graph_io.write_edges_csv(tmp_path / "edges.csv", [])
