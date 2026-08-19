"""Tests for the shared agent run tracker."""

from __future__ import annotations

import pytest

from src.common.run_tracker import AgentRunTracker, summarize_artifact


def test_run_tracker_records_completed_step():
    tracker = AgentRunTracker()
    tracker.start()
    value = tracker.run_step("load_graph", lambda: {"status": "completed", "node_count": 5}, "loaded")
    tracker.complete("completed")
    state = tracker.to_dict()
    assert value["node_count"] == 5
    assert state["status"] == "completed"
    assert state["steps"][0]["name"] == "load_graph"
    assert state["steps"][0]["status"] == "completed"


def test_run_tracker_marks_failed_step():
    tracker = AgentRunTracker()
    tracker.start()
    with pytest.raises(RuntimeError):
        tracker.run_step("fail_step", lambda: (_ for _ in ()).throw(RuntimeError("boom")), "x")
    assert tracker.to_dict()["status"] == "failed"


def test_summarize_artifact_handles_common_shapes():
    assert summarize_artifact("hello") == {"length": 5}
    assert summarize_artifact([1, 2]) == {"count": 2}
    summary = summarize_artifact({"status": "completed", "node_count": 3, "charts": {"a": 1}})
    assert summary["node_count"] == 3
    assert summary["chart_count"] == 1
