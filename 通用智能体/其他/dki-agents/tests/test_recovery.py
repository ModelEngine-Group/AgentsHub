from pathlib import Path

from dki.pipeline.recovery import (
    FailureCategory,
    RecoveryAction,
    decide_recovery,
    recovery_patch,
)
from dki.pipeline.state import TaskReport, TaskStatus


def test_task_status_truth_table():
    failed = TaskReport.new()
    failed.n_failed = 2
    failed.mark_finished()
    assert failed.status == TaskStatus.FAILED

    partial = TaskReport.new()
    partial.n_failed = 1
    partial.n_output = 1
    partial.mark_finished()
    assert partial.status == TaskStatus.PARTIAL_SUCCESS

    complete = TaskReport.new()
    complete.n_output = 1
    complete.mark_finished()
    assert complete.status == TaskStatus.COMPLETED

    stopped = TaskReport.new()
    stopped.status = TaskStatus.STOPPED
    stopped.mark_finished()
    assert stopped.status == TaskStatus.STOPPED


def test_task_report_atomic_save_and_legacy_fields(tmp_path):
    report = TaskReport.new("x")
    report.failure_category = "OOM"
    report.child_task_ids.append("child1")
    report.recovery_events.append({"category": "OOM", "action": "REDUCE_BATCH_ONCE"})
    path = report.save(tmp_path)
    loaded = TaskReport.load(tmp_path, report.task_id)
    assert path.exists()
    assert loaded["schema_version"] == 3
    assert loaded["failure_category"] == "OOM"
    assert loaded["child_task_ids"] == ["child1"]
    assert loaded["recovery_events"][0]["action"] == "REDUCE_BATCH_ONCE"
    assert not list(tmp_path.glob("*.tmp"))


def test_deterministic_failures_never_retry():
    decision = decide_recovery("model hash mismatch", idempotent=True)
    assert decision.category == FailureCategory.MODEL_HASH
    assert decision.action == RecoveryAction.NO_RETRY
    assert decision.max_attempts == 0


def test_non_idempotent_transient_is_quarantined():
    decision = decide_recovery("connection reset", idempotent=False)
    assert decision.action == RecoveryAction.QUARANTINE


def test_oom_and_acl_are_bounded_once():
    oom = decide_recovery("NPU out of memory", idempotent=True)
    acl = decide_recovery("ACL fatal context error", has_idempotency_key=True)
    assert oom.action == RecoveryAction.REDUCE_BATCH_ONCE and oom.max_attempts == 1
    assert recovery_patch(oom, {"batch": 64}) == {"batch": 32}
    assert acl.action == RecoveryAction.RESTART_WORKER_ONCE and acl.max_attempts == 1
