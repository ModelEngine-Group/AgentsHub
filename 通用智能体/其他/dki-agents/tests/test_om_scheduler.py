"""Service-level NPU scheduler semantics (CPU, deterministic clock).

Covers max-batch/max-wait batching, stable reorder, absolute-deadline drop,
cancellation, static-batch tail padding+mask, failure circuit, and kill switch.
"""

import threading
import time

import pytest

from dki.bench.npu.om_scheduler import (
    NpuInferenceScheduler,
    RequestState,
    SchedulerError,
)


class _EchoBatchRuntime:
    """Records batch sizes; returns f(payload) per input, same order."""

    def __init__(self, func=lambda x: x, record=None):
        self.func = func
        self.batch_sizes = record if record is not None else []
        self.closed = False

    def execute_batch(self, payloads):
        self.batch_sizes.append(len(payloads))
        return [self.func(p) for p in payloads]

    def close(self):
        self.closed = True


def test_batches_by_max_size_and_preserves_order():
    runtime = _EchoBatchRuntime(func=lambda x: x * 10)
    sched = NpuInferenceScheduler(runtime, max_batch_size=4, max_wait_ms=50)
    handles = [sched.submit(i, deadline_s=5) for i in range(4)]
    results = [h.result(timeout_s=5) for h in handles]
    assert results == [0, 10, 20, 30]
    assert runtime.batch_sizes[0] == 4  # filled a full batch in one execute
    sched.kill()


def test_max_wait_flushes_partial_batch():
    runtime = _EchoBatchRuntime()
    sched = NpuInferenceScheduler(runtime, max_batch_size=8, max_wait_ms=20)
    handle = sched.submit("solo", deadline_s=5)
    assert handle.result(timeout_s=5) == "solo"
    assert runtime.batch_sizes == [1]
    sched.kill()


def test_absolute_deadline_drops_before_execution():
    release = threading.Event()

    def slow(_):
        release.wait(2)
        return _

    runtime = _EchoBatchRuntime(func=slow)
    sched = NpuInferenceScheduler(runtime, max_batch_size=1, max_wait_ms=0)
    blocker = sched.submit("block", deadline_s=5)
    # This one expires while the worker is busy on the blocker.
    late = sched.submit("late", deadline_s=0.05)
    time.sleep(0.2)
    release.set()
    with pytest.raises(SchedulerError):
        late.result(timeout_s=5)
    assert late.state == RequestState.DEADLINE
    assert blocker.result(timeout_s=5) == "block"
    assert sched.telemetry.dropped_deadline == 1
    sched.kill()


def test_cancellation_before_execution():
    release = threading.Event()
    runtime = _EchoBatchRuntime(func=lambda x: (release.wait(2), x)[1])
    sched = NpuInferenceScheduler(runtime, max_batch_size=1, max_wait_ms=0)
    blocker = sched.submit("block", deadline_s=5)
    victim = sched.submit("cancel-me", deadline_s=5)
    assert victim.cancel() is True
    release.set()
    with pytest.raises(SchedulerError):
        victim.result(timeout_s=5)
    assert victim.state == RequestState.CANCELLED
    assert blocker.result(timeout_s=5) == "block"
    assert sched.telemetry.cancelled == 1
    sched.kill()


def test_static_batch_pads_tail_and_masks_padding():
    seen = []

    class _StaticRuntime:
        def execute_batch(self, payloads):
            seen.append(list(payloads))
            return [p for p in payloads]  # returns all rows incl padding

        def close(self):
            pass

    sched = NpuInferenceScheduler(
        _StaticRuntime(), max_batch_size=2, max_wait_ms=50,
        static_batch_size=4, pad_payload="PAD",
    )
    h0 = sched.submit("a", deadline_s=5)
    h1 = sched.submit("b", deadline_s=5)
    assert h0.result(timeout_s=5) == "a"
    assert h1.result(timeout_s=5) == "b"
    # runtime saw a padded batch of 4, but only 2 real results were returned.
    assert seen[0] == ["a", "b", "PAD", "PAD"]
    assert sched.telemetry.batched == 2
    sched.kill()


def test_batch_failure_opens_circuit_and_isolates_text():
    class _BoomRuntime:
        def execute_batch(self, payloads):
            raise RuntimeError("raw text 高血压 leak attempt")

        def close(self):
            pass

    sched = NpuInferenceScheduler(_BoomRuntime(), max_batch_size=1, max_wait_ms=0)
    handle = sched.submit("x", deadline_s=5)
    with pytest.raises(SchedulerError) as raised:
        handle.result(timeout_s=5)
    assert "高血压" not in str(raised.value)  # only error type propagates
    assert sched.telemetry.circuit_open is True
    with pytest.raises(SchedulerError):
        sched.submit("y", deadline_s=5)
    sched.kill()


def test_kill_switch_cancels_inflight_queue():
    release = threading.Event()
    runtime = _EchoBatchRuntime(func=lambda x: (release.wait(2), x)[1])
    sched = NpuInferenceScheduler(runtime, max_batch_size=1, max_wait_ms=0)
    blocker = sched.submit("block", deadline_s=5)
    queued = sched.submit("queued", deadline_s=5)
    time.sleep(0.1)
    release.set()
    sched.kill()
    assert queued.state in {RequestState.KILLED, RequestState.DONE}
    assert runtime.closed is True


def test_telemetry_contains_no_free_text_fields():
    runtime = _EchoBatchRuntime()
    sched = NpuInferenceScheduler(runtime, max_batch_size=2, max_wait_ms=10)
    sched.submit("secret-note", deadline_s=5).result(timeout_s=5)
    snap = sched.telemetry.snapshot()
    assert set(snap) >= {"submitted", "completed", "avg_queue_wait_ms", "avg_batch_fill"}
    assert "secret-note" not in str(snap)
    sched.kill()
