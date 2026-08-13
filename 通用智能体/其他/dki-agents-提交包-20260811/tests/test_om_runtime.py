import os
import threading
import time

import pytest

from dki.bench.npu.om_manifest import OmCompatibilityError, OmManifest, TensorSpec
from dki.bench.npu.om_runtime import FailureKind, OmRuntime, OmRuntimeError, ProcessOmRuntime, ResourceStack
from dki.evidence import sha256_file


class FakeAdapter:
    def __init__(self, events, fail_open=False, fail_execute=False):
        self.events = events
        self.fail_open = fail_open
        self.fail_execute = fail_execute
        self.closed = False

    def open(self):
        self.events.append("open")
        if self.fail_open:
            raise RuntimeError("open")

    def execute(self, value):
        self.events.append(("execute", value))
        if self.fail_execute:
            self.fail_execute = False
            raise RuntimeError("execute")
        return value * 2

    def close(self):
        if not self.closed:
            self.events.append("close")
            self.closed = True


def test_resource_stack_closes_reverse_once():
    events = []
    stack = ResourceStack()
    stack.push(lambda: events.append(1))
    stack.push(lambda: events.append(2))
    stack.close()
    stack.close()
    assert events == [2, 1]


def test_runtime_restarts_whole_adapter_once():
    events = []
    count = 0

    def factory():
        nonlocal count
        count += 1
        return FakeAdapter(events, fail_execute=count == 1)

    runtime = OmRuntime(factory)
    assert runtime.execute(4) == 8
    assert runtime.health.restarts == 1
    assert events == ["open", ("execute", 4), "close", "open", ("execute", 4)]
    runtime.close()
    runtime.close()
    assert events[-1] == "close"


def test_runtime_fails_closed_after_restart_budget():
    events = []
    runtime = OmRuntime(lambda: FakeAdapter(events, fail_execute=True), max_restarts=0)
    with pytest.raises(OmRuntimeError) as raised:
        runtime.execute(1)
    assert raised.value.kind == FailureKind.ACL_FATAL
    assert runtime.health.circuit_open


def test_runtime_fails_closed_when_replacement_execute_fails():
    events = []
    runtime = OmRuntime(lambda: FakeAdapter(events, fail_execute=True), max_restarts=1)
    with pytest.raises(OmRuntimeError) as raised:
        runtime.execute(1)
    assert raised.value.kind == FailureKind.ACL_FATAL
    assert runtime.health.circuit_open
    assert runtime.health.restarts == 1
    assert runtime.health.failures == 2


def test_runtime_rejects_negative_restart_budget():
    with pytest.raises(ValueError, match="non-negative"):
        OmRuntime(lambda: FakeAdapter([]), max_restarts=-1)


def test_runtime_rejects_wrong_thread_after_first_use_claims_owner():
    runtime = OmRuntime(lambda: FakeAdapter([]))
    assert runtime.execute(1) == 2
    errors = []
    thread = threading.Thread(target=lambda: _capture(lambda: runtime.execute(1), errors))
    thread.start()
    thread.join()
    assert isinstance(errors[0], OmRuntimeError)
    assert errors[0].kind == FailureKind.COMPATIBILITY


def _capture(action, errors):
    try:
        action()
    except BaseException as exc:
        errors.append(exc)


def test_process_runtime_owns_adapter_in_spawn_child():
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        maxRestarts=0,
    )
    result = runtime.execute(4, timeout_s=30)
    assert result["value"] == 8
    assert result["pid"] != os.getpid()
    runtime.close()
    assert runtime.health.state == "closed"


def test_process_runtime_sends_ordered_batch_in_one_ipc_command():
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        maxRestarts=0,
    )
    results = runtime.execute_batch([3, 1, 4], timeout_s=30)
    assert [item["value"] for item in results] == [6, 2, 8]
    assert all(item["pid"] != os.getpid() for item in results)
    runtime.close()


def test_process_runtime_batch_cardinality_mismatch_fails_closed():
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        {"mode": "bad_batch_cardinality"},
        maxRestarts=0,
    )
    with pytest.raises(OmRuntimeError) as raised:
        # 30s matches the other spawn tests: slow-spawn hosts (aarch64 CI) need
        # the worker up before the cardinality check can classify COMPATIBILITY.
        runtime.execute_batch([1, 2], timeout_s=30)
    assert raised.value.kind == FailureKind.COMPATIBILITY
    assert runtime.health.circuit_open
    runtime.close()


def test_process_runtime_enforces_hard_timeout_and_opens_circuit():
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        {"mode": "hang"},
        maxRestarts=0,
    )
    with pytest.raises(OmRuntimeError) as raised:
        runtime.execute(1, timeout_s=0.1)
    assert raised.value.kind == FailureKind.TIMEOUT
    assert runtime.health.circuit_open
    runtime.close()


def test_process_runtime_rejects_more_than_one_restart():
    with pytest.raises(ValueError, match="zero or one"):
        ProcessOmRuntime(
            "tests.process_runtime_fixtures.process_test_adapter",
            maxRestarts=2,
        )


def test_process_runtime_crash_uses_bounded_restart_then_circuit():
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        {"mode": "crash"},
        maxRestarts=1,
    )
    started = time.monotonic()
    with pytest.raises(OmRuntimeError) as raised:
        runtime.execute(1, timeout_s=30)
    assert raised.value.kind == FailureKind.WORKER_CRASH
    assert time.monotonic() - started < 30
    assert runtime.health.restarts == 1
    assert runtime.health.circuit_open
    runtime.close()


def test_process_runtime_timeout_includes_waiting_for_execution_lock():
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        {"mode": "short_hang"},
        maxRestarts=0,
    )
    first_errors = []
    thread = threading.Thread(
        target=lambda: _capture(lambda: runtime.execute(1, timeout_s=2), first_errors),
    )
    thread.start()
    time.sleep(0.1)
    started = time.monotonic()
    with pytest.raises(OmRuntimeError) as raised:
        runtime.execute(2, timeout_s=0.05)
    elapsed = time.monotonic() - started
    assert raised.value.kind == FailureKind.TIMEOUT
    assert elapsed < 0.15
    thread.join()
    runtime.close()


def test_process_runtime_rejects_oversized_requests():
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        maxRestarts=0, maxBatchItems=1, maxPayloadBytes=128,
    )
    with pytest.raises(OmRuntimeError, match="batch has") as batch_error:
        runtime.execute_batch(["a", "b"])
    assert batch_error.value.kind == FailureKind.CONFIG
    with pytest.raises(OmRuntimeError, match="exceeds") as byte_error:
        runtime.execute("x" * 1024)
    assert byte_error.value.kind == FailureKind.CONFIG
    assert not runtime.health.circuit_open
    runtime.close()


def test_process_runtime_rejects_nonserializable_payload_immediately(monkeypatch):
    runtime = ProcessOmRuntime(
        "tests.process_runtime_fixtures.process_test_adapter",
        maxRestarts=0,
    )
    start_calls = []
    monkeypatch.setattr(runtime, "_start", lambda timeout_s: start_calls.append(timeout_s))
    with pytest.raises(OmRuntimeError) as raised:
        runtime.execute(lambda: None, timeout_s=1)
    assert raised.value.kind == FailureKind.CONFIG
    assert start_calls == []
    assert not runtime.health.circuit_open
    runtime.close()


def test_manifest_checks_hash_and_runtime(tmp_path):
    om = tmp_path / "model.om"
    om.write_bytes(b"om")
    input_spec = TensorSpec("input_ids", "int64", (64, 256))
    output_spec = TensorSpec("logits", "float32", (64, 256, 13))
    manifest = OmManifest(1, "bert", sha256_file(om), "t", "l", "Ascend910B2C", "8.5.1", (input_spec,), (output_spec,))
    manifest.verify_file(om)
    tokenizer = tmp_path / "tokenizer.json"
    labels = tmp_path / "labels.json"
    tokenizer.write_text("tokenizer", encoding="utf-8")
    labels.write_text("labels", encoding="utf-8")
    bound = OmManifest(
        1, "bert", sha256_file(om), sha256_file(tokenizer), sha256_file(labels),
        "Ascend910B2C", "8.5.1", (input_spec,), (output_spec,),
    )
    bound.verify_artifacts(om, tokenizer, labels)
    tokenizer.write_text("changed", encoding="utf-8")
    with pytest.raises(OmCompatibilityError, match="tokenizer hash mismatch"):
        bound.verify_artifacts(om, tokenizer, labels)
    manifest.verify_runtime(soc_version="Ascend910B2C", cann_version="8.5.1", inputs=[input_spec], outputs=[output_spec])
    with pytest.raises(OmCompatibilityError, match="SoC"):
        manifest.verify_runtime(soc_version="Ascend910B4", cann_version="8.5.1", inputs=[input_spec], outputs=[output_spec])
