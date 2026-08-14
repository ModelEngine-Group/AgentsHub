from __future__ import annotations

import json
import subprocess
import urllib.error

import pytest

from runtime_common import npu_runtime


def test_try_import_success_and_failure() -> None:
    assert npu_runtime._try_import("json") == (True, None)
    available, error = npu_runtime._try_import("module_that_does_not_exist_chroniccare")
    assert available is False and error


def test_run_probe_success_missing_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        npu_runtime.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=" ok ", stderr=""),
    )
    success = npu_runtime._run_probe(["npu-smi", "info"])
    assert success["available"] is True and success["stdout"] == "ok"

    def missing(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(npu_runtime.subprocess, "run", missing)
    assert "missing" in npu_runtime._run_probe(["missing"])["error"]

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.01)

    monkeypatch.setattr(npu_runtime.subprocess, "run", timeout)
    assert "timeout after" in npu_runtime._run_probe(["slow"], timeout=0.01)["error"]


def test_check_http_health_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size: int) -> bytes:
            return b"healthy"

    monkeypatch.setattr(npu_runtime.urllib.request, "urlopen", lambda request, timeout: Response())
    assert npu_runtime._check_http_health("http://service/health")["available"] is True

    def unavailable(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(npu_runtime.urllib.request, "urlopen", unavailable)
    failed = npu_runtime._check_http_health("http://service/health")
    assert failed["available"] is False and "offline" in failed["error"]


def test_detect_npu_runtime_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        npu_runtime,
        "_try_import",
        lambda name: (False, f"{name} unavailable"),
    )
    monkeypatch.setattr(npu_runtime, "_run_probe", lambda command: {"available": False})
    monkeypatch.setattr(
        npu_runtime,
        "_check_http_health",
        lambda url: {"url": url, "available": False},
    )
    monkeypatch.setattr(npu_runtime.os.path, "exists", lambda path: False)
    monkeypatch.setenv("ASCEND_HOME_PATH", "/private/host/cann")
    result = npu_runtime.detect_npu_runtime()
    assert result["backend"] == "cpu_fallback"
    assert result["cann_env"]["ASCEND_HOME_PATH"] == "configured"
    assert all(value == "configured" for value in result["cann_env"].values())
    assert result["fallback_required"] is True
    assert result["npu_available"] is False
    assert len(result["notes"]) >= 4


def test_select_backend_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        npu_runtime,
        "detect_npu_runtime",
        lambda: {
            "model_service_detected": True,
            "torch_npu_device_available": True,
            "backend": "original",
        },
    )
    assert npu_runtime.select_npu_backend(prefer_service=True)["backend"] == "http_model_service"
    assert npu_runtime.select_npu_backend(prefer_service=False)["backend"] == "torch_npu"


def test_run_with_fallback_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        npu_runtime,
        "select_npu_backend",
        lambda prefer_service=True: {"npu_available": True, "backend": "torch_npu", "reason": "ok"},
    )
    npu = npu_runtime.run_with_fallback(
        lambda: {"status": "success", "duration_sec": 0.1},
        lambda: {"status": "cpu"},
        use_npu=True,
    )
    assert npu["backend"] == "torch_npu" and npu["fallback_used"] is False

    def broken_npu():
        raise RuntimeError("boom")

    fallback = npu_runtime.run_with_fallback(
        broken_npu,
        lambda: {"status": "cpu", "duration_sec": 0.2},
        use_npu=True,
    )
    assert fallback["backend"] == "cpu_fallback"
    assert fallback["fallback_used"] is True
    assert "boom" in fallback["fallback_reason"]

    with pytest.raises(RuntimeError, match="boom"):
        npu_runtime.run_with_fallback(
            broken_npu,
            lambda: {"status": "cpu"},
            use_npu=True,
            fallback=False,
        )

    monkeypatch.setattr(
        npu_runtime,
        "select_npu_backend",
        lambda prefer_service=True: {"npu_available": False, "backend": "cpu_fallback", "reason": "no device"},
    )
    cpu = npu_runtime.run_with_fallback(
        lambda: {"status": "npu"},
        lambda: {"status": "cpu"},
        use_npu=False,
    )
    assert cpu["fallback_used"] is False
    assert "use_npu=false" in cpu["fallback_reason"]
    with pytest.raises(RuntimeError, match="no device"):
        npu_runtime.run_with_fallback(
            lambda: {"status": "npu"},
            lambda: {"status": "cpu"},
            use_npu=True,
            fallback=False,
        )


def test_npu_report_and_json_serialization() -> None:
    payload = {
        "status": "success",
        "npu_available": True,
        "backend": "torch_npu",
        "recommended_npu_targets": ["entity"],
        "notes": ["ready"],
        "safety_note": "安全",
    }
    report = npu_runtime.to_markdown_report("NPU 状态", payload)
    serialized = npu_runtime.dumps_json(payload)
    assert "# NPU 状态" in report and "- entity" in report and "ready" in report
    assert json.loads(serialized)["backend"] == "torch_npu"
