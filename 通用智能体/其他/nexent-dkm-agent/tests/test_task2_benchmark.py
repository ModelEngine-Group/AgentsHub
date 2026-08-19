import sys

from src.operators.npu_ops.kg_benchmark import benchmark_task2_kg_ops, detect_npu_runtime


SAMPLE_TEXT = """
记录 1:
患者赵六。既往有高血压病史，现诊断为高血压并发冠心病，建议心电图检查。
"""


def test_task2_kg_benchmark_reports_cpu_metrics():
    report = benchmark_task2_kg_ops(SAMPLE_TEXT, iterations=1)

    assert report["task"] == "task2_kg_agent"
    assert report["input"]["record_count"] == 1
    assert report["cpu"]["status"] == "completed"
    assert report["cpu"]["iterations"] == 1
    assert report["cpu"]["triple_count"] > 0
    assert report["cpu"]["latency_ms_avg"] >= 0
    assert report["cpu"]["throughput_records_per_sec"] > 0
    assert report["npu"]["status"] in {"available", "unavailable"}


def test_detect_npu_runtime_is_non_destructive():
    runtime = detect_npu_runtime()

    assert "status" in runtime
    assert "backend" in runtime
    if runtime["status"] == "unavailable":
        assert runtime["reason"]


def test_detect_npu_runtime_reports_torch_npu_probe(monkeypatch):
    """When torch_npu is available, report a small tensor probe separately."""

    class FakeTensor:
        device = "npu:0"

        def npu(self):
            return self

        def __matmul__(self, other):
            return self

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def synchronize():
            return None

    class FakeTorch:
        npu = FakeNpu()

        @staticmethod
        def randn(*_shape):
            return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch_npu", object())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    runtime = detect_npu_runtime(probe_iterations=2, probe_size=4)

    assert runtime["status"] == "available"
    assert runtime["backend"] == "Ascend PyTorch NPU"
    assert runtime["runtime_probe"]["status"] == "completed"
    assert runtime["runtime_probe"]["device_count"] == 1
    assert runtime["runtime_probe"]["iterations"] == 2
    assert runtime["runtime_probe"]["matrix_shape"] == [4, 4]
    assert runtime["runtime_probe"]["latency_ms_avg"] >= 0


def test_task2_benchmark_forwards_npu_probe_options(monkeypatch):
    import src.operators.npu_ops.kg_benchmark as kg_benchmark

    captured = {}

    def fake_detect_npu_runtime(**kwargs):
        captured.update(kwargs)
        return {"status": "available", "backend": "fake"}

    monkeypatch.setattr(kg_benchmark, "detect_npu_runtime", fake_detect_npu_runtime)

    report = kg_benchmark.benchmark_task2_kg_ops(
        SAMPLE_TEXT,
        iterations=1,
        npu_probe=False,
        npu_probe_iterations=7,
        npu_probe_size=8,
    )

    assert captured == {"probe": False, "probe_iterations": 7, "probe_size": 8}
    assert report["input"]["npu_probe"] == {
        "enabled": False,
        "iterations": 7,
        "matrix_size": 8,
    }
    assert report["npu"]["backend"] == "fake"
