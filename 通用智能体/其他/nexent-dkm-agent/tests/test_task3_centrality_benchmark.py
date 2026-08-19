from benchmarks.task3_centrality_benchmark import benchmark_task3_centrality_integration
from src.common.device import DeviceSpec
from src.operators.analysis_ops.graph_analytics import compute_centrality
from src.operators.npu_ops.graph_tensor_ops import (
    generate_synthetic_graph,
    generate_synthetic_graph_multi_type,
    compute_type_centrality_npu,
)


def test_task3_centrality_benchmark_reports_cpu_and_optimized_paths():
    report = benchmark_task3_centrality_integration(
        node_count=20,
        edge_count=40,
        iterations=2,
        prefer_device="cpu",
        seed=3,
    )

    assert report["task"] == "task3_centrality_integration"
    assert report["graph"]["node_count"] == 20
    assert report["graph"]["edge_count"] == 40
    assert report["graph"]["seed"] == 3
    assert report["graph"]["multi_type"] is False
    assert report["cpu_compute_centrality"]["status"] == "completed"
    assert report["cpu_compute_centrality"]["top_hubs_backend"] == "python"
    assert report["optimized_compute_centrality"]["status"] == "completed"
    assert report["optimized_compute_centrality"]["top_hubs_backend"] == "python"
    assert report["cached_npu_top_hubs_cpu_type_centrality"]["status"] == "completed"
    assert report["cached_npu_top_hubs_cpu_type_centrality"]["top_hubs_backend"] == "python"
    assert report["cached_npu_top_hubs_cpu_type_centrality"]["cache_status"] == "unavailable"
    assert report["correctness"]["status"] == "passed"
    assert report["cached_correctness"]["status"] == "passed"
    assert report["speedup"] is not None
    assert report["cached_speedup"] is not None


def test_task3_centrality_benchmark_accepts_amortized_runs():
    report = benchmark_task3_centrality_integration(
        node_count=20,
        edge_count=40,
        iterations=2,
        prefer_device="cpu",
        seed=3,
        amortized_runs=[1, 3],
    )

    assert report["amortized_cached_centrality"]["runs"] == [1, 3]
    assert len(report["amortized_cached_centrality"]["cpu_total_ms"]) == 2
    assert len(report["amortized_cached_centrality"]["cached_total_ms"]) == 2


def test_compute_type_centrality_npu_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    graph = generate_synthetic_graph(node_count=10, edge_count=20, seed=7)
    result = compute_type_centrality_npu(graph, prefer_device="auto")

    assert result["status"] == "unavailable"
    assert result["type_centrality"] == {}
    assert "npu" in result["reason"].lower()


def test_compute_centrality_produces_type_centrality_on_cpu():
    graph = generate_synthetic_graph(node_count=20, edge_count=40, seed=11)
    result = compute_centrality(graph, prefer_device="cpu")

    assert result["status"] == "completed"
    assert isinstance(result["type_centrality"], dict)
    assert "Synthetic" in result["type_centrality"]
    assert result["type_centrality"]["Synthetic"]["count"] == 20


def test_multi_type_graph_has_multiple_type_centrality():
    graph = generate_synthetic_graph_multi_type(node_count=30, edge_count=60, seed=13)
    result = compute_centrality(graph, prefer_device="cpu")

    assert result["status"] == "completed"
    types = result["type_centrality"]
    assert len(types) > 1
    type_names = set(types.keys())
    assert "Disease" in type_names
    assert "Symptom" in type_names
    assert "Drug" in type_names
    total_count = sum(t["count"] for t in types.values())
    assert total_count == 30


def test_multi_type_centrality_benchmark_report():
    report = benchmark_task3_centrality_integration(
        node_count=30,
        edge_count=60,
        iterations=2,
        prefer_device="cpu",
        seed=17,
        multi_type=True,
    )

    assert report["graph"]["multi_type"] is True
    assert report["graph"]["type_count"] > 1
    assert report["cpu_compute_centrality"]["status"] == "completed"
    assert report["correctness"]["status"] == "passed"
