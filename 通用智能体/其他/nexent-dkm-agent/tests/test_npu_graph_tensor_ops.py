from src.common.device import DeviceSpec
from src.operators.npu_ops.graph_tensor_ops import (
    benchmark_degree_centrality_npu_prepared,
    benchmark_graph_degree_centrality,
    compute_degree_centrality_cpu,
    compute_degree_centrality_npu,
    compute_degree_topk_npu_cached,
    generate_synthetic_graph,
    prepare_graph_degree_tensor_cache,
    profile_graph_degree_breakdown,
)


def test_generate_synthetic_graph_is_deterministic():
    first = generate_synthetic_graph(node_count=8, edge_count=12, seed=7)
    second = generate_synthetic_graph(node_count=8, edge_count=12, seed=7)

    assert first == second
    assert len(first["nodes"]) == 8
    assert len(first["edges"]) == 12
    assert first["metadata"] == {"synthetic": True, "seed": 7}


def test_generate_synthetic_graph_rejects_invalid_size():
    import pytest

    with pytest.raises(ValueError, match="node_count"):
        generate_synthetic_graph(node_count=0, edge_count=1)
    with pytest.raises(ValueError, match="too high"):
        generate_synthetic_graph(node_count=3, edge_count=4)


def test_cpu_degree_centrality_counts_undirected_edges():
    graph = {
        "nodes": [
            {"id": "n0", "name": "n0", "type": "Synthetic"},
            {"id": "n1", "name": "n1", "type": "Synthetic"},
            {"id": "n2", "name": "n2", "type": "Synthetic"},
        ],
        "edges": [
            {"source": "n0", "target": "n1"},
            {"source": "n0", "target": "n2"},
        ],
    }

    result = compute_degree_centrality_cpu(graph)

    assert result["status"] == "completed"
    assert result["degrees"] == [2, 1, 1]
    assert result["top_hubs"][0]["id"] == "n0"
    assert result["top_hubs"][0]["degree_centrality"] == 1.0


def test_npu_degree_centrality_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    graph = generate_synthetic_graph(node_count=4, edge_count=3, seed=1)
    result = compute_degree_centrality_npu(graph)

    assert result["status"] == "unavailable"
    assert "npu" in result["reason"].lower()


def test_degree_tensor_cache_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    graph = generate_synthetic_graph(node_count=4, edge_count=3, seed=1)
    result = prepare_graph_degree_tensor_cache(graph)

    assert result["status"] == "unavailable"
    assert result["operator"] == "graph_degree_tensor_cache"
    assert result["device"] == "cpu"
    assert "npu" in result["reason"].lower()


def test_degree_topk_cached_api_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    graph = generate_synthetic_graph(node_count=4, edge_count=3, seed=1)
    result = compute_degree_topk_npu_cached(graph, top_k=3)

    assert result["status"] == "unavailable"
    assert result["operator"] == "graph_degree_topk"
    assert result["kernel"] == "bincount"
    assert result["result_mode"] == "topk"
    assert result["top_hubs"] == []
    assert "npu" in result["reason"].lower()


def test_graph_degree_benchmark_reports_correctness_or_unavailable(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_graph_degree_centrality(
        node_count=10,
        edge_count=15,
        iterations=2,
        seed=3,
        amortized_runs=[1, 2, 5],
    )

    assert report["task"] == "task3_graph_tensor_degree"
    assert report["graph"] == {"node_count": 10, "edge_count": 15, "seed": 3}
    assert report["cpu"]["status"] == "completed"
    assert report["cpu"]["latency_ms_avg"] >= 0
    assert report["cpu"]["throughput_edges_per_sec"] > 15
    assert report["npu"]["status"] == "unavailable"
    assert report["npu_prepared"]["status"] == "unavailable"
    assert report["correctness"]["status"] == "not_run"
    assert report["speedup"] is None
    assert report["prepared_speedup"] is None
    assert report["amortized"]["runs"] == [1, 2, 5]
    assert len(report["amortized"]["cpu_total_ms"]) == 3
    assert report["amortized"]["npu_prepared_total_ms"] == [None, None, None]
    assert report["amortized"]["speedup"] == [None, None, None]
    assert report["amortized"]["breakeven_runs"] is None


def test_prepared_npu_benchmark_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    graph = generate_synthetic_graph(node_count=10, edge_count=15, seed=3)
    result = benchmark_degree_centrality_npu_prepared(graph=graph, iterations=2)

    assert result["metrics"]["status"] == "unavailable"
    assert result["metrics"]["measurement_mode"] == "prepared_kernel"
    assert result["result"]["status"] == "unavailable"


def test_graph_degree_breakdown_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    graph = generate_synthetic_graph(node_count=10, edge_count=15, seed=3)
    breakdown = profile_graph_degree_breakdown(graph=graph)

    assert breakdown["status"] == "unavailable"
    assert breakdown["measurement_mode"] == "breakdown_profile"
    assert breakdown["device"] == "cpu"
    assert set(breakdown["steps"]) >= {
        "node_index_build_ms",
        "edge_index_build_ms",
        "tensor_create_cpu_ms",
        "h2d_transfer_ms",
        "npu_buffer_alloc_ms",
        "kernel_zero_ms",
        "kernel_index_add_ms",
        "d2h_copy_ms",
        "tolist_ms",
        "format_result_ms",
    }
    assert all(value is None for value in breakdown["steps"].values())


def test_graph_degree_benchmark_includes_breakdown_when_requested(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_graph_degree_centrality(
        node_count=10,
        edge_count=15,
        iterations=2,
        seed=3,
        profile_breakdown=True,
    )

    assert report["breakdown"]["status"] == "unavailable"
    assert report["breakdown"]["measurement_mode"] == "breakdown_profile"


def test_graph_degree_benchmark_reports_requested_modes_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_graph_degree_centrality(
        node_count=10,
        edge_count=15,
        iterations=2,
        seed=3,
        benchmark_modes=[
            "baseline_index_add_full_format",
            "cached_index_add_full_format",
            "cached_index_add_topk",
            "cached_bincount_topk",
        ],
    )

    assert [mode["name"] for mode in report["mode_benchmarks"]] == [
        "baseline_index_add_full_format",
        "cached_index_add_full_format",
        "cached_index_add_topk",
        "cached_bincount_topk",
    ]
    assert {mode["status"] for mode in report["mode_benchmarks"]} == {"unavailable"}
    assert all(mode["measurement_mode"] == "mode_benchmark" for mode in report["mode_benchmarks"])


def test_graph_degree_benchmark_reports_edge_throughput(monkeypatch):
    import src.operators.npu_ops.graph_tensor_ops as graph_tensor_ops

    monkeypatch.setattr(
        graph_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )
    ticks = iter([0.0, 0.01, 0.01, 0.03])
    monkeypatch.setattr(graph_tensor_ops.time, "perf_counter", lambda: next(ticks))

    report = benchmark_graph_degree_centrality(
        node_count=10,
        edge_count=15,
        iterations=2,
        seed=3,
    )

    assert report["cpu"]["throughput_edges_per_sec"] == 1000.0
