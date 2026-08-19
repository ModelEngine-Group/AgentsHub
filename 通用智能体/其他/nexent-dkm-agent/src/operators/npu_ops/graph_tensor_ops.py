"""Tensorized graph-degree operators for CPU/NPU benchmark experiments."""

from __future__ import annotations

import importlib
import random
import time
from statistics import mean
from typing import Any

from src.common.device import get_device

GRAPH_DEGREE_BENCHMARK_MODES = (
    "baseline_index_add_full_format",
    "cached_index_add_full_format",
    "cached_index_add_topk",
    "cached_bincount_topk",
)


def generate_synthetic_graph(
    node_count: int,
    edge_count: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a deterministic undirected synthetic graph."""

    if node_count < 1:
        raise ValueError("node_count must be >= 1")
    max_edges = node_count * (node_count - 1) // 2
    if edge_count < 0:
        raise ValueError("edge_count must be >= 0")
    if edge_count > max_edges:
        raise ValueError(f"edge_count is too high for {node_count} nodes")

    rng = random.Random(seed)
    edges_seen: set[tuple[int, int]] = set()
    edges = []
    while len(edges) < edge_count:
        src = rng.randrange(node_count)
        tgt = rng.randrange(node_count)
        if src == tgt:
            continue
        a, b = sorted((src, tgt))
        key = (a, b)
        if key in edges_seen:
            continue
        edges_seen.add(key)
        edges.append(
            {
                "source": f"node_{a}",
                "target": f"node_{b}",
                "predicate": "synthetic_relation",
            }
        )

    return {
        "nodes": [
            {"id": f"node_{idx}", "name": f"node_{idx}", "type": "Synthetic"}
            for idx in range(node_count)
        ],
        "edges": edges,
        "metadata": {"synthetic": True, "seed": seed},
    }


MEDICAL_NODE_TYPES = [
    "Disease",
    "Symptom",
    "Drug",
    "Procedure",
    "BodyPart",
    "Department",
]


def generate_synthetic_graph_multi_type(
    node_count: int,
    edge_count: int,
    seed: int = 42,
    node_types: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a synthetic graph with multiple node types (simulates medical KG).

    Nodes are assigned types in a round-robin fashion from ``node_types``.
    Edge predicates are chosen based on source/target type pairs to simulate
    realistic medical relationships.
    """

    types = node_types or MEDICAL_NODE_TYPES
    graph = generate_synthetic_graph(node_count=node_count, edge_count=edge_count, seed=seed)

    # Reassign node types round-robin
    for idx, node in enumerate(graph["nodes"]):
        node["type"] = types[idx % len(types)]

    # Assign more realistic predicates based on type pairs
    type_pair_predicates = {
        ("Disease", "Symptom"): "has_symptom",
        ("Symptom", "Disease"): "indicates",
        ("Disease", "Drug"): "treated_by",
        ("Drug", "Disease"): "treats",
        ("Disease", "Procedure"): "diagnosed_by",
        ("Procedure", "Disease"): "diagnoses",
        ("Disease", "BodyPart"): "affects",
        ("BodyPart", "Disease"): "affected_by",
        ("Disease", "Department"): "belongs_to",
        ("Department", "Disease"): "contains",
        ("Drug", "Symptom"): "causes_side_effect",
        ("Symptom", "Drug"): "side_effect_of",
        ("Drug", "Drug"): "interacts_with",
        ("Procedure", "BodyPart"): "targets",
        ("BodyPart", "Procedure"): "targeted_by",
    }

    node_type_map = {node["id"]: node["type"] for node in graph["nodes"]}
    for edge in graph["edges"]:
        src_type = node_type_map.get(edge["source"], "Synthetic")
        tgt_type = node_type_map.get(edge["target"], "Synthetic")
        edge["predicate"] = type_pair_predicates.get(
            (src_type, tgt_type),
            "related_to",
        )

    graph["metadata"]["multi_type"] = True
    graph["metadata"]["node_types"] = types
    return graph


def compute_degree_centrality_cpu(graph: dict[str, Any]) -> dict[str, Any]:
    """Compute undirected degree centrality using Python lists."""

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    index = {node["id"]: idx for idx, node in enumerate(nodes)}
    degrees = [0] * len(nodes)
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in index and tgt in index:
            degrees[index[src]] += 1
            degrees[index[tgt]] += 1

    return _format_degree_result(
        graph=graph,
        degrees=degrees,
        backend="python",
        device="cpu",
    )


def compute_degree_centrality_npu(
    graph: dict[str, Any],
    prefer_device: str = "auto",
) -> dict[str, Any]:
    """Compute undirected degree centrality with torch tensors on Ascend NPU."""

    device = get_device(prefer_device)
    if device.kind != "npu":
        return {
            "status": "unavailable",
            "backend": "torch",
            "device": device.device,
            "reason": f"NPU device is unavailable: {device.reason}",
        }

    try:
        torch = importlib.import_module("torch")
        node_count = len(graph.get("nodes", []))
        sources, targets = _edge_indices(graph)
        src = torch.tensor(sources, dtype=torch.long, device=device.device)
        tgt = torch.tensor(targets, dtype=torch.long, device=device.device)
        ones = torch.ones(len(sources), dtype=torch.int64, device=device.device)
        degrees = torch.zeros(node_count, dtype=torch.int64, device=device.device)
        degrees.index_add_(0, src, ones)
        degrees.index_add_(0, tgt, ones)
        _synchronize_npu(torch)
        degree_values = degrees.cpu().tolist()
    except Exception as exc:
        return {
            "status": "failed",
            "backend": "torch_npu",
            "device": device.device,
            "reason": str(exc) or type(exc).__name__,
        }

    return _format_degree_result(
        graph=graph,
        degrees=[int(value) for value in degree_values],
        backend="torch_npu",
        device=device.device,
    )


def prepare_graph_degree_tensor_cache(
    graph: dict[str, Any],
    prefer_device: str = "auto",
) -> dict[str, Any]:
    """Prepare reusable NPU tensors for graph-degree operators.

    This is the formal operator cache API behind the cached benchmark modes.
    Callers can reuse the returned cache for repeated top-k degree queries on
    the same graph without rebuilding edge indices or recreating tensors.
    """

    device = get_device(prefer_device)
    if device.kind != "npu":
        return {
            "status": "unavailable",
            "operator": "graph_degree_tensor_cache",
            "backend": "torch",
            "device": device.device,
            "measurement_mode": "cached_tensor_prepare",
            "reason": f"NPU device is unavailable: {device.reason}",
        }

    prepared = _prepare_npu_degree_inputs(graph, prefer_device=prefer_device, resolved_device=device)
    if prepared.get("status") != "completed":
        return {
            **prepared,
            "operator": "graph_degree_tensor_cache",
            "measurement_mode": "cached_tensor_prepare",
        }

    try:
        torch = prepared["torch"]
        prepared["all_nodes"] = torch.cat((prepared["src"], prepared["tgt"]))
        _synchronize_npu(torch)
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "graph_degree_tensor_cache",
            "backend": "torch_npu",
            "device": prepared.get("device", device.device),
            "measurement_mode": "cached_tensor_prepare",
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        **prepared,
        "operator": "graph_degree_tensor_cache",
        "measurement_mode": "cached_tensor_prepare",
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
        "cache_reusable": True,
    }


def compute_degree_topk_npu_cached(
    graph_or_cache: dict[str, Any],
    prefer_device: str = "auto",
    top_k: int = 10,
    kernel: str = "bincount",
) -> dict[str, Any]:
    """Compute top-k degree hubs with cached NPU graph tensors.

    `graph_or_cache` can be either a graph dictionary or the cache returned by
    `prepare_graph_degree_tensor_cache`. The default `bincount` kernel is the
    production API counterpart of the fastest benchmark mode.
    """

    if kernel not in {"index_add", "bincount"}:
        raise ValueError("kernel must be one of: index_add, bincount")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    prepared = graph_or_cache
    if prepared.get("operator") != "graph_degree_tensor_cache" and "torch" not in prepared:
        prepared = prepare_graph_degree_tensor_cache(graph_or_cache, prefer_device=prefer_device)

    if prepared.get("status") != "completed":
        return {
            "status": prepared.get("status", "unavailable"),
            "operator": "graph_degree_topk",
            "backend": prepared.get("backend", "torch_npu"),
            "device": prepared.get("device"),
            "kernel": kernel,
            "result_mode": "topk",
            "top_k": top_k,
            "top_hubs": [],
            "reason": prepared.get("reason", "NPU tensor cache is unavailable"),
        }

    try:
        result = _run_cached_degree_variant(
            prepared=prepared,
            kernel=kernel,
            result_mode="topk",
            top_k=top_k,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "graph_degree_topk",
            "backend": "torch_npu",
            "device": prepared.get("device"),
            "kernel": kernel,
            "result_mode": "topk",
            "top_k": top_k,
            "top_hubs": [],
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        **result,
        "operator": "graph_degree_topk",
        "kernel": kernel,
        "result_mode": "topk",
        "top_k": top_k,
        "uses_cached_tensors": True,
    }


def benchmark_degree_centrality_npu_prepared(
    graph: dict[str, Any],
    iterations: int = 20,
    prefer_device: str = "auto",
) -> dict[str, Any]:
    """Benchmark NPU degree centrality after graph tensors are on device."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    device = get_device(prefer_device)
    if device.kind != "npu":
        prepared = {
            "status": "unavailable",
            "backend": "torch",
            "device": device.device,
            "measurement_mode": "prepared_kernel",
            "reason": f"NPU device is unavailable: {device.reason}",
        }
        metrics = _npu_prepared_unavailable_metrics(
            prepared=prepared,
            iterations=iterations,
            prepare_duration=None,
        )
        return {"metrics": metrics, "result": prepared}

    started = time.perf_counter()
    prepared = _prepare_npu_degree_inputs(graph, prefer_device=prefer_device, resolved_device=device)
    prepare_duration = time.perf_counter() - started if prepared.get("status") == "completed" else None
    if prepared.get("status") != "completed":
        metrics = _npu_prepared_unavailable_metrics(
            prepared=prepared,
            iterations=iterations,
            prepare_duration=prepare_duration,
        )
        return {"metrics": metrics, "result": prepared}

    _run_prepared_npu_degree(prepared, copy_result=False)
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        _run_prepared_npu_degree(prepared, copy_result=False)
        durations.append(time.perf_counter() - started)

    final_result = _run_prepared_npu_degree(prepared, copy_result=True)
    metrics = {
        **_timing_metrics(durations, iterations, len(graph.get("edges", []))),
        "status": "completed",
        "backend": "torch_npu",
        "device": prepared["device"],
        "measurement_mode": "prepared_kernel",
        "includes_tensor_setup": False,
        "includes_result_copy": False,
        "prepare_latency_ms": round(prepare_duration * 1000, 4) if prepare_duration is not None else None,
        "top_degree": final_result["top_hubs"][0]["degree"] if final_result.get("top_hubs") else 0,
    }
    return {"metrics": metrics, "result": final_result}


def profile_graph_degree_breakdown(
    graph: dict[str, Any],
    prefer_device: str = "auto",
) -> dict[str, Any]:
    """Profile graph-to-NPU degree centrality as separate timing steps."""

    device = get_device(prefer_device)
    if device.kind != "npu":
        return {
            "status": "unavailable",
            "backend": "torch",
            "device": device.device,
            "measurement_mode": "breakdown_profile",
            "reason": f"NPU device is unavailable: {device.reason}",
            "steps": _empty_breakdown_steps(),
        }

    steps: dict[str, float | None] = _empty_breakdown_steps()
    started_total = time.perf_counter()
    try:
        torch = importlib.import_module("torch")

        started = time.perf_counter()
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        steps["graph_access_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        index = {node["id"]: idx for idx, node in enumerate(nodes)}
        steps["node_index_build_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        sources = []
        targets = []
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in index and tgt in index:
                sources.append(index[src])
                targets.append(index[tgt])
        steps["edge_index_build_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        src_cpu = torch.tensor(sources, dtype=torch.long, device="cpu")
        tgt_cpu = torch.tensor(targets, dtype=torch.long, device="cpu")
        steps["tensor_create_cpu_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        src = src_cpu.to(device.device)
        tgt = tgt_cpu.to(device.device)
        _synchronize_npu(torch)
        steps["h2d_transfer_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        ones = torch.ones(len(sources), dtype=torch.int64, device=device.device)
        degrees = torch.zeros(len(nodes), dtype=torch.int64, device=device.device)
        _synchronize_npu(torch)
        steps["npu_buffer_alloc_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        degrees.zero_()
        _synchronize_npu(torch)
        steps["kernel_zero_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        degrees.index_add_(0, src, ones)
        degrees.index_add_(0, tgt, ones)
        _synchronize_npu(torch)
        steps["kernel_index_add_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        degree_cpu = degrees.cpu()
        _synchronize_npu(torch)
        steps["d2h_copy_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        degree_values = degree_cpu.tolist()
        steps["tolist_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        result = _format_degree_result(
            graph=graph,
            degrees=[int(value) for value in degree_values],
            backend="torch_npu",
            device=device.device,
        )
        steps["format_result_ms"] = _elapsed_ms(started)
    except Exception as exc:
        return {
            "status": "failed",
            "backend": "torch_npu",
            "device": device.device,
            "measurement_mode": "breakdown_profile",
            "reason": str(exc) or type(exc).__name__,
            "steps": steps,
        }

    return {
        "status": "completed",
        "backend": "torch_npu",
        "device": device.device,
        "measurement_mode": "breakdown_profile",
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
        "total_profiled_ms": _elapsed_ms(started_total),
        "steps": steps,
        "top_degree": result["top_hubs"][0]["degree"] if result.get("top_hubs") else 0,
    }


def benchmark_graph_degree_centrality(
    node_count: int = 1000,
    edge_count: int = 10000,
    iterations: int = 20,
    seed: int = 42,
    prefer_device: str = "auto",
    amortized_runs: list[int] | tuple[int, ...] = (1, 2, 5, 10, 20),
    profile_breakdown: bool = False,
    benchmark_modes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Benchmark CPU degree centrality against the NPU tensor implementation."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    graph = generate_synthetic_graph(node_count=node_count, edge_count=edge_count, seed=seed)

    cpu_warmup = compute_degree_centrality_cpu(graph)
    cpu_durations = []
    cpu_last = cpu_warmup
    for _ in range(iterations):
        started = time.perf_counter()
        cpu_last = compute_degree_centrality_cpu(graph)
        cpu_durations.append(time.perf_counter() - started)

    npu_warmup = compute_degree_centrality_npu(graph, prefer_device=prefer_device)
    npu_durations = []
    npu_last = npu_warmup
    if npu_warmup.get("status") == "completed":
        for _ in range(iterations):
            started = time.perf_counter()
            npu_last = compute_degree_centrality_npu(graph, prefer_device=prefer_device)
            npu_durations.append(time.perf_counter() - started)

    prepared = benchmark_degree_centrality_npu_prepared(
        graph=graph,
        iterations=iterations,
        prefer_device=prefer_device,
    )
    correctness = _compare_degree_results(cpu_last, npu_last)
    prepared_correctness = _compare_degree_results(cpu_last, prepared["result"])
    cpu_metrics = _timing_metrics(cpu_durations, iterations, edge_count)
    npu_metrics = (
        {
            **_timing_metrics(npu_durations, iterations, edge_count),
            "status": "completed",
            "backend": npu_last["backend"],
            "device": npu_last["device"],
            "measurement_mode": "end_to_end",
            "includes_tensor_setup": True,
            "includes_result_copy": True,
            "top_degree": npu_last["top_hubs"][0]["degree"] if npu_last.get("top_hubs") else 0,
        }
        if npu_durations
        else {
            "status": npu_last.get("status", "unavailable"),
            "backend": npu_last.get("backend", "torch_npu"),
            "device": npu_last.get("device"),
            "measurement_mode": "end_to_end",
            "reason": npu_last.get("reason"),
        }
    )
    speedup = None
    if npu_durations and npu_metrics["latency_ms_avg"] > 0:
        speedup = round(cpu_metrics["latency_ms_avg"] / npu_metrics["latency_ms_avg"], 4)
    prepared_speedup = None
    prepared_metrics = prepared["metrics"]
    if prepared_metrics.get("status") == "completed" and prepared_metrics["latency_ms_avg"] > 0:
        prepared_speedup = round(cpu_metrics["latency_ms_avg"] / prepared_metrics["latency_ms_avg"], 4)
    amortized = _amortized_metrics(
        cpu_latency_ms=cpu_metrics["latency_ms_avg"],
        prepared_latency_ms=prepared_metrics.get("latency_ms_avg"),
        prepare_latency_ms=prepared_metrics.get("prepare_latency_ms"),
        runs=amortized_runs,
    )

    report = {
        "task": "task3_graph_tensor_degree",
        "graph": {"node_count": node_count, "edge_count": edge_count, "seed": seed},
        "cpu": {
            **cpu_metrics,
            "status": "completed",
            "backend": "python",
            "device": "cpu",
            "top_degree": cpu_last["top_hubs"][0]["degree"] if cpu_last.get("top_hubs") else 0,
        },
        "npu": npu_metrics,
        "npu_prepared": prepared_metrics,
        "correctness": correctness,
        "prepared_correctness": prepared_correctness,
        "speedup": speedup,
        "prepared_speedup": prepared_speedup,
        "amortized": amortized,
        "notes": [
            "CPU measures Python list-based undirected degree centrality.",
            "NPU end_to_end measures graph index parsing, device tensor creation, index_add_, synchronization, and result copy.",
            "NPU prepared_kernel measures repeated zero_ + index_add_ execution after tensors are already on npu:0.",
            "Amortized totals estimate one NPU tensor preparation plus repeated prepared-kernel runs on the same graph.",
            "Synthetic graph generation is excluded from CPU/NPU timing.",
        ],
    }
    if profile_breakdown:
        report["breakdown"] = profile_graph_degree_breakdown(
            graph=graph,
            prefer_device=prefer_device,
        )
    if benchmark_modes:
        report["mode_benchmarks"] = benchmark_graph_degree_modes(
            graph=graph,
            cpu_result=cpu_last,
            cpu_latency_ms=cpu_metrics["latency_ms_avg"],
            npu_latency_ms=npu_metrics.get("latency_ms_avg"),
            iterations=iterations,
            prefer_device=prefer_device,
            modes=benchmark_modes,
        )
    return report


def compute_type_centrality_npu(
    graph_or_cache: dict[str, Any],
    prefer_device: str = "auto",
    kernel: str = "bincount",
) -> dict[str, Any]:
    """Compute type-level centrality aggregation with NPU tensor acceleration.

    Uses the cached NPU degree vector (when available) and scatter-based type
    aggregation to replace two major Python loops in ``compute_centrality``:
    ``_compute_degree_counts`` and ``_build_type_centrality``.

    When no NPU cache is provided, builds tensors from scratch on the NPU device.
    Falls back to CPU if NPU is unavailable.
    """

    prepared = graph_or_cache
    if prepared.get("operator") != "graph_degree_tensor_cache" and "torch" not in prepared:
        prepared = prepare_graph_degree_tensor_cache(graph_or_cache, prefer_device=prefer_device)

    if prepared.get("status") != "completed":
        return {
            "status": prepared.get("status", "unavailable"),
            "backend": prepared.get("backend", "torch"),
            "device": prepared.get("device"),
            "type_centrality": {},
            "reason": prepared.get("reason", "NPU tensor cache is unavailable"),
        }

    try:
        torch = prepared["torch"]
        graph = prepared["graph"]
        nodes = graph.get("nodes", [])

        # Compute degree vector on NPU
        if kernel == "bincount":
            degrees = _compute_degrees_bincount(prepared)
        else:
            degrees = _compute_degrees_index_add(prepared)

        # Build type-to-index mapping
        type_to_idx: dict[str, int] = {}
        type_indices_cpu: list[int] = []
        for node in nodes:
            t = node.get("type", "")
            if t not in type_to_idx:
                type_to_idx[t] = len(type_to_idx)
            type_indices_cpu.append(type_to_idx[t])

        num_types = len(type_to_idx)
        if num_types == 0:
            return {
                "status": "completed",
                "backend": "torch_npu",
                "device": prepared["device"],
                "type_centrality": {},
            }

        # Move type indices to NPU and use scatter_add for aggregation
        type_indices = torch.tensor(type_indices_cpu, dtype=torch.long, device=degrees.device)

        # Aggregate: count, degree_sum, max_degree per type — all on NPU
        ones = torch.ones(len(nodes), dtype=degrees.dtype, device=degrees.device)
        type_count = torch.zeros(num_types, dtype=degrees.dtype, device=degrees.device)
        type_count.scatter_add_(0, type_indices, ones)

        type_degree_sum = torch.zeros(num_types, dtype=degrees.dtype, device=degrees.device)
        type_degree_sum.scatter_add_(0, type_indices, degrees.float())

        # For max_degree per type: use a per-type loop (small num_types) on NPU result
        degrees_cpu = degrees.cpu().tolist()
        _synchronize_npu(torch)

        type_count_cpu = type_count.cpu().tolist()
        type_degree_sum_cpu = type_degree_sum.cpu().tolist()

        # Build per-type top_node using CPU (small loop over nodes)
        idx_to_type = {v: k for k, v in type_to_idx.items()}
        type_top_node: dict[int, tuple[str, int]] = {}
        for node_idx, node in enumerate(nodes):
            tid = type_indices_cpu[node_idx]
            deg = int(degrees_cpu[node_idx])
            if tid not in type_top_node or deg > type_top_node[tid][1]:
                type_top_node[tid] = (node.get("name", ""), deg)

        type_centrality: dict[str, dict[str, Any]] = {}
        for tid in range(num_types):
            node_type = idx_to_type[tid]
            count = int(type_count_cpu[tid])
            degree_sum = float(type_degree_sum_cpu[tid])
            max_deg = type_top_node.get(tid, ("", 0))[1]
            top_node = type_top_node.get(tid, ("", 0))[0]
            type_centrality[node_type] = {
                "count": count,
                "avg_degree": round(degree_sum / max(count, 1), 2),
                "max_degree": max_deg,
                "top_node": top_node,
            }

    except Exception as exc:
        return {
            "status": "failed",
            "backend": "torch_npu",
            "device": prepared.get("device"),
            "type_centrality": {},
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        "status": "completed",
        "backend": "torch_npu",
        "device": prepared["device"],
        "type_centrality": type_centrality,
    }


def benchmark_graph_degree_modes(
    graph: dict[str, Any],
    cpu_result: dict[str, Any],
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
    iterations: int = 20,
    prefer_device: str = "auto",
    modes: list[str] | tuple[str, ...] = GRAPH_DEGREE_BENCHMARK_MODES,
) -> list[dict[str, Any]]:
    """Benchmark graph-degree implementation variants for optimization experiments."""

    requested_modes = _normalize_benchmark_modes(modes)
    device = get_device(prefer_device)
    if device.kind != "npu":
        return [
            _unavailable_mode_benchmark(mode, device=device.device, reason=f"NPU device is unavailable: {device.reason}")
            for mode in requested_modes
        ]

    results = []
    cached: dict[str, Any] | None = None
    for mode in requested_modes:
        if mode == "baseline_index_add_full_format":
            results.append(
                _benchmark_baseline_index_add_full_format(
                    graph=graph,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    prefer_device=prefer_device,
                )
            )
            continue

        if cached is None:
            cached = prepare_graph_degree_tensor_cache(graph, prefer_device=prefer_device)
        if cached.get("status") != "completed":
            results.append(
                _unavailable_mode_benchmark(
                    mode,
                    device=cached.get("device", device.device),
                    reason=cached.get("reason", "NPU tensor preparation failed"),
                    status=cached.get("status", "failed"),
                )
            )
            continue

        if mode == "cached_index_add_full_format":
            results.append(
                _benchmark_cached_mode(
                    graph=graph,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    prepared=cached,
                    name=mode,
                    kernel="index_add",
                    result_mode="full_format",
                )
            )
        elif mode == "cached_index_add_topk":
            results.append(
                _benchmark_cached_mode(
                    graph=graph,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    prepared=cached,
                    name=mode,
                    kernel="index_add",
                    result_mode="topk",
                )
            )
        elif mode == "cached_bincount_topk":
            results.append(
                _benchmark_cached_mode(
                    graph=graph,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    prepared=cached,
                    name=mode,
                    kernel="bincount",
                    result_mode="topk",
                )
            )
    return results


def _benchmark_baseline_index_add_full_format(
    graph: dict[str, Any],
    cpu_result: dict[str, Any],
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
    iterations: int,
    prefer_device: str,
) -> dict[str, Any]:
    durations = []
    last_result: dict[str, Any] = {"status": "not_run"}
    for _ in range(iterations):
        started = time.perf_counter()
        last_result = compute_degree_centrality_npu(graph, prefer_device=prefer_device)
        durations.append(time.perf_counter() - started)
        if last_result.get("status") != "completed":
            return _unavailable_mode_benchmark(
                "baseline_index_add_full_format",
                device=last_result.get("device"),
                reason=last_result.get("reason", "baseline NPU execution failed"),
                status=last_result.get("status", "failed"),
            )
    metrics = _mode_timing_metrics(
        name="baseline_index_add_full_format",
        durations=durations,
        iterations=iterations,
        edge_count=len(graph.get("edges", [])),
        cpu_latency_ms=cpu_latency_ms,
        npu_latency_ms=npu_latency_ms,
    )
    return {
        **metrics,
        "backend": "torch_npu",
        "device": last_result.get("device"),
        "kernel": "index_add",
        "uses_cached_tensors": False,
        "result_mode": "full_format",
        "includes_edge_index_build": True,
        "includes_tensor_create": True,
        "includes_full_result_format": True,
        "correctness": _compare_degree_results(cpu_result, last_result),
        "top_degree": _top_degree(last_result),
    }


def _benchmark_cached_mode(
    graph: dict[str, Any],
    cpu_result: dict[str, Any],
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
    iterations: int,
    prepared: dict[str, Any],
    name: str,
    kernel: str,
    result_mode: str,
) -> dict[str, Any]:
    durations = []
    last_result: dict[str, Any] = {"status": "not_run"}
    try:
        _run_cached_mode_once(prepared=prepared, kernel=kernel, result_mode=result_mode)
        for _ in range(iterations):
            started = time.perf_counter()
            last_result = _run_cached_mode_once(
                prepared=prepared,
                kernel=kernel,
                result_mode=result_mode,
            )
            durations.append(time.perf_counter() - started)
    except Exception as exc:
        return _unavailable_mode_benchmark(
            name,
            device=prepared.get("device"),
            reason=str(exc) or type(exc).__name__,
            status="failed",
        )

    correctness_result = _run_cached_degree_variant(
        prepared=prepared,
        kernel=kernel,
        result_mode="full_format",
    )
    metrics = _mode_timing_metrics(
        name=name,
        durations=durations,
        iterations=iterations,
        edge_count=len(graph.get("edges", [])),
        cpu_latency_ms=cpu_latency_ms,
        npu_latency_ms=npu_latency_ms,
    )
    return {
        **metrics,
        "backend": "torch_npu",
        "device": prepared.get("device"),
        "kernel": kernel,
        "uses_cached_tensors": True,
        "result_mode": result_mode,
        "includes_edge_index_build": False,
        "includes_tensor_create": False,
        "includes_full_result_format": result_mode == "full_format",
        "correctness": _compare_degree_results(cpu_result, correctness_result),
        "correctness_check_included_in_timing": False,
        "top_degree": _top_degree(last_result),
    }


def _run_cached_mode_once(
    prepared: dict[str, Any],
    kernel: str,
    result_mode: str,
) -> dict[str, Any]:
    if result_mode == "topk":
        return compute_degree_topk_npu_cached(prepared, kernel=kernel, top_k=10)
    return _run_cached_degree_variant(prepared=prepared, kernel=kernel, result_mode=result_mode)


def _run_cached_degree_variant(
    prepared: dict[str, Any],
    kernel: str,
    result_mode: str,
    top_k: int = 10,
) -> dict[str, Any]:
    if kernel == "index_add":
        degrees = _compute_degrees_index_add(prepared)
    elif kernel == "bincount":
        degrees = _compute_degrees_bincount(prepared)
    else:
        raise ValueError(f"unsupported graph degree kernel: {kernel}")

    if result_mode == "full_format":
        degree_values = degrees.cpu().tolist()
        return _format_degree_result(
            graph=prepared["graph"],
            degrees=[int(value) for value in degree_values],
            backend="torch_npu",
            device=prepared["device"],
        )
    if result_mode == "topk":
        return _format_topk_degree_result(prepared=prepared, degrees=degrees, k=top_k)
    raise ValueError(f"unsupported graph degree result mode: {result_mode}")


def _compute_degrees_index_add(prepared: dict[str, Any]) -> Any:
    torch = prepared["torch"]
    degrees = prepared["degrees"]
    degrees.zero_()
    degrees.index_add_(0, prepared["src"], prepared["ones"])
    degrees.index_add_(0, prepared["tgt"], prepared["ones"])
    _synchronize_npu(torch)
    return degrees


def _compute_degrees_bincount(prepared: dict[str, Any]) -> Any:
    torch = prepared["torch"]
    if "all_nodes" not in prepared:
        prepared["all_nodes"] = torch.cat((prepared["src"], prepared["tgt"]))
        _synchronize_npu(torch)
    degrees = torch.bincount(
        prepared["all_nodes"],
        minlength=len(prepared["graph"].get("nodes", [])),
    )
    _synchronize_npu(torch)
    return degrees


def _format_topk_degree_result(prepared: dict[str, Any], degrees: Any, k: int = 10) -> dict[str, Any]:
    torch = prepared["torch"]
    nodes = prepared["graph"].get("nodes", [])
    top_k = min(k, len(nodes))
    if top_k < 1:
        return {
            "status": "completed",
            "backend": "torch_npu",
            "device": prepared["device"],
            "node_count": 0,
            "edge_count": len(prepared["graph"].get("edges", [])),
            "top_hubs": [],
        }
    values, indices = torch.topk(degrees, k=top_k)
    _synchronize_npu(torch)
    top_values = values.cpu().tolist()
    top_indices = indices.cpu().tolist()
    max_possible = max(len(nodes) - 1, 1)
    top_hubs = []
    for idx, degree in zip(top_indices, top_values):
        node = nodes[int(idx)]
        top_hubs.append(
            {
                "id": node.get("id", ""),
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "degree": int(degree),
                "degree_centrality": round(int(degree) / max_possible, 6),
            }
        )
    top_hubs.sort(key=lambda item: (-item["degree"], item["id"]))
    return {
        "status": "completed",
        "backend": "torch_npu",
        "device": prepared["device"],
        "node_count": len(nodes),
        "edge_count": len(prepared["graph"].get("edges", [])),
        "top_hubs": top_hubs,
    }


def _mode_timing_metrics(
    name: str,
    durations: list[float],
    iterations: int,
    edge_count: int,
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
) -> dict[str, Any]:
    metrics = {
        "name": name,
        "status": "completed",
        "measurement_mode": "mode_benchmark",
        **_timing_metrics(durations, iterations, edge_count),
    }
    latency_ms = metrics.get("latency_ms_avg")
    metrics["speedup_vs_cpu"] = (
        round(cpu_latency_ms / latency_ms, 4)
        if cpu_latency_ms is not None and latency_ms
        else None
    )
    metrics["speedup_vs_npu_end_to_end"] = (
        round(npu_latency_ms / latency_ms, 4)
        if npu_latency_ms is not None and latency_ms
        else None
    )
    return metrics


def _unavailable_mode_benchmark(
    name: str,
    device: str | None,
    reason: str,
    status: str = "unavailable",
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "measurement_mode": "mode_benchmark",
        "backend": "torch_npu",
        "device": device,
        "reason": reason,
        "latency_ms_avg": None,
        "latency_ms_min": None,
        "latency_ms_max": None,
        "throughput_edges_per_sec": None,
        "speedup_vs_cpu": None,
        "speedup_vs_npu_end_to_end": None,
    }


def _normalize_benchmark_modes(modes: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for mode in modes:
        if mode == "all":
            normalized.extend(GRAPH_DEGREE_BENCHMARK_MODES)
            continue
        if mode not in GRAPH_DEGREE_BENCHMARK_MODES:
            raise ValueError(f"unsupported graph degree benchmark mode: {mode}")
        normalized.append(mode)
    return list(dict.fromkeys(normalized))


def _top_degree(result: dict[str, Any]) -> int:
    top_hubs = result.get("top_hubs") or []
    return int(top_hubs[0]["degree"]) if top_hubs else 0


def _edge_indices(graph: dict[str, Any]) -> tuple[list[int], list[int]]:
    nodes = graph.get("nodes", [])
    index = {node["id"]: idx for idx, node in enumerate(nodes)}
    sources = []
    targets = []
    for edge in graph.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src in index and tgt in index:
            sources.append(index[src])
            targets.append(index[tgt])
    return sources, targets


def _format_degree_result(
    graph: dict[str, Any],
    degrees: list[int],
    backend: str,
    device: str,
) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    max_possible = max(len(nodes) - 1, 1)
    top_hubs = []
    for idx, node in enumerate(nodes):
        degree = degrees[idx] if idx < len(degrees) else 0
        top_hubs.append(
            {
                "id": node.get("id", ""),
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "degree": int(degree),
                "degree_centrality": round(degree / max_possible, 6),
            }
        )
    top_hubs.sort(key=lambda item: (-item["degree"], item["id"]))
    return {
        "status": "completed",
        "backend": backend,
        "device": device,
        "node_count": len(nodes),
        "edge_count": len(graph.get("edges", [])),
        "degrees": [int(value) for value in degrees],
        "top_hubs": top_hubs[:10],
    }


def _prepare_npu_degree_inputs(
    graph: dict[str, Any],
    prefer_device: str = "auto",
    resolved_device: Any | None = None,
) -> dict[str, Any]:
    device = resolved_device or get_device(prefer_device)
    if device.kind != "npu":
        return {
            "status": "unavailable",
            "backend": "torch",
            "device": device.device,
            "measurement_mode": "prepared_kernel",
            "reason": f"NPU device is unavailable: {device.reason}",
        }

    try:
        torch = importlib.import_module("torch")
        node_count = len(graph.get("nodes", []))
        sources, targets = _edge_indices(graph)
        src = torch.tensor(sources, dtype=torch.long, device=device.device)
        tgt = torch.tensor(targets, dtype=torch.long, device=device.device)
        ones = torch.ones(len(sources), dtype=torch.int64, device=device.device)
        degrees = torch.zeros(node_count, dtype=torch.int64, device=device.device)
        _synchronize_npu(torch)
    except Exception as exc:
        return {
            "status": "failed",
            "backend": "torch_npu",
            "device": device.device,
            "measurement_mode": "prepared_kernel",
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        "status": "completed",
        "backend": "torch_npu",
        "device": device.device,
        "measurement_mode": "prepared_kernel",
        "graph": graph,
        "torch": torch,
        "src": src,
        "tgt": tgt,
        "ones": ones,
        "degrees": degrees,
    }


def _run_prepared_npu_degree(prepared: dict[str, Any], copy_result: bool) -> dict[str, Any]:
    torch = prepared["torch"]
    degrees = prepared["degrees"]
    degrees.zero_()
    degrees.index_add_(0, prepared["src"], prepared["ones"])
    degrees.index_add_(0, prepared["tgt"], prepared["ones"])
    _synchronize_npu(torch)
    if not copy_result:
        return {
            "status": "completed",
            "backend": "torch_npu",
            "device": prepared["device"],
            "measurement_mode": "prepared_kernel",
        }
    degree_values = degrees.cpu().tolist()
    return _format_degree_result(
        graph=prepared["graph"],
        degrees=[int(value) for value in degree_values],
        backend="torch_npu",
        device=prepared["device"],
    )


def _compare_degree_results(
    cpu_result: dict[str, Any],
    npu_result: dict[str, Any],
) -> dict[str, Any]:
    if npu_result.get("status") != "completed":
        return {
            "status": "not_run",
            "reason": npu_result.get("reason", "NPU result is unavailable"),
        }
    matches = cpu_result.get("degrees") == npu_result.get("degrees")
    return {
        "status": "passed" if matches else "failed",
        "degree_vector_equal": matches,
        "max_abs_diff": _max_abs_diff(cpu_result.get("degrees", []), npu_result.get("degrees", [])),
    }


def _timing_metrics(durations: list[float], iterations: int, edge_count: int) -> dict[str, Any]:
    if not durations:
        return {"iterations": iterations, "latency_ms_avg": None, "latency_ms_min": None, "latency_ms_max": None}
    total = sum(durations)
    return {
        "iterations": iterations,
        "latency_ms_avg": round(mean(durations) * 1000, 4),
        "latency_ms_min": round(min(durations) * 1000, 4),
        "latency_ms_max": round(max(durations) * 1000, 4),
        "throughput_edges_per_sec": _throughput(edge_count * iterations, total),
    }


def _throughput(iterations: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return round(iterations / seconds, 4)


def _empty_breakdown_steps() -> dict[str, float | None]:
    return {
        "graph_access_ms": None,
        "node_index_build_ms": None,
        "edge_index_build_ms": None,
        "tensor_create_cpu_ms": None,
        "h2d_transfer_ms": None,
        "npu_buffer_alloc_ms": None,
        "kernel_zero_ms": None,
        "kernel_index_add_ms": None,
        "d2h_copy_ms": None,
        "tolist_ms": None,
        "format_result_ms": None,
    }


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _npu_prepared_unavailable_metrics(
    prepared: dict[str, Any],
    iterations: int,
    prepare_duration: float | None,
) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "latency_ms_avg": None,
        "latency_ms_min": None,
        "latency_ms_max": None,
        "throughput_edges_per_sec": None,
        "status": prepared.get("status", "unavailable"),
        "backend": prepared.get("backend", "torch_npu"),
        "device": prepared.get("device"),
        "measurement_mode": "prepared_kernel",
        "includes_tensor_setup": False,
        "includes_result_copy": False,
        "prepare_latency_ms": round(prepare_duration * 1000, 4) if prepare_duration is not None else None,
        "reason": prepared.get("reason"),
    }


def _amortized_metrics(
    cpu_latency_ms: float | None,
    prepared_latency_ms: float | None,
    prepare_latency_ms: float | None,
    runs: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    run_counts = _normalize_run_counts(runs)
    cpu_totals = [
        round(cpu_latency_ms * run_count, 4) if cpu_latency_ms is not None else None
        for run_count in run_counts
    ]
    if prepared_latency_ms is None or prepare_latency_ms is None:
        return {
            "runs": run_counts,
            "cpu_total_ms": cpu_totals,
            "npu_prepared_total_ms": [None for _ in run_counts],
            "speedup": [None for _ in run_counts],
            "breakeven_runs": None,
        }

    npu_totals = [
        round(prepare_latency_ms + prepared_latency_ms * run_count, 4)
        for run_count in run_counts
    ]
    speedups = [
        round(cpu_total / npu_total, 4) if cpu_total is not None and npu_total > 0 else None
        for cpu_total, npu_total in zip(cpu_totals, npu_totals)
    ]
    breakeven_runs = next(
        (run_count for run_count, speedup in zip(run_counts, speedups) if speedup is not None and speedup >= 1),
        None,
    )
    return {
        "runs": run_counts,
        "cpu_total_ms": cpu_totals,
        "npu_prepared_total_ms": npu_totals,
        "speedup": speedups,
        "breakeven_runs": breakeven_runs,
    }


def _normalize_run_counts(runs: list[int] | tuple[int, ...]) -> list[int]:
    run_counts = [int(value) for value in runs]
    if not run_counts:
        raise ValueError("amortized_runs must not be empty")
    if any(value < 1 for value in run_counts):
        raise ValueError("amortized_runs must contain positive integers")
    return run_counts


def _max_abs_diff(left: list[int], right: list[int]) -> int | None:
    if len(left) != len(right):
        return None
    if not left:
        return 0
    return max(abs(a - b) for a, b in zip(left, right))


def _synchronize_npu(torch: Any) -> None:
    npu = getattr(torch, "npu", None)
    synchronize = getattr(npu, "synchronize", None)
    if callable(synchronize):
        synchronize()
