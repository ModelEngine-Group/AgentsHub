"""Tensorized Task-2 KG relation-scoring operators for CPU/NPU benchmarks."""

from __future__ import annotations

from dataclasses import asdict
import importlib
import time
from statistics import mean
from typing import Any

from src.common.device import get_device

DEFAULT_RELATION_TYPES = [
    "has_symptom",
    "treated_by",
    "diagnosed_by",
    "recommended_treatment",
    "complication_of",
]


def generate_synthetic_relation_candidates(
    candidate_count: int,
    feature_dim: int,
    relation_count: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate deterministic tensor inputs for Task-2 relation scoring."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    if feature_dim < 1:
        raise ValueError("feature_dim must be >= 1")
    if relation_count < 1:
        raise ValueError("relation_count must be >= 1")

    torch = importlib.import_module("torch")
    generator = torch.Generator()
    generator.manual_seed(seed)
    features = torch.randn(candidate_count, feature_dim, generator=generator, dtype=torch.float32)
    weights = torch.randn(relation_count, feature_dim, generator=generator, dtype=torch.float32)
    bias = torch.randn(relation_count, generator=generator, dtype=torch.float32)

    return {
        "candidate_count": candidate_count,
        "feature_dim": feature_dim,
        "relation_count": relation_count,
        "relation_types": _relation_types(relation_count),
        "features": features,
        "weights": weights,
        "bias": bias,
        "seed": seed,
    }


def score_relation_candidates_cpu(candidates: dict[str, Any]) -> dict[str, Any]:
    """Score Task-2 relation candidates with CPU torch tensors."""

    try:
        importlib.import_module("torch")
        features, weights, bias = _candidate_tensors(candidates, device="cpu")
        logits = features @ weights.transpose(0, 1) + bias
        return _format_scoring_result(
            candidates=candidates,
            logits=logits.detach().cpu(),
            backend="torch",
            device="cpu",
            operator="task2_relation_candidate_scoring",
        )
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch",
            "device": "cpu",
            "reason": str(exc) or type(exc).__name__,
        }


def score_relation_candidates_npu(
    candidates: dict[str, Any],
    prefer_device: str = "auto",
) -> dict[str, Any]:
    """Score Task-2 relation candidates on Ascend NPU when available."""

    device = get_device(prefer_device)
    if device.kind != "npu":
        return {
            "status": "unavailable",
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch",
            "device": device.device,
            "runtime": asdict(device),
            "predicted_relation_ids": [],
            "predicted_relations": [],
            "top_scores": [],
            "reason": f"NPU device is unavailable: {device.reason}",
        }

    try:
        torch = importlib.import_module("torch")
        features, weights, bias = _candidate_tensors(candidates, device=device.device)
        logits = features @ weights.transpose(0, 1) + bias
        _synchronize_npu(torch)
        logits_cpu = logits.detach().cpu()
        _synchronize_npu(torch)
        return _format_scoring_result(
            candidates=candidates,
            logits=logits_cpu,
            backend="torch_npu",
            device=device.device,
            operator="task2_relation_candidate_scoring",
            runtime=asdict(device),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch_npu",
            "device": device.device,
            "runtime": asdict(device),
            "predicted_relation_ids": [],
            "predicted_relations": [],
            "top_scores": [],
            "reason": str(exc) or type(exc).__name__,
        }


def compare_relation_scores(
    cpu_result: dict[str, Any],
    npu_result: dict[str, Any],
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare CPU and NPU relation scoring outputs."""

    if npu_result.get("status") != "completed":
        return {
            "status": "not_run",
            "reason": npu_result.get("reason", "NPU result is unavailable"),
        }
    if cpu_result.get("status") != "completed":
        return {
            "status": "not_run",
            "reason": cpu_result.get("reason", "CPU result is unavailable"),
        }

    try:
        torch = importlib.import_module("torch")
        left = cpu_result["logits"]
        right = npu_result["logits"]
        same_shape = tuple(left.shape) == tuple(right.shape)
        if same_shape:
            diff = (left - right).abs()
            max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
            score_close = bool(torch.allclose(left, right, atol=tolerance, rtol=tolerance))
        else:
            max_abs_diff = None
            score_close = False
        prediction_equal = cpu_result.get("predicted_relation_ids") == npu_result.get("predicted_relation_ids")
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc) or type(exc).__name__,
        }

    passed = same_shape and score_close and prediction_equal
    return {
        "status": "passed" if passed else "failed",
        "score_close": score_close,
        "prediction_equal": prediction_equal,
        "max_abs_diff": round(max_abs_diff, 8) if max_abs_diff is not None else None,
        "tolerance": tolerance,
    }


def profile_relation_scoring_breakdown(
    candidates: dict[str, Any] | None = None,
    candidate_count: int = 4096,
    feature_dim: int = 256,
    relation_count: int = 5,
    seed: int = 42,
    prefer_device: str = "auto",
    warmup_iterations: int = 0,
    profile_iterations: int = 1,
) -> dict[str, Any]:
    """Profile Task-2 NPU relation scoring as separate timing steps.

    Measures where time is spent during a single NPU relation-scoring pass:
    device selection, host-to-device transfers (features, weights/bias),
    matmul, NPU argmax, logits device-to-host, labels device-to-host,
    CPU argmax, and result formatting.

    When ``profile_iterations > 1``, each step is measured multiple times and
    the report contains both per-iteration averages and the cold (first-run)
    values. This helps distinguish one-time kernel compilation / JIT overhead
    from steady-state timing.
    """

    device = get_device(prefer_device)
    if device.kind != "npu":
        return {
            "status": "unavailable",
            "backend": "torch",
            "device": device.device,
            "measurement_mode": "breakdown_profile",
            "reason": f"NPU device is unavailable: {device.reason}",
            "steps": _empty_relation_breakdown_steps(),
        }

    if candidates is None:
        candidates = generate_synthetic_relation_candidates(
            candidate_count=candidate_count,
            feature_dim=feature_dim,
            relation_count=relation_count,
            seed=seed,
        )

    if profile_iterations < 1:
        profile_iterations = 1

    _torch = importlib.import_module("torch")
    _device = get_device(prefer_device)

    all_steps: list[dict[str, float]] = []
    total_durations: list[float] = []

    for run_idx in range(warmup_iterations + profile_iterations):
        steps: dict[str, float] = {k: 0.0 for k in _empty_relation_breakdown_steps()}
        started_total = time.perf_counter()
        try:
            started = time.perf_counter()
            _ = get_device(prefer_device)
            steps["device_select_ms"] = _elapsed_ms(started)

            features_cpu = candidates["features"]
            weights_cpu = candidates["weights"]
            bias_cpu = candidates["bias"]

            started = time.perf_counter()
            features_npu = features_cpu.to(device=_device.device, dtype=features_cpu.new_empty(()).float().dtype)
            _synchronize_npu(_torch)
            steps["h2d_features_ms"] = _elapsed_ms(started)

            started = time.perf_counter()
            weights_npu = weights_cpu.to(device=_device.device, dtype=weights_cpu.new_empty(()).float().dtype)
            bias_npu = bias_cpu.to(device=_device.device, dtype=bias_cpu.new_empty(()).float().dtype)
            _synchronize_npu(_torch)
            steps["h2d_weights_bias_ms"] = _elapsed_ms(started)

            started = time.perf_counter()
            logits = features_npu @ weights_npu.transpose(0, 1) + bias_npu
            _synchronize_npu(_torch)
            steps["matmul_ms"] = _elapsed_ms(started)

            started = time.perf_counter()
            npu_argmax_values, npu_argmax_indices = logits.max(dim=1)
            _synchronize_npu(_torch)
            steps["argmax_npu_ms"] = _elapsed_ms(started)

            started = time.perf_counter()
            logits_cpu = logits.detach().cpu()
            _synchronize_npu(_torch)
            steps["logits_d2h_ms"] = _elapsed_ms(started)

            started = time.perf_counter()
            npu_argmax_indices.detach().cpu()
            npu_argmax_values.detach().cpu()
            _synchronize_npu(_torch)
            steps["labels_d2h_ms"] = _elapsed_ms(started)

            started = time.perf_counter()
            cpu_argmax_values, cpu_argmax_indices = logits_cpu.max(dim=1)
            steps["argmax_cpu_ms"] = _elapsed_ms(started)

            started = time.perf_counter()
            predicted_ids = [int(v) for v in cpu_argmax_indices.tolist()]
            relation_types = candidates["relation_types"]
            predicted_relations = [relation_types[idx] for idx in predicted_ids]
            top_scores = [round(float(v), 6) for v in cpu_argmax_values.tolist()]
            _ = (predicted_ids, predicted_relations, top_scores)
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

        total_durations.append(_elapsed_ms(started_total))
        if run_idx >= warmup_iterations:
            all_steps.append(steps)

    cold_steps = all_steps[0] if all_steps else {k: None for k in _empty_relation_breakdown_steps()}
    avg_steps: dict[str, float | None] = {}
    for key in cold_steps:
        values = [s[key] for s in all_steps]
        avg_steps[key] = round(mean(values), 4) if values else None

    result: dict[str, Any] = {
        "status": "completed",
        "backend": "torch_npu",
        "device": device.device,
        "measurement_mode": "breakdown_profile",
        "candidate_count": int(candidates["candidate_count"]),
        "feature_dim": int(candidates["feature_dim"]),
        "relation_count": int(candidates["relation_count"]),
        "warmup_iterations": warmup_iterations,
        "profile_iterations": profile_iterations,
        "total_profiled_ms": round(mean(total_durations[warmup_iterations:]), 4) if len(total_durations) > warmup_iterations else None,
        "steps": avg_steps if profile_iterations > 1 else cold_steps,
    }
    if profile_iterations > 1:
        result["cold_steps"] = cold_steps
        result["avg_steps"] = avg_steps
    return result


RELATION_TENSOR_BENCHMARK_MODES = (
    "baseline_full_logits",
    "cached_full_logits",
    "cached_argmax_labels",
    "cached_topk_labels",
    "cpu_topk_labels",
)


def prepare_relation_tensor_cache(
    candidates: dict[str, Any],
    prefer_device: str = "auto",
) -> dict[str, Any]:
    """Prepare reusable NPU tensors for relation scoring.

    Moves features, weights, and bias to the NPU device once so that
    repeated scoring passes skip host-to-device transfer.
    """

    device = get_device(prefer_device)
    if device.kind != "npu":
        return {
            "status": "unavailable",
            "operator": "relation_tensor_cache",
            "backend": "torch",
            "device": device.device,
            "reason": f"NPU device is unavailable: {device.reason}",
        }

    try:
        torch = importlib.import_module("torch")
        features = candidates["features"].to(
            device=device.device, dtype=candidates["features"].new_empty(()).float().dtype
        )
        weights = candidates["weights"].to(
            device=device.device, dtype=candidates["weights"].new_empty(()).float().dtype
        )
        bias = candidates["bias"].to(
            device=device.device, dtype=candidates["bias"].new_empty(()).float().dtype
        )
        _synchronize_npu(torch)
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "relation_tensor_cache",
            "backend": "torch_npu",
            "device": device.device,
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        "status": "completed",
        "operator": "relation_tensor_cache",
        "backend": "torch_npu",
        "device": device.device,
        "torch": torch,
        "features": features,
        "weights": weights,
        "bias": bias,
        "candidates": candidates,
        "cache_reusable": True,
    }


def score_cached_full_logits(cache: dict[str, Any]) -> dict[str, Any]:
    """Score with cached NPU tensors, returning full logits (CPU format)."""

    if cache.get("status") != "completed":
        return {
            "status": cache.get("status", "unavailable"),
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch_npu",
            "device": cache.get("device"),
            "predicted_relation_ids": [],
            "predicted_relations": [],
            "top_scores": [],
            "reason": cache.get("reason", "NPU tensor cache is unavailable"),
        }

    try:
        torch = cache["torch"]
        logits = cache["features"] @ cache["weights"].transpose(0, 1) + cache["bias"]
        _synchronize_npu(torch)
        logits_cpu = logits.detach().cpu()
        _synchronize_npu(torch)
        return _format_scoring_result(
            candidates=cache["candidates"],
            logits=logits_cpu,
            backend="torch_npu",
            device=cache["device"],
            operator="task2_relation_candidate_scoring",
        )
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch_npu",
            "device": cache.get("device"),
            "predicted_relation_ids": [],
            "predicted_relations": [],
            "top_scores": [],
            "reason": str(exc) or type(exc).__name__,
        }


def score_cached_argmax_labels(cache: dict[str, Any]) -> dict[str, Any]:
    """Score with cached NPU tensors, returning only argmax labels and scores.

    Skips copying full logits back to CPU. Only copies the predicted label
    indices and top scores — the minimum data needed for relation prediction.
    """

    if cache.get("status") != "completed":
        return {
            "status": cache.get("status", "unavailable"),
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch_npu",
            "device": cache.get("device"),
            "predicted_relation_ids": [],
            "predicted_relations": [],
            "top_scores": [],
            "reason": cache.get("reason", "NPU tensor cache is unavailable"),
        }

    try:
        torch = cache["torch"]
        logits = cache["features"] @ cache["weights"].transpose(0, 1) + cache["bias"]
        _synchronize_npu(torch)

        values, indices = logits.max(dim=1)
        _synchronize_npu(torch)

        label_ids = indices.detach().cpu().tolist()
        scores = values.detach().cpu().tolist()

        relation_types = cache["candidates"]["relation_types"]
        predicted_ids = [int(idx) for idx in label_ids]
        predicted_relations = [relation_types[idx] for idx in predicted_ids]
        top_scores = [round(float(v), 6) for v in scores]
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch_npu",
            "device": cache.get("device"),
            "predicted_relation_ids": [],
            "predicted_relations": [],
            "top_scores": [],
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        "status": "completed",
        "operator": "task2_relation_candidate_scoring",
        "backend": "torch_npu",
        "device": cache["device"],
        "scoring_mode": "cached_argmax_labels",
        "candidate_count": int(cache["candidates"]["candidate_count"]),
        "feature_dim": int(cache["candidates"]["feature_dim"]),
        "relation_count": int(cache["candidates"]["relation_count"]),
        "relation_types": relation_types,
        "predicted_relation_ids": predicted_ids,
        "predicted_relations": predicted_relations,
        "top_scores": top_scores,
        "skipped_full_logits_copy": True,
    }


def score_cached_topk_labels(
    cache: dict[str, Any],
    top_k: int = 10,
) -> dict[str, Any]:
    """Score with cached NPU tensors, returning only the top-k predictions.

    Instead of argmax over all candidates, uses ``torch.topk`` on NPU to find
    the *k* highest-scoring candidates, then copies only those k entries back
    to CPU. This avoids:
    - Copying all 65k logits to CPU (d2h savings)
    - Formatting all 65k results in Python (format savings)
    """

    if cache.get("status") != "completed":
        return {
            "status": cache.get("status", "unavailable"),
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch_npu",
            "device": cache.get("device"),
            "top_k": top_k,
            "top_k_indices": [],
            "top_k_scores": [],
            "top_k_relations": [],
            "reason": cache.get("reason", "NPU tensor cache is unavailable"),
        }

    try:
        torch = cache["torch"]
        features = cache["features"]
        weights = cache["weights"]
        bias = cache["bias"]

        # Matmul on NPU: (N, D) @ (D, R) + (R,) -> (N, R)
        logits = features @ weights.transpose(0, 1) + bias  # (N, R)
        _synchronize_npu(torch)

        # Max per candidate (across relation types): (N,)
        candidate_scores, candidate_labels = logits.max(dim=1)
        _synchronize_npu(torch)

        # top-k among all candidates — done entirely on NPU
        k = min(top_k, candidate_scores.shape[0])
        topk_scores, topk_indices = torch.topk(candidate_scores, k)
        _synchronize_npu(torch)

        # Only copy k values back to CPU
        topk_indices_cpu = topk_indices.detach().cpu().tolist()
        topk_scores_cpu = topk_scores.detach().cpu().tolist()

        # Gather the corresponding labels for top-k candidates
        topk_label_ids = [int(candidate_labels[idx]) for idx in topk_indices_cpu]
        relation_types = cache["candidates"]["relation_types"]
        topk_relations = [relation_types[label_id] for label_id in topk_label_ids]

    except Exception as exc:
        return {
            "status": "failed",
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch_npu",
            "device": cache.get("device"),
            "top_k": top_k,
            "top_k_indices": [],
            "top_k_scores": [],
            "top_k_relations": [],
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        "status": "completed",
        "operator": "task2_relation_candidate_scoring",
        "backend": "torch_npu",
        "device": cache["device"],
        "candidate_count": int(cache["candidates"]["candidate_count"]),
        "feature_dim": int(cache["candidates"]["feature_dim"]),
        "relation_count": int(cache["candidates"]["relation_count"]),
        "relation_types": relation_types,
        "top_k": top_k,
        "top_k_indices": [int(i) for i in topk_indices_cpu],
        "top_k_scores": [round(float(v), 6) for v in topk_scores_cpu],
        "top_k_label_ids": topk_label_ids,
        "top_k_relations": topk_relations,
        "skipped_full_logits_copy": True,
        "skipped_full_format": True,
    }


def score_cpu_topk_labels(
    candidates: dict[str, Any],
    top_k: int = 10,
) -> dict[str, Any]:
    """Score on CPU with torch.topk for fair comparison with cached_topk_labels.

    Performs the same logical work as ``score_cached_topk_labels`` but entirely
    on CPU: matmul, max per candidate, topk, and format only top-k results.
    """

    try:
        torch = importlib.import_module("torch")
        features = candidates["features"]
        weights = candidates["weights"]
        bias = candidates["bias"]

        logits = features @ weights.transpose(0, 1) + bias
        candidate_scores, candidate_labels = logits.max(dim=1)

        k = min(top_k, candidate_scores.shape[0])
        topk_scores, topk_indices = torch.topk(candidate_scores, k)

        topk_label_ids = [int(candidate_labels[idx]) for idx in topk_indices.tolist()]
        relation_types = candidates["relation_types"]
        topk_relations = [relation_types[label_id] for label_id in topk_label_ids]
    except Exception as exc:
        return {
            "status": "failed",
            "operator": "task2_relation_candidate_scoring",
            "backend": "torch",
            "device": "cpu",
            "top_k": top_k,
            "top_k_indices": [],
            "top_k_scores": [],
            "top_k_relations": [],
            "reason": str(exc) or type(exc).__name__,
        }

    return {
        "status": "completed",
        "operator": "task2_relation_candidate_scoring",
        "backend": "torch",
        "device": "cpu",
        "candidate_count": int(candidates["candidate_count"]),
        "feature_dim": int(candidates["feature_dim"]),
        "relation_count": int(candidates["relation_count"]),
        "relation_types": relation_types,
        "top_k": top_k,
        "top_k_indices": [int(i) for i in topk_indices.tolist()],
        "top_k_scores": [round(float(v), 6) for v in topk_scores.tolist()],
        "top_k_label_ids": topk_label_ids,
        "top_k_relations": topk_relations,
        "skipped_full_format": True,
    }


def benchmark_relation_tensor_modes(
    candidates: dict[str, Any],
    cpu_result: dict[str, Any],
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
    iterations: int = 20,
    prefer_device: str = "auto",
    modes: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Benchmark Task-2 relation scoring optimization variants."""

    requested = _normalize_relation_modes(modes)
    device = get_device(prefer_device)
    if device.kind != "npu":
        return [
            _unavailable_mode_result(
                mode, device=device.device, reason=f"NPU device is unavailable: {device.reason}"
            )
            for mode in requested
        ]

    results = []
    cache: dict[str, Any] | None = None

    for mode in requested:
        if mode == "baseline_full_logits":
            results.append(
                _benchmark_baseline_full_logits(
                    candidates=candidates,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    prefer_device=prefer_device,
                )
            )
            continue

        if cache is None:
            started = time.perf_counter()
            cache = prepare_relation_tensor_cache(candidates, prefer_device=prefer_device)
            prepare_ms = _elapsed_ms(started)
        if cache.get("status") != "completed":
            results.append(
                _unavailable_mode_result(
                    mode,
                    device=cache.get("device", device.device),
                    reason=cache.get("reason", "NPU tensor preparation failed"),
                    status=cache.get("status", "failed"),
                )
            )
            continue

        if mode == "cached_full_logits":
            results.append(
                _benchmark_cached_mode(
                    candidates=candidates,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    cache=cache,
                    name=mode,
                    score_fn=score_cached_full_logits,
                    prepare_ms=prepare_ms,
                )
            )
        elif mode == "cached_argmax_labels":
            results.append(
                _benchmark_cached_mode(
                    candidates=candidates,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    cache=cache,
                    name=mode,
                    score_fn=score_cached_argmax_labels,
                    prepare_ms=prepare_ms,
                )
            )
        elif mode == "cached_topk_labels":
            results.append(
                _benchmark_topk_mode(
                    candidates=candidates,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    cache=cache,
                    name=mode,
                    score_fn=score_cached_topk_labels,
                    prepare_ms=prepare_ms,
                )
            )
        elif mode == "cpu_topk_labels":
            results.append(
                _benchmark_cpu_topk_mode(
                    candidates=candidates,
                    cpu_result=cpu_result,
                    cpu_latency_ms=cpu_latency_ms,
                    npu_latency_ms=npu_latency_ms,
                    iterations=iterations,
                    top_k=10,
                )
            )

    return _annotate_relation_mode_comparisons(results)


def _annotate_relation_mode_comparisons(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label baseline scope and add the steady-state top-k comparison."""

    by_name = {result.get("name"): result for result in results}
    for result in results:
        if "speedup_vs_cpu" in result:
            result["speedup_vs_cpu_full_format"] = result.get("speedup_vs_cpu")

    cached_topk = by_name.get("cached_topk_labels")
    cpu_topk = by_name.get("cpu_topk_labels")
    if not cached_topk or not cpu_topk:
        return results
    if cached_topk.get("status") != "completed" or cpu_topk.get("status") != "completed":
        return results

    npu_latency_ms = cached_topk.get("latency_ms_avg")
    cpu_latency_ms = cpu_topk.get("latency_ms_avg")
    steady_state_speedup = (
        round(cpu_latency_ms / npu_latency_ms, 4)
        if cpu_latency_ms is not None and npu_latency_ms
        else None
    )
    cached_topk.update(
        {
            "comparison_scope": "steady_state_topk",
            "includes_npu_cache_preparation": False,
            "steady_state_speedup_vs_cpu_topk": steady_state_speedup,
            "comparison_note": (
                "Both paths run matmul, max, top-k, and top-k formatting; "
                "NPU cache preparation is excluded from the timed steady-state path."
            ),
        }
    )
    cpu_topk["comparison_scope"] = "steady_state_topk"
    return results


def _benchmark_baseline_full_logits(
    candidates: dict[str, Any],
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
        last_result = score_relation_candidates_npu(candidates, prefer_device=prefer_device)
        durations.append(time.perf_counter() - started)
        if last_result.get("status") != "completed":
            return _unavailable_mode_result(
                "baseline_full_logits",
                device=last_result.get("device"),
                reason=last_result.get("reason", "baseline NPU execution failed"),
                status=last_result.get("status", "failed"),
            )

    candidate_count = int(candidates["candidate_count"])
    metrics = _mode_timing_metrics(
        name="baseline_full_logits",
        durations=durations,
        iterations=iterations,
        candidate_count=candidate_count,
        cpu_latency_ms=cpu_latency_ms,
        npu_latency_ms=npu_latency_ms,
    )
    return {
        **metrics,
        "backend": "torch_npu",
        "device": last_result.get("device"),
        "uses_cached_tensors": False,
        "skips_full_logits_copy": False,
        "includes_h2d_transfer": True,
        "includes_full_format": True,
        "correctness": _compare_mode_predictions(cpu_result, last_result),
        "top_relation": last_result.get("predicted_relations", [None])[0],
    }


def _benchmark_cached_mode(
    candidates: dict[str, Any],
    cpu_result: dict[str, Any],
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
    iterations: int,
    cache: dict[str, Any],
    name: str,
    score_fn: Any,
    prepare_ms: float,
) -> dict[str, Any]:
    durations = []
    last_result: dict[str, Any] = {"status": "not_run"}
    try:
        score_fn(cache)
        for _ in range(iterations):
            started = time.perf_counter()
            last_result = score_fn(cache)
            durations.append(time.perf_counter() - started)
    except Exception as exc:
        return _unavailable_mode_result(
            name,
            device=cache.get("device"),
            reason=str(exc) or type(exc).__name__,
            status="failed",
        )

    candidate_count = int(candidates["candidate_count"])
    skips_full_logits = last_result.get("skipped_full_logits_copy", False)

    metrics = _mode_timing_metrics(
        name=name,
        durations=durations,
        iterations=iterations,
        candidate_count=candidate_count,
        cpu_latency_ms=cpu_latency_ms,
        npu_latency_ms=npu_latency_ms,
    )

    correctness_result = score_cached_full_logits(cache)
    return {
        **metrics,
        "backend": "torch_npu",
        "device": cache.get("device"),
        "uses_cached_tensors": True,
        "skips_full_logits_copy": skips_full_logits,
        "includes_h2d_transfer": False,
        "includes_full_format": not skips_full_logits,
        "prepare_latency_ms": round(prepare_ms, 4),
        "correctness": _compare_mode_predictions(cpu_result, correctness_result),
        "correctness_check_included_in_timing": False,
        "top_relation": last_result.get("predicted_relations", [None])[0],
    }


def _benchmark_topk_mode(
    candidates: dict[str, Any],
    cpu_result: dict[str, Any],
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
    iterations: int,
    cache: dict[str, Any],
    name: str,
    score_fn: Any,
    prepare_ms: float,
    top_k: int = 10,
) -> dict[str, Any]:
    durations = []
    last_result: dict[str, Any] = {"status": "not_run"}
    try:
        score_fn(cache, top_k=top_k)
        for _ in range(iterations):
            started = time.perf_counter()
            last_result = score_fn(cache, top_k=top_k)
            durations.append(time.perf_counter() - started)
    except Exception as exc:
        return _unavailable_mode_result(
            name,
            device=cache.get("device"),
            reason=str(exc) or type(exc).__name__,
            status="failed",
        )

    candidate_count = int(candidates["candidate_count"])
    metrics = _mode_timing_metrics(
        name=name,
        durations=durations,
        iterations=iterations,
        candidate_count=candidate_count,
        cpu_latency_ms=cpu_latency_ms,
        npu_latency_ms=npu_latency_ms,
    )

    # Correctness: verify top-k indices match CPU's top-k
    correctness = _compare_topk_predictions(cpu_result, last_result, top_k=top_k)

    return {
        **metrics,
        "backend": "torch_npu",
        "device": cache.get("device"),
        "uses_cached_tensors": True,
        "skips_full_logits_copy": True,
        "skips_full_format": True,
        "includes_h2d_transfer": False,
        "includes_full_format": False,
        "prepare_latency_ms": round(prepare_ms, 4),
        "top_k": top_k,
        "correctness": correctness,
        "correctness_check_included_in_timing": False,
        "top_relation": last_result.get("top_k_relations", [None])[0],
    }


def _benchmark_cpu_topk_mode(
    candidates: dict[str, Any],
    cpu_result: dict[str, Any],
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
    iterations: int,
    top_k: int = 10,
) -> dict[str, Any]:
    durations = []
    last_result: dict[str, Any] = {"status": "not_run"}
    try:
        score_cpu_topk_labels(candidates, top_k=top_k)
        for _ in range(iterations):
            started = time.perf_counter()
            last_result = score_cpu_topk_labels(candidates, top_k=top_k)
            durations.append(time.perf_counter() - started)
    except Exception as exc:
        return _unavailable_mode_result(
            "cpu_topk_labels",
            device="cpu",
            reason=str(exc) or type(exc).__name__,
            status="failed",
        )

    candidate_count = int(candidates["candidate_count"])
    metrics = _mode_timing_metrics(
        name="cpu_topk_labels",
        durations=durations,
        iterations=iterations,
        candidate_count=candidate_count,
        cpu_latency_ms=cpu_latency_ms,
        npu_latency_ms=npu_latency_ms,
    )

    correctness = _compare_topk_predictions(cpu_result, last_result, top_k=top_k)

    return {
        **metrics,
        "backend": "torch",
        "device": "cpu",
        "uses_cached_tensors": False,
        "skips_full_logits_copy": False,
        "skips_full_format": True,
        "includes_h2d_transfer": False,
        "includes_full_format": False,
        "top_k": top_k,
        "correctness": correctness,
        "correctness_check_included_in_timing": False,
        "top_relation": last_result.get("top_k_relations", [None])[0],
    }


def _compare_topk_predictions(
    cpu_result: dict[str, Any],
    topk_result: dict[str, Any],
    top_k: int = 10,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    if topk_result.get("status") != "completed":
        return {"status": "not_run", "reason": topk_result.get("reason", "topk result unavailable")}
    if cpu_result.get("status") != "completed":
        return {"status": "not_run", "reason": "CPU result unavailable"}

    # Get CPU's top-k indices by sorting top_scores descending
    cpu_scores = cpu_result.get("top_scores", [])
    cpu_topk_indices = sorted(range(len(cpu_scores)), key=lambda i: cpu_scores[i], reverse=True)[:top_k]
    cpu_topk_set = set(cpu_topk_indices)

    topk_indices = topk_result.get("top_k_indices", [])
    topk_set = set(topk_indices)

    indices_match = cpu_topk_set == topk_set

    # Compare scores for matching indices
    if indices_match and cpu_scores:
        cpu_topk_scores = sorted([cpu_scores[i] for i in cpu_topk_indices], reverse=True)
        npu_topk_scores = sorted(topk_result.get("top_k_scores", []), reverse=True)
        if len(cpu_topk_scores) == len(npu_topk_scores):
            max_diff = max(abs(a - b) for a, b in zip(cpu_topk_scores, npu_topk_scores))
            scores_close = max_diff <= tolerance
        else:
            max_diff = None
            scores_close = False
    else:
        max_diff = None
        scores_close = indices_match

    passed = indices_match and scores_close
    return {
        "status": "passed" if passed else "failed",
        "topk_indices_match": indices_match,
        "scores_close": scores_close,
        "max_score_diff": round(max_diff, 6) if max_diff is not None else None,
        "tolerance": tolerance,
        "top_k": top_k,
    }


def _mode_timing_metrics(
    name: str,
    durations: list[float],
    iterations: int,
    candidate_count: int,
    cpu_latency_ms: float | None,
    npu_latency_ms: float | None,
) -> dict[str, Any]:
    metrics = {
        "name": name,
        "status": "completed",
        "measurement_mode": "mode_benchmark",
        **_timing_metrics(durations, iterations, candidate_count),
    }
    latency_ms = metrics.get("latency_ms_avg")
    metrics["speedup_vs_cpu"] = (
        round(cpu_latency_ms / latency_ms, 4) if cpu_latency_ms is not None and latency_ms else None
    )
    metrics["speedup_vs_npu_end_to_end"] = (
        round(npu_latency_ms / latency_ms, 4) if npu_latency_ms is not None and latency_ms else None
    )
    return metrics


def _unavailable_mode_result(
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
        "throughput_candidates_per_sec": None,
        "speedup_vs_cpu": None,
        "speedup_vs_npu_end_to_end": None,
    }


def _normalize_relation_modes(modes: list[str] | tuple[str, ...] | None) -> list[str]:
    if modes is None:
        return list(RELATION_TENSOR_BENCHMARK_MODES)
    normalized = []
    for mode in modes:
        if mode == "all":
            normalized.extend(RELATION_TENSOR_BENCHMARK_MODES)
            continue
        if mode not in RELATION_TENSOR_BENCHMARK_MODES:
            raise ValueError(f"unsupported relation tensor benchmark mode: {mode}")
        normalized.append(mode)
    return list(dict.fromkeys(normalized))


def _compare_mode_predictions(
    cpu_result: dict[str, Any],
    mode_result: dict[str, Any],
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    if mode_result.get("status") != "completed":
        return {"status": "not_run", "reason": mode_result.get("reason", "mode result unavailable")}
    if cpu_result.get("status") != "completed":
        return {"status": "not_run", "reason": "CPU result unavailable"}

    prediction_equal = cpu_result.get("predicted_relation_ids") == mode_result.get("predicted_relation_ids")
    cpu_scores = cpu_result.get("top_scores", [])
    mode_scores = mode_result.get("top_scores", [])
    if len(cpu_scores) == len(mode_scores) and cpu_scores and mode_scores:
        max_diff = max(abs(a - b) for a, b in zip(cpu_scores, mode_scores))
        scores_close = max_diff <= tolerance
    else:
        max_diff = None
        scores_close = cpu_scores == mode_scores
    return {
        "status": "passed" if prediction_equal and scores_close else "failed",
        "prediction_equal": prediction_equal,
        "scores_close": scores_close,
        "max_score_diff": round(max_diff, 6) if max_diff is not None else None,
        "tolerance": tolerance,
    }


def benchmark_task2_relation_tensor_ops(
    candidate_count: int = 4096,
    feature_dim: int = 256,
    relation_count: int = 5,
    iterations: int = 20,
    seed: int = 42,
    prefer_device: str = "auto",
    profile_breakdown: bool = False,
    benchmark_modes: list[str] | tuple[str, ...] | None = None,
    breakdown_warmup: int = 3,
    breakdown_iterations: int = 5,
    candidates: dict[str, Any] | None = None,
    input_source: str = "synthetic",
) -> dict[str, Any]:
    """Benchmark Task-2 tensorized relation scoring on CPU and NPU.

    When ``candidates`` is provided (e.g. encoded from a real medical corpus),
    it is benchmarked directly instead of generating synthetic inputs; the
    ``candidate_count`` / ``feature_dim`` / ``relation_count`` fields are then
    derived from the supplied candidates.
    """

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if candidates is None:
        candidates = generate_synthetic_relation_candidates(
            candidate_count=candidate_count,
            feature_dim=feature_dim,
            relation_count=relation_count,
            seed=seed,
        )
    else:
        candidate_count = int(candidates["candidate_count"])
        feature_dim = int(candidates["feature_dim"])
        relation_count = int(candidates["relation_count"])
    selected_device = get_device(prefer_device)

    cpu_last = score_relation_candidates_cpu(candidates)
    cpu_durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        cpu_last = score_relation_candidates_cpu(candidates)
        cpu_durations.append(time.perf_counter() - started)

    npu_last = score_relation_candidates_npu(candidates, prefer_device=prefer_device)
    npu_durations = []
    if npu_last.get("status") == "completed":
        for _ in range(iterations):
            started = time.perf_counter()
            npu_last = score_relation_candidates_npu(candidates, prefer_device=prefer_device)
            npu_durations.append(time.perf_counter() - started)

    cpu_metrics = {
        **_timing_metrics(cpu_durations, iterations, candidate_count),
        "status": cpu_last.get("status", "completed"),
        "backend": cpu_last.get("backend", "torch"),
        "device": cpu_last.get("device", "cpu"),
    }
    npu_metrics = (
        {
            **_timing_metrics(npu_durations, iterations, candidate_count),
            "status": npu_last.get("status", "completed"),
            "backend": npu_last.get("backend", "torch_npu"),
            "device": npu_last.get("device"),
            "runtime": npu_last.get("runtime"),
        }
        if npu_durations
        else {
            "iterations": iterations,
            "latency_ms_avg": None,
            "latency_ms_min": None,
            "latency_ms_max": None,
            "throughput_candidates_per_sec": None,
            "status": npu_last.get("status", "unavailable"),
            "backend": npu_last.get("backend", "torch_npu"),
            "device": npu_last.get("device"),
            "runtime": npu_last.get("runtime"),
            "reason": npu_last.get("reason"),
        }
    )
    speedup = None
    if npu_metrics.get("latency_ms_avg"):
        speedup = round(cpu_metrics["latency_ms_avg"] / npu_metrics["latency_ms_avg"], 4)

    report = {
        "task": "task2_relation_tensor_scoring",
        "input": {
            "candidate_count": candidate_count,
            "feature_dim": feature_dim,
            "relation_count": relation_count,
            "iterations": iterations,
            "seed": seed,
            "source": input_source,
        },
        "cpu": cpu_metrics,
        "npu": npu_metrics,
        "speedup": speedup,
        "correctness": compare_relation_scores(cpu_last, npu_last),
        "runtime": {
            "selected_device": asdict(selected_device),
            "relation_types": candidates["relation_types"],
        },
        "notes": [
            "This benchmark measures a tensorized Task-2 relation-candidate scoring operator.",
            (
                "Synthetic features stand in for pair embeddings from a medical relation extractor."
                if input_source == "synthetic"
                else "Real (Disease, Object) candidate pairs are encoded via build_features_from_records (no synthetic torch.randn)."
            ),
            "CPU and NPU paths use the same feature, weight, and bias tensors.",
            "NPU latency includes moving feature, weight, and bias tensors to the selected NPU device and copying logits back to CPU.",
        ],
    }
    if profile_breakdown:
        report["breakdown"] = profile_relation_scoring_breakdown(
            candidates=candidates,
            prefer_device=prefer_device,
            warmup_iterations=breakdown_warmup,
            profile_iterations=breakdown_iterations,
        )
    if benchmark_modes:
        report["mode_benchmarks"] = benchmark_relation_tensor_modes(
            candidates=candidates,
            cpu_result=cpu_last,
            cpu_latency_ms=cpu_metrics["latency_ms_avg"],
            npu_latency_ms=npu_metrics.get("latency_ms_avg"),
            iterations=iterations,
            prefer_device=prefer_device,
            modes=benchmark_modes,
        )
    return report


def _candidate_tensors(candidates: dict[str, Any], device: str) -> tuple[Any, Any, Any]:
    features = candidates["features"].to(device=device, dtype=_torch_float32(candidates["features"]))
    weights = candidates["weights"].to(device=device, dtype=_torch_float32(candidates["weights"]))
    bias = candidates["bias"].to(device=device, dtype=_torch_float32(candidates["bias"]))
    return features, weights, bias


def _torch_float32(tensor: Any) -> Any:
    return tensor.new_empty(()).float().dtype


def _format_scoring_result(
    candidates: dict[str, Any],
    logits: Any,
    backend: str,
    device: str,
    operator: str,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values, indices = logits.max(dim=1)
    predicted_ids = [int(value) for value in indices.tolist()]
    relation_types = candidates["relation_types"]
    return {
        "status": "completed",
        "operator": operator,
        "backend": backend,
        "device": device,
        "runtime": runtime,
        "candidate_count": int(candidates["candidate_count"]),
        "feature_dim": int(candidates["feature_dim"]),
        "relation_count": int(candidates["relation_count"]),
        "relation_types": relation_types,
        "logits": logits,
        "predicted_relation_ids": predicted_ids,
        "predicted_relations": [relation_types[idx] for idx in predicted_ids],
        "top_scores": [round(float(value), 6) for value in values.tolist()],
    }


def _relation_types(relation_count: int) -> list[str]:
    if relation_count <= len(DEFAULT_RELATION_TYPES):
        return DEFAULT_RELATION_TYPES[:relation_count]
    extra = [f"relation_{idx}" for idx in range(len(DEFAULT_RELATION_TYPES), relation_count)]
    return [*DEFAULT_RELATION_TYPES, *extra]


def _timing_metrics(durations: list[float], iterations: int, candidate_count: int) -> dict[str, Any]:
    if not durations:
        return {
            "iterations": iterations,
            "latency_ms_avg": None,
            "latency_ms_min": None,
            "latency_ms_max": None,
            "throughput_candidates_per_sec": None,
        }
    total = sum(durations)
    return {
        "iterations": iterations,
        "latency_ms_avg": round(mean(durations) * 1000, 4),
        "latency_ms_min": round(min(durations) * 1000, 4),
        "latency_ms_max": round(max(durations) * 1000, 4),
        "throughput_candidates_per_sec": _throughput(candidate_count * iterations, total),
    }


def _throughput(items: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return round(items / seconds, 4)


def _synchronize_npu(torch: Any) -> None:
    npu = getattr(torch, "npu", None)
    synchronize = getattr(npu, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _empty_relation_breakdown_steps() -> dict[str, float | None]:
    return {
        "device_select_ms": None,
        "h2d_features_ms": None,
        "h2d_weights_bias_ms": None,
        "matmul_ms": None,
        "argmax_npu_ms": None,
        "logits_d2h_ms": None,
        "labels_d2h_ms": None,
        "argmax_cpu_ms": None,
        "format_result_ms": None,
    }
