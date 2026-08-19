from __future__ import annotations

from src.common.device import DeviceSpec
from src.operators.npu_ops.kg_tensor_ops import (
    RELATION_TENSOR_BENCHMARK_MODES,
    benchmark_relation_tensor_modes,
    benchmark_task2_relation_tensor_ops,
    compare_relation_scores,
    generate_synthetic_relation_candidates,
    prepare_relation_tensor_cache,
    profile_relation_scoring_breakdown,
    score_cached_argmax_labels,
    score_cached_full_logits,
    score_cached_topk_labels,
    score_relation_candidates_cpu,
    score_relation_candidates_npu,
)


def test_generate_synthetic_relation_candidates_is_deterministic():
    first = generate_synthetic_relation_candidates(
        candidate_count=6,
        feature_dim=4,
        relation_count=3,
        seed=11,
    )
    second = generate_synthetic_relation_candidates(
        candidate_count=6,
        feature_dim=4,
        relation_count=3,
        seed=11,
    )

    assert first["candidate_count"] == 6
    assert first["feature_dim"] == 4
    assert first["relation_count"] == 3
    assert first["relation_types"] == second["relation_types"]
    assert first["features"].equal(second["features"])
    assert first["weights"].equal(second["weights"])
    assert first["bias"].equal(second["bias"])


def test_cpu_relation_scoring_returns_expected_predictions():
    import torch

    candidates = {
        "candidate_count": 3,
        "feature_dim": 2,
        "relation_count": 2,
        "relation_types": ["has_symptom", "treated_by"],
        "features": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "weights": torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        "bias": torch.tensor([0.0, 0.0]),
        "seed": 0,
    }

    result = score_relation_candidates_cpu(candidates)

    assert result["status"] == "completed"
    assert result["backend"] == "torch"
    assert result["device"] == "cpu"
    assert result["predicted_relation_ids"] == [0, 1, 1]
    assert result["predicted_relations"] == ["has_symptom", "treated_by", "treated_by"]
    assert result["top_scores"] == [1.0, 2.0, 2.0]


def test_compare_relation_scores_passes_for_matching_outputs():
    candidates = generate_synthetic_relation_candidates(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        seed=5,
    )
    left = score_relation_candidates_cpu(candidates)
    right = score_relation_candidates_cpu(candidates)

    comparison = compare_relation_scores(left, right)

    assert comparison["status"] == "passed"
    assert comparison["prediction_equal"] is True
    assert comparison["max_abs_diff"] == 0.0


def test_npu_relation_scoring_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )
    candidates = generate_synthetic_relation_candidates(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        seed=7,
    )

    result = score_relation_candidates_npu(candidates)

    assert result["status"] == "unavailable"
    assert result["operator"] == "task2_relation_candidate_scoring"
    assert result["backend"] == "torch"
    assert result["device"] == "cpu"
    assert result["predicted_relation_ids"] == []
    assert "npu" in result["reason"].lower()


def test_task2_relation_tensor_benchmark_report_structure_without_npu(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_task2_relation_tensor_ops(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        iterations=2,
        seed=13,
    )

    assert report["task"] == "task2_relation_tensor_scoring"
    assert report["input"] == {
        "candidate_count": 8,
        "feature_dim": 4,
        "relation_count": 3,
        "iterations": 2,
        "seed": 13,
        "source": "synthetic",
    }
    assert report["cpu"]["status"] == "completed"
    assert report["cpu"]["latency_ms_avg"] >= 0
    assert report["cpu"]["throughput_candidates_per_sec"] > 0
    assert report["npu"]["status"] == "unavailable"
    assert report["speedup"] is None
    assert report["correctness"]["status"] == "not_run"
    assert report["runtime"]["selected_device"]["kind"] == "cpu"


def test_profile_breakdown_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    result = profile_relation_scoring_breakdown(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        seed=17,
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == "torch"
    assert result["device"] == "cpu"
    assert result["measurement_mode"] == "breakdown_profile"
    assert "npu" in result["reason"].lower()
    steps = result["steps"]
    assert all(value is None for value in steps.values())


def test_profile_breakdown_has_expected_step_keys(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    result = profile_relation_scoring_breakdown(
        candidate_count=4,
        feature_dim=2,
        relation_count=2,
        seed=19,
    )

    expected_keys = {
        "device_select_ms",
        "h2d_features_ms",
        "h2d_weights_bias_ms",
        "matmul_ms",
        "argmax_npu_ms",
        "logits_d2h_ms",
        "labels_d2h_ms",
        "argmax_cpu_ms",
        "format_result_ms",
    }
    assert set(result["steps"].keys()) == expected_keys


def test_profile_breakdown_in_benchmark_report_when_flagged(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_task2_relation_tensor_ops(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        iterations=2,
        seed=23,
        profile_breakdown=True,
    )

    assert "breakdown" in report
    assert report["breakdown"]["status"] == "unavailable"
    assert report["breakdown"]["measurement_mode"] == "breakdown_profile"
    assert "steps" in report["breakdown"]


def test_profile_breakdown_absent_in_benchmark_report_when_not_flagged(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_task2_relation_tensor_ops(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        iterations=2,
        seed=29,
        profile_breakdown=False,
    )

    assert "breakdown" not in report


def test_profile_breakdown_uses_provided_candidates(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    candidates = generate_synthetic_relation_candidates(
        candidate_count=10,
        feature_dim=8,
        relation_count=4,
        seed=31,
    )

    result = profile_relation_scoring_breakdown(candidates=candidates)

    assert result["status"] == "unavailable"
    assert "candidate_count" not in result
    assert "steps" in result


def test_relation_tensor_cache_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )
    candidates = generate_synthetic_relation_candidates(
        candidate_count=8, feature_dim=4, relation_count=3, seed=37,
    )

    cache = prepare_relation_tensor_cache(candidates)

    assert cache["status"] == "unavailable"
    assert cache["operator"] == "relation_tensor_cache"
    assert "npu" in cache["reason"].lower()


def test_score_cached_full_logits_reports_unavailable_without_cache():
    result = score_cached_full_logits({"status": "unavailable", "device": "cpu"})

    assert result["status"] == "unavailable"
    assert result["predicted_relation_ids"] == []


def test_score_cached_argmax_labels_reports_unavailable_without_cache():
    result = score_cached_argmax_labels({"status": "unavailable", "device": "cpu"})

    assert result["status"] == "unavailable"
    assert result["predicted_relation_ids"] == []


def test_benchmark_modes_reports_unavailable_without_npu(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )
    candidates = generate_synthetic_relation_candidates(
        candidate_count=8, feature_dim=4, relation_count=3, seed=41,
    )
    cpu_result = score_relation_candidates_cpu(candidates)

    results = benchmark_relation_tensor_modes(
        candidates=candidates,
        cpu_result=cpu_result,
        cpu_latency_ms=1.0,
        npu_latency_ms=None,
        iterations=2,
        modes=["all"],
    )

    assert len(results) == len(RELATION_TENSOR_BENCHMARK_MODES)
    for result in results:
        assert result["status"] == "unavailable"
        assert result["speedup_vs_cpu"] is None


def test_benchmark_modes_in_report_when_requested(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_task2_relation_tensor_ops(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        iterations=2,
        seed=43,
        benchmark_modes=["all"],
    )

    assert "mode_benchmarks" in report
    assert len(report["mode_benchmarks"]) == len(RELATION_TENSOR_BENCHMARK_MODES)
    for mode_result in report["mode_benchmarks"]:
        assert mode_result["status"] == "unavailable"


def test_benchmark_modes_absent_when_not_requested(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_task2_relation_tensor_ops(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        iterations=2,
        seed=47,
    )

    assert "mode_benchmarks" not in report


def test_normalize_relation_modes_rejects_invalid():
    import pytest

    with pytest.raises(ValueError, match="unsupported relation tensor benchmark mode"):
        benchmark_relation_tensor_modes(
            candidates={},
            cpu_result={},
            cpu_latency_ms=None,
            npu_latency_ms=None,
            modes=["nonexistent_mode"],
        )


def test_cached_topk_labels_in_benchmark_modes_list():
    assert "cached_topk_labels" in RELATION_TENSOR_BENCHMARK_MODES


def test_score_cached_topk_labels_reports_unavailable_without_cache():
    result = score_cached_topk_labels({"status": "unavailable", "device": "cpu"}, top_k=5)

    assert result["status"] == "unavailable"
    assert result["top_k"] == 5
    assert result["top_k_indices"] == []
    assert result["top_k_scores"] == []
    assert result["top_k_relations"] == []


def test_benchmark_modes_include_cached_topk_labels(monkeypatch):
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    monkeypatch.setattr(
        kg_tensor_ops,
        "get_device",
        lambda _prefer="auto": DeviceSpec(kind="cpu", device="cpu", reason="no npu"),
    )

    report = benchmark_task2_relation_tensor_ops(
        candidate_count=8,
        feature_dim=4,
        relation_count=3,
        iterations=2,
        seed=53,
        benchmark_modes=["all"],
    )

    assert "mode_benchmarks" in report
    mode_names = [m["name"] for m in report["mode_benchmarks"]]
    assert "cached_topk_labels" in mode_names


def test_topk_mode_comparison_labels_full_format_and_steady_state_baselines():
    import src.operators.npu_ops.kg_tensor_ops as kg_tensor_ops

    results = [
        {
            "name": "cached_topk_labels",
            "status": "completed",
            "latency_ms_avg": 2.0,
            "speedup_vs_cpu": 50.0,
        },
        {
            "name": "cpu_topk_labels",
            "status": "completed",
            "latency_ms_avg": 10.0,
            "speedup_vs_cpu": 10.0,
        },
    ]

    annotated = kg_tensor_ops._annotate_relation_mode_comparisons(results)
    cached_topk = annotated[0]

    assert cached_topk["speedup_vs_cpu_full_format"] == 50.0
    assert cached_topk["steady_state_speedup_vs_cpu_topk"] == 5.0
    assert cached_topk["comparison_scope"] == "steady_state_topk"
    assert cached_topk["includes_npu_cache_preparation"] is False
    assert "cache preparation" in cached_topk["comparison_note"]
