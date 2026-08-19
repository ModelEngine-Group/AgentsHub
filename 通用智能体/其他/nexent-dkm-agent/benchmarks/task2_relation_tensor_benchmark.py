"""Run Task-2 relation tensor CPU/NPU benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.npu_ops.kg_tensor_ops import (
    RELATION_TENSOR_BENCHMARK_MODES,
    benchmark_task2_relation_tensor_ops,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Task-2 tensor relation scoring.")
    parser.add_argument("--candidate-count", type=int, default=4096, help="Number of relation candidates.")
    parser.add_argument("--feature-dim", type=int, default=256, help="Candidate embedding feature dimension.")
    parser.add_argument("--relation-count", type=int, default=5, help="Number of relation labels.")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic input seed.")
    parser.add_argument(
        "--prefer-device",
        choices=["auto", "npu", "cuda", "cpu"],
        default="auto",
        help="Preferred runtime device.",
    )
    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Profile NPU relation scoring step-by-step breakdown.",
    )
    parser.add_argument(
        "--breakdown-warmup",
        type=int,
        default=3,
        help="Warmup iterations before breakdown profiling (default 3).",
    )
    parser.add_argument(
        "--breakdown-iterations",
        type=int,
        default=5,
        help="Profiled iterations for breakdown averaging (default 5).",
    )
    parser.add_argument(
        "--benchmark-modes",
        nargs="*",
        default=None,
        choices=["all", *RELATION_TENSOR_BENCHMARK_MODES],
        help=f"Benchmark optimization modes. 'all' runs all {len(RELATION_TENSOR_BENCHMARK_MODES)} modes.",
    )
    parser.add_argument(
        "--real-corpus",
        default=None,
        help=(
            "Path to a medical text corpus. When set, real (Disease, Object) "
            "candidate pairs are extracted and encoded instead of synthetic inputs."
        ),
    )
    parser.add_argument(
        "--monitor-npu",
        action="store_true",
        help="Sample npu-smi AICore%%/utilization/power during the benchmark.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="npu-smi sampling interval in seconds (default 0.5).",
    )
    parser.add_argument(
        "--npu-id",
        type=int,
        default=None,
        help="npu-smi NPU id to sample (default: auto-detect).",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def _build_real_corpus_candidates(corpus_path: str) -> dict:
    """Extract and encode real (Disease, Object) candidate pairs from a corpus."""

    from src.operators.kg_ops import build_relation_candidates, build_scoring_inputs
    from src.operators.kg_ops.entity_extractor import extract_medical_entities

    text = Path(corpus_path).read_text(encoding="utf-8")
    extraction = extract_medical_entities(text)
    candidate_pairs = build_relation_candidates(extraction["records"])
    if not candidate_pairs:
        raise ValueError(
            f"No (Disease, Object) relation candidates found in corpus: {corpus_path}"
        )
    return build_scoring_inputs(candidate_pairs)


def _evaluate_real_corpus_correctness(corpus_path: str) -> dict:
    """Verify CPU tensor scoring reproduces rule-based triples on a real corpus."""

    from src.operators.kg_ops import (
        extract_medical_entities,
        extract_relations,
        extract_relations_tensorized,
    )

    text = Path(corpus_path).read_text(encoding="utf-8")
    records = extract_medical_entities(text)["records"]
    rule_triples = extract_relations(records)
    tensor_result = extract_relations_tensorized(records, backend="cpu")

    def _norm(triples):
        return sorted((t["subject"], t["predicate"], t["object"]) for t in triples)

    passed = (
        tensor_result.get("status") == "completed"
        and _norm(rule_triples) == _norm(tensor_result.get("triples", []))
    )
    return {
        "status": "passed" if passed else "failed",
        "backend": tensor_result.get("scoring_backend", "cpu"),
        "candidate_count": tensor_result.get("candidate_count", 0),
        "triple_count": len(tensor_result.get("triples", [])),
        "metric": "cpu_tensor_matches_rule_triples",
    }


def _serialize_report(report: dict) -> dict:
    """Strip non-serializable torch tensors from the report before JSON dump."""

    import torch

    clean = {}
    for key, value in report.items():
        if isinstance(value, torch.Tensor):
            continue
        if isinstance(value, dict):
            clean[key] = _serialize_report(value)
        elif isinstance(value, list):
            clean[key] = [
                _serialize_report(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            clean[key] = value
    return clean


def main() -> int:
    args = parse_args()

    real_candidates = None
    input_source = "synthetic"
    if args.real_corpus:
        real_candidates = _build_real_corpus_candidates(args.real_corpus)
        input_source = f"real_corpus:{Path(args.real_corpus).name}"

    sampler = None
    if args.monitor_npu:
        from benchmarks.npu_monitor import NpuUtilizationSampler

        sampler = NpuUtilizationSampler(npu_id=args.npu_id, interval_s=args.monitor_interval)
        sampler.start()

    try:
        report = benchmark_task2_relation_tensor_ops(
            candidate_count=args.candidate_count,
            feature_dim=args.feature_dim,
            relation_count=args.relation_count,
            iterations=args.iterations,
            seed=args.seed,
            prefer_device=args.prefer_device,
            profile_breakdown=args.profile_breakdown,
            benchmark_modes=args.benchmark_modes,
            breakdown_warmup=args.breakdown_warmup,
            breakdown_iterations=args.breakdown_iterations,
            candidates=real_candidates,
            input_source=input_source,
        )
    finally:
        if sampler is not None:
            sampler.stop()

    if sampler is not None:
        report["npu_utilization"] = sampler.result()

    if args.real_corpus:
        report["correctness"] = _evaluate_real_corpus_correctness(args.real_corpus)

    serializable = _serialize_report(report)
    payload = json.dumps(serializable, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
