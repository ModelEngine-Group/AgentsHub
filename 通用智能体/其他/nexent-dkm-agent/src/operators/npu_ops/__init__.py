"""NPU operator wrappers and benchmark helpers."""

from src.operators.npu_ops.analysis_benchmark import benchmark_task3_analysis_ops
from src.operators.npu_ops.graph_tensor_ops import (
    GRAPH_DEGREE_BENCHMARK_MODES,
    benchmark_degree_centrality_npu_prepared,
    benchmark_graph_degree_centrality,
    benchmark_graph_degree_modes,
    compute_degree_centrality_cpu,
    compute_degree_centrality_npu,
    compute_degree_topk_npu_cached,
    generate_synthetic_graph,
    prepare_graph_degree_tensor_cache,
)
from src.operators.npu_ops.kg_benchmark import benchmark_task2_kg_ops, detect_npu_runtime
from src.operators.npu_ops.kg_tensor_ops import (
    benchmark_task2_relation_tensor_ops,
    compare_relation_scores,
    generate_synthetic_relation_candidates,
    score_relation_candidates_cpu,
    score_relation_candidates_npu,
)

__all__ = [
    "GRAPH_DEGREE_BENCHMARK_MODES",
    "benchmark_degree_centrality_npu_prepared",
    "benchmark_graph_degree_centrality",
    "benchmark_graph_degree_modes",
    "benchmark_task2_relation_tensor_ops",
    "benchmark_task2_kg_ops",
    "benchmark_task3_analysis_ops",
    "compare_relation_scores",
    "compute_degree_centrality_cpu",
    "compute_degree_centrality_npu",
    "compute_degree_topk_npu_cached",
    "detect_npu_runtime",
    "generate_synthetic_graph",
    "generate_synthetic_relation_candidates",
    "prepare_graph_degree_tensor_cache",
    "score_relation_candidates_cpu",
    "score_relation_candidates_npu",
]
