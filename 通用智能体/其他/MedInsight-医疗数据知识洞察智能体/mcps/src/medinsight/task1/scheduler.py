from __future__ import annotations

from typing import Any


THROUGHPUT_KEYWORDS = ("性能", "快速", "加速", "批量", "高并发", "吞吐", "大规模", "优化")


def build_batch_scheduler(
    task: str,
    profile: dict[str, Any],
    enabled: bool = True,
) -> dict[str, Any]:
    """Choose a deterministic execution policy for Task1 operators."""
    record_count = int(profile.get("record_count") or 0)
    avg_text_length = float(profile.get("avg_text_length") or 0)
    task_text = task or ""
    throughput_first = any(keyword in task_text for keyword in THROUGHPUT_KEYWORDS) or record_count >= 100

    if not enabled:
        return {
            "name": "batch_scheduler",
            "enabled": False,
            "strategy": "sequential_baseline",
            "batch_size": 1,
            "parallel_workers": 1,
            "enable_content_cache": False,
            "use_optimized_text_ops": False,
            "reason": "Benchmark baseline: disabled adaptive scheduling for sequential comparison.",
        }

    if throughput_first:
        batch_size = 128 if record_count >= 500 else 64 if record_count >= 100 else 32
        return {
            "name": "batch_scheduler",
            "enabled": True,
            "strategy": "throughput_first",
            "batch_size": batch_size,
            "parallel_workers": 1,
            "enable_content_cache": False,
            "use_optimized_text_ops": True,
            "avg_text_length": avg_text_length,
            "reason": "Use optimized short/long text operators; avoid cache and thread overhead.",
        }

    return {
        "name": "batch_scheduler",
        "enabled": True,
        "strategy": "quality_first",
        "batch_size": 32,
        "parallel_workers": 1,
        "enable_content_cache": False,
        "use_optimized_text_ops": True,
        "avg_text_length": avg_text_length,
        "reason": "Small or quality-oriented workload; keep deterministic text operators.",
    }
