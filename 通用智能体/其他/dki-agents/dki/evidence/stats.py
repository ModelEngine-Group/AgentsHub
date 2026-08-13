"""确定性统计工具：bootstrap 置信区间与基础汇总。

用于把“5 次正式重复的均值”升级为带不确定度的统计合同（mean/std/median/p95 +
bootstrap 95% CI）。bootstrap 使用固定种子，结果可复现、可审计。
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        raise ValueError("percentile of empty sequence")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = 10000,
    seed: int = 20260720,
) -> Dict[str, float]:
    """返回均值的 bootstrap 置信区间（确定性：固定 seed）。

    小样本（如 5 次重复）也可用；返回 point/lower/upper/iterations。
    """
    data = [float(v) for v in values]
    if not data:
        raise ValueError("bootstrap_ci requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    point = statistics.fmean(data)
    if len(data) == 1:
        return {"point": point, "lower": point, "upper": point, "iterations": 0, "confidence": confidence}
    import random

    rng = random.Random(seed)
    n = len(data)
    means: List[float] = []
    for _ in range(iterations):
        resample = [data[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(resample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": round(point, 6),
        "lower": round(_percentile(means, alpha), 6),
        "upper": round(_percentile(means, 1.0 - alpha), 6),
        "iterations": iterations,
        "confidence": confidence,
    }


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """mean/std/median/min/max/p95 + bootstrap CI（供性能统计合同使用）。"""
    data = [float(v) for v in values]
    if not data:
        raise ValueError("summarize requires at least one value")
    ordered = sorted(data)
    return {
        "n": len(data),
        "mean": round(statistics.fmean(data), 6),
        "stdev": round(statistics.stdev(data), 6) if len(data) > 1 else 0.0,
        "median": round(statistics.median(data), 6),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": round(_percentile(ordered, 0.95), 6),
        "ci95": bootstrap_ci(data),
    }
