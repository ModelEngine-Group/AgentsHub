"""任务一混合清洗的执行策略与并发调度。"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def should_start_background_job(wait: bool) -> bool:
    """非阻塞请求始终进入后台，避免按文件数误判实际耗时。"""

    return not wait


def task1_parallel_workers(job_count: int) -> int:
    """返回任务一格式分组的并发上限，最多覆盖五种受支持格式。"""

    configured = os.environ.get("CCF_TASK1_PARALLEL_GROUPS", "5")
    try:
        limit = int(configured)
    except ValueError:
        limit = 5
    return max(1, min(job_count, limit, 5))


def run_parallel_group_jobs(
    groups: Iterable[T],
    worker: Callable[[T], R],
    on_complete: Callable[[T, R, int, int], None] | None = None,
) -> dict[T, R]:
    """并发执行彼此独立的格式分组，并在主线程汇总结果。"""

    group_list = list(groups)
    if not group_list:
        return {}
    results: dict[T, R] = {}
    with ThreadPoolExecutor(
        max_workers=task1_parallel_workers(len(group_list)),
        thread_name_prefix="task1-format",
    ) as executor:
        future_to_group = {executor.submit(worker, group): group for group in group_list}
        for completed, future in enumerate(as_completed(future_to_group), start=1):
            group = future_to_group[future]
            result = future.result()
            results[group] = result
            if on_complete is not None:
                on_complete(group, result, completed, len(group_list))
    return results
