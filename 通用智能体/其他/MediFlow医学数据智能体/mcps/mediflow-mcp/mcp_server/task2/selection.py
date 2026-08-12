# -*- coding: utf-8 -*-
"""
任务二记录选择模块。
"""

from __future__ import annotations

from collections import defaultdict


def select_balanced_records(records: list[dict], max_records: int) -> list[dict]:
    """跨源文件选择记录，避免只截取单个文件前缀。"""
    if max_records <= 0 or len(records) <= max_records:
        return records

    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = str(record.get("source_file") or record.get("source_format") or "unknown")
        buckets[key].append(record)

    selected: list[dict] = []
    keys = list(buckets)
    cursor = 0
    while len(selected) < max_records and keys:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop(0))
        keys = [item for item in keys if buckets[item]]
        cursor += 1
    return selected
