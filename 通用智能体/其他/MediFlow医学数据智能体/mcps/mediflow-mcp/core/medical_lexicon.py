# -*- coding: utf-8 -*-
"""任务二离线词典资产加载。

词典和已知关系保存在独立 JSON 文件中，训练数据更新时只需重建资产，
无需修改抽取算子代码。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


TYPE_PRIORITY = {
    "dis": 0,
    "dru": 1,
    "sym": 2,
    "ite": 3,
    "pro": 4,
    "dep": 5,
    "mic": 6,
    "bod": 7,
}


def _asset_dir() -> Path:
    configured = os.getenv("CCF_TASK2_ASSET_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "task2"


@lru_cache(maxsize=1)
def load_benchmark_terms() -> tuple[tuple[str, str], ...]:
    """加载训练集生成的术语及其主类型。"""
    path = _asset_dir() / "entity_lexicon.json"
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms = payload.get("terms", {})
    result: list[tuple[str, str]] = []
    for term, counts in terms.items():
        if not isinstance(counts, dict) or not counts:
            continue
        entity_type = sorted(
            counts,
            key=lambda item: (-int(counts[item]), TYPE_PRIORITY.get(item, 99), item),
        )[0]
        result.append((str(term), entity_type))
    return tuple(sorted(result, key=lambda item: (-len(item[0]), item[0], item[1])))


@lru_cache(maxsize=1)
def load_known_relation_pairs() -> dict[tuple[str, str], str]:
    """加载训练集中已有的主语、宾语与关系映射。"""
    path = _asset_dir() / "relation_pairs.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], str] = {}
    for key, counts in payload.get("pairs", {}).items():
        if "\u0001" not in key or not isinstance(counts, dict) or not counts:
            continue
        subject, obj = key.split("\u0001", 1)
        predicate = max(counts, key=lambda item: (int(counts[item]), item))
        result[(subject, obj)] = predicate
    return result


@lru_cache(maxsize=1)
def load_relation_terms() -> tuple[tuple[str, str], ...]:
    """加载关系训练集中高频出现的主客体术语。"""
    path = _asset_dir() / "relation_pairs.json"
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: list[tuple[str, str]] = []
    for term, counts in payload.get("terms", {}).items():
        if not isinstance(counts, dict) or not counts:
            continue
        entity_type = sorted(
            counts,
            key=lambda item: (-int(counts[item]), TYPE_PRIORITY.get(item, 99), item),
        )[0]
        result.append((str(term), entity_type))
    return tuple(sorted(result, key=lambda item: (-len(item[0]), item[0], item[1])))


def clear_lexicon_caches() -> None:
    """重建资产后清除进程内缓存。"""
    load_benchmark_terms.cache_clear()
    load_known_relation_pairs.cache_clear()
    load_relation_terms.cache_clear()
