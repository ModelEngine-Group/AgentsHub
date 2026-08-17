# -*- coding: utf-8 -*-
"""任务二分组可靠性配置。

可靠性来自验证集上的分组精确率，只用于内部入库决策，不表示逐条事实概率。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Reliability:
    level: str
    score: float
    support: int


_FALLBACK = {
    "dictionary_exact": Reliability("medium", 0.70, 0),
    "known_pair": Reliability("high", 0.90, 0),
    "sentence_rule": Reliability("medium", 0.65, 0),
    "context_rule": Reliability("low", 0.40, 0),
    "explicit_section_frame": Reliability("medium", 0.85, 0),
    "llm": Reliability("medium", 0.70, 0),
}


def _profile_path() -> Path:
    configured = os.getenv("CCF_TASK2_RELIABILITY_PROFILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "task2" / "reliability_profile.json"


@lru_cache(maxsize=1)
def _groups() -> dict[str, dict]:
    path = _profile_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("groups", {})


def reliability_for(stage: str, method: str, category: str) -> Reliability:
    """按处理阶段、抽取方法和类型读取分组可靠性。"""
    item = _groups().get(f"{stage}|{method}|{category}")
    if not item:
        return _FALLBACK.get(method, Reliability("low", 0.40, 0))
    return Reliability(
        level=str(item.get("level") or "low"),
        score=float(item.get("precision") or 0.0),
        support=int(item.get("predicted") or 0),
    )


def clear_reliability_cache() -> None:
    _groups.cache_clear()
