"""实体归一化与链接（Entity Normalization & Linking）。

医疗文本中同一概念常有多种写法（缩写、英文、别名、繁简），生产级知识图谱
必须将其归一到规范名（canonical name），否则同一实体会碎裂成多个节点。

策略（分级，命中即返回）：
1. 精确别名映射：synonyms.json 中「规范名 -> 别名列表」的反向索引；
2. 大小写/全半角规范化后再查表；
3. 编辑距离模糊匹配（阈值内取最近规范名），兜底容错 OCR/拼写噪声。

返回 (canonical_name, method, score)，method ∈ {exact, alias, normalized, fuzzy, none}。
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

SYN_PATH = Path(__file__).resolve().parents[2] / "data" / "dictionaries" / "synonyms.json"


def _norm_text(s: str) -> str:
    """全半角统一、去空白、小写：用于健壮匹配。"""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return s.replace(" ", "")


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class EntityNormalizer:
    """基于同义词表 + 规范化 + 模糊匹配的实体链接器。"""

    def __init__(self, synonyms: dict[str, list[str]] | None = None,
                 fuzzy_max_dist: int = 0, fuzzy_min_len: int = 4) -> None:
        raw = synonyms or json.loads(SYN_PATH.read_text(encoding="utf-8"))
        self.canonical: dict[str, str] = {}   # 原始别名 -> 规范名
        self.norm_index: dict[str, str] = {}  # 规范化别名 -> 规范名
        for canon, aliases in raw.items():
            for a in [canon, *aliases]:
                self.canonical[a] = canon
                self.norm_index[_norm_text(a)] = canon
        # 模糊匹配默认关闭：中文短词一次编辑即改变语义（心悸↔心梗），风险极高。
        # 仅当显式开启且词长≥fuzzy_min_len 时才启用，用于容忍 OCR/拼写噪声。
        self.fuzzy_max_dist = fuzzy_max_dist
        self.fuzzy_min_len = fuzzy_min_len
        self._all_norm = list(self.norm_index.keys())

    def link(self, mention: str) -> tuple[str, str, float]:
        if mention in self.canonical:
            method = "exact" if mention == self.canonical[mention] else "alias"
            return self.canonical[mention], method, 1.0
        nm = _norm_text(mention)
        if nm in self.norm_index:
            return self.norm_index[nm], "normalized", 0.95
        # 模糊匹配（默认关闭）：仅对足够长的词、长度接近的候选计算
        if self.fuzzy_max_dist > 0 and len(nm) >= self.fuzzy_min_len:
            best, best_d = None, self.fuzzy_max_dist + 1
            for cand in self._all_norm:
                if abs(len(cand) - len(nm)) > self.fuzzy_max_dist or len(cand) < self.fuzzy_min_len:
                    continue
                d = _edit_distance(nm, cand)
                if d < best_d:
                    best, best_d = cand, d
            if best is not None and best_d <= self.fuzzy_max_dist:
                score = 1.0 - best_d / max(len(nm), 1)
                return self.norm_index[best], "fuzzy", round(score, 3)
        return mention, "none", 0.0

    def normalize(self, mention: str) -> str:
        return self.link(mention)[0]
