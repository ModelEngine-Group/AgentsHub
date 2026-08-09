"""实体识别算子（医疗 NER）。

CPU 基线：逐文档、逐词典项 str.find 扫描（朴素实现，作为性能对比基线）。
CPU 优化：单遍 Aho-Corasick 自动机多模式匹配，复杂度 O(n)。
NPU 版本见 mgi.operators.npu.ner_npu。
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from .base import Operator, OperatorMetadata, registry
from ..kg.schema import Entity
from ..kg.normalize import EntityNormalizer

DICT_PATH = Path(__file__).resolve().parents[2] / "data" / "dictionaries" / "medical_terms.json"


def load_dictionary(path: Path | str = DICT_PATH) -> dict[str, str]:
    """加载 词条 -> 实体类型 词典。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    term2type: dict[str, str] = {}
    for etype, terms in raw.items():
        for t in terms:
            term2type[t] = etype
    return term2type


class AhoCorasick:
    """轻量 Aho-Corasick 自动机：多模式串单遍匹配。"""

    def __init__(self, patterns: dict[str, str]) -> None:
        self.goto: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.out: list[list[tuple[str, str]]] = [[]]
        for word, tag in patterns.items():
            self._insert(word, tag)
        self._build()

    def _insert(self, word: str, tag: str) -> None:
        node = 0
        for ch in word:
            nxt = self.goto[node].get(ch)
            if nxt is None:
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
                nxt = len(self.goto) - 1
                self.goto[node][ch] = nxt
            node = nxt
        self.out[node].append((word, tag))

    def _build(self) -> None:
        q: deque[int] = deque()
        for child in self.goto[0].values():
            q.append(child)
        while q:
            node = q.popleft()
            for ch, child in self.goto[node].items():
                q.append(child)
                f = self.fail[node]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[child] = self.goto[f].get(ch, 0) if self.goto[f].get(ch, 0) != child else 0
                self.out[child] = self.out[child] + self.out[self.fail[child]]

    def find(self, text: str) -> list[tuple[int, int, str, str]]:
        """返回 (start, end, word, tag) 列表。"""
        node = 0
        hits: list[tuple[int, int, str, str]] = []
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for word, tag in self.out[node]:
                hits.append((i - len(word) + 1, i + 1, word, tag))
        return hits


def _resolve_overlaps(hits: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    """最长匹配优先，去重叠。"""
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: list[tuple[int, int, str, str]] = []
    last_end = -1
    for h in hits:
        if h[0] >= last_end:
            kept.append(h)
            last_end = h[1]
    return kept


class EntityExtractOperator(Operator):
    """实体识别算子：从医疗文本中抽取疾病/症状/药物等实体。

    mode="naive"  逐词条 str.find（CPU 朴素基线）
    mode="ac"     Aho-Corasick 单遍匹配（CPU 优化，默认）
    """

    metadata = OperatorMetadata(
        name="entity_extract",
        description="医疗实体识别：识别疾病、症状、药物、检查、治疗方式、部位、人群等实体",
        op_type="mapper",
        inputs={"batch": "list[str] 文档列表", "mode": "naive|ac"},
        output_type="list[list[Entity]]",
    )

    def __init__(self, dictionary: dict[str, str] | None = None,
                 normalizer: EntityNormalizer | None = None, normalize: bool = True) -> None:
        self.term2type = dictionary or load_dictionary()
        # 别名注入：将同义词表中的所有别名也加入词典，提升召回（缩写/英文/繁简）
        self.normalizer = normalizer or (EntityNormalizer() if normalize else None)
        if self.normalizer is not None:
            self._augment_with_aliases()
        self._ac = AhoCorasick(self.term2type)

    def _augment_with_aliases(self) -> None:
        """把同义词表中的别名按其规范名的实体类型加入匹配词典。"""
        canon2type = {t: et for t, et in self.term2type.items()}
        for alias, canon in self.normalizer.canonical.items():
            etype = canon2type.get(canon)
            if etype and alias not in self.term2type:
                self.term2type[alias] = etype

    def process(self, batch: list[str], mode: str = "ac", **kwargs: Any) -> list[list[Entity]]:
        if mode == "naive":
            return [self._extract_naive(doc) for doc in batch]
        return [self._extract_ac(doc) for doc in batch]

    def _link(self, w: str, t: str, s: int, e: int) -> Entity:
        if self.normalizer is None:
            return Entity(text=w, etype=t, start=s, end=e)
        canon, method, _ = self.normalizer.link(w)
        return Entity(text=w, etype=t, start=s, end=e, norm=canon, link_method=method)

    def _extract_ac(self, text: str) -> list[Entity]:
        hits = _resolve_overlaps(self._ac.find(text))
        return [self._link(w, t, s, e) for s, e, w, t in hits]

    def _extract_naive(self, text: str) -> list[Entity]:
        hits: list[tuple[int, int, str, str]] = []
        for term, etype in self.term2type.items():
            start = 0
            while True:
                idx = text.find(term, start)
                if idx < 0:
                    break
                hits.append((idx, idx + len(term), term, etype))
                start = idx + 1
        return [self._link(w, t, s, e) for s, e, w, t in _resolve_overlaps(hits)]


def build_default() -> EntityExtractOperator:
    op = EntityExtractOperator()
    registry.register(op)
    return op
