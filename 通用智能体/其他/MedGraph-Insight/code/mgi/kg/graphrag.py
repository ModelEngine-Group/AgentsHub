"""GraphRAG 语义检索问答。

在规则图查询之上叠加检索增强能力：
1. 字符 n-gram TF-IDF 语义检索（numpy 实现，无需 sklearn），
   把问题映射到「实体中心文档」（实体名+关系尾+证据句），解决问题中
   未出现精确实体串（同义词/英文名/口语化）时的实体链接问题；
2. 实体归一化（别名/英文名/缩写→规范名）增强召回；
3. 子图证据检索：回答附带来源证据句，可解释、可回溯；
4. 多跳推理：支持「A 的并发症有哪些症状/用什么药」等链式问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .graph import MedicalKG
from .normalize import EntityNormalizer
from .schema import RELATION_TYPES

# 意图关键词 -> 关系
INTENT_RULES: list[tuple[list[str], str]] = [
    (["症状", "表现", "临床表现", "不适"], "HAS_SYMPTOM"),
    (["吃什么药", "用什么药", "药物", "用药", "服用"], "TREATED_BY_DRUG"),
    (["怎么治", "治疗", "疗法", "干预"], "TREATED_BY"),
    (["检查", "确诊", "诊断", "化验", "筛查"], "DIAGNOSED_BY"),
    (["并发症", "并发", "合并症"], "COMPLICATION_OF"),
    (["哪些人", "人群", "高发", "好发", "易患"], "HIGH_RISK_GROUP"),
    (["部位", "累及", "影响", "器官"], "AFFECTS"),
]


def _char_ngrams(text: str, n: tuple[int, int] = (1, 2)) -> list[str]:
    """字符 n-gram 分词（对中文稳健，免分词器）。"""
    s = re.sub(r"\s+", "", text)
    grams: list[str] = []
    for k in range(n[0], n[1] + 1):
        grams.extend(s[i:i + k] for i in range(len(s) - k + 1))
    return grams


class TfidfIndex:
    """字符 n-gram TF-IDF 检索（numpy 稠密实现）。"""

    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        tokd = [_char_ngrams(d) for d in docs]
        vocab: dict[str, int] = {}
        for toks in tokd:
            for t in toks:
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab
        n_docs, n_vocab = len(docs), max(len(vocab), 1)
        tf = np.zeros((n_docs, n_vocab), dtype=np.float64)
        for i, toks in enumerate(tokd):
            for t in toks:
                tf[i, vocab[t]] += 1.0
        df = (tf > 0).sum(axis=0)
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf = idf
        mat = tf * idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.matrix = mat / np.where(norms == 0, 1.0, norms)

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(max(len(self.vocab), 1), dtype=np.float64)
        for t in _char_ngrams(text):
            j = self.vocab.get(t)
            if j is not None:
                vec[j] += 1.0
        vec *= self.idf
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm else vec

    def query(self, text: str, k: int = 5) -> list[tuple[int, float]]:
        sims = self.matrix @ self._vectorize(text)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]


@dataclass
class RAGAnswer:
    question: str
    answer: str
    entity: str | None = None
    relation: str | None = None
    hops: int = 1
    evidence: list[dict] = field(default_factory=list)
    retrieval: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer,
            "entity": self.entity, "relation": self.relation, "hops": self.hops,
            "evidence": self.evidence, "retrieval": self.retrieval,
        }


class GraphRAG:
    """图检索增强问答：语义检索 + 子图证据 + 多跳推理。"""

    def __init__(self, kg: MedicalKG, normalizer: EntityNormalizer | None = None) -> None:
        self.kg = kg
        self.nz = normalizer or EntityNormalizer()
        self.heads = [n for n in kg.entities() if kg.neighbors(n)]
        self.index = TfidfIndex([self._doc_text(h) for h in self.heads]) if self.heads else None

    def _doc_text(self, head: str) -> str:
        parts = [head]
        for rel, tail, _ in self.kg.neighbors(head):
            parts.append(RELATION_TYPES.get(rel, ("", "", rel))[2])
            parts.append(tail)
        return " ".join(parts)

    # ---- 实体链接：精确串 > 归一化别名 > 语义检索 --------------------
    def link_entity(self, question: str) -> tuple[str | None, float, str]:
        subs = [h for h in self.heads if h in question]
        if subs:
            return max(subs, key=len), 1.0, "substring"
        # 别名/英文名/缩写归一化：把问题里的规范名候选找出来
        for canon in self.heads:
            for alias, mapped in self.nz.canonical.items():
                if mapped == canon and len(alias) >= 2 and alias in question:
                    return canon, 0.9, "alias"
        if self.index is not None:
            hits = self.index.query(question, k=1)
            if hits:
                i, score = hits[0]
                return self.heads[i], round(score, 3), "semantic"
        return None, 0.0, "none"

    @staticmethod
    def detect_intent(question: str) -> str | None:
        # 并发症类问题优先（避免被"症状"等更靠前的规则截胡）
        if any(k in question for k in ["并发症", "并发", "合并症"]):
            return "COMPLICATION_OF"
        for kws, rel in INTENT_RULES:
            if any(k in question for k in kws):
                return rel
        return None

    def _edge_evidence(self, head: str, relation: str) -> list[dict]:
        out = []
        hits = self.kg.neighbors(head, relation)
        if relation == "COMPLICATION_OF":
            hits += self.kg.reverse_neighbors(head, relation)
        for rel, tail, conf in hits:
            ev = ""
            data = self.kg.g.get_edge_data(head, tail)
            if data and rel in data:
                ev = data[rel].get("evidence", "")
            out.append({"relation": rel, "tail": tail, "confidence": conf, "evidence": ev})
        return out

    def answer(self, question: str, multi_hop: bool = True) -> dict:
        entity, score, method = self.link_entity(question)
        retr = [{"entity": entity, "score": score, "method": method}] if entity else []
        if entity is None:
            return RAGAnswer(question, "未能在图谱中检索到相关实体。", retrieval=retr).as_dict()
        relation = self.detect_intent(question)
        if relation is None:
            hits = self.kg.neighbors(entity)[:12]
            summary = "；".join(f"[{RELATION_TYPES[r][2]}] {t}" for r, t, _ in hits)
            return RAGAnswer(question, f"{entity} 的相关知识：{summary}", entity=entity,
                             retrieval=retr,
                             evidence=[{"relation": r, "tail": t, "confidence": c}
                                       for r, t, c in hits]).as_dict()

        # 多跳：并发症链式问题 —— 「A 的并发症有哪些症状/用药」
        second = self._second_intent(question, relation)
        if multi_hop and relation == "COMPLICATION_OF" and second is not None:
            return self._multi_hop(question, entity, second, retr)

        # 治疗类问题：合并药物治疗与非药物治疗（口语化"怎么治"不区分）
        rels = [relation]
        if relation in ("TREATED_BY", "TREATED_BY_DRUG") and (
                "治疗" in question or "怎么治" in question or "疗法" in question):
            rels = ["TREATED_BY_DRUG", "TREATED_BY"]
        ev: list[dict] = []
        for r in rels:
            ev.extend(self._edge_evidence(entity, r))
        if not ev:
            return RAGAnswer(question, f"图谱中暂无 {entity} 的相关记录。", entity=entity,
                             relation=relation, retrieval=retr).as_dict()
        tails = sorted({e["tail"] for e in ev})
        zh = "/".join(RELATION_TYPES[r][2] for r in rels)
        return RAGAnswer(question, f"{entity} 的{zh}包括：{'、'.join(tails)}。",
                         entity=entity, relation=relation, evidence=ev,
                         retrieval=retr).as_dict()

    def _second_intent(self, question: str, first: str) -> str | None:
        rel = self.detect_intent(question.replace("并发症", "").replace("并发", ""))
        return rel if rel and rel != first else None

    def _multi_hop(self, question: str, entity: str, second: str, retr: list) -> dict:
        comps = [t for _, t, _ in self.kg.neighbors(entity, "COMPLICATION_OF")]
        comps += [h for _, h, _ in self.kg.reverse_neighbors(entity, "COMPLICATION_OF")]
        chain, evidence = [], []
        for c in sorted(set(comps)):
            tails = sorted({t for _, t, _ in self.kg.neighbors(c, second)})
            if tails:
                chain.append(f"{c}（{'、'.join(tails)}）")
                evidence.append({"hop1": c, "relation": second, "hop2": tails})
        zh = RELATION_TYPES[second][2]
        if not chain:
            ans = f"图谱中 {entity} 的并发症暂无{zh}记录。"
        else:
            ans = f"{entity} 的并发症及其{zh}：" + "；".join(chain) + "。"
        return RAGAnswer(question, ans, entity=entity, relation="COMPLICATION_OF+" + second,
                         hops=2, evidence=evidence, retrieval=retr).as_dict()
