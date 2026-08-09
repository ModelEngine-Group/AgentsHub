"""关系抽取算子。

基于「句内实体对 + 触发词模式」的关系抽取：
对每个句子内的实体对，按 Schema 允许的 (头类型, 尾类型) 组合，
用触发词/句式模式判定关系类型并给出置信度。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .base import Operator, OperatorMetadata
from ..kg.schema import Entity, Triple

SENT_SPLIT = re.compile(r"[。！？；\n]")

# 否定/不确定触发词（否定检测：避免把"无胸痛""排除心梗"抽成阳性关系）
NEGATION_CUES = ["无", "未见", "未出现", "不伴", "否认", "排除", "非", "无明显", "不存在", "缺乏"]
UNCERTAIN_CUES = ["可能", "疑似", "怀疑", "考虑为", "不除外", "待排"]

# (关系, 头类型, 尾类型, 触发词列表, 基础置信度)
PATTERNS: list[tuple[str, str, str, list[str], float]] = [
    ("HAS_SYMPTOM", "Disease", "Symptom",
     ["表现为", "症状", "出现", "伴有", "可见", "主要临床表现", "常见", "typical", "感到"], 0.9),
    ("TREATED_BY_DRUG", "Disease", "Drug",
     ["服用", "使用", "给予", "口服", "药物", "常用药", "首选", "可用", "静脉滴注"], 0.9),
    ("TREATED_BY", "Disease", "Treatment",
     ["采用", "进行", "治疗方式", "行", "推荐", "可通过", "需要", "给予", "首选", "联合"], 0.85),
    ("DIAGNOSED_BY", "Disease", "Examination",
     ["检查", "确诊", "诊断", "检测", "筛查", "测定", "监测"], 0.85),
    ("COMPLICATION_OF", "Disease", "Disease",
     ["并发", "并发症", "继发", "引发", "导致", "诱发", "合并"], 0.8),
    ("AFFECTS", "Disease", "BodyPart",
     ["累及", "损害", "病变", "影响", "侵犯"], 0.8),
    ("HIGH_RISK_GROUP", "Disease", "Population",
     ["高发", "多见于", "好发于", "易患", "常见于"], 0.8),
    ("CONTRAINDICATED", "Drug", "Population",
     ["禁用", "慎用", "禁忌", "不宜"], 0.85),
]


class RelationExtractOperator(Operator):
    """关系抽取算子：句级实体对 + 触发词模式。"""

    metadata = OperatorMetadata(
        name="relation_extract",
        description="医疗关系抽取：从（文档, 实体列表）中抽取疾病-症状/药物/检查等关系候选",
        op_type="mapper",
        inputs={"batch": "list[tuple[str, list[Entity]]] (文档, 实体) 列表"},
        output_type="list[list[Triple]]",
    )

    def process(self, batch: list[tuple[str, list[Entity]]], **kwargs: Any) -> list[list[Triple]]:
        return [self._extract_doc(text, ents) for text, ents in batch]

    def _extract_doc(self, text: str, entities: list[Entity]) -> list[Triple]:
        """篇章级抽取（单主题临床文档假设）。

        临床笔记/科普文档通常围绕一个主病展开，症状/检查/用药/人群等均从属于该主病。
        因此以「篇章主题疾病」为疾病类关系的唯一头实体，避免句中出现的并发症/危险因素
        疾病被错误地当作头实体，与所有尾实体做笛卡尔积（精确率杀手）。
        """
        triples: list[Triple] = []
        offset = 0
        dis = [e for e in entities if e.etype == "Disease"]
        topic: Entity | None = None
        if dis:
            freq = Counter(e.canonical for e in dis)
            topic = max(dis, key=lambda e: (freq[e.canonical], -e.start))
        for sent in SENT_SPLIT.split(text):
            if sent.strip():
                start, end = offset, offset + len(sent)
                sent_ents = [e for e in entities if e.start >= start and e.end <= end]
                triples.extend(self._extract_sentence(sent, start, sent_ents, topic))
            offset += len(sent) + 1
        return triples

    @staticmethod
    def _assert_status(sent: str, tail: Entity, offset: int) -> str:
        """否定检测：判断尾实体在句中的断言状态。

        在尾实体前的一个窗口内查找否定/不确定线索词（ConText 算法简化版）：
        否定词出现在实体前 -> negated；不确定词 -> uncertain；否则 present。
        """
        pos = tail.start - offset
        if pos < 0:
            return "present"
        # 否定作用域限定在同一小句内：截断到实体前最近的子句分隔符
        left = sent[:pos]
        cut = max((left.rfind(c) for c in "，,、；;：:"), default=-1)
        window = left[cut + 1:]
        if any(cue in window for cue in NEGATION_CUES):
            return "negated"
        if any(cue in window for cue in UNCERTAIN_CUES):
            return "uncertain"
        return "present"

    def _extract_sentence(self, sent: str, offset: int, ents: list[Entity],
                          topic: Entity | None = None) -> list[Triple]:
        out: list[Triple] = []
        start, end = offset, offset + len(sent)
        topic_in_sent = topic is not None and start <= topic.start < end
        for rel, ht, tt, triggers, conf in PATTERNS:
            tails = [e for e in ents if e.etype == tt]
            if ht == "Disease":
                # 疾病类关系：以主题疾病为唯一头实体
                if topic is None:
                    continue
                heads = [topic]
                if rel == "COMPLICATION_OF":
                    # 并发症：尾实体为句中除主病外的其他疾病
                    tails = [e for e in ents if e.etype == "Disease"
                             and e.canonical != topic.canonical]
            else:
                heads = [e for e in ents if e.etype == ht]  # 如 药物-禁忌人群
            if not heads or not tails:
                continue
            trigger = next((t for t in triggers if t in sent), None)
            if trigger is None:
                continue
            head_off_sent = ht == "Disease" and not topic_in_sent
            for h in heads:
                for t in tails:
                    if h.canonical == t.canonical:
                        continue
                    if head_off_sent:
                        c = conf - 0.05  # 主题跨句：轻微降权，不做距离衰减
                    else:
                        dist = abs((t.start - offset) - (h.start - offset))
                        c = max(0.55, conf - 0.002 * dist)
                    assertion = self._assert_status(sent, t, offset)
                    if assertion != "present":
                        c = round(c * 0.5, 3)  # 否定/不确定关系降权
                    out.append(Triple(
                        head=h.canonical, head_type=ht, relation=rel,
                        tail=t.canonical, tail_type=tt,
                        confidence=round(c, 3), evidence=sent.strip(),
                        assertion=assertion,
                    ))
        return out
