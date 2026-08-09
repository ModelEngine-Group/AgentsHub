"""三元组生成与校验算子。

对关系抽取的候选三元组做：
1. Schema 校验（关系类型与头尾实体类型必须匹配）
2. 自环过滤
3. 去重合并（同一三元组多处出现时保留最高置信度并累计支持数）
4. 置信度阈值过滤
"""

from __future__ import annotations

from typing import Any

from .base import Operator, OperatorMetadata
from ..kg.schema import RELATION_TYPES, Triple, ValidationReport


class TripleBuildOperator(Operator):
    """三元组生成与校验算子。"""

    metadata = OperatorMetadata(
        name="triple_build",
        description="三元组生成与校验：Schema 校验、去重合并、置信度过滤，输出可入图三元组",
        op_type="validator",
        inputs={"batch": "list[Triple] 候选三元组", "min_confidence": "float 置信度阈值"},
        output_type="list[Triple]",
    )

    def __init__(self) -> None:
        self.last_report = ValidationReport()

    def process(self, batch: list[Triple], min_confidence: float = 0.55,
                keep_negated: bool = False, **kwargs: Any) -> list[Triple]:
        report = ValidationReport(total=len(batch))
        merged: dict[tuple[str, str, str], Triple] = {}
        support: dict[tuple[str, str, str], int] = {}
        for t in batch:
            spec = RELATION_TYPES.get(t.relation)
            if spec is None or (t.head_type, t.tail_type) != (spec[0], spec[1]):
                report.dropped_schema += 1
                report.issues.append(f"schema不匹配: {t.as_tuple()} ({t.head_type}->{t.tail_type})")
                continue
            if t.head == t.tail:
                report.dropped_self_loop += 1
                continue
            # 否定检测：默认剔除否定断言的三元组（避免"无胸痛"被入图为阳性）
            if not keep_negated and t.assertion == "negated":
                report.dropped_negated += 1
                continue
            if t.confidence < min_confidence:
                continue
            key = t.as_tuple()
            if key in merged:
                report.dropped_duplicate += 1
                support[key] += 1
                if t.confidence > merged[key].confidence:
                    merged[key] = t
            else:
                merged[key] = t
                support[key] = 1
        out = [
            Triple(
                head=t.head, head_type=t.head_type, relation=t.relation,
                tail=t.tail, tail_type=t.tail_type, source=t.source,
                confidence=min(1.0, round(t.confidence + 0.05 * (support[k] - 1), 3)),
                evidence=t.evidence, assertion=t.assertion,
            )
            for k, t in merged.items()
        ]
        report.valid = len(out)
        self.last_report = report
        return out
