"""Medical-text PII redaction with deterministic audit counts."""
from __future__ import annotations

import re

from medigraph.operators.base import BaseOperator, OperatorMeta

_PATTERNS = {
    "id_card": re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)"),
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    "patient_id": re.compile(r"(?i)(?P<label>患者编号|病历号|住院号|门诊号|patient\s*id)\s*[:：]?\s*[A-Za-z0-9-]{4,}"),
}


class PIIRedactOperator(BaseOperator):
    def __init__(self):
        self.meta = OperatorMeta(
            name="pii_redact",
            version="1.0.0",
            description=(
                "脱敏医疗文本中的手机号、身份证、邮箱、银行卡号、病历号，返回逐类计数审计。"
                "输入 {text}, 输出 {text,redaction_report}。"
            ),
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "redaction_report": {"type": "object"}},
                "required": ["text", "redaction_report"],
            },
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        text = str(inputs.get("text", "") or "")
        counts = {}
        for category, pattern in _PATTERNS.items():
            counter = 0

            def replacement(match: re.Match) -> str:
                nonlocal counter
                counter += 1
                if category == "patient_id" and match.groupdict().get("label"):
                    return f"{match.group('label')}:[{category.upper()}_{counter}]"
                return f"[{category.upper()}_{counter}]"

            text = pattern.sub(replacement, text)
            counts[category] = counter
        return {
            "text": text,
            "redaction_report": {
                "total": sum(counts.values()),
                "by_type": counts,
                "policy": "regex_v1",
                "review_required": True,
            },
        }
