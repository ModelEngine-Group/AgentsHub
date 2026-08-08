"""Medical NER with a confidence-routed local fast path and LLM fallback.

Extracts typed medical entities (disease/symptom/drug/exam/biomarker/gene/...)
from a text chunk using constrained JSON decoding. The LLM is reused across
operators for shared latency stats.
"""
from __future__ import annotations

from typing import Any

from config.settings import get_extraction_config
from medigraph.extraction.cascade import (
    load_entity_linker,
    load_fast_extractor,
    load_neural_extractor,
    merge_entities,
)
from medigraph.operators.base import BaseOperator, OperatorMeta
from medigraph.schema.ontology import is_valid_entity_type, ontology_prompt_block
from medigraph.schema.normalize import canonical_key, canonical_name, is_structural_noise

_SYSTEM = "你是严谨的医学信息抽取引擎，只输出 JSON，不要任何解释。"

_FEWSHOT = """示例（学习抽取粒度与类型，不要照抄示例内容）：

示例文本1："2型糖尿病患者常有多饮、多尿和体重减轻，治疗首选二甲双胍，建议检查糖化血红蛋白，可并发糖尿病肾病。"
示例输出1：{{"entities": [{{"name": "2型糖尿病", "type": "Disease", "confidence": 0.97}}, {{"name": "多饮", "type": "Symptom", "confidence": 0.95}}, {{"name": "多尿", "type": "Symptom", "confidence": 0.95}}, {{"name": "体重减轻", "type": "Symptom", "confidence": 0.92}}, {{"name": "二甲双胍", "type": "Drug", "confidence": 0.96}}, {{"name": "糖化血红蛋白", "type": "Examination", "confidence": 0.93}}, {{"name": "糖尿病肾病", "type": "Disease", "confidence": 0.9}}]}}

示例文本2："嗜铬细胞瘤多位于肾上腺髓质，CgA 常呈阳性，与 RET 基因突变相关，可表现为阵发性高血压。"
示例输出2：{{"entities": [{{"name": "嗜铬细胞瘤", "type": "Tumor", "confidence": 0.96}}, {{"name": "肾上腺髓质", "type": "Body", "confidence": 0.9}}, {{"name": "CgA", "type": "Biomarker", "confidence": 0.92}}, {{"name": "RET", "type": "Gene", "confidence": 0.9}}, {{"name": "阵发性高血压", "type": "Symptom", "confidence": 0.88}}]}}
"""

_PROMPT = """从下面的医学文本中抽取医学实体。严格遵循给定本体的实体类型。

{ontology}

{fewshot}
要求：
1. 只抽取文本中明确出现的、有医学含义的实体，不要臆造。
2. type 字段必须是上面实体类型的英文 key（如 Disease/Symptom/Drug/Biomarker/Gene/Tumor 等）。
3. name 取**最小可识别的医学术语**（如「多尿」而非「出现多尿的症状」、「高血压」而非「阵发性血压升高表现」），去掉编号与修饰性整句，保持原文大小写；同一术语只输出一次。
4. **不要把文档结构当作实体**：忽略章节标题/导航/目录，例如
   "Definition / general"、"Laboratory"、"Radiology images"、"Gross description"、
   "Microscopic description"、"Differential diagnosis"、"Treatment"、"Case reports"
   这类页面小节标题——它们不是医学实体。
5. confidence 为 0~1 的抽取置信度。
6. 输出 JSON 对象：{{"entities": [{{"name": "...", "type": "...", "confidence": 0.0}}]}}

文本：
\"\"\"
{text}
\"\"\"
"""


class MedicalNEROperator(BaseOperator):
    def __init__(self, llm: Any | None = None, backend: str | None = None):
        self.llm = llm
        self.config = get_extraction_config()
        self.backend = backend or self.config.backend
        self.meta = OperatorMeta(
            name="medical_ner",
            version="2.0.0",
            description=(
                "置信度路由的医疗实体抽取：auto 模式优先运行自训练 neural GPLinker，"
                "环境不可用或低置信时退回本地词典快路/LLM，并链接到规范实体 ID。"
                "输入 {text}, 输出 {entities, routing}。"
            ),
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            output_schema={"type": "object", "properties": {"entities": {"type": "array"}}},
        )

    def _ensure_llm(self):
        if self.llm is None:
            from medigraph.llm.client import LLMClient
            self.llm = LLMClient()
        return self.llm

    def run(self, inputs: dict, **kwargs) -> dict:
        text = (inputs.get("text", "") or "").strip()
        if not text:
            return {"entities": [], "routing": {"level": "none", "reason": "empty_text"}}

        neural_entities: list[dict] = []
        fast_entities: list[dict] = []
        uncertainty = {"uncertain": True, "reason": "local_path_disabled", "mean_confidence": 0.0}

        if self.backend in {"auto", "neural"}:
            neural = load_neural_extractor()
            if neural is not None:
                result = neural.extract(
                    text[:4000],
                    threshold=self.config.neural_threshold,
                    rel_threshold=self.config.neural_rel_threshold,
                )
                neural_entities = result.get("entities", [])
                uncertainty = self._uncertainty(neural_entities)
                if self.backend == "neural" or (
                    neural_entities and not uncertainty["uncertain"]
                ):
                    return {
                        "entities": self._link(neural_entities),
                        "routing": {
                            "level": "L1_neural",
                            "llm_called": False,
                            "model_version": getattr(neural, "version", "neural_gplinker"),
                            **uncertainty,
                        },
                    }
            elif self.backend == "neural":
                return {
                    "entities": [],
                    "routing": {
                        "level": "L1_neural_unavailable",
                        "llm_called": False,
                        "uncertain": True,
                        "reason": "neural_model_or_dependencies_unavailable",
                        "mean_confidence": 0.0,
                    },
                }

        fast = None
        if self.backend != "llm":
            fast = load_fast_extractor()
            if fast is not None:
                fast_entities = fast.extract_entities(text[:4000], overlap_policy="maximal")
                uncertainty = fast.uncertainty(fast_entities, self.config.route_threshold)
                if self.backend == "fast" or (
                    not uncertainty["uncertain"] and fast_entities
                ):
                    entities = self._link(fast_entities)
                    return {
                        "entities": entities,
                        "routing": {
                            "level": "L1_fast",
                            "llm_called": False,
                            **uncertainty,
                        },
                    }

        if self.backend == "fast" or not self.config.llm_fallback:
            return {
                "entities": self._link(fast_entities),
                "routing": {"level": "L1_fast", "llm_called": False, **uncertainty},
            }

        llm_entities = self._run_llm(text)
        entities = merge_entities(merge_entities(neural_entities, fast_entities), llm_entities)
        return {
            "entities": self._link(entities),
            "routing": {
                "level": (
                    "L1_neural+L1_fast+L2_llm" if neural_entities and fast_entities
                    else "L1_neural+L2_llm" if neural_entities
                    else "L1_fast+L2_llm" if fast_entities
                    else "L2_llm"
                ),
                "llm_called": True,
                "neural_predictions": len(neural_entities),
                "fast_predictions": len(fast_entities),
                "llm_predictions": len(llm_entities),
                **uncertainty,
            },
        }

    def _run_llm(self, text: str) -> list[dict]:
        llm = self._ensure_llm()
        prompt = _PROMPT.format(ontology=ontology_prompt_block(), fewshot=_FEWSHOT, text=text[:4000])
        data = llm.chat_json(prompt, system=_SYSTEM, default={"entities": []})

        raw_entities = data.get("entities", []) if isinstance(data, dict) else []
        seen: set[str] = set()
        entities = []
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            name = canonical_name(str(e.get("name", "")))
            etype = str(e.get("type", "")).strip()
            if not name or not is_valid_entity_type(etype):
                continue
            if is_structural_noise(name):  # drop section-title / navigation noise
                continue
            key = f"{etype}::{canonical_key(name)}"
            if key in seen:
                continue
            seen.add(key)
            try:
                conf = float(e.get("confidence", 0.8))
            except (TypeError, ValueError):
                conf = 0.8
            entities.append(
                {
                    "name": name,
                    "type": etype,
                    "confidence": round(max(0.0, min(1.0, conf)), 3),
                    "extractor": "llm_schema",
                }
            )
        return entities

    @staticmethod
    def _link(entities: list[dict]) -> list[dict]:
        linker = load_entity_linker()
        return linker.link_entities(entities) if linker is not None else entities

    def _uncertainty(self, items: list[dict]) -> dict:
        if not items:
            return {"uncertain": True, "reason": "no_prediction", "mean_confidence": 0.0}
        mean_conf = sum(float(item.get("confidence", 0.0) or 0.0) for item in items) / len(items)
        return {
            "uncertain": mean_conf < self.config.route_threshold,
            "reason": "below_route_threshold" if mean_conf < self.config.route_threshold else "confident",
            "mean_confidence": round(mean_conf, 4),
        }
