"""Medical relation extraction with local joint fast path and LLM fallback.

Given a text chunk and a list of entities, extracts relation triples
(head, relation, tail) limited to the ontology's relation types. If no entities
are supplied it can still run by asking the model to extract both ends.
"""
from __future__ import annotations

from typing import Any

from config.settings import get_extraction_config
from medigraph.extraction.cascade import load_fast_extractor, load_neural_extractor, merge_triples
from medigraph.operators.base import BaseOperator, OperatorMeta
from medigraph.schema.relation_mapping import map_cmeie_triples_to_clinical
from medigraph.schema.ontology import is_valid_relation, ontology_prompt_block

_SYSTEM = "你是严谨的医学关系抽取引擎，只输出 JSON，不要任何解释。"

_FEWSHOT = """示例（学习关系方向与三元组粒度，不要照抄示例内容）：

示例文本1："2型糖尿病患者常有多饮、多尿，治疗首选二甲双胍，建议检查糖化血红蛋白，可并发糖尿病肾病。"
示例输出1：{{"triples": [{{"head":"2型糖尿病","head_type":"Disease","relation":"has_symptom","tail":"多饮","tail_type":"Symptom","confidence":0.95}}, {{"head":"2型糖尿病","head_type":"Disease","relation":"has_symptom","tail":"多尿","tail_type":"Symptom","confidence":0.95}}, {{"head":"2型糖尿病","head_type":"Disease","relation":"recommend_drug","tail":"二甲双胍","tail_type":"Drug","confidence":0.94}}, {{"head":"2型糖尿病","head_type":"Disease","relation":"need_examination","tail":"糖化血红蛋白","tail_type":"Examination","confidence":0.9}}, {{"head":"2型糖尿病","head_type":"Disease","relation":"complication","tail":"糖尿病肾病","tail_type":"Disease","confidence":0.88}}]}}

示例文本2："嗜铬细胞瘤多位于肾上腺髓质，CgA 常呈阳性，与 RET 基因突变相关，可表现为阵发性高血压。"
示例输出2：{{"triples": [{{"head":"嗜铬细胞瘤","head_type":"Tumor","relation":"located_in","tail":"肾上腺髓质","tail_type":"Body","confidence":0.9}}, {{"head":"嗜铬细胞瘤","head_type":"Tumor","relation":"positive_marker","tail":"CgA","tail_type":"Biomarker","confidence":0.92}}, {{"head":"嗜铬细胞瘤","head_type":"Tumor","relation":"associated_gene","tail":"RET","tail_type":"Gene","confidence":0.9}}, {{"head":"嗜铬细胞瘤","head_type":"Tumor","relation":"has_symptom","tail":"阵发性高血压","tail_type":"Symptom","confidence":0.85}}]}}
"""

_PROMPT = """根据给定的医学本体，从文本中抽取实体之间的关系三元组。

{ontology}

{fewshot}
已识别的实体（name#type）：
{entities}

要求：
1. relation 必须是上面关系类型的英文 key（如 has_symptom/recommend_drug/positive_marker 等）。
2. head/tail 必须使用已识别实体里**原文一致**的最小术语名；head_type/tail_type 必须符合该关系的两端类型约束。
3. 关系方向以疾病/肿瘤为 head（如 疾病-has_symptom-症状），不要颠倒。
4. 文本明确支持的关系尽量都抽出（提高召回），但不要臆造未提及的关系。
5. confidence 为 0~1。
6. 输出 JSON：{{"triples": [{{"head":"...","head_type":"...","relation":"...","tail":"...","tail_type":"...","confidence":0.0}}]}}

文本：
\"\"\"
{text}
\"\"\"
"""


class MedicalREOperator(BaseOperator):
    def __init__(self, llm: Any | None = None, backend: str | None = None):
        self.llm = llm
        self.config = get_extraction_config()
        self.backend = backend or self.config.backend
        self.meta = OperatorMeta(
            name="medical_re",
            version="2.0.0",
            description=(
                "联合式医疗关系抽取：auto 模式优先运行自训练 neural GPLinker，并把可安全映射的"
                " CMeIE 谓词转换到临床图谱本体；环境不可用或低置信时退回词典快路/LLM。"
                "输入 {text, entities?}, 输出 {triples,routing}。"
            ),
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "entities": {"type": "array"}},
                "required": ["text"],
            },
            output_schema={"type": "object", "properties": {"triples": {"type": "array"}}},
        )

    def _ensure_llm(self):
        if self.llm is None:
            from medigraph.llm.client import LLMClient
            self.llm = LLMClient()
        return self.llm

    def run(self, inputs: dict, **kwargs) -> dict:
        text = (inputs.get("text", "") or "").strip()
        if not text:
            return {"triples": [], "routing": {"level": "none", "reason": "empty_text"}}
        entities = inputs.get("entities", []) or []

        neural_triples: list[dict] = []
        fast_triples: list[dict] = []
        uncertainty = {"uncertain": True, "reason": "local_path_disabled", "mean_confidence": 0.0}

        if self.backend in {"auto", "neural"}:
            neural = load_neural_extractor()
            if neural is not None:
                result = neural.extract(
                    text[:4000],
                    threshold=self.config.neural_threshold,
                    rel_threshold=self.config.neural_rel_threshold,
                )
                neural_triples = map_cmeie_triples_to_clinical(
                    result.get("triples", []),
                    entities,
                    result.get("entities", []),
                )
                uncertainty = self._uncertainty(neural_triples)
                if self.backend == "neural" or (
                    neural_triples and not uncertainty["uncertain"]
                ):
                    return {
                        "triples": neural_triples,
                        "routing": {
                            "level": "L1_neural",
                            "llm_called": False,
                            "raw_cmeie_triples": len(result.get("triples", [])),
                            "mapped_triples": len(neural_triples),
                            "model_version": getattr(neural, "version", "neural_gplinker"),
                            **uncertainty,
                        },
                    }
            elif self.backend == "neural":
                return {
                    "triples": [],
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
                fast_triples = fast.extract_relations(text[:4000], entities or None)
                uncertainty = fast.uncertainty(fast_triples, self.config.route_threshold)
                if self.backend == "fast" or (
                    not uncertainty["uncertain"] and fast_triples
                ):
                    return {
                        "triples": fast_triples,
                        "routing": {"level": "L1_fast", "llm_called": False, **uncertainty},
                    }

        if self.backend == "fast" or not self.config.llm_fallback:
            return {
                "triples": fast_triples,
                "routing": {"level": "L1_fast", "llm_called": False, **uncertainty},
            }

        llm_triples = self._run_llm(text, entities)
        triples = merge_triples(merge_triples(neural_triples, fast_triples), llm_triples)
        return {
            "triples": triples,
            "routing": {
                "level": (
                    "L1_neural+L1_fast+L2_llm" if neural_triples and fast_triples
                    else "L1_neural+L2_llm" if neural_triples
                    else "L1_fast+L2_llm" if fast_triples
                    else "L2_llm"
                ),
                "llm_called": True,
                "neural_predictions": len(neural_triples),
                "fast_predictions": len(fast_triples),
                "llm_predictions": len(llm_triples),
                **uncertainty,
            },
        }

    def _run_llm(self, text: str, entities: list[dict]) -> list[dict]:
        ent_lines = "\n".join(
            f"  - {e.get('name')}#{e.get('type')}" for e in entities if isinstance(e, dict) and e.get("name")
        ) or "  (无，请你同时识别实体两端)"

        llm = self._ensure_llm()
        prompt = _PROMPT.format(ontology=ontology_prompt_block(), fewshot=_FEWSHOT, entities=ent_lines, text=text[:4000])
        data = llm.chat_json(prompt, system=_SYSTEM, default={"triples": []})

        raw = data.get("triples", []) if isinstance(data, dict) else []
        triples = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            head, tail = str(t.get("head", "")).strip(), str(t.get("tail", "")).strip()
            rel = str(t.get("relation", "")).strip()
            if not head or not tail or not is_valid_relation(rel):
                continue
            try:
                conf = float(t.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            triples.append(
                {
                    "head": head,
                    "head_type": str(t.get("head_type", "")).strip(),
                    "relation": rel,
                    "tail": tail,
                    "tail_type": str(t.get("tail_type", "")).strip(),
                    "confidence": round(max(0.0, min(1.0, conf)), 3),
                    "extractor": "llm_schema",
                }
            )
        return triples

    def _uncertainty(self, items: list[dict]) -> dict:
        if not items:
            return {"uncertain": True, "reason": "no_prediction", "mean_confidence": 0.0}
        mean_conf = sum(float(item.get("confidence", 0.0) or 0.0) for item in items) / len(items)
        return {
            "uncertain": mean_conf < self.config.route_threshold,
            "reason": "below_route_threshold" if mean_conf < self.config.route_threshold else "confident",
            "mean_confidence": round(mean_conf, 4),
        }
