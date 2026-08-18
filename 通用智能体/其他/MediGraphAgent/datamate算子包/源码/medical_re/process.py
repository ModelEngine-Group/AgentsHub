# -*- coding: utf-8 -*-
"""DataMate operator: medical relation extraction (LLM-backed, self-contained).

Reads sample['text']. The text may be raw medical text, OR the JSON output of the
MedicalNER operator ({"entities":[...]}) plus original text -- this operator
extracts ontology-constrained relation triples and writes them back as JSON
{"triples":[...]}.

The host Nexent/MCP path can use the shipped neural GPLinker checkpoint. This
DataMate marketplace operator remains small and self-contained, so it exposes a
clear ``DataMate_LLM`` route in its output.
"""
import json
import re
from typing import Any, Dict

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

from datamate.core.base_op import Mapper

RELATION_TYPES = {
    "has_symptom": "有症状", "recommend_drug": "推荐药物", "need_examination": "需做检查",
    "complication": "并发", "contraindication": "禁忌", "treated_in_department": "就诊于",
    "positive_marker": "阳性标志物", "negative_marker": "阴性标志物", "associated_gene": "关联基因",
    "has_morphology": "形态学表现", "located_in": "位于", "subtype_of": "亚型",
    "treated_by_procedure": "手术/操作治疗", "adverse_reaction": "药物不良反应",
}
OPERATOR_VERSION = "datamate-medical-re-2.0.0"

_SYSTEM = "你是严谨的医学关系抽取引擎，只输出 JSON，不要任何解释。"
_LESION = {"Disease", "Tumor"}
RELATION_CONSTRAINTS = {
    "has_symptom": (_LESION, {"Symptom"}),
    "recommend_drug": (_LESION, {"Drug"}),
    "need_examination": (_LESION, {"Examination"}),
    "complication": (_LESION, _LESION),
    "contraindication": ({"Drug"}, _LESION | {"Symptom"}),
    "treated_in_department": (_LESION, {"Department"}),
    "positive_marker": (_LESION, {"Biomarker"}),
    "negative_marker": (_LESION, {"Biomarker"}),
    "associated_gene": (_LESION, {"Gene"}),
    "has_morphology": (_LESION, {"Morphology"}),
    "located_in": (_LESION | {"Symptom"}, {"Body"}),
    "subtype_of": (_LESION, _LESION),
    "treated_by_procedure": (_LESION, {"Procedure"}),
    "adverse_reaction": ({"Drug"}, {"Symptom", "Disease"}),
}

_FEWSHOT = """示例1：
文本："2型糖尿病患者常有多饮，治疗首选二甲双胍，建议检查糖化血红蛋白。"
输出：{"triples":[
{"head":"2型糖尿病","head_type":"Disease","relation":"has_symptom","tail":"多饮","tail_type":"Symptom","confidence":0.95},
{"head":"2型糖尿病","head_type":"Disease","relation":"recommend_drug","tail":"二甲双胍","tail_type":"Drug","confidence":0.94},
{"head":"2型糖尿病","head_type":"Disease","relation":"need_examination","tail":"糖化血红蛋白","tail_type":"Examination","confidence":0.9}]}

示例2：
文本："嗜铬细胞瘤多位于肾上腺髓质，CgA 常呈阳性，与 RET 基因突变相关。"
输出：{"triples":[
{"head":"嗜铬细胞瘤","head_type":"Tumor","relation":"located_in","tail":"肾上腺髓质","tail_type":"Body","confidence":0.9},
{"head":"嗜铬细胞瘤","head_type":"Tumor","relation":"positive_marker","tail":"CgA","tail_type":"Biomarker","confidence":0.92},
{"head":"嗜铬细胞瘤","head_type":"Tumor","relation":"associated_gene","tail":"RET","tail_type":"Gene","confidence":0.9}]}
"""


def _ontology_block() -> str:
    lines = []
    for relation, display in RELATION_TYPES.items():
        head_types, tail_types = RELATION_CONSTRAINTS[relation]
        lines.append(
            f"- {relation} ({display}): {'/'.join(sorted(head_types))} -> "
            f"{'/'.join(sorted(tail_types))}"
        )
    return "\n".join(lines)


_PROMPT = """从医学文本中抽取实体间关系三元组。

允许的 relation 与方向约束：
{ontology}

{fewshot}
要求：
1. 只抽取文本明确支持的关系。
2. head/tail 优先使用候选实体中的原文名称和类型。
3. 严格遵守上面的关系方向与两端类型约束，不要颠倒 head/tail。
4. 不要仅凭候选实体名称臆造关系。
5. 给出 head/head_type/relation/tail/tail_type/confidence(0~1)。
输出 JSON：{{"triples":[{{"head":"","head_type":"","relation":"","tail":"","tail_type":"","confidence":0.0}}]}}

医学原文：
\"\"\"
{text}
\"\"\"

候选实体 JSON：
{entities}
"""


class MedicalRE(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_base = kwargs.get("apiBase", "https://api.siliconflow.cn/v1").strip()
        self.api_key = kwargs.get("apiKey", "").strip()
        self.model = kwargs.get("model", "Qwen/Qwen3.6-35B-A3B").strip()
        self.temperature = float(kwargs.get("temperature", 0.2))
        self.max_chars = int(kwargs.get("maxChars", 4000))
        self.max_chunks = int(kwargs.get("maxChunks", 8))
        # Relation extraction emits one JSON object per triple, so a content-rich
        # clinical record can keep the model generating well past two minutes.
        # 120s timed out on real 265-character outpatient records.
        self.timeout = float(kwargs.get("timeout", 240))
        if self.max_chars < 100:
            raise ValueError("maxChars must be at least 100")
        if self.max_chunks < 0:
            raise ValueError("maxChunks must be >= 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def _call_llm(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key or "EMPTY", base_url=self.api_base, timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extract_json(text: str) -> Any:
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self.read_file_first(sample)
        text = (sample.get(self.text_key, "") or "").strip()
        if not text:
            return sample

        source_text = str(sample.get("_medical_source_text", "") or "").strip()
        entity_payload: Any = {}
        parsed_input = self._extract_json(text)
        if isinstance(parsed_input, dict) and isinstance(parsed_input.get("entities"), list):
            entity_payload = parsed_input
        if not source_text:
            # MedicalRE also supports standalone use, where its input is raw text.
            source_text = text

        chunks = sample.get("_medical_chunks")
        chunks = [str(chunk).strip() for chunk in chunks] if isinstance(chunks, list) else []
        if not chunks:
            chunks = [source_text]
        chunks = [chunk for chunk in chunks if chunk]
        if self.max_chunks:
            chunks = chunks[:self.max_chunks]

        candidate_entities = entity_payload.get("entities", []) if isinstance(entity_payload, dict) else []
        triples = []
        seen = set()
        for index, chunk in enumerate(chunks):
            lowered = chunk.lower()
            chunk_entities = [
                entity for entity in candidate_entities
                if isinstance(entity, dict)
                and str(entity.get("name", "")).strip()
                and str(entity.get("name", "")).strip().lower() in lowered
            ]
            prompt = _PROMPT.format(
                ontology=_ontology_block(),
                fewshot=_FEWSHOT,
                text=chunk[:self.max_chars],
                entities=json.dumps({"entities": chunk_entities}, ensure_ascii=False),
            )
            try:
                data = self._extract_json(self._call_llm(prompt)) or {}
            except Exception as e:  # noqa: BLE001
                logger.error(f"MedicalRE LLM call failed on chunk {index + 1}/{len(chunks)}: {e}")
                raise
            for triple in data.get("triples", []) if isinstance(data, dict) else []:
                if not isinstance(triple, dict):
                    continue
                relation = str(triple.get("relation", "")).strip()
                head = str(triple.get("head", "")).strip()
                tail = str(triple.get("tail", "")).strip()
                if relation not in RELATION_TYPES or not head or not tail:
                    continue
                key = (head.lower(), relation, tail.lower())
                if key in seen:
                    continue
                seen.add(key)
                triples.append({
                    "head": head,
                    "head_type": str(triple.get("head_type", "")).strip(),
                    "relation": relation,
                    "tail": tail,
                    "tail_type": str(triple.get("tail_type", "")).strip(),
                    "confidence": triple.get("confidence", 0.7),
                })
        logger.info(f"MedicalRE: {len(triples)} unique triples from {len(chunks)} chunk(s)")
        sample[self.text_key] = json.dumps(
            {
                "triples": triples,
                "operator_version": OPERATOR_VERSION,
                "backend": "llm_schema",
                "routing": {
                    "level": "DataMate_LLM",
                    "llm_called": True,
                    "model": self.model,
                    "chunks": len(chunks),
                },
                "note": "Nexent/MCP path uses neural GPLinker when available; DataMate package is lightweight LLM-backed.",
            },
            ensure_ascii=False,
            indent=2,
        )
        return sample
