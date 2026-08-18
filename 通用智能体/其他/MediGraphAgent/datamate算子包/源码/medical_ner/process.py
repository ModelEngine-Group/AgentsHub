# -*- coding: utf-8 -*-
"""DataMate operator: medical named-entity recognition (LLM-backed).

Self-contained (no external project deps) so it can be zipped and uploaded to the
DataMate operator marketplace. Calls an OpenAI-compatible API (SiliconFlow /
DashScope / local vLLM) using apiBase/apiKey/model provided via the operator's
UI settings. Reads sample['text'], writes the extracted entities (JSON) back into
sample['text'] so the exported file contains the structured result.

Note: the heavy neural GPLinker L1 used by the Nexent/MCP path lives in the host
MediGraph project. This DataMate package intentionally stays lightweight for the
operator marketplace and reports its own route as ``DataMate_LLM``.
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

ENTITY_TYPES = {
    "Disease": "疾病", "Symptom": "症状", "Drug": "药物", "Examination": "检查",
    "Procedure": "手术", "Body": "身体部位", "Department": "科室",
    "Tumor": "肿瘤", "Biomarker": "标志物", "Gene": "基因", "Morphology": "形态学特征",
}
OPERATOR_VERSION = "datamate-medical-ner-2.0.0"

_SYSTEM = "你是严谨的医学信息抽取引擎，只输出 JSON，不要任何解释。"
_PROMPT = """从下面的医学文本中抽取医学实体。实体类型只能取：{types}。
要求：只抽取明确出现的、有医学含义的实体；type 用英文 key；confidence 为 0~1。
不要把文档结构当实体（忽略 "Definition / general"、"Laboratory"、"Radiology images"、
"Gross description"、"Treatment" 等页面小节标题/导航）。
输出 JSON：{{"entities":[{{"name":"...","type":"...","confidence":0.0}}]}}

文本：
\"\"\"
{text}
\"\"\"
"""


class MedicalNER(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_base = kwargs.get("apiBase", "https://api.siliconflow.cn/v1").strip()
        self.api_key = kwargs.get("apiKey", "").strip()
        self.model = kwargs.get("model", "Qwen/Qwen3.6-35B-A3B").strip()
        self.temperature = float(kwargs.get("temperature", 0.2))
        self.max_chars = int(kwargs.get("maxChars", 4000))
        self.max_chunks = int(kwargs.get("maxChunks", 8))
        # Content-rich clinical records make the model emit a long entity list;
        # 120s was not enough for real 265-character outpatient records.
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
        text = str(sample.get(self.text_key, "") or "").strip()
        if not text:
            return sample

        chunks = sample.get("_medical_chunks")
        chunks = [str(chunk).strip() for chunk in chunks] if isinstance(chunks, list) else []
        parsed_input = self._extract_json(text)
        if not chunks and isinstance(parsed_input, dict) and isinstance(parsed_input.get("chunks"), list):
            chunks = [str(chunk).strip() for chunk in parsed_input["chunks"]]
        source_text = str(sample.get("_medical_source_text", "") or "").strip()
        if not source_text:
            source_text = "\n\n".join(chunks) if chunks else text
        if not chunks:
            chunks = [source_text]
        chunks = [chunk for chunk in chunks if chunk]
        if self.max_chunks:
            chunks = chunks[:self.max_chunks]

        # Keep the raw/cleaned document available to downstream MedicalRE.
        sample["_medical_source_text"] = source_text
        entities = []
        seen = set()
        for index, chunk in enumerate(chunks):
            prompt = _PROMPT.format(types="/".join(ENTITY_TYPES), text=chunk[:self.max_chars])
            try:
                data = self._extract_json(self._call_llm(prompt)) or {}
            except Exception as e:  # noqa: BLE001
                logger.error(f"MedicalNER LLM call failed on chunk {index + 1}/{len(chunks)}: {e}")
                raise
            for ent in data.get("entities", []) if isinstance(data, dict) else []:
                if not isinstance(ent, dict):
                    continue
                name = str(ent.get("name", "")).strip()
                etype = str(ent.get("type", "")).strip()
                if not name or etype not in ENTITY_TYPES:
                    continue
                key = (etype, name.lower())
                if key in seen:
                    continue
                seen.add(key)
                entities.append({
                    "name": name,
                    "type": etype,
                    "confidence": ent.get("confidence", 0.8),
                })
        logger.info(f"MedicalNER: {len(entities)} unique entities from {len(chunks)} chunk(s)")
        sample[self.text_key] = json.dumps(
            {
                "entities": entities,
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
