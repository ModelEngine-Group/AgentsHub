# -*- coding: utf-8 -*-
"""DataMate 医学关系抽取算子。

该算子读取医学文本和已识别实体，调用项目 core 层生成实体之间的
疾病-症状、疾病-药物、疾病-检查等关系，供后续三元组生成算子使用。
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from datamate.core.base_op import Mapper


def _ensure_core_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        Path(os.environ["CCF_PROJECT_ROOT"]) if os.environ.get("CCF_PROJECT_ROOT") else None,
        here.parents[2] if len(here.parents) > 2 else None,
        Path("/opt/runtime/ccf_medical_ai"),
    ]
    for candidate in candidates:
        if candidate and (candidate / "core").exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
    loaded_core = sys.modules.get("core")
    loaded_file = str(getattr(loaded_core, "__file__", "") or "")
    if loaded_core and "ccf_medical_ai" not in loaded_file and "ccf-medical-ai" not in loaded_file:
        sys.modules.pop("core", None)


_ensure_core_path()

from core import LLMClient  # noqa: E402
from core.medical_extraction_service import extract_medical_knowledge, normalize_backend  # noqa: E402


def _secret_value(kwargs: dict, key_name: str = "apiKey") -> str:
    value = kwargs.get(key_name) or os.environ.get("CCF_LLM_API_KEY", "")
    if value:
        return value
    key_file = os.environ.get("CCF_LLM_API_KEY_FILE", "/run/secrets/ccf_llm_api_key")
    try:
        return Path(key_file).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _llm_client(kwargs: dict) -> LLMClient:
    base_url = kwargs.get("ollamaUrl") or os.environ.get(
        "CCF_LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions"
    )
    model = kwargs.get("modelName") or os.environ.get("CCF_LLM_MODEL", "deepseek-chat")
    allow_ollama_default = os.environ.get("CCF_ALLOW_OLLAMA_DEFAULT", "").lower() in {"1", "true", "yes"}
    if "11434" in str(base_url) and not allow_ollama_default:
        base_url = os.environ.get("CCF_LLM_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
    if str(model).startswith("qwen") and not allow_ollama_default:
        model = os.environ.get("CCF_LLM_MODEL", "deepseek-chat")
    return LLMClient(
        base_url=str(base_url),
        model=str(model),
        timeout=int(kwargs.get("timeoutSeconds", os.environ.get("CCF_LLM_TIMEOUT", 120))),
        api_key=_secret_value(kwargs),
    )


class MedicalRelationExtractor(Mapper):
    """从医疗文本抽取关系，默认使用离线词典/规则 backend。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = normalize_backend(kwargs.get("backend") or os.environ.get("CCF_TASK2_BACKEND", "offline"))
        self.kg_db_path = str(
            kwargs.get("kgDbPath")
            or os.environ.get("CCF_KG_DB_PATH")
            or "/opt/runtime/ccf_medical_ai/data/task2_medical_kg.db"
        )
        self.llm = _llm_client(kwargs) if self.backend in {"llm", "hybrid"} else None

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample.get(self.text_key, "")
        if not text.strip():
            sample["relations"] = []
            return sample
        try:
            result = extract_medical_knowledge(
                text,
                backend=self.backend,
                kg_db_path=self.kg_db_path,
                llm=self.llm,
            )
            if not sample.get("entities"):
                sample["entities"] = [entity.to_dict() for entity in result.entities]
            sample["relations"] = [relation.to_dict() for relation in result.relations]
            sample["task2_backend"] = result.backend
            sample["relation_extraction_seconds"] = result.elapsed_seconds
            if result.llm_error:
                sample["llm_error"] = result.llm_error
        except Exception as exc:
            logger.warning(f"MedicalRelationExtractor failed: {exc}")
            sample["relations"] = []
        logger.info(
            f"MedicalRelationExtractor[{self.backend}]: {len(sample['relations'])} relations, "
            f"{time.time() - start:.2f}s"
        )
        return sample
