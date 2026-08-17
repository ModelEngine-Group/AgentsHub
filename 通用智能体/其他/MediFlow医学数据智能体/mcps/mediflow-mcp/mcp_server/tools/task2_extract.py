"""
任务二实体、关系和三元组抽取 MCP 工具。
"""
from mcp_server.tools import mcp
from core.llm_client import LLMClient
from mcp_server.task2.text_extraction_service import (
    extract_text_knowledge,
    validate_text_backend,
)
from mcp_server.config import KG_DB, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        _llm = LLMClient(base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY or None)
    return _llm


def _llm_for_backend(backend: str) -> LLMClient | None:
    selected = validate_text_backend(backend)
    return _get_llm() if selected in {"llm", "hybrid"} else None


def _extract(text: str, backend: str) -> dict:
    selected = validate_text_backend(backend)
    return extract_text_knowledge(
        text,
        backend=selected,
        kg_db_path=KG_DB,
        llm=_llm_for_backend(selected),
    )


@mcp.tool
def extract_medical_knowledge_from_text(
    text: str,
    backend: str = "offline",
) -> dict:
    """一次抽取单段医疗文本的实体、关系、三元组、级联统计和性能指标。"""

    return _extract(text, backend)


@mcp.tool
def extract_medical_entities(text: str, backend: str = "offline") -> dict:
    """抽取医学实体；空结果也返回结构化成功对象。"""

    result = _extract(text, backend)
    return {
        "status": result["status"],
        "backend": result["backend"],
        "item_count": result["counts"]["entity_count"],
        "entities": result["entities"],
        "performance": result["performance"],
        "extraction_errors": result["extraction_errors"],
    }

@mcp.tool
def extract_medical_relations(text: str, backend: str = "offline") -> dict:
    """抽取医学关系；空结果也返回结构化成功对象。"""

    result = _extract(text, backend)
    return {
        "status": result["status"],
        "backend": result["backend"],
        "item_count": result["counts"]["relation_count"],
        "relations": result["relations"],
        "performance": result["performance"],
        "extraction_errors": result["extraction_errors"],
    }

@mcp.tool
def generate_medical_triples(text: str, backend: str = "offline") -> dict:
    """生成医学 SPO 三元组；空结果也返回结构化成功对象。"""

    result = _extract(text, backend)
    return {
        "status": result["status"],
        "backend": result["backend"],
        "item_count": result["counts"]["triple_count"],
        "triples": result["triples"],
        "performance": result["performance"],
        "extraction_errors": result["extraction_errors"],
    }
