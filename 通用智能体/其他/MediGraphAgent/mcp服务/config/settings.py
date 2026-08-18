"""Central configuration for MediGraph-Agent.

All environment-variable access lives here (single source of truth), so the rest
of the code never reads os.environ directly. Values come from a .env file (loaded
via python-dotenv) or from the real environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder holding this config/ package. Load its .env if present.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
# Raw text corpus for Task-1 ETL + Task-2 extraction (built by data/prep/build_dataset.py
# from CMeIE-V2 + DiaKG + pathology). Name kept as RAW_DEMO_DIR for back-compat.
RAW_DEMO_DIR = DATA_DIR / "corpus"
CORPUS_DIR = DATA_DIR / "corpus"
KG_GRAPH = DATA_DIR / "kg" / "cm3kg_graph.json"
FAST_EXTRACTOR_ARTIFACT = MODEL_DIR / "fast_extractor.json"
ENTITY_LINKER_ARTIFACT = MODEL_DIR / "entity_linker.json"
CALIBRATION_ARTIFACT = MODEL_DIR / "temperature_calibration.json"
NEURAL_EXTRACTOR_DIR = MODEL_DIR / "neural_extractor"


@dataclass
class LLMConfig:
    """Resolved LLM connection settings for the active provider."""

    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_retries: int
    timeout: int
    enable_thinking: bool = False
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_llm_config(provider: str | None = None) -> LLMConfig:
    """Build the LLM config for the requested provider (default from env)."""
    provider = (provider or _get("LLM_PROVIDER", "siliconflow")).lower()

    if provider == "dashscope":
        api_key = _get("DASHSCOPE_API_KEY")
        base_url = _get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = _get("DASHSCOPE_MODEL", "qwen-plus")
    elif provider == "deepseek":
        api_key = _get("DEEPSEEK_API_KEY")
        base_url = _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = _get("DEEPSEEK_MODEL", "deepseek-chat")
    else:
        provider = "siliconflow"
        api_key = _get("SILICONFLOW_API_KEY")
        base_url = _get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        model = _get("SILICONFLOW_MODEL", "Qwen/Qwen3.5-35B-A3B")

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=float(_get("LLM_TEMPERATURE", "0.2") or 0.2),
        max_retries=int(_get("LLM_MAX_RETRIES", "3") or 3),
        timeout=int(_get("LLM_TIMEOUT", "120") or 120),
        enable_thinking=_get("LLM_ENABLE_THINKING", "false").lower() == "true",
        embedding_model=_get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
    )


@dataclass
class GraphConfig:
    backend: str  # local | neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str


def get_graph_config() -> GraphConfig:
    return GraphConfig(
        backend=_get("GRAPH_BACKEND", "local").lower() or "local",
        neo4j_uri=_get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=_get("NEO4J_USER", "neo4j"),
        neo4j_password=_get("NEO4J_PASSWORD", "neo4j-password"),
    )


@dataclass
class AnalyticsConfig:
    """Relational analytics backend selection.

    sqlite  -> file DB under outputs/ (zero-config default, used by CI and demos)
    postgres-> pooled PostgreSQL; same logical schema, adds real indexes/EXPLAIN
    """

    backend: str  # sqlite | postgres
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_database: str
    pool_min: int
    pool_max: int

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} user={self.pg_user} "
            f"password={self.pg_password} dbname={self.pg_database}"
        )


def get_analytics_config() -> AnalyticsConfig:
    backend = _get("ANALYTICS_BACKEND", "sqlite").lower() or "sqlite"
    if backend not in {"sqlite", "postgres"}:
        backend = "sqlite"
    return AnalyticsConfig(
        backend=backend,
        pg_host=_get("POSTGRES_HOST", "127.0.0.1"),
        pg_port=int(_get("POSTGRES_PORT", "5433") or 5433),
        pg_user=_get("POSTGRES_USER", "postgres"),
        pg_password=_get("POSTGRES_PASSWORD", ""),
        pg_database=_get("POSTGRES_DB", "medigraph"),
        pool_min=int(_get("PG_POOL_MIN", "1") or 1),
        pool_max=int(_get("PG_POOL_MAX", "8") or 8),
    )


@dataclass
class CacheConfig:
    """Redis result cache for extraction / NL2SQL responses."""

    enabled: bool
    url: str
    ttl_seconds: int


def get_cache_config() -> CacheConfig:
    return CacheConfig(
        enabled=_get("CACHE_ENABLED", "false").lower() == "true",
        url=_get("REDIS_URL", "redis://127.0.0.1:6380/0"),
        ttl_seconds=int(_get("CACHE_TTL_SECONDS", "3600") or 3600),
    )


@dataclass
class ExtractionConfig:
    """Confidence-routed local/LLM extraction configuration."""

    backend: str  # auto | neural | fast | llm
    neural_model_dir: Path
    fast_artifact: Path
    linker_artifact: Path
    calibration_artifact: Path
    route_threshold: float
    neural_threshold: float
    neural_rel_threshold: float
    llm_fallback: bool


def get_extraction_config() -> ExtractionConfig:
    backend = _get("EXTRACTION_BACKEND", "auto").lower() or "auto"
    if backend not in {"auto", "neural", "fast", "llm"}:
        backend = "auto"
    return ExtractionConfig(
        backend=backend,
        neural_model_dir=Path(_get("NEURAL_EXTRACTOR_DIR", str(NEURAL_EXTRACTOR_DIR))),
        fast_artifact=Path(_get("FAST_EXTRACTOR_ARTIFACT", str(FAST_EXTRACTOR_ARTIFACT))),
        linker_artifact=Path(_get("ENTITY_LINKER_ARTIFACT", str(ENTITY_LINKER_ARTIFACT))),
        calibration_artifact=Path(_get("CALIBRATION_ARTIFACT", str(CALIBRATION_ARTIFACT))),
        route_threshold=float(_get("EXTRACTION_ROUTE_THRESHOLD", "0.55") or 0.55),
        neural_threshold=float(_get("NEURAL_EXTRACTOR_THRESHOLD", "0.0") or 0.0),
        neural_rel_threshold=float(_get("NEURAL_EXTRACTOR_REL_THRESHOLD", "0.0") or 0.0),
        llm_fallback=_get("EXTRACTION_LLM_FALLBACK", "true").lower() == "true",
    )
