"""LLM-enhanced NL2SQL for task 3.

Uses an OpenAI-compatible LLM to translate arbitrary natural-language
questions into SQL against the graph analytics schema.  Falls back
to the template-based translator when the LLM is unavailable.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from src.common.llm_config import openai_extra_kwargs
from src.operators.analysis_ops.nl2sql import execute_read_only_sql

logger = logging.getLogger(__name__)

# Reasoning models consume part of the token budget before emitting SQL; keep a
# generous default so the SELECT is not truncated to empty content.
_DEFAULT_LLM_MAX_TOKENS = 4096

_SCHEMA_PROMPT = """\
You are a SQL expert. Given the following SQLite schema and a question,
generate a single SELECT query. Only output the SQL, no explanation.

Schema:
CREATE TABLE nodes (
  id TEXT PRIMARY KEY,
  name TEXT,
  type TEXT,
  mention_count INTEGER
);
CREATE TABLE edges (
  source TEXT,
  target TEXT,
  predicate TEXT,
  confidence REAL
);

Valid node types: Disease, Symptom, Drug, Examination, Treatment
Valid predicates: has_symptom, treated_by, diagnosed_by, recommended_treatment, complication_of

Rules:
- Only generate SELECT statements
- Use proper JOINs when combining nodes and edges
- LIMIT results to 20 rows max
- ORDER BY count DESC when aggregating
- Use node names in Chinese as they appear in the data

Question: {question}
"""

def translate_question_to_sql_llm_only(
    question: str,
    conn: sqlite3.Connection,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate via LLM only (no template fallback). Used for path-isolated benchmarks."""

    if not question:
        return {
            "status": "skipped",
            "intent": "no_question",
            "sql": "SELECT 1",
            "rows": [],
            "translator": "llm",
        }
    if not llm_config:
        return {
            "status": "skipped",
            "intent": "llm_not_configured",
            "sql": "SELECT 1",
            "rows": [],
            "translator": "llm",
        }

    try:
        sql = _llm_translate(question, llm_config)
        if sql:
            safe_result = _safe_execute(conn, sql)
            if safe_result is not None:
                return {
                    "status": "completed",
                    "intent": "llm_generated",
                    "sql": safe_result["sql"],
                    "rows": safe_result["rows"],
                    "translator": "llm",
                }
    except Exception:
        logger.warning("LLM-only NL2SQL failed.", exc_info=True)

    return {
        "status": "failed",
        "intent": "llm_failed",
        "sql": "SELECT 1",
        "rows": [],
        "translator": "llm",
    }


def translate_question_to_sql_local_only(
    question: str,
    conn: sqlite3.Connection,
    local_model_path: str | None = None,
) -> dict[str, Any]:
    """Translate via local model only (no LLM/template fallback). Used for path benchmarks."""

    if not question:
        return {
            "status": "skipped",
            "intent": "no_question",
            "sql": "SELECT 1",
            "rows": [],
            "translator": "local_model",
        }
    if not local_model_path:
        return {
            "status": "skipped",
            "intent": "local_model_not_configured",
            "sql": "SELECT 1",
            "rows": [],
            "translator": "local_model",
        }

    try:
        from src.operators.analysis_ops.local_model_nl2sql import predict_sql

        sql = predict_sql(local_model_path, question)
        if sql:
            safe_result = _safe_execute(conn, sql)
            if safe_result is not None:
                return {
                    "status": "completed",
                    "intent": "local_model_generated",
                    "sql": safe_result["sql"],
                    "rows": safe_result["rows"],
                    "translator": "local_model",
                }
    except Exception:
        logger.warning("Local-model-only NL2SQL failed.", exc_info=True)

    return {
        "status": "failed",
        "intent": "local_model_failed",
        "sql": "SELECT 1",
        "rows": [],
        "translator": "local_model",
    }


def translate_question_to_sql_with_llm(
    question: str,
    conn: sqlite3.Connection,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a natural-language question to SQL using LLM, with template fallback.

    When llm_config is provided and the LLM responds successfully,
    uses the LLM-generated SQL.  Otherwise falls back to template-based
    translation from ``nl2sql.translate_question_to_sql``.
    """
    if not question:
        return {
            "status": "skipped",
            "intent": "no_question",
            "sql": "SELECT 1",
            "rows": [],
            "translator": "none",
        }

    # Try LLM first
    if llm_config:
        try:
            sql = _llm_translate(question, llm_config)
            if sql:
                safe_result = _safe_execute(conn, sql)
                if safe_result is not None:
                    return {
                        "status": "completed",
                        "intent": "llm_generated",
                        "sql": safe_result["sql"],
                        "rows": safe_result["rows"],
                        "translator": "llm",
                    }
                logger.warning("LLM SQL validation or execution failed; falling back to template.")
        except Exception:
            logger.warning("LLM NL2SQL failed; falling back to template.", exc_info=True)

    # Template fallback (entity-aware: filter by a specific disease when named)
    from src.operators.analysis_ops.nl2sql import (
        disease_names_from_connection,
        drug_names_from_connection,
        execute_sql,
        symptom_names_from_connection,
        treatment_names_from_connection,
        translate_question_to_sql,
    )

    disease_names = disease_names_from_connection(conn)
    symptom_names = symptom_names_from_connection(conn)
    drug_names = drug_names_from_connection(conn)
    treatment_names = treatment_names_from_connection(conn)
    template_result = translate_question_to_sql(
        question,
        disease_names=disease_names,
        symptom_names=symptom_names,
        drug_names=drug_names,
        treatment_names=treatment_names,
    )
    rows = execute_sql(conn, template_result["sql"])
    return {**template_result, "rows": rows, "translator": "template"}


def translate_question_with_fallbacks(
    question: str,
    conn: sqlite3.Connection,
    llm_config: dict[str, Any] | None = None,
    local_model_path: str | None = None,
) -> dict[str, Any]:
    """Translate a question to SQL with local-model > LLM > template precedence.

    The locally fine-tuned model (when a path is given) is tried first for a
    fully self-hosted, no-API path; its SQL is still validated and executed
    through the read-only guard. On any miss it delegates to the LLM path,
    which itself falls back to the safe template translator.
    """

    if question and local_model_path:
        try:
            from src.operators.analysis_ops.local_model_nl2sql import predict_sql

            sql = predict_sql(local_model_path, question)
            if sql:
                safe_result = _safe_execute(conn, sql)
                if safe_result is not None:
                    return {
                        "status": "completed",
                        "intent": "local_model_generated",
                        "sql": safe_result["sql"],
                        "rows": safe_result["rows"],
                        "translator": "local_model",
                    }
                logger.warning(
                    "Local-model SQL validation/execution failed; trying LLM/template."
                )
        except Exception:
            logger.warning("Local-model NL2SQL failed; trying LLM/template.", exc_info=True)

    return translate_question_to_sql_with_llm(question, conn, llm_config=llm_config)


def _llm_translate(question: str, config: dict[str, Any]) -> str | None:
    """Call LLM to generate SQL from a question."""
    import openai

    prompt = _SCHEMA_PROMPT.format(question=question)
    client = openai.OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )
    response = client.chat.completions.create(
        model=config.get("model_name", "glm-5.1"),
        messages=[
            {"role": "system", "content": "You are a SQL expert. Only output the SQL query."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=config.get("max_tokens", _DEFAULT_LLM_MAX_TOKENS),
        timeout=config.get("timeout", 60.0),
        **openai_extra_kwargs(config),
    )
    raw = response.choices[0].message.content or ""
    return _extract_sql(raw)


def _extract_sql(raw: str) -> str | None:
    """Extract SQL from LLM response, handling code fences."""
    text = raw.strip()
    # Strip code fences
    if "```" in text:
        match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    # Strip trailing semicolons
    text = text.rstrip(";").strip()
    if text.upper().startswith("SELECT"):
        return text
    return None


def _safe_execute(conn: sqlite3.Connection, sql: str) -> dict[str, Any] | None:
    """Execute SQL safely, returning None on error."""
    try:
        return execute_read_only_sql(conn, sql)
    except (sqlite3.Error, ValueError):
        return None
