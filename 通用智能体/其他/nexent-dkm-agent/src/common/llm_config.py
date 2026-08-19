"""Secret-safe loaders for local OpenAI-compatible LLM configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ENV_KEY_MAP = {
    "OPENAI_API_KEY": "api_key",
    "OPENAI_BASE_URL": "base_url",
    "OPENAI_MODEL": "model_name",
    "OPENAI_TIMEOUT": "timeout",
    "OPENAI_MAX_TOKENS": "max_tokens",
    "OPENAI_THINKING": "thinking",
    "DEEPSEEK_THINKING": "thinking",
}
JSON_KEYS = {
    "api_key",
    "base_url",
    "model_name",
    "timeout",
    "max_tokens",
    "thinking",
}


def load_llm_config(path: str | Path | None) -> dict[str, Any] | None:
    """Load a local `.env` or JSON LLM config without exposing secrets."""

    if not path:
        return None
    config_path = Path(path)
    if not config_path.exists():
        return None

    try:
        if config_path.suffix.lower() in {".json", ".jsonc"}:
            values = load_json_config_values(config_path)
        else:
            values = load_env_config_values(config_path)
        return normalize_llm_config(values)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def load_json_config_values(path: str | Path) -> dict[str, Any]:
    """Return supported JSON config fields from a local config file."""

    raw_text = Path(path).read_text(encoding="utf-8").lstrip("\ufeff")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("LLM config file must contain a JSON object.")
    values: dict[str, Any] = {}
    for key, value in payload.items():
        mapped_key = key if key in JSON_KEYS else ENV_KEY_MAP.get(key)
        if mapped_key and value not in (None, ""):
            values[mapped_key] = value
    return values


def load_env_config_values(path: str | Path) -> dict[str, Any]:
    """Return supported config values from a simple env-style file."""

    values: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = raw_value.strip().strip('"').strip("'")
        mapped_key = ENV_KEY_MAP.get(key, key if key in JSON_KEYS else None)
        if mapped_key and value:
            values[mapped_key] = value
    return values


def normalize_llm_config(values: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and normalize config fields used by LLM callers."""

    api_key = _as_non_empty_string(values.get("api_key"))
    base_url = _normalize_base_url(values.get("base_url"))
    if not api_key or not base_url:
        return None

    config: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": _as_non_empty_string(values.get("model_name")) or "glm-5.1",
    }
    for key in ("timeout", "max_tokens"):
        number = _coerce_number(values.get(key))
        if number is not None:
            config[key] = number
    thinking = _normalize_thinking(values.get("thinking"))
    if thinking:
        config["thinking"] = thinking
    return config


def openai_extra_body(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return provider-specific OpenAI-compatible body extensions."""

    if not config:
        return {}
    thinking = _normalize_thinking(config.get("thinking"))
    return {"thinking": thinking} if thinking else {}


def openai_extra_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return kwargs accepted by the OpenAI SDK for provider extensions."""

    extra_body = openai_extra_body(config)
    return {"extra_body": extra_body} if extra_body else {}


def _normalize_base_url(value: Any) -> str | None:
    raw = _as_non_empty_string(value)
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username or parsed.password:
        return None
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def _as_non_empty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_number(value: Any) -> float | int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _normalize_thinking(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        thinking_type = _as_non_empty_string(value.get("type"))
    else:
        thinking_type = _as_non_empty_string(value)
    if not thinking_type:
        return None
    thinking_type = thinking_type.lower()
    if thinking_type not in {"enabled", "disabled"}:
        return None
    return {"type": thinking_type}
