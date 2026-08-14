"""Minimal OpenAI-compatible client used by the medical demo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
_ENV_LOADED = False
LLM_ENV_KEYS = {
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_API_PATH",
    "OPENAI_TIMEOUT_SECONDS",
    "OPENAI_TEMPERATURE",
    "OPENAI_MAX_TOKENS",
    "LLM_MODEL",
}


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key not in LLM_ENV_KEYS:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _load_env_file(Path(os.getenv("MODELENGINE_ENV_FILE", DEFAULT_ENV_FILE)))
    _ENV_LOADED = True


@dataclass(frozen=True)
class OpenAICompatConfig:
    base_url: str
    api_key: str
    model: str
    api_path: str = "v1/chat/completions"
    timeout_seconds: float = 30.0
    temperature: float = 0.2
    max_tokens: int = 512


def get_openai_compat_config() -> OpenAICompatConfig | None:
    _ensure_env_loaded()
    base_url = os.getenv("OPENAI_API_BASE", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not base_url or not api_key or not model:
        return None

    api_path = os.getenv("OPENAI_API_PATH", "v1/chat/completions").strip()
    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    return OpenAICompatConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        api_path=api_path,
        timeout_seconds=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def is_configured() -> bool:
    return get_openai_compat_config() is not None


def _build_endpoint(config: OpenAICompatConfig) -> str:
    base = config.base_url.rstrip("/") + "/"
    return parse.urljoin(base, config.api_path.lstrip("/"))


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    config = get_openai_compat_config()
    if config is None:
        raise RuntimeError("OpenAI-compatible API is not configured")

    payload = {
        "model": model or config.model,
        "messages": messages,
        "temperature": config.temperature if temperature is None else temperature,
        "max_tokens": config.max_tokens if max_tokens is None else max_tokens,
        "stream": False,
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = _build_endpoint(config)
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds or config.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response missing choices: {data}")

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise RuntimeError(f"LLM response missing text content: {data}")
