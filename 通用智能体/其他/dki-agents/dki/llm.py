"""统一 LLM 客户端（OpenAI 兼容端点；配置读 dki.paths 的 LLM_* 常量）。"""

import re
import time
from typing import Any, Literal, Optional

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LlmNotConfigured(RuntimeError):
    """未配置 LLM 环境变量（LLM_API_KEY / LLM_MODEL_NAME / LLM_API_URL）。"""


def llm_available() -> bool:
    from dki.paths import LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME

    return bool(LLM_API_KEY and LLM_API_URL and LLM_MODEL_NAME)


def _is_mimo_profile(api_url: str, model_name: str) -> bool:
    """Return whether the endpoint follows Xiaomi MiMo chat semantics.

    MiMo reasoning models may spend the whole legacy ``max_tokens`` budget in
    ``reasoning_content`` and consequently return an empty final ``content``.
    The official OpenAI-compatible API exposes ``thinking.type=disabled`` and
    ``max_completion_tokens`` for deterministic compiler-style calls.
    """

    endpoint = api_url.casefold()
    model = model_name.casefold()
    return "xiaomimimo" in endpoint or model.startswith("mimo-")


def _is_minimax_profile(api_url: str, model_name: str) -> bool:
    """Return whether the endpoint follows MiniMax M-series semantics."""

    endpoint = api_url.casefold()
    model = model_name.casefold()
    return "minimax" in endpoint or model.startswith("minimax-")


def _is_minimax_m3_profile(api_url: str, model_name: str) -> bool:
    """Return whether the active MiniMax profile is an M3 model."""

    return _is_minimax_profile(api_url, model_name) and model_name.casefold().startswith(
        "minimax-m3"
    )


def _uses_completion_token_budget(api_url: str, model_name: str) -> bool:
    """Return whether the provider uses the current completion-token field."""

    return _is_mimo_profile(api_url, model_name) or _is_minimax_profile(
        api_url, model_name
    )


def nl2sql_completion_token_budget(default: int = 1024) -> int:
    """Return the bounded NL2SQL completion budget for the active provider.

    MiniMax M2.x cannot disable reasoning.  A 1024-token request can end at
    the closing reasoning tag before producing SQL, so the Agent compiler uses
    a small provider-specific 2048-token floor.  Other providers retain the
    caller's deterministic budget.
    """

    if not isinstance(default, int) or default <= 0:
        raise ValueError("default completion budget must be a positive integer")
    from dki.paths import LLM_API_URL, LLM_MODEL_NAME

    if _is_minimax_profile(LLM_API_URL, LLM_MODEL_NAME) and not _is_minimax_m3_profile(
        LLM_API_URL, LLM_MODEL_NAME
    ):
        return max(default, 2048)
    return default


def chat(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: Optional[float] = 120,
    reasoning_mode: Literal["disabled", "adaptive"] = "disabled",
) -> str:
    """单轮对话，返回助手文本。未配置时抛 LlmNotConfigured。

    ``reasoning_mode`` explicitly controls MiniMax-M3 thinking.  Other
    provider profiles retain their existing request semantics.
    """
    if reasoning_mode not in ("disabled", "adaptive"):
        raise ValueError("reasoning_mode must be 'disabled' or 'adaptive'")

    from dki.paths import LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME

    if not llm_available():
        raise LlmNotConfigured(
            "请在 .env 配置 LLM_API_KEY / LLM_MODEL_NAME / LLM_API_URL（任意 OpenAI 兼容端点）"
        )
    from openai import OpenAI

    # A bare float timeout leaves the connect/read phases sharing one budget, and
    # a pooled socket that the server half-closed can park the caller in poll()
    # far longer than ``timeout``.  Pin each phase explicitly and cap retries so a
    # single wedged connection can never stall a whole evaluation run.
    try:
        import httpx

        request_timeout: Any = httpx.Timeout(
            timeout,
            connect=min(30.0, timeout) if timeout else 30.0,
            read=timeout,
            write=min(60.0, timeout) if timeout else 60.0,
            pool=min(30.0, timeout) if timeout else 30.0,
        )
    except Exception:
        request_timeout = timeout

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_API_URL,
        timeout=request_timeout,
        max_retries=2,
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    request = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if _uses_completion_token_budget(LLM_API_URL, LLM_MODEL_NAME):
        request["max_completion_tokens"] = max_tokens
        if _is_mimo_profile(LLM_API_URL, LLM_MODEL_NAME):
            request["extra_body"] = {
                "thinking": {
                    "type": "adaptive"
                    if reasoning_mode == "adaptive"
                    else "disabled"
                }
            }
        elif _is_minimax_profile(LLM_API_URL, LLM_MODEL_NAME):
            # Preserve final content separately from M-series reasoning; this
            # does not disable reasoning on M2.x, where that is unsupported.
            extra_body = {"reasoning_split": True}
            if _is_minimax_m3_profile(LLM_API_URL, LLM_MODEL_NAME):
                extra_body["thinking"] = {"type": reasoning_mode}
            request["extra_body"] = extra_body
    else:
        request["max_tokens"] = max_tokens
    # Transport-level resets (gateway connection caps, TLS EOF under burst
    # load) surface as APIConnectionError after the SDK's quick retries.  A
    # bounded exponential backoff rides out short resets without weakening
    # fail-closed semantics: after the last attempt the error still raises.
    try:
        from openai import APIConnectionError, APITimeoutError

        transient_errors: tuple[type[Exception], ...] = (
            APIConnectionError,
            APITimeoutError,
        )
    except ImportError:  # stubbed SDK in tests: no outer retry taxonomy
        transient_errors = ()

    last_error: Optional[Exception] = None
    for delay in (0.0, 1.0, 4.0, 10.0):
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(**request)
            break
        except transient_errors as exc:  # noqa: B030 - tuple built above
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    # 推理型模型（如 MiniMax M 系列）会在正文前输出 <think>…</think>，
    # 下游按结构化文本解析（SQL/JSON），必须剥离
    return _THINK_RE.sub("", resp.choices[0].message.content or "")
