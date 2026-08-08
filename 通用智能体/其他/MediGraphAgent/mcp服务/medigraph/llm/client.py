"""OpenAI-compatible LLM client with provider switching, retries and JSON mode.

Used by the LLM-backed operators (NER / RE / validator fallback) and by the
agents (DataProc planner, GraphRAG answerer). Works against SiliconFlow or
Aliyun DashScope -- both expose an OpenAI-compatible /chat/completions endpoint.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

# Allow running both as a package (medigraph.llm.client) and standalone.
try:
    from config.settings import LLMConfig, get_llm_config
except ModuleNotFoundError:  # pragma: no cover - fallback for odd sys.path
    from CCF.config.settings import LLMConfig, get_llm_config  # type: ignore


@dataclass
class CallStats:
    """Lightweight latency / token accounting for benchmarking operators."""

    calls: int = 0
    total_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    #: Time-to-first-token, recorded for streaming calls only. This is the metric
    #: streaming actually improves -- total latency is unchanged, but the user sees
    #: output after the first chunk instead of after the whole completion.
    ttfts: list[float] = field(default_factory=list)
    stream_calls: int = 0

    def record(self, seconds: float, usage: Any) -> None:
        self.calls += 1
        self.total_seconds += seconds
        self.latencies.append(seconds)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def record_stream(self, ttft: float, seconds: float, usage: Any = None) -> None:
        """Record a streaming call: TTFT plus the usual total-latency accounting."""
        self.stream_calls += 1
        self.ttfts.append(ttft)
        self.record(seconds, usage)

    def summary(self) -> dict:
        avg = self.total_seconds / self.calls if self.calls else 0.0
        out = {
            "calls": self.calls,
            "total_seconds": round(self.total_seconds, 3),
            "avg_latency_s": round(avg, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        if self.ttfts:
            out["stream_calls"] = self.stream_calls
            out["avg_ttft_s"] = round(sum(self.ttfts) / len(self.ttfts), 3)
            out["p50_ttft_s"] = round(sorted(self.ttfts)[len(self.ttfts) // 2], 3)
        return out


class LLMClient:
    """Thin wrapper over the OpenAI SDK pointed at an OpenAI-compatible host."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or get_llm_config()
        if not self.config.api_key:
            print(
                f"[LLMClient] WARNING: no API key for provider "
                f"'{self.config.provider}'. Set it in mcp服务/.env (copy .env.example first).",
                file=sys.stderr,
            )
        self.client = OpenAI(
            api_key=self.config.api_key or "EMPTY",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,  # we handle retries ourselves for clearer logging
        )
        self.stats = CallStats()

    # ------------------------------------------------------------------ #
    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        enable_thinking: bool | None = None,
    ) -> str:
        """Single-turn chat completion with retry. Returns raw text content.

        enable_thinking controls Qwen3/3.5/3.6 "thinking" mode. Defaults to the
        client config (off for low-latency extraction). The flag is only sent to
        Qwen3* models; it is dropped automatically if the host rejects it.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if json_mode:
            # Most OpenAI-compatible hosts accept this; if not, we still rely on
            # prompt instructions + _extract_json below.
            kwargs["response_format"] = {"type": "json_object"}

        think = self.config.enable_thinking if enable_thinking is None else enable_thinking
        if "qwen3" in self.config.model.lower():
            kwargs["extra_body"] = {"enable_thinking": think}

        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            start = time.time()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - start
                self.stats.record(elapsed, getattr(resp, "usage", None))
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - surface provider errors
                last_err = exc
                # Drop optional params some hosts reject, then retry.
                if json_mode and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                kwargs.pop("extra_body", None)
                if attempt < self.config.max_retries:
                    wait = min(2 ** attempt, 10)
                    print(
                        f"[LLMClient] call failed (attempt {attempt}/{self.config.max_retries}): "
                        f"{exc}. retrying in {wait}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"[LLMClient] call failed (attempt {attempt}/{self.config.max_retries}): {exc}.",
                        file=sys.stderr,
                    )
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {last_err}")

    def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        default: Any = None,
    ) -> Any:
        """Chat completion that returns parsed JSON, with robust extraction."""
        raw = self.chat(prompt, system=system, temperature=temperature, json_mode=True)
        parsed = self._extract_json(raw)
        if parsed is None:
            return default
        return parsed

    # ------------------------------------------------------------------ #
    def _build_kwargs(
        self,
        prompt: str,
        system: str | None,
        temperature: float | None,
        enable_thinking: bool | None,
    ) -> dict[str, Any]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        think = self.config.enable_thinking if enable_thinking is None else enable_thinking
        if "qwen3" in self.config.model.lower():
            kwargs["extra_body"] = {"enable_thinking": think}
        return kwargs

    def chat_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
    ) -> Iterator[str]:
        """Yield content deltas as the model produces them.

        Retries are attempted only *before the first token is emitted*: once a
        delta has been handed to the caller it cannot be un-yielded, so a mid-stream
        failure propagates instead of silently restarting and duplicating output.

        Records TTFT and total latency into `self.stats`, so a caller can report the
        streaming win (first byte) separately from total generation time.
        """
        kwargs = self._build_kwargs(prompt, system, temperature, enable_thinking)
        kwargs["stream"] = True

        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            start = time.time()
            first_token_at: float | None = None
            emitted = False
            try:
                stream = self.client.chat.completions.create(**kwargs)
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    piece = getattr(delta, "content", None) if delta else None
                    if not piece:
                        continue
                    if first_token_at is None:
                        first_token_at = time.time()
                    emitted = True
                    yield piece
                elapsed = time.time() - start
                ttft = (first_token_at - start) if first_token_at else elapsed
                self.stats.record_stream(ttft, elapsed)
                return
            except Exception as exc:  # noqa: BLE001 - surface provider errors
                last_err = exc
                if emitted:
                    # Partial output already delivered; retrying would duplicate it.
                    raise
                kwargs.pop("extra_body", None)
                if attempt < self.config.max_retries:
                    wait = min(2 ** attempt, 10)
                    print(
                        f"[LLMClient] stream failed before first token "
                        f"(attempt {attempt}/{self.config.max_retries}): {exc}. retrying in {wait}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"LLM stream failed after {self.config.max_retries} attempts: {last_err}"
        )

    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed texts via the OpenAI-compatible embeddings endpoint.

        Default model is a small, fast embedder available on SiliconFlow. Returns
        one vector per input; on failure returns empty lists (callers degrade to
        graph-only retrieval).
        """
        if not texts:
            return []
        model = model or self.config.embedding_model
        try:
            resp = self.client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as exc:  # noqa: BLE001
            print(f"[LLMClient] embeddings failed ({exc}); falling back to no-vector.",
                  file=sys.stderr)
            return [[] for _ in texts]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_json(text: str) -> Any:
        """Best-effort JSON extraction from an LLM response."""
        if not text:
            return None
        text = text.strip()
        # Strip ```json ... ``` fences if present.
        fence = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fallback: grab the outermost JSON object/array.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
        return None
