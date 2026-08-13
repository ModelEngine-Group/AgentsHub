"""Focused tests for provider-specific OpenAI-compatible request handling."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dki import llm


class _FakeCompletions:
    def __init__(self, calls: list[dict], content: str) -> None:
        self._calls = calls
        self._content = content

    def create(self, **kwargs):
        self._calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    calls: list[dict] = []
    response_content = "SELECT 1"

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(self.calls, self.response_content)
        )


@pytest.fixture(autouse=True)
def _fake_openai(monkeypatch: pytest.MonkeyPatch):
    _FakeOpenAI.calls = []
    _FakeOpenAI.response_content = "SELECT 1"
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setattr("dki.paths.LLM_API_KEY", "configured-test-key")
    yield


def test_mimo_disables_thinking_and_uses_completion_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "mimo-v2.5")

    assert llm.chat("compile", temperature=0.0, max_tokens=320) == "SELECT 1"

    request = _FakeOpenAI.calls[-1]
    assert request["max_completion_tokens"] == 320
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "max_tokens" not in request


def test_mimo_can_request_adaptive_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "mimo-v2.5")

    assert (
        llm.chat(
            "compile",
            temperature=0.0,
            max_tokens=8192,
            reasoning_mode="adaptive",
        )
        == "SELECT 1"
    )
    request = _FakeOpenAI.calls[-1]
    assert request["max_completion_tokens"] == 8192
    assert request["extra_body"] == {"thinking": {"type": "adaptive"}}


def test_non_mimo_keeps_legacy_openai_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "generic-model")

    assert llm.chat("compile", max_tokens=123) == "SELECT 1"

    request = _FakeOpenAI.calls[-1]
    assert request["max_tokens"] == 123
    assert "max_completion_tokens" not in request
    assert "extra_body" not in request


def test_minimax_uses_reasoning_split_and_completion_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://api.minimaxi.com/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "MiniMax-M2.7-highspeed")

    assert llm.chat("compile", max_tokens=2048) == "SELECT 1"

    request = _FakeOpenAI.calls[-1]
    assert request["max_completion_tokens"] == 2048
    assert "max_tokens" not in request
    assert request["extra_body"] == {"reasoning_split": True}


def test_minimax_gets_a_bounded_nl2sql_completion_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://api.minimaxi.com/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "MiniMax-M2.7-highspeed")

    assert llm.nl2sql_completion_token_budget() == 2048
    assert llm.nl2sql_completion_token_budget(4096) == 4096


def test_minimax_m3_disables_thinking_and_keeps_default_nl2sql_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://api.minimaxi.com/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "MiniMax-M3")

    assert llm.chat("compile", max_tokens=1024) == "SELECT 1"
    request = _FakeOpenAI.calls[-1]
    assert request["max_completion_tokens"] == 1024
    assert request["extra_body"] == {
        "reasoning_split": True,
        "thinking": {"type": "disabled"},
    }
    assert llm.nl2sql_completion_token_budget() == 1024


def test_minimax_m3_can_enable_adaptive_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://api.minimaxi.com/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "MiniMax-M3")

    assert (
        llm.chat(
            "compile",
            max_tokens=1536,
            reasoning_mode="adaptive",
        )
        == "SELECT 1"
    )
    request = _FakeOpenAI.calls[-1]
    assert request["max_completion_tokens"] == 1536
    assert "max_tokens" not in request
    assert request["extra_body"] == {
        "reasoning_split": True,
        "thinking": {"type": "adaptive"},
    }


@pytest.mark.parametrize("reasoning_mode", ["enabled", "auto", "", None])
def test_chat_rejects_invalid_reasoning_mode(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_mode: object,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://api.minimaxi.com/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "MiniMax-M3")

    with pytest.raises(
        ValueError,
        match="reasoning_mode must be 'disabled' or 'adaptive'",
    ):
        llm.chat("compile", reasoning_mode=reasoning_mode)  # type: ignore[arg-type]

    assert _FakeOpenAI.calls == []


def test_non_minimax_keeps_the_requested_nl2sql_completion_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "generic-model")

    assert llm.nl2sql_completion_token_budget() == 1024


def test_reasoning_prefix_is_removed_from_final_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "generic-model")
    _FakeOpenAI.response_content = "<think>private reasoning</think>\nSELECT 1"

    assert llm.chat("compile") == "SELECT 1"
