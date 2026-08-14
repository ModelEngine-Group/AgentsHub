from __future__ import annotations

import json
from contextvars import ContextVar, Token
from typing import Any, Dict

from runtime_common.common import resolve_path

LAST_COHORT_PATH = "outputs/runtime_context/last_cohort.json"
CONVERSATION_STATE_PATH = "outputs/runtime_context/conversation_state.json"
FALLBACK_CONTEXT_DIR = "outputs/local_runtime/runtime_context"
_CURRENT_CONVERSATION_ID: ContextVar[str | None] = ContextVar(
    "chroniccare_conversation_id", default=None
)


def bind_conversation_context(conversation_id: str | None) -> Token:
    value = str(conversation_id or "").strip() or None
    return _CURRENT_CONVERSATION_ID.set(value)


def reset_conversation_context(token: Token) -> None:
    _CURRENT_CONVERSATION_ID.reset(token)


def get_current_conversation_id() -> str | None:
    return _CURRENT_CONVERSATION_ID.get()


def active_data_version() -> str | None:
    manifest = _read_json("data/raw/data_manifest.json")
    return str(manifest.get("data_version") or "").strip() or None



def _candidate_paths(path_str: str) -> list:
    primary = resolve_path(path_str)
    fallback = resolve_path(f"{FALLBACK_CONTEXT_DIR}/{primary.name}")
    return [primary, fallback]


def _read_json(path_str: str) -> Dict[str, Any]:
    candidates = [path for path in _candidate_paths(path_str) if path.exists()]
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _write_json(path_str: str, payload: Dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: OSError | None = None
    for path in _candidate_paths(path_str):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def load_last_cohort() -> Dict[str, Any]:
    return _read_json(LAST_COHORT_PATH)


def load_conversation_state() -> Dict[str, Any]:
    return _read_json(CONVERSATION_STATE_PATH)


def save_last_cohort(payload: Dict[str, Any]) -> None:
    if not payload:
        return
    _write_json(LAST_COHORT_PATH, payload)
    state = load_conversation_state()
    history = list(state.get("history") or [])
    history.append(payload)
    state["last_cohort"] = payload
    state["history"] = history[-20:]
    _write_json(CONVERSATION_STATE_PATH, state)


def has_pronoun_reference(query: str) -> bool:
    text = str(query or "")
    return any(token in text for token in ("他们", "这些患者", "该群体", "上述患者", "这批患者", "这些人"))



# Explicit per-conversation cohort context (v2). The legacy last-cohort helpers above
# remain for compatibility with existing tools.
def _conversation_path(conversation_id: str) -> str:
    safe = "".join(ch for ch in str(conversation_id or "default") if ch.isalnum() or ch in "-_")[:80] or "default"
    return f"outputs/runtime_context/conversations/{safe}.json"


def save_conversation_cohort(conversation_id: str, payload: Dict[str, Any], *, ttl_seconds: int = 3600) -> Dict[str, Any]:
    from datetime import datetime, timedelta
    from uuid import uuid4
    now = datetime.now().astimezone()
    sanitized = {k: v for k, v in dict(payload or {}).items() if k not in {"patient_ids", "patients", "rows", "patient_details"}}
    sanitized.update({
        "conversation_id": conversation_id,
        "cohort_id": sanitized.get("cohort_id") or f"cohort_{uuid4().hex[:12]}",
        "cohort_label": sanitized.get("cohort_label") or "未命名队列",
        "cohort_definition": sanitized.get("cohort_definition") or sanitized.get("filters") or {},
        "source_question": sanitized.get("source_question") or sanitized.get("question"),
        "source_tool": sanitized.get("source_tool") or "unknown",
        "filters": sanitized.get("filters") or {},
        "time_window": sanitized.get("time_window") or sanitized.get("window"),
        "patient_count": sanitized.get("patient_count") or sanitized.get("cohort_patient_count"),
        "data_version": sanitized.get("data_version"),
        "as_of_date": sanitized.get("as_of_date"),
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=max(60, ttl_seconds))).isoformat(),
    })
    path = _conversation_path(conversation_id)
    state = _read_json(path)
    history = list(state.get("cohorts") or [])
    history.append(sanitized)
    _write_json(path, {"conversation_id": conversation_id, "cohorts": history[-20:], "active_cohort_id": sanitized["cohort_id"]})
    return sanitized


def load_conversation_cohorts(conversation_id: str, *, current_data_version: str | None = None) -> list[Dict[str, Any]]:
    from datetime import datetime
    state = _read_json(_conversation_path(conversation_id))
    now = datetime.now().astimezone()
    valid = []
    for item in state.get("cohorts") or []:
        try:
            if datetime.fromisoformat(str(item.get("expires_at"))) <= now:
                continue
        except (TypeError, ValueError):
            continue
        if current_data_version and item.get("data_version") != current_data_version:
            continue
        valid.append(item)
    return valid


def resolve_cohort_reference(query: str, conversation_id: str, *, current_data_version: str | None = None) -> Dict[str, Any]:
    text = str(query or "")
    if not has_pronoun_reference(text) and "前一个群体" not in text:
        return {"status": "no_reference", "cohort": None}
    cohorts = load_conversation_cohorts(conversation_id, current_data_version=current_data_version)
    if not cohorts:
        return {"status": "needs_clarification", "cohort": None, "question": "他们指哪个群体？请先说明患者范围。"}
    if "前一个群体" in text:
        if len(cohorts) < 2:
            return {"status": "needs_clarification", "cohort": None, "question": "当前没有可用的前一个群体。"}
        return {"status": "resolved", "cohort": cohorts[-2], "resolution": "previous"}
    return {"status": "resolved", "cohort": cohorts[-1], "resolution": "latest"}


def resolve_active_cohort(query: str) -> Dict[str, Any]:
    """Resolve cohort state for the bound real conversation.

    Global state is used only when no conversation ID reaches the service,
    preserving an explicit legacy compatibility path.
    """
    conversation_id = get_current_conversation_id()
    if conversation_id:
        resolution = resolve_cohort_reference(
            query, conversation_id, current_data_version=active_data_version()
        )
        if resolution["status"] == "no_reference":
            cohorts = load_conversation_cohorts(
                conversation_id, current_data_version=active_data_version()
            )
            resolution["cohort"] = cohorts[-1] if cohorts else None
        resolution["context_mode"] = "conversation_isolated"
        resolution["conversation_id"] = conversation_id
        return resolution
    cohort = load_last_cohort() or None
    return {
        "status": "resolved" if cohort else "no_reference",
        "cohort": cohort,
        "context_mode": "legacy_global_compatibility",
        "conversation_id": None,
    }
