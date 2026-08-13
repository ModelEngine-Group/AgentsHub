"""Deterministic, bounded pipeline failure diagnosis and recovery policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class FailureCategory(str, Enum):
    CONFIG = "CONFIG"
    INPUT = "INPUT"
    CONTRACT = "CONTRACT"
    MODEL_HASH = "MODEL_HASH"
    TRANSIENT = "TRANSIENT"
    QUEUE = "QUEUE"
    OOM = "OOM"
    ACL_FATAL = "ACL_FATAL"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(str, Enum):
    NO_RETRY = "NO_RETRY"
    RETRY_BACKOFF = "RETRY_BACKOFF"
    REDUCE_BATCH_ONCE = "REDUCE_BATCH_ONCE"
    RESTART_WORKER_ONCE = "RESTART_WORKER_ONCE"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class RecoveryDecision:
    category: FailureCategory
    action: RecoveryAction
    max_attempts: int
    reason: str


_PATTERNS = (
    (FailureCategory.MODEL_HASH, ("hash mismatch", "checksum", "model hash")),
    (FailureCategory.OOM, ("out of memory", "oom", "memory allocation")),
    (FailureCategory.QUEUE, ("queue full", "backpressure", "admission")),
    (FailureCategory.ACL_FATAL, ("acl fatal", "acl.mdl", "device reset", "context")),
    (FailureCategory.CONFIG, ("unknown backend", "invalid config", "配置", "unknown operator")),
    (FailureCategory.CONTRACT, ("schema", "contract", "shape mismatch", "dtype")),
    (FailureCategory.INPUT, ("invalid input", "decode error", "malformed", "越界")),
    (
        FailureCategory.TRANSIENT,
        ("transient", "timeout", "temporarily", "connection reset", "rate limit"),
    ),
)


def classify_failure(message: str) -> FailureCategory:
    normalized = message.casefold()
    for category, patterns in _PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return category
    return FailureCategory.UNKNOWN


def decide_recovery(
    message: str,
    *,
    attempt: int = 0,
    idempotent: bool = False,
    has_idempotency_key: bool = False,
) -> RecoveryDecision:
    category = classify_failure(message)
    safe_to_replay = idempotent or has_idempotency_key
    if category in {FailureCategory.CONFIG, FailureCategory.INPUT, FailureCategory.CONTRACT, FailureCategory.MODEL_HASH}:
        return RecoveryDecision(category, RecoveryAction.NO_RETRY, 0, "deterministic failure")
    if not safe_to_replay:
        return RecoveryDecision(category, RecoveryAction.QUARANTINE, 0, "replay is not idempotent")
    if category == FailureCategory.OOM:
        return RecoveryDecision(category, RecoveryAction.REDUCE_BATCH_ONCE, 1, "one bounded batch reduction")
    if category == FailureCategory.ACL_FATAL:
        return RecoveryDecision(category, RecoveryAction.RESTART_WORKER_ONCE, 1, "one whole-worker restart")
    if category in {FailureCategory.TRANSIENT, FailureCategory.QUEUE}:
        return RecoveryDecision(category, RecoveryAction.RETRY_BACKOFF, 2, "bounded transient retry")
    return RecoveryDecision(category, RecoveryAction.QUARANTINE, 0, "unknown failure")


def recovery_patch(decision: RecoveryDecision, current: Mapping[str, object]) -> dict:
    """Return a minimal allow-listed dry-run patch; never mutate input config."""
    if decision.action == RecoveryAction.REDUCE_BATCH_ONCE:
        batch = max(1, int(current.get("batch", 1)) // 2)
        return {"batch": batch}
    if decision.action == RecoveryAction.RESTART_WORKER_ONCE:
        return {"restart_worker": True}
    if decision.action == RecoveryAction.RETRY_BACKOFF:
        return {"retry": min(2, int(current.get("retry", 0)) + 1)}
    return {}
