"""Lossless token-window planning and deterministic entity reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TokenWindow:
    index: int
    token_start: int
    token_end: int
    core_start: int
    core_end: int
    char_start: int
    char_end: int


class WindowingError(ValueError):
    pass


def plan_token_windows(
    offsets: Sequence[Sequence[int]],
    *,
    capacity: int,
    overlap: int,
) -> List[TokenWindow]:
    """Plan windows over content-token offsets.

    ``offsets`` excludes model special/padding tokens and contains global text
    coordinates. Cores partition token space, while neighboring context overlaps.
    """
    if capacity <= 0:
        raise WindowingError("capacity must be positive")
    if overlap < 0 or overlap >= capacity:
        raise WindowingError("overlap must satisfy 0 <= overlap < capacity")
    normalized: List[Tuple[int, int]] = []
    previous_end = -1
    for raw in offsets:
        if len(raw) != 2:
            raise WindowingError(f"offset must contain two values: {raw!r}")
        start, end = int(raw[0]), int(raw[1])
        if start < 0 or end <= start or start < previous_end:
            raise WindowingError(f"invalid/non-monotonic offset [{start}, {end})")
        normalized.append((start, end))
        previous_end = end
    if not normalized:
        return []
    stride = capacity - overlap
    windows: List[TokenWindow] = []
    start = 0
    while start < len(normalized):
        end = min(start + capacity, len(normalized))
        left_context = 0 if start == 0 else overlap // 2
        right_context = 0 if end == len(normalized) else overlap - overlap // 2
        core_start = start + left_context
        core_end = end - right_context
        if core_end <= core_start:
            raise WindowingError("window has an empty ownership core")
        windows.append(TokenWindow(
            index=len(windows),
            token_start=start,
            token_end=end,
            core_start=core_start,
            core_end=core_end,
            char_start=normalized[start][0],
            char_end=normalized[end - 1][1],
        ))
        if end == len(normalized):
            break
        start += stride
    return windows


def owns_anchor(window: TokenWindow, anchor_token: int) -> bool:
    return window.core_start <= anchor_token < window.core_end


def merge_window_entities(
    window_results: Iterable[tuple[TokenWindow, Iterable[Mapping[str, object]]]],
) -> List[dict]:
    """Merge global-coordinate entities independent of worker completion order.

    Each result needs ``anchor_token`` in global token coordinates.  Ownership
    cores prevent boundary duplicates; exact span/type dedup handles identical
    repeated windows without collapsing same-name entities at different spans.
    """
    candidates = []
    for window, entities in window_results:
        for entity in entities:
            anchor = int(entity["anchor_token"])
            if not owns_anchor(window, anchor):
                continue
            start, end = int(entity["span"][0]), int(entity["span"][1])
            if start < window.char_start or end > window.char_end or end <= start:
                raise WindowingError(
                    f"entity [{start}, {end}) is outside window {window.index}"
                )
            item = dict(entity)
            item["span"] = [start, end]
            item["window_index"] = window.index
            candidates.append(item)
    candidates.sort(key=lambda item: (
        int(item["span"][0]), int(item["span"][1]), str(item["type"]),
        int(item["window_index"]), str(item.get("mention", "")),
    ))
    merged = []
    seen = set()
    for item in candidates:
        key = (int(item["span"][0]), int(item["span"][1]), str(item["type"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def choose_shape(token_count: int, sequence_lengths: Sequence[int], special_tokens: int = 2) -> int | None:
    """Return the smallest static sequence length that can contain the text."""
    required = token_count + special_tokens
    for length in sorted(set(int(value) for value in sequence_lengths)):
        if length >= required:
            return length
    return None
