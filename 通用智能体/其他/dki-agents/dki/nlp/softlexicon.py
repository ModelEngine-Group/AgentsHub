"""Fit-only SoftLexicon surfaces and deterministic BMES token features.

Feature layout per token: num_types * 4 channels (B,M,E,S) in {0,1}.
Surfaces come only from the train-fit split to avoid selection leakage.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Sequence, Set

import numpy as np


BMES = ("B", "M", "E", "S")
SOFTLEXICON_SCHEMA_VERSION = 1


def build_surface_lexicon(
    rows: Sequence[dict],
    type2id: Dict[str, int],
) -> Dict[str, Set[int]]:
    lex: Dict[str, Set[int]] = {}
    for row in rows:
        text = str(row.get("text", ""))
        for ent in row.get("entities", []):
            t = ent.get("type")
            if t not in type2id:
                continue
            start, end = int(ent["start"]), int(ent["end"])
            surface = text[start:end]
            if len(surface) < 1:
                continue
            lex.setdefault(surface, set()).add(type2id[t])
    return lex


def build_cmeie_argument_surface_lexicon(
    rows: Sequence[dict],
    type2id: Dict[str, int],
    *,
    min_chars: int = 2,
) -> Dict[str, Set[int]]:
    """Build a native-type CMeIE lexicon from fit SPO arguments only.

    A surface is admitted only when it is present in the row text.  The
    function never reads a calibration, outer, or official-dev row; callers
    are responsible for supplying the already validated physical fit shard.
    """
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 1:
        raise ValueError("min_chars must be a positive integer")
    lex: Dict[str, Set[int]] = {}
    for row in rows:
        text = str(row.get("text", ""))
        for spo in row.get("spo_list", []):
            for role in ("subject", "object"):
                surface = str(spo.get(role, ""))
                label = str(spo.get(f"{role}_type", ""))
                if (
                    label not in type2id
                    or len(surface) < min_chars
                    or surface not in text
                ):
                    continue
                lex.setdefault(surface, set()).add(type2id[label])
    return lex


def argument_surface_lexicon_payload(
    lexicon: Dict[str, Set[int]],
    labels: Sequence[str],
    *,
    min_chars: int,
) -> dict[str, Any]:
    """Return the canonical, portable CMeIE SoftLexicon payload."""
    if (
        not labels
        or list(labels) != sorted(set(labels))
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError("labels must be sorted unique non-empty strings")
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 1:
        raise ValueError("min_chars must be a positive integer")
    surfaces: dict[str, list[str]] = {}
    for surface in sorted(lexicon):
        if not isinstance(surface, str) or len(surface) < min_chars:
            raise ValueError("lexicon surface is shorter than min_chars")
        type_ids = sorted(lexicon[surface])
        if not type_ids or any(
            isinstance(type_id, bool)
            or not isinstance(type_id, int)
            or not 0 <= type_id < len(labels)
            for type_id in type_ids
        ):
            raise ValueError("lexicon contains an invalid type id")
        surfaces[surface] = [labels[type_id] for type_id in type_ids]
    return {
        "schema_version": SOFTLEXICON_SCHEMA_VERSION,
        "kind": "cmeie_fit_argument_softlexicon",
        "algorithm": "longest_nonoverlap_per_type_bmes",
        "min_chars": min_chars,
        "labels": list(labels),
        "feature_layout": [
            f"{label}:{position}"
            for label in labels
            for position in BMES
        ],
        "feature_dim": len(labels) * len(BMES),
        "surface_count": len(surfaces),
        "type_assignment_count": sum(len(values) for values in surfaces.values()),
        "surfaces": surfaces,
    }


def canonical_surface_lexicon_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize one lexicon payload for exact file/hash binding."""
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def surface_lexicon_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_surface_lexicon_bytes(payload)).hexdigest()


def char_bmes_features(
    text: str,
    lexicon: Dict[str, Set[int]],
    num_types: int,
) -> np.ndarray:
    """Return (len(text), num_types*4) float32 BMES indicators."""
    n = len(text)
    feat = np.zeros((n, num_types * 4), dtype=np.float32)
    if n == 0 or not lexicon:
        return feat
    # Longer surfaces first; allow overlapping types but prefer longer spans per type.
    surfaces = sorted(lexicon.keys(), key=len, reverse=True)
    occupied = [[False] * n for _ in range(num_types)]
    for surface in surfaces:
        types = lexicon[surface]
        L = len(surface)
        if L < 1:
            continue
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            end = idx + L
            for tid in types:
                if any(occupied[tid][idx:end]):
                    continue
                for j in range(idx, end):
                    occupied[tid][j] = True
                base = tid * 4
                if L == 1:
                    feat[idx, base + 3] = 1.0  # S
                else:
                    feat[idx, base + 0] = 1.0  # B
                    for mid in range(idx + 1, end - 1):
                        feat[mid, base + 1] = 1.0  # M
                    feat[end - 1, base + 2] = 1.0  # E
            start = idx + 1
    return feat


def token_softlex_from_offsets(
    offsets: np.ndarray,
    char_feat: np.ndarray,
    max_len: int,
) -> np.ndarray:
    """Aggregate char BMES → token features with max-pool; pad/truncate to max_len."""
    dim = char_feat.shape[1] if char_feat.ndim == 2 else 0
    out = np.zeros((max_len, dim), dtype=np.float32)
    seq = min(max_len, len(offsets))
    for ti in range(seq):
        cs, ce = int(offsets[ti][0]), int(offsets[ti][1])
        if ce <= cs or cs >= len(char_feat):
            continue
        ce = min(ce, len(char_feat))
        if ce > cs:
            out[ti] = char_feat[cs:ce].max(axis=0)
    return out


def encode_softlex_batch(
    rows: Sequence[dict],
    offsets_batch: np.ndarray,
    lexicon: Dict[str, Set[int]],
    num_types: int,
    max_len: int,
) -> np.ndarray:
    """offsets_batch: (n, max_len, 2) → (n, max_len, num_types*4)."""
    n = len(rows)
    dim = num_types * 4
    out = np.zeros((n, max_len, dim), dtype=np.float32)
    for i, row in enumerate(rows):
        text = str(row.get("text", ""))
        char_feat = char_bmes_features(text, lexicon, num_types)
        out[i] = token_softlex_from_offsets(offsets_batch[i], char_feat, max_len)
    return out
