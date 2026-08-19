"""Audit CMeIE character-to-token label alignment without reading outer/dev.

The report is aggregate-only: it never stores source text, entity mentions, or
gold triples.  It is intended to run against the exact fast tokenizer used by
training before any expensive NPU retraining is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.cmeie_spo import (
    all_occurrences,
    char_span_to_tokens,
    same_sentence,
    select_mention_pairs,
)


CANONICAL_SENTENCE_BREAKS = frozenset("。！？；!?\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_same_sentence(
    text: str,
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    between = text[min(left[1], right[1]) : max(left[0], right[0])]
    return not any(char in CANONICAL_SENTENCE_BREAKS for char in between)


def _pairs(
    text: str,
    subject_spans: Iterable[tuple[int, int]],
    object_spans: Iterable[tuple[int, int]],
    predicate,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [
        (subject_span, object_span)
        for subject_span in subject_spans
        for object_span in object_spans
        if predicate(text, subject_span, object_span)
    ]


def _pair_aligns(
    offsets: list[tuple[int, int]],
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    return (
        char_span_to_tokens(offsets, pair[0]) is not None
        and char_span_to_tokens(offsets, pair[1]) is not None
    )


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _publish_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"alignment audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tokenizer_fingerprint(tokenizer) -> dict[str, Any]:
    vocabulary = json.dumps(
        sorted(tokenizer.get_vocab().items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    backend = str(tokenizer.backend_tokenizer).encode("utf-8")
    return {
        "class": tokenizer.__class__.__name__,
        "is_fast": bool(tokenizer.is_fast),
        "vocabulary_size": len(tokenizer.get_vocab()),
        "vocabulary_sha256": _sha256_bytes(vocabulary),
        "backend_sha256": _sha256_bytes(backend),
        "model_max_length": int(tokenizer.model_max_length),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate-only CMeIE tokenizer alignment audit"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "calibration"), default="fit")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"alignment audit already exists: {args.output}")
    if args.max_length < 8:
        raise SystemExit("--max-length must be at least 8")
    if len(args.source_tree_sha256) != 64:
        raise SystemExit("--source-tree-sha256 must be a full SHA-256")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("source_train_sha256") != _sha256(args.train):
        raise SystemExit("train hash differs from frozen split manifest")
    if args.split not in manifest.get("splits", {}):
        raise SystemExit(f"split is absent from manifest: {args.split}")

    rows = _load_jsonl(args.train)
    indices = manifest["splits"][args.split]["indices"]
    selected = [rows[index] for index in indices]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("alignment audit requires a fast tokenizer")

    categories: Counter[str] = Counter()
    joint_categories: Counter[str] = Counter()
    predicates: dict[str, Counter[str]] = {}
    token_lengths: list[int] = []
    row_counts: Counter[str] = Counter()
    current_pair_count = 0
    canonical_pair_count = 0
    current_only_pair_count = 0
    selected_pair_count = 0
    selected_aligned_pair_count = 0
    multi_pair_spo_count = 0
    full_offset_gap_rows = 0

    for row in selected:
        text = str(row.get("text", ""))
        full = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=False,
            add_special_tokens=True,
        )
        truncated = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=True,
        )
        full_offsets = [tuple(pair) for pair in full["offset_mapping"]]
        truncated_offsets = [tuple(pair) for pair in truncated["offset_mapping"]]
        token_lengths.append(len(full["input_ids"]))
        if len(full["input_ids"]) > args.max_length:
            row_counts["rows_over_max_length"] += 1

        covered = [
            (left, right)
            for left, right in full_offsets
            if right > left
        ]
        if text and (
            not covered
            or min(left for left, _ in covered) != 0
            or max(right for _, right in covered) != len(text)
        ):
            full_offset_gap_rows += 1

        for spo in row.get("spo_list", []):
            subject = str(spo.get("subject", ""))
            obj = str(spo.get("object", ""))
            predicate = str(spo.get("predicate", ""))
            predicate_counts = predicates.setdefault(predicate, Counter())
            subject_spans = all_occurrences(text, subject)
            object_spans = all_occurrences(text, obj)

            if not subject_spans or not object_spans:
                category = "source_term_absent"
            else:
                current_pairs = _pairs(
                    text, subject_spans, object_spans, same_sentence
                )
                canonical_pairs = _pairs(
                    text,
                    subject_spans,
                    object_spans,
                    _canonical_same_sentence,
                )
                current_pair_count += len(current_pairs)
                canonical_pair_count += len(canonical_pairs)
                current_only_pair_count += len(set(current_pairs) - set(canonical_pairs))

                if not current_pairs:
                    category = (
                        "sentence_rule_disagreement"
                        if canonical_pairs
                        else "cross_sentence_only"
                    )
                elif any(
                    _pair_aligns(truncated_offsets, pair) for pair in current_pairs
                ):
                    category = "aligned"
                elif any(_pair_aligns(full_offsets, pair) for pair in current_pairs):
                    category = "truncated"
                else:
                    category = "offset_mapping_gap_or_normalization"

            categories[category] += 1
            predicate_counts[category] += 1

            joint_pairs = select_mention_pairs(
                text,
                subject,
                obj,
                allow_cross_sentence_fallback=True,
            )
            selected_pair_count += len(joint_pairs)
            multi_pair_spo_count += int(len(joint_pairs) > 1)
            aligned_joint_pairs = [
                pair
                for pair in joint_pairs
                if _pair_aligns(truncated_offsets, pair)
            ]
            selected_aligned_pair_count += len(aligned_joint_pairs)
            if not joint_pairs:
                joint_category = "source_term_absent"
            elif aligned_joint_pairs:
                joint_category = "aligned"
            elif any(
                _pair_aligns(full_offsets, pair)
                for pair in joint_pairs
            ):
                joint_category = "truncated"
            else:
                joint_category = "offset_mapping_gap_or_normalization"
            joint_categories[joint_category] += 1

    total_spo = sum(categories.values())
    dropped = total_spo - categories["aligned"]
    joint_dropped = total_spo - joint_categories["aligned"]
    payload = {
        "schema_version": 1,
        "kind": "cmeie_alignment_audit",
        "dataset_role": f"{args.split}_diagnostic_only",
        "outer_accessed": False,
        "official_dev_accessed": False,
        "aggregate_only": True,
        "source_tree_sha256": args.source_tree_sha256,
        "train_sha256": _sha256(args.train),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "audit_script_sha256": _sha256(Path(__file__)),
        "alignment_module_sha256": _sha256(
            ROOT / "dki" / "nlp" / "cmeie_spo.py"
        ),
        "base_model": args.base_model,
        "tokenizer": _tokenizer_fingerprint(tokenizer),
        "max_length": args.max_length,
        "row_count": len(selected),
        "spo_count": total_spo,
        "categories": dict(sorted(categories.items())),
        "dropped_label_count_one_pass": dropped,
        "dropped_label_rate": dropped / max(1, total_spo),
        "joint_gold_fallback_policy": {
            "same_sentence_preferred": True,
            "all_deterministic_nearest_same_sentence_links": True,
            "closest_pair_only": False,
            "multiple_surface_occurrences_retained": True,
            "cross_sentence_fallback": "single_closest_pair_only",
            "selected_pair_count": selected_pair_count,
            "selected_aligned_pair_count": selected_aligned_pair_count,
            "multi_pair_spo_count": multi_pair_spo_count,
            "categories": dict(sorted(joint_categories.items())),
            "dropped_label_count_one_pass": joint_dropped,
            "dropped_label_rate": joint_dropped / max(1, total_spo),
        },
        "sentence_pair_counts": {
            "current": current_pair_count,
            "canonical": canonical_pair_count,
            "current_only": current_only_pair_count,
        },
        "token_length": {
            "min": min(token_lengths, default=0),
            "p50": _percentile(token_lengths, 0.50),
            "p90": _percentile(token_lengths, 0.90),
            "p95": _percentile(token_lengths, 0.95),
            "p99": _percentile(token_lengths, 0.99),
            "max": max(token_lengths, default=0),
            "rows_over_max_length": row_counts["rows_over_max_length"],
            "rows_with_full_offset_gap": full_offset_gap_rows,
        },
        "per_predicate_categories": {
            predicate: dict(sorted(counts.items()))
            for predicate, counts in sorted(predicates.items())
        },
    }
    _publish_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
