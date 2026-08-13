"""CMeIE dual-column eval: gold-pair predicate F1 vs end-to-end SPO Micro-F1.

End-to-end path:
  1) entity spans from --ner-backend (gold | cmeee | dict)
  2) enumerate subject/object pairs within each sentence
  3) multi-label predicate classifier (trained CMeIE checkpoint)
  4) SPO Micro-F1 against official gold triples

Does not retrain; only reports. Gold-pair column remains the conditional upper-bound proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.cmeie_pair import (
    allowed_label_ids,
    load_relation_contract,
    mark_pair,
    normalize_entity_type,
    signature_text,
)
from dki.bench.npu.om_manifest import sha256_tree
from dki.evidence import sha256_file

DATA = ROOT / "data" / "cblue" / "CMeIE-V2"


def _load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return {"micro_f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


def _predicate_breakdown(gold_rows, predicted_rows, predicates):
    breakdown = {}
    for predicate in predicates:
        true_positive = false_positive = false_negative = 0
        for gold, predicted in zip(gold_rows, predicted_rows):
            gold_filtered = {triple for triple in gold if triple[1] == predicate}
            predicted_filtered = {
                triple for triple in predicted if triple[1] == predicate
            }
            true_positive += len(gold_filtered & predicted_filtered)
            false_positive += len(predicted_filtered - gold_filtered)
            false_negative += len(gold_filtered - predicted_filtered)
        breakdown[predicate] = {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in _metric(
                true_positive, false_positive, false_negative,
            ).items()
        }
    return breakdown


def _score_spo_rows(gold_rows, predicted_rows):
    true_positive = false_positive = false_negative = 0
    for gold, predicted in zip(gold_rows, predicted_rows):
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    return _metric(true_positive, false_positive, false_negative)


def _decode_spo_rows(
    *,
    n_rows,
    pairs,
    allowed_rows,
    probability_rows,
    predicates,
    default_threshold,
    predicate_thresholds=None,
):
    predicted_rows = [set() for _ in range(n_rows)]
    predicted_candidate_pairs = 0
    predicted_label_activations = 0
    thresholds = predicate_thresholds or {}
    for pair, allowed, probabilities in zip(
        pairs, allowed_rows, probability_rows,
    ):
        row_index, subject, obj = pair
        hits = []
        for label_id in allowed:
            predicate = predicates[label_id]
            cutoff = thresholds.get(predicate, default_threshold)
            if cutoff is not None and probabilities[label_id] >= cutoff:
                hits.append(label_id)
        if hits:
            predicted_candidate_pairs += 1
            predicted_label_activations += len(hits)
        for label_id in hits:
            predicted_rows[row_index].add(
                (subject, predicates[label_id], obj)
            )
    return predicted_rows, {
        "n_predicted_candidate_pairs": predicted_candidate_pairs,
        "n_predicted_label_activations": predicted_label_activations,
    }


def _calibrate_threshold_policy(
    *,
    threshold_grid,
    gold_rows,
    pairs,
    allowed_rows,
    probability_rows,
    predicates,
):
    per_threshold = {}
    for threshold in threshold_grid:
        predicted_rows, decode_stats = _decode_spo_rows(
            n_rows=len(gold_rows),
            pairs=pairs,
            allowed_rows=allowed_rows,
            probability_rows=probability_rows,
            predicates=predicates,
            default_threshold=threshold,
        )
        aggregate = _score_spo_rows(gold_rows, predicted_rows)
        per_threshold[threshold] = {
            "aggregate": aggregate,
            "by_predicate": _predicate_breakdown(
                gold_rows, predicted_rows, predicates,
            ),
            "decode": decode_stats,
        }
    best_global = max(
        threshold_grid,
        key=lambda threshold: (
            per_threshold[threshold]["aggregate"]["micro_f1"],
            per_threshold[threshold]["aggregate"]["precision"],
            threshold,
        ),
    )
    selected = {predicate: best_global for predicate in predicates}
    gold_counts = {
        predicate: (
            per_threshold[best_global]["by_predicate"][predicate]["tp"]
            + per_threshold[best_global]["by_predicate"][predicate]["fn"]
        )
        for predicate in predicates
    }

    def policy_metric(policy):
        true_positive = false_positive = 0
        for predicate, threshold in policy.items():
            if threshold is None:
                continue
            stats = per_threshold[threshold]["by_predicate"][predicate]
            true_positive += stats["tp"]
            false_positive += stats["fp"]
        false_negative = sum(gold_counts.values()) - true_positive
        return _metric(true_positive, false_positive, false_negative)

    iterations = 0
    for _ in range(8):
        iterations += 1
        changed = False
        for predicate in predicates:
            current = selected[predicate]
            candidates = [None, *threshold_grid]
            best = max(
                candidates,
                key=lambda candidate: (
                    policy_metric({**selected, predicate: candidate})["micro_f1"],
                    policy_metric({**selected, predicate: candidate})["precision"],
                    -1.0 if candidate is None else candidate,
                ),
            )
            if best != current:
                selected[predicate] = best
                changed = True
        if not changed:
            break

    selected_predictions, selected_decode = _decode_spo_rows(
        n_rows=len(gold_rows),
        pairs=pairs,
        allowed_rows=allowed_rows,
        probability_rows=probability_rows,
        predicates=predicates,
        default_threshold=best_global,
        predicate_thresholds=selected,
    )
    selected_metric = _score_spo_rows(gold_rows, selected_predictions)
    report = {
        "threshold_grid": threshold_grid,
        "best_global_threshold": best_global,
        "best_global_metrics": per_threshold[best_global]["aggregate"],
        "selected_strategy": "coordinate_micro_f1_per_predicate",
        "selected_thresholds": selected,
        "selected_metrics": selected_metric,
        "coordinate_iterations": iterations,
        "global_sweep": {
            str(threshold): per_threshold[threshold]["aggregate"]
            for threshold in threshold_grid
        },
    }
    return selected_predictions, selected_decode, report


def _write_json_create_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gold_spo(row):
    out = set()
    for spo in row.get("spo_list", []):
        out.add((spo["subject"], spo["predicate"], spo["object"]))
    return out


def _entities_gold(row):
    typed_names = set()
    for spo in row.get("spo_list", []):
        for key in ("subject", "object"):
            name = spo[key]
            if name:
                typed_names.add((name, spo.get(f"{key}_type") or "Unknown"))
    ents = []
    for name, entity_type in sorted(typed_names):
        start = 0
        while True:
            index = row["text"].find(name, start)
            if index < 0:
                break
            ents.append({
                "name": name,
                "type": entity_type,
                "span": [index, index + len(name)],
            })
            start = index + 1
    return ents


def _sentence_context(text: str, span):
    if not span:
        return None
    index = 0
    start = 0
    for offset, char in enumerate(text):
        if char in "\u3002\uff01\uff1f\uff1b;.!?\n":
            if span[0] <= offset:
                return index, start, offset + 1, text[start:offset + 1]
            index += 1
            start = offset + 1
    if start < len(text) and span[0] < len(text):
        return index, start, len(text), text[start:]
    return None


def _span_gap(left_span, right_span) -> int:
    left_start, left_end = (int(value) for value in left_span)
    right_start, right_end = (int(value) for value in right_span)
    if left_start < right_end and right_start < left_end:
        return 0
    return max(left_start, right_start) - min(left_end, right_end)


def _rows_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["npu", "cuda", "cpu"])
    parser.add_argument(
        "--re-model", type=Path, default=None,
        help="CMeIE checkpoint dir (default: models/medical_re_cmeie or best candidate)",
    )
    parser.add_argument(
        "--ner-backend",
        choices=["gold", "cmeee", "cmeee-gazetteer", "cmeie-gazetteer", "dict"],
        default="gold",
    )
    parser.add_argument("--cmeee-model", type=Path, default=ROOT / "models" / "medical_ner_cmeee")
    parser.add_argument(
        "--gazetteer-min-chars", type=int, default=2,
        help="minimum CMeIE training-lexicon surface length",
    )
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ner-batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-grid",
        help="comma-separated calibration grid; train-internal only",
    )
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--threshold-map", type=Path)
    parser.add_argument("--max-pair-char-gap", type=int, default=None)
    parser.add_argument(
        "--data-split",
        choices=["dev", "train-internal"],
        default="dev",
    )
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=43)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--gold-pair-metrics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cblue" / "cmeie_dual_eval.json")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.ner_batch_size <= 0:
        parser.error("batch-size and ner-batch-size must be positive")
    if args.gazetteer_min_chars <= 0:
        parser.error("gazetteer-min-chars must be positive")
    if args.max_pair_char_gap is not None and args.max_pair_char_gap < 0:
        parser.error("max-pair-char-gap must be non-negative")
    if args.data_split == "train-internal" and not 0 < args.internal_val_ratio < 1:
        parser.error("internal-val-ratio must be between zero and one")
    if args.output.exists():
        parser.error("output is a create-once evaluation artifact")
    if args.threshold_grid and args.threshold_map:
        parser.error("threshold-grid and threshold-map are mutually exclusive")
    if args.threshold_grid and args.threshold is not None:
        parser.error("threshold-grid and threshold are mutually exclusive")
    if args.threshold_map and args.threshold is not None:
        parser.error("threshold-map and threshold are mutually exclusive")
    if args.threshold_grid and args.data_split != "train-internal":
        parser.error("threshold calibration is restricted to train-internal")
    if args.threshold_grid and args.calibration_output is None:
        parser.error("threshold-grid requires calibration-output")
    if args.calibration_output and not args.threshold_grid:
        parser.error("calibration-output requires threshold-grid")
    if args.calibration_output and args.calibration_output.exists():
        parser.error("calibration-output is a create-once artifact")
    if args.threshold_map and not args.threshold_map.is_file():
        parser.error("threshold-map does not exist")
    threshold_grid = None
    if args.threshold_grid:
        try:
            threshold_grid = sorted({
                float(value.strip())
                for value in args.threshold_grid.split(",")
                if value.strip()
            })
        except ValueError as error:
            parser.error(f"invalid threshold-grid: {error}")
        if not threshold_grid or any(
            threshold < 0 or threshold > 1 for threshold in threshold_grid
        ):
            parser.error("threshold-grid values must be between zero and one")
    if args.re_model is None:
        candidates = [
            ROOT / "models" / "medical_re_cmeie",
            ROOT / "models" / "medical_re_cmeie_tl_v3_s42",
            ROOT / "models" / "candidates" / "cmeie_multisig_fitonly_s42_20260723",
            ROOT / "models" / "candidates" / "cmeie_multisig_fitonly_s43_20260723",
        ]
        args.re_model = next((p for p in candidates if (p / "relation_config.json").is_file()), candidates[0])

    import numpy as np
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.data_split == "train-internal":
        from scripts.train_re_cmeie import _split_rows

        source_data_path = DATA / "train.jsonl"
        fit_rows, rows = _split_rows(
            _load(source_data_path),
            args.internal_val_ratio,
            args.split_seed,
        )
        gazetteer_rows = fit_rows
    else:
        source_data_path = DATA / "dev.jsonl"
        rows = _load(source_data_path)
        # The official dev rows are evaluation-only.  The gazetteer is always
        # sourced from official train, never from the rows being evaluated.
        gazetteer_rows = _load(DATA / "train.jsonl")
    if args.limit_dev > 0:
        rows = rows[: args.limit_dev]

    config = load_relation_contract(args.re_model)
    predicates = json.loads((args.re_model / "predicates.json").read_text(encoding="utf-8"))
    default_threshold = float(
        args.threshold if args.threshold is not None
        else config.get("threshold", config.get("selected_threshold", 0.5))
    )
    re_model_sha256 = sha256_tree(args.re_model.resolve())
    threshold_map_payload = None
    predicate_thresholds = None
    if args.threshold_map:
        threshold_map_payload = json.loads(
            args.threshold_map.read_text(encoding="utf-8")
        )
        if threshold_map_payload.get("re_model_sha256") != re_model_sha256:
            parser.error("threshold-map is bound to a different RE model")
        predicate_thresholds = threshold_map_payload.get("selected_thresholds")
        if not isinstance(predicate_thresholds, dict):
            parser.error("threshold-map has no selected_thresholds object")
        if set(predicate_thresholds) != set(predicates):
            parser.error("threshold-map predicate set does not match checkpoint")
        for predicate, threshold in predicate_thresholds.items():
            if threshold is not None and (
                not isinstance(threshold, (int, float))
                or threshold < 0
                or threshold > 1
            ):
                parser.error(
                    f"threshold-map has invalid value for {predicate!r}"
                )
        default_threshold = float(
            threshold_map_payload.get(
                "best_global_threshold", default_threshold,
            )
        )
    max_len = int(args.max_len or config.get("max_len", 256))
    tokenizer = AutoTokenizer.from_pretrained(args.re_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.re_model)
    model = model.to(args.device if args.device != "npu" else "npu").eval()

    cmeee = None
    if args.ner_backend in {"cmeee", "cmeee-gazetteer"}:
        from dki.bench.npu.cmeee_ner_infer import CmeeeGlobalPointerBackend
        cmeee = CmeeeGlobalPointerBackend(
            model_dir=str(args.cmeee_model), device=args.device, map_to_project=False,
        )
    gazetteer = None
    if args.ner_backend in {"cmeie-gazetteer", "cmeee-gazetteer"}:
        from dki.nlp.cmeie_gazetteer import CmeieTrainGazetteer

        gazetteer = CmeieTrainGazetteer(
            gazetteer_rows, min_chars=args.gazetteer_min_chars,
        )

    def ner_entities(row):
        if args.ner_backend == "gold":
            return _entities_gold(row)
        if args.ner_backend in {"cmeee", "cmeee-gazetteer"}:
            raw = cmeee.extract([row["text"]])[0]
            converted = [
                {
                    "name": e["mention"],
                    "type": e.get("cmeee_type") or e["type"],
                    "span": e.get("span"),
                }
                for e in raw
            ]
            return (
                gazetteer.augment_types(row["text"], converted)
                if args.ner_backend == "cmeee-gazetteer"
                else converted
            )
        if args.ner_backend == "cmeie-gazetteer":
            return gazetteer.extract(row["text"])
        # dict: longest vocabulary hits as a lightweight fallback
        from dki.operators.medical_entity_extractor.process import load_vocab, _resolve_spans
        vocab = load_vocab(None)
        hits = [w for w in vocab if w in row["text"]]
        return [
            {"name": e["name"], "type": e["type"], "span": e.get("span")}
            for e in _resolve_spans(row["text"], hits, vocab)
        ]

    # --- gold-pair column (conditional): reuse an explicitly bound report ---
    gold_pair_path = (
        args.gold_pair_metrics
        or ROOT / "outputs" / "cblue" / "cmeie_metrics.json"
    )
    gold_pair = None
    if gold_pair_path.is_file():
        gold_pair = json.loads(gold_pair_path.read_text(encoding="utf-8"))

    # --- candidate-to-SPO evaluation; globally batched across documents ---
    started = time.perf_counter()
    if args.ner_backend in {"cmeee", "cmeee-gazetteer"}:
        ner_started = time.perf_counter()
        entities_by_row = []
        for start in range(0, len(rows), args.ner_batch_size):
            raw_batch = cmeee.extract(
                [row["text"] for row in rows[start:start + args.ner_batch_size]]
            )
            for row, raw in zip(rows[start:start + args.ner_batch_size], raw_batch):
                converted = [
                    {
                        "name": entity["mention"],
                        "type": entity.get("cmeee_type") or entity["type"],
                        "span": entity.get("span"),
                    }
                    for entity in raw
                ]
                entities_by_row.append(
                    gazetteer.augment_types(row["text"], converted)
                    if args.ner_backend == "cmeee-gazetteer"
                    else converted
                )
        ner_seconds = time.perf_counter() - ner_started
    else:
        ner_started = time.perf_counter()
        entities_by_row = [ner_entities(row) for row in rows]
        ner_seconds = time.perf_counter() - ner_started

    gold_mentions_by_row = []
    gold_typed_mentions_by_row = []
    predicted_mentions_by_row = []
    predicted_typed_mentions_by_row = []
    for row, entities in zip(rows, entities_by_row):
        gold_mentions = set()
        gold_typed_mentions = set()
        for spo in row.get("spo_list", []):
            for role in ("subject", "object"):
                mention = spo.get(role)
                if not mention:
                    continue
                entity_type = normalize_entity_type(spo.get(f"{role}_type"))
                gold_mentions.add(mention)
                gold_typed_mentions.add((mention, entity_type))
        gold_mentions_by_row.append(gold_mentions)
        gold_typed_mentions_by_row.append(gold_typed_mentions)
        predicted_mentions_by_row.append({
            entity["name"] for entity in entities if entity.get("name")
        })
        predicted_typed_mentions_by_row.append({
            (entity["name"], normalize_entity_type(entity.get("type")))
            for entity in entities if entity.get("name")
        })

    def entity_metric(gold_rows, predicted_rows):
        true_positive = sum(
            len(gold & predicted)
            for gold, predicted in zip(gold_rows, predicted_rows)
        )
        false_positive = sum(
            len(predicted - gold)
            for gold, predicted in zip(gold_rows, predicted_rows)
        )
        false_negative = sum(
            len(gold - predicted)
            for gold, predicted in zip(gold_rows, predicted_rows)
        )
        return _metric(true_positive, false_positive, false_negative)

    mention_metric = entity_metric(gold_mentions_by_row, predicted_mentions_by_row)
    typed_mention_metric = entity_metric(
        gold_typed_mentions_by_row, predicted_typed_mentions_by_row,
    )

    gold_by_row = [_gold_spo(row) for row in rows]
    pairs = []
    marked = []
    signatures = []
    allowed_rows = []
    candidate_started = time.perf_counter()
    for row_index, (row, ents) in enumerate(zip(rows, entities_by_row)):
        if len(ents) >= 2:
            for i, subject in enumerate(ents):
                for j, obj in enumerate(ents):
                    if i == j or subject["name"] == obj["name"]:
                        continue
                    subject_span = subject.get("span")
                    object_span = obj.get("span")
                    subject_context = _sentence_context(row["text"], subject_span)
                    object_context = _sentence_context(row["text"], object_span)
                    if (
                        subject_context is None
                        or object_context is None
                        or subject_context[0] != object_context[0]
                    ):
                        continue
                    if (
                        args.max_pair_char_gap is not None
                        and _span_gap(subject_span, object_span) > args.max_pair_char_gap
                    ):
                        continue
                    _, sentence_start, _, sentence = subject_context
                    marked_text = mark_pair(
                        sentence,
                        subject["name"],
                        obj["name"],
                        subject_span=[
                            subject_span[0] - sentence_start,
                            subject_span[1] - sentence_start,
                        ],
                        object_span=[
                            object_span[0] - sentence_start,
                            object_span[1] - sentence_start,
                        ],
                    )
                    if not marked_text:
                        continue
                    pairs.append((row_index, subject["name"], obj["name"]))
                    marked.append(marked_text)
                    signatures.append(signature_text(subject["type"], obj["type"]))
                    allowed_rows.append(
                        allowed_label_ids(
                            config, subject["type"], obj["type"], len(predicates)
                        )
                    )
    candidate_seconds = time.perf_counter() - candidate_started

    probability_rows = []
    predicate_started = time.perf_counter()
    for start in range(0, len(marked), args.batch_size):
        batch = marked[start:start + args.batch_size]
        batch_signatures = signatures[start:start + args.batch_size]
        enc = tokenizer(
            batch, batch_signatures,
            truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        )
        enc = {key: value.to(model.device) for key, value in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.float().cpu()
        probability_rows.extend(torch.sigmoid(logits).tolist())
    predicate_seconds = time.perf_counter() - predicate_started

    calibration_report = None
    if threshold_grid:
        pred_by_row, decode_stats, calibration_report = (
            _calibrate_threshold_policy(
                threshold_grid=threshold_grid,
                gold_rows=gold_by_row,
                pairs=pairs,
                allowed_rows=allowed_rows,
                probability_rows=probability_rows,
                predicates=predicates,
            )
        )
        predicate_thresholds = calibration_report["selected_thresholds"]
        default_threshold = calibration_report["best_global_threshold"]
        calibration_payload = {
            "schema_version": 1,
            "benchmark": "CBLUE CMeIE-V2",
            "selection_scope": (
                "train_derived_text_group_disjoint_internal_text_to_spo"
            ),
            "official_dev_used_for_selection": False,
            "re_model": str(args.re_model.resolve()),
            "re_model_sha256": re_model_sha256,
            "source_data_sha256": sha256_file(source_data_path),
            "eval_rows_sha256": _rows_sha256(rows),
            "ner_backend": args.ner_backend,
            "ner_model": (
                str(args.cmeee_model.resolve())
                if args.ner_backend in {"cmeee", "cmeee-gazetteer"}
                else None
            ),
            "ner_provenance": (
                gazetteer.provenance()
                if args.ner_backend in {"cmeie-gazetteer", "cmeee-gazetteer"}
                else None
            ),
            "split_seed": args.split_seed,
            "internal_val_ratio": args.internal_val_ratio,
            **calibration_report,
        }
        _write_json_create_once(args.calibration_output, calibration_payload)
    else:
        pred_by_row, decode_stats = _decode_spo_rows(
            n_rows=len(rows),
            pairs=pairs,
            allowed_rows=allowed_rows,
            probability_rows=probability_rows,
            predicates=predicates,
            default_threshold=default_threshold,
            predicate_thresholds=predicate_thresholds,
        )

    e2e = _score_spo_rows(gold_by_row, pred_by_row)
    predicate_breakdown = _predicate_breakdown(
        gold_by_row, pred_by_row, predicates,
    )
    full_text_to_spo = args.ner_backend != "gold"
    evaluation_scope = (
        "text_to_spo_end_to_end"
        if full_text_to_spo
        else "gold_ner_conditioned_sentence_pair_to_spo"
    )
    payload = {
        "schema_version": 2,
        "benchmark": "CBLUE CMeIE-V2",
        "dataset_role": (
            "diagnostic_limited_official_dev"
            if args.limit_dev > 0 and args.data_split == "dev"
            else (
                "diagnostic_limited_train_internal"
                if args.limit_dev > 0
                else (
                    "train_derived_text_group_disjoint_internal_selection"
                    if args.data_split == "train-internal"
                    else "development_locked_report"
                )
            )
        ),
        "eval_split": (
            "train_derived_text_group_disjoint_internal"
            if args.data_split == "train-internal"
            else "official_dev"
        ),
        "n_dev": len(rows),
        "source_data_path": str(source_data_path.resolve()),
        "source_data_sha256": sha256_file(source_data_path),
        "eval_rows_sha256": _rows_sha256(rows),
        "re_model": str(args.re_model.resolve()),
        "re_model_sha256": re_model_sha256,
        "ner_provenance": (
            gazetteer.provenance()
            if args.ner_backend in {"cmeie-gazetteer", "cmeee-gazetteer"}
            else None
        ),
        "threshold_policy": {
            "mode": (
                "per_predicate_calibrated"
                if predicate_thresholds is not None
                else "global"
            ),
            "default_threshold": default_threshold,
            "predicate_thresholds": predicate_thresholds,
            "evidence_path": (
                str((args.threshold_map or args.calibration_output).resolve())
                if args.threshold_map or args.calibration_output
                else None
            ),
            "evidence_sha256": (
                sha256_file(args.threshold_map or args.calibration_output)
                if args.threshold_map or args.calibration_output
                else None
            ),
        },
        "gold_pair_metrics_path": (
            str(gold_pair_path.resolve()) if gold_pair_path.is_file() else None
        ),
        "gold_pair_metrics_sha256": (
            sha256_file(gold_pair_path) if gold_pair_path.is_file() else None
        ),
        "columns": {
            "entity_detection": {
                "mention_exact": {
                    key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in mention_metric.items()
                },
                "typed_mention_exact": {
                    key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in typed_mention_metric.items()
                },
                "note": "Document-local exact mention diagnostics on CMeIE SPO arguments.",
            },
            "predicate_on_gold_pairs": {
                "micro_f1": gold_pair.get("micro_f1") if gold_pair else None,
                "precision": gold_pair.get("precision") if gold_pair else None,
                "recall": gold_pair.get("recall") if gold_pair else None,
                "evaluation_scope": (
                    gold_pair.get("evaluation_scope")
                    if gold_pair else "gold_entity_pairs_and_gold_type_signatures_not_end_to_end_spo"
                ),
                "note": "条件化 gold-pair 上界代理，不宣称端到端 SPO",
            },
            "end_to_end_spo": {
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e2e.items()},
                "ner_backend": args.ner_backend,
                "re_threshold": (
                    "per_predicate_calibrated"
                    if predicate_thresholds is not None
                    else default_threshold
                ),
                "max_pair_char_gap": args.max_pair_char_gap,
                "evaluation_scope": evaluation_scope,
                "full_text_to_spo": full_text_to_spo,
                "note": (
                    "预测 NER→句内候选→谓词→SPO 的完整 Micro-F1"
                    if full_text_to_spo
                    else "使用 gold NER 的条件化 SPO Micro-F1，不宣称完整 text-to-SPO"
                ),
            },
        },
        "runtime": {
            "ner_seconds": round(ner_seconds, 3),
            "candidate_build_seconds": round(candidate_seconds, 3),
            "predicate_seconds": round(predicate_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
            "n_candidate_pairs": len(pairs),
            "n_allowed_candidate_pairs": sum(bool(labels) for labels in allowed_rows),
            **decode_stats,
            "n_detected_entities": sum(len(entities) for entities in entities_by_row),
            "predicate_batch_size": args.batch_size,
            "ner_batch_size": args.ner_batch_size,
        },
        "predicate_breakdown": predicate_breakdown,
        "selection_protocol": {
            "official_dev_used_for_selection": False,
            "split_seed": (
                args.split_seed if args.data_split == "train-internal" else None
            ),
            "internal_val_ratio": (
                args.internal_val_ratio
                if args.data_split == "train-internal"
                else None
            ),
            "calibration_generated": calibration_report is not None,
            "calibration_reused": threshold_map_payload is not None,
        },
    }
    _write_json_create_once(args.output, payload)
    print(json.dumps(payload["columns"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
