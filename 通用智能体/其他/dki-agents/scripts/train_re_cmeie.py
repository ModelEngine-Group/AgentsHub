"""CMeIE-V2 gold-pair relation classification with marked, multi-label pairs.

The official-dev score remains a gold-entity-pair upper-bound proxy, not
end-to-end SPO extraction. Checkpoint and threshold selection use only a
text-group-disjoint split of official train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.cmeie_pair import MARKERS, mark_pair, signature_text

DATA = ROOT / "data" / "cblue" / "CMeIE-V2"
OUT_MODEL = ROOT / "models" / "medical_re_cmeie"
OUT_METRICS = ROOT / "outputs" / "cblue" / "cmeie_metrics.json"
def _load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _predicates(rows):
    return sorted({spo["predicate"] for row in rows for spo in row.get("spo_list", [])})


def _split_rows(rows, ratio, seed):
    if not 0 < ratio < 1 or len(rows) < 2:
        raise ValueError("internal split requires at least two rows and 0 < ratio < 1")
    groups = {}
    for row in rows:
        groups.setdefault(row["text"], []).append(row)
    if len(groups) < 2:
        raise ValueError("internal split requires at least two distinct texts")
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest(),
    )
    target = max(1, min(len(rows) - 1, round(len(rows) * ratio)))
    internal, fit = [], []
    for _, group in ordered:
        (internal if len(internal) < target else fit).extend(group)
    if not fit:
        fit.extend(internal[-len(ordered[-1][1]):])
        del internal[-len(ordered[-1][1]):]
    if {row["text"] for row in fit} & {row["text"] for row in internal}:
        raise RuntimeError("text leakage across CMeIE internal split")
    return fit, internal


def _mark_pair(text, subject, obj):
    """Insert directional entity markers; preserve all original characters."""
    return mark_pair(text, subject, obj)


def _pair_examples(
    row,
    pred2id,
    neg_per_pos=1,
    rng=None,
    *,
    all_negatives=False,
    allowed_signatures=None,
):
    """Return (marked text, subject, object, label ids, type signatures)."""
    text = row["text"]
    positives = {}
    signatures = _pair_type_signatures(row)
    entity_types = {}
    entities = []
    for spo in row.get("spo_list", []):
        subject, obj, predicate = spo["subject"], spo["object"], spo["predicate"]
        if predicate not in pred2id or not subject or not obj:
            continue
        pair = (subject, obj)
        positives.setdefault(pair, set()).add(pred2id[predicate])
        subject_type = str(spo.get("subject_type", ""))
        object_type = str(spo.get("object_type", ""))
        if subject_type:
            entity_types.setdefault(subject, set()).add(subject_type)
        if object_type:
            entity_types.setdefault(obj, set()).add(object_type)
        if subject in text and subject not in entities:
            entities.append(subject)
        if obj in text and obj not in entities:
            entities.append(obj)

    def build(pair, labels):
        subject, obj = pair
        pair_signatures = signatures.get(pair)
        if not pair_signatures:
            pair_signatures = {
                (subject_type, object_type)
                for subject_type in entity_types.get(subject, {""})
                for object_type in entity_types.get(obj, {""})
            }
        pair_signatures = tuple(sorted(pair_signatures))
        if (
            allowed_signatures is not None
            and not any(signature in allowed_signatures for signature in pair_signatures)
        ):
            return None
        marked = _mark_pair(text, subject, obj)
        if marked is None:
            return None
        return (marked, subject, obj, tuple(sorted(labels)), pair_signatures)

    examples = [
        example
        for pair, labels in positives.items()
        if (example := build(pair, labels)) is not None
    ]
    negatives = []
    for subject in entities:
        for obj in entities:
            if subject != obj and (subject, obj) not in positives:
                example = build((subject, obj), ())
                if example is not None:
                    negatives.append(example)
    rng = rng or random
    rng.shuffle(negatives)
    if all_negatives:
        examples.extend(negatives)
    else:
        examples.extend(negatives[:max(1, len(positives)) * max(0, neg_per_pos)])
    return examples


def _decode_probabilities(probabilities, threshold, force_nonempty=False, allowed_labels=None):
    decoded = []
    if allowed_labels is not None and len(allowed_labels) != len(probabilities):
        raise ValueError("allowed-label constraints must match probability rows")
    for row_index, row in enumerate(probabilities):
        allowed = (
            set(range(len(row))) if allowed_labels is None
            else set(allowed_labels[row_index])
        )
        labels = {
            index for index, probability in enumerate(row)
            if index in allowed and probability >= threshold
        }
        if force_nonempty and not labels and allowed:
            labels.add(max(allowed, key=row.__getitem__))
        decoded.append(labels)
    return decoded


def _metric(gold_sets, predicted_sets, threshold):
    tp = sum(len(gold & pred) for gold, pred in zip(gold_sets, predicted_sets))
    fp = sum(len(pred - gold) for gold, pred in zip(gold_sets, predicted_sets))
    fn = sum(len(gold - pred) for gold, pred in zip(gold_sets, predicted_sets))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"threshold": threshold, "micro_f1": f1, "precision": precision,
            "recall": recall, "tp": tp, "fp": fp, "fn": fn,
            "pred_empty": sum(not pred for pred in predicted_sets),
            "pred_rel": sum(len(pred) for pred in predicted_sets)}


def _select_threshold(records, fixed=0.5):
    return max(records, key=lambda item: (
        item["micro_f1"], -abs(item["threshold"] - fixed),
        item["precision"], item["threshold"],
    ))


def _pair_type_signatures(row):
    """Return every official type signature attached to each entity pair.

    CMeIE occasionally annotates the same surface pair with different object
    types and predicates.  Keeping only the lexicographically first signature
    silently makes the remaining gold predicates inadmissible at decode time.
    """
    signatures = {}
    for spo in row.get("spo_list", []):
        subject, obj = spo.get("subject"), spo.get("object")
        if not subject or not obj:
            continue
        signatures.setdefault((subject, obj), set()).add((
            str(spo.get("subject_type", "")),
            str(spo.get("object_type", "")),
        ))
    return signatures


def _allowed_labels(signatures, signature_labels, n_labels):
    """Union admissible predicates across all signatures for a gold pair."""
    allowed = set()
    for signature in signatures:
        allowed.update(signature_labels.get(tuple(signature), ()))
    return allowed


def _build_signature_labels(rows, pred2id):
    """Build decode constraints from the fit partition only.

    The internal validation fold must not contribute predicate admissibility:
    doing so leaks validation annotations into checkpoint/threshold selection.
    Unknown validation/dev signatures deliberately fall back to all labels.
    """
    signature_labels = {}
    for row in rows:
        for spo in row.get("spo_list", []):
            predicate = spo.get("predicate")
            if predicate not in pred2id:
                continue
            signature = (
                str(spo.get("subject_type", "")),
                str(spo.get("object_type", "")),
            )
            signature_labels.setdefault(signature, set()).add(pred2id[predicate])
    return signature_labels


def _is_same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _promote_model_directory(output_model: Path, write_artifacts) -> None:
    output_model.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_model.name}.staging-", dir=output_model.parent))
    backup = output_model.with_name(f".{output_model.name}.backup-{os.getpid()}")
    try:
        write_artifacts(staging)
        if backup.exists():
            shutil.rmtree(backup)
        if output_model.exists():
            os.replace(output_model, backup)
        try:
            os.replace(staging, output_model)
        except BaseException:
            if backup.exists() and not output_model.exists():
                os.replace(backup, output_model)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-per-pos", type=int, default=0,
                        help="auxiliary pair-existence negatives per positive; gold-pair default is zero")
    parser.add_argument(
        "--all-negatives",
        action="store_true",
        help="train and calibrate on every sentence-local fit-ontology candidate pair",
    )
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output-model", type=Path, default=OUT_MODEL)
    parser.add_argument("--output-metrics", type=Path, default=OUT_METRICS)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.neg_per_pos < 0:
        parser.error("epochs/batch-size must be positive and neg-per-pos non-negative")
    diagnostic_run = args.limit_train > 0 or args.limit_dev > 0
    if diagnostic_run and (
        _is_same_path(args.output_model, OUT_MODEL)
        or _is_same_path(args.output_metrics, OUT_METRICS)
    ):
        parser.error(
            "limited runs require explicit noncanonical --output-model and "
            "--output-metrics paths"
        )

    import numpy as np
    import torch
    import torch.nn.functional as F
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "npu" else "npu")
    all_rows = _load(DATA / "train.jsonl")
    predicates = _predicates(all_rows)
    pred2id = {predicate: index for index, predicate in enumerate(predicates)}
    rows = all_rows[:args.limit_train] if args.limit_train else all_rows
    train_rows, internal_rows = _split_rows(rows, args.internal_val_ratio, args.seed)
    signature_labels = _build_signature_labels(train_rows, pred2id)
    rng = random.Random(args.seed)
    train_samples = [
        example for row in train_rows
        for example in _pair_examples(
            row,
            pred2id,
            args.neg_per_pos,
            rng,
            all_negatives=args.all_negatives,
            allowed_signatures=set(signature_labels),
        )
    ]
    internal_samples = [
        example for row in internal_rows
        for example in _pair_examples(
            row,
            pred2id,
            args.neg_per_pos,
            rng,
            all_negatives=args.all_negatives,
            allowed_signatures=set(signature_labels),
        )
    ]
    rng.shuffle(train_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": list(MARKERS)})
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(predicates), problem_type="multi_label_classification",
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    positive_counts = torch.zeros(len(predicates))
    for _, _, _, labels, _ in train_samples:
        for label in labels:
            positive_counts[label] += 1
    negatives = len(train_samples) - positive_counts
    pos_weight = (negatives / positive_counts.clamp_min(1)).clamp(1.0, args.pos_weight_cap).to(device)

    def encode(batch):
        marked = [item[0] for item in batch]
        signatures = [
            "；".join(signature_text(subject_type, object_type) for subject_type, object_type in item[4])
            for item in batch
        ]
        encoded = tokenizer(
            marked, signatures, truncation=True, max_length=args.max_len,
            padding=True, return_tensors="pt",
        )
        labels = torch.zeros(len(batch), len(predicates), dtype=torch.float32)
        for row_index, item in enumerate(batch):
            for label in item[3]:
                labels[row_index, label] = 1.0
        return {key: value.to(device) for key, value in encoded.items()}, labels.to(device)

    def predict(samples, thresholds, force_nonempty=False):
        probabilities, gold_sets = [], []
        allowed_sets = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start:start + args.batch_size]
                encoded, _ = encode(batch)
                probabilities.extend(model(**encoded).logits.sigmoid().cpu().tolist())
                gold_sets.extend(set(item[3]) for item in batch)
                allowed_sets.extend(
                    _allowed_labels(item[4], signature_labels, len(predicates))
                    for item in batch
                )
        return [
            _metric(
                gold_sets,
                _decode_probabilities(
                    probabilities, threshold,
                    force_nonempty=force_nonempty,
                    allowed_labels=allowed_sets,
                ),
                threshold,
            )
            for threshold in thresholds
        ]

    thresholds = tuple(round(value / 100, 2) for value in range(30, 71, 5))
    best_state, best_epoch, best_internal = None, None, None
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_samples)
        total_loss = 0.0
        for start in range(0, len(train_samples), args.batch_size):
            batch = train_samples[start:start + args.batch_size]
            encoded, labels = encode(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CMeIE loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
        internal = _select_threshold(
            predict(internal_samples, thresholds, force_nonempty=False)
        )
        if best_internal is None or internal["micro_f1"] > best_internal["micro_f1"]:
            best_internal, best_epoch = internal, epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        print(
            f"[epoch {epoch + 1}/{args.epochs}] loss={total_loss / max(1, len(train_samples) / args.batch_size):.4f} "
            f"internal_f1={internal['micro_f1']:.4f} threshold={internal['threshold']:.2f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("no CMeIE checkpoint selected")
    model.load_state_dict(best_state)
    selected_threshold = best_internal["threshold"]

    # Frozen candidate: official dev is loaded only now, and only gold pairs are scored.
    dev_rows = _load(DATA / "dev.jsonl")
    if args.limit_dev:
        dev_rows = dev_rows[:args.limit_dev]
    dev_samples = []
    gold_spo = []
    for row in dev_rows:
        pairs = {}
        signatures = _pair_type_signatures(row)
        for spo in row.get("spo_list", []):
            predicate = spo["predicate"]
            if predicate not in pred2id:
                raise ValueError(f"official dev contains unknown predicate: {predicate!r}")
            pair = (spo["subject"], spo["object"])
            pairs.setdefault(pair, set()).add(pred2id[predicate])
        for (subject, obj), labels in pairs.items():
            pair_signatures = tuple(sorted(signatures.get((subject, obj), {("", "")})))
            marked = _mark_pair(row["text"], subject, obj)
            if marked is None:
                continue
            dev_samples.append((marked, subject, obj, tuple(sorted(labels)), pair_signatures))
            gold_spo.append({(subject, label, obj) for label in labels})

    # Compute both fixed and internal-selected operating points in one frozen pass.
    records = predict(dev_samples, sorted({0.5, selected_threshold}), force_nonempty=True)
    by_threshold = {record["threshold"]: record for record in records}
    fixed, selected = by_threshold[0.5], by_threshold[selected_threshold]
    metrics = {
        "schema_version": 2,
        "benchmark": "CBLUE CMeIE-V2 official dev, gold-pair multi-label classification",
        "dataset_role": (
            "diagnostic_limited_official_dev"
            if diagnostic_run else "development_locked_report"
        ),
        "evaluation_scope": (
            "gold_entity_pairs_and_gold_type_signatures_not_end_to_end_spo"
        ),
        "gold_type_signature_conditioning": True,
        "gold_pair_decode": (
            "fit-ontology-constrained thresholded multi-label with mandatory top-1 predicate"
        ),
        "selection_protocol": {
            "checkpoint_and_threshold": (
                "train-derived text-group-disjoint internal sentence-local "
                "candidate pairs including negatives"
            ),
            "official_dev_used_for_selection": False,
        },
        "base_model": args.base_model, "n_predicates": len(predicates),
        "n_fit_type_signatures": len(signature_labels),
        "schema_constraint_scope": "fit_partition_only",
        "n_train_pairs": len(train_samples), "n_internal_pairs": len(internal_samples),
        "n_dev": len(dev_rows), "n_dev_gold_pairs": len(dev_samples),
        "best_epoch": best_epoch, "selected_threshold": selected_threshold,
        "micro_f1": round(selected["micro_f1"], 4),
        "precision": round(selected["precision"], 4), "recall": round(selected["recall"], 4),
        "tp": selected["tp"], "fp": selected["fp"], "fn": selected["fn"],
        "pred_na": selected["pred_empty"], "pred_rel": selected["pred_rel"],
        "fixed_threshold_0_5": {
            key: round(value, 4) if isinstance(value, float) else value for key, value in fixed.items()
        },
        "train_seconds": round(time.time() - started, 1),
        "train_config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_len": args.max_len, "neg_per_pos": args.neg_per_pos,
            "all_negatives": args.all_negatives,
            "internal_val_ratio": args.internal_val_ratio,
            "pos_weight_cap": args.pos_weight_cap, "seed": args.seed,
            "entity_markers": list(MARKERS), "objective": "multi_label_bce",
            "candidate_scope": "sentence_local_real_mentions",
            "threshold_selection_scope": "positive_and_negative_candidate_pairs",
            "schema_constraints": "fit-partition-only subject_type/object_type admissible predicates",
            "gold_type_signature_conditioning": True,
        },
        "cblue_baseline_ref": {"RoBERTa/ERNIE": "~55-60"},
        "note": "Gold entity-pair relation classification is an upper-bound proxy; end-to-end SPO is not claimed.",
    }
    output_metrics, output_model = args.output_metrics, args.output_model
    relation_config = {
        "schema_version": 1, "threshold": selected_threshold, "multi_label": True,
        "max_len": args.max_len,
        "markers": MARKERS,
        "evaluation_scope": "sentence_local_candidate_pairs_with_negative_calibration",
        "candidate_scope": "sentence_local_real_mentions",
        "all_negatives": args.all_negatives,
        "schema_constraint_scope": "fit_partition_only",
        "schema_constraints": {
            f"{subject_type}\t{object_type}": sorted(labels)
            for (subject_type, object_type), labels in signature_labels.items()
        },
    }

    def write_model(staging: Path) -> None:
        model.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
        (staging / "predicates.json").write_text(
            json.dumps(predicates, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (staging / "relation_config.json").write_text(
            json.dumps(relation_config, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    _promote_model_directory(output_model, write_model)
    _write_json_atomic(output_metrics, metrics)
    print(json.dumps({key: metrics[key] for key in (
        "micro_f1", "precision", "recall", "tp", "fp", "fn", "pred_na", "pred_rel",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
