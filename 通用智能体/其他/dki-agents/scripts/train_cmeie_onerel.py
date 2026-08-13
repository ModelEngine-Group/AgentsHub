"""Train and select the CMeIE OneRel large-encoder candidate on Ascend NPU.

The formal route keeps every pretrained encoder weight frozen and trains only
query/value LoRA updates plus the single-stage OneRel heads.  Calibration is
the only selection split read by this script; outer and official dev paths are
intentionally absent from the interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THRESHOLDS = (-0.5, 0.0, 0.5)
FORMAL_MAX_LENGTH = 320
FORMAL_PAIR_RANK = 192
FORMAL_LORA_RANK = 8
FORMAL_LORA_ALPHA = 16.0
FORMAL_HEAD_ONLY_EPOCHS = 2
FORMAL_EPOCHS = 6
FORMAL_HARD_NEGATIVE_RATIO = 16
ONLINE_MINING_THRESHOLD = 0.0
FIXED_PROPOSAL_REFINEMENT_THRESHOLD = -0.5
FORMAL_REFINEMENT_THRESHOLD = -0.5
REFINEMENT_HARD_NEGATIVE_RATIO = 8
REFINEMENT_RAW_CANDIDATE_CAP = 100_000
REFINEMENT_SELECTED_CANDIDATE_CAP = 4096
OVERFIT_REFINEMENT_STEPS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> tuple[str, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    for item in sorted(candidate for candidate in path.iterdir() if candidate.is_file()):
        entries.append(
            {
                "path": item.name,
                "size": item.stat().st_size,
                "sha256": _sha256(item),
            }
        )
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            (
                f"{entry['path']}\0{entry['size']}\0{entry['sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest(), entries


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _save_torch_once(path: Path, payload: dict[str, Any], torch: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"checkpoint already exists: {path}")
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _gold(row: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            str(spo.get("subject", "")),
            str(spo.get("predicate", "")),
            str(spo.get("object", "")),
        )
        for spo in row.get("spo_list", [])
    }


def _predicates(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(spo.get("predicate", ""))
            for row in rows
            for spo in row.get("spo_list", [])
            if str(spo.get("predicate", ""))
        }
    )


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)


def _tokenize(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    device: Any,
    *,
    max_length: int,
) -> tuple[dict[str, Any], list[list[tuple[int, int]]], Any]:
    import torch

    encoded = tokenizer(
        [str(row.get("text", "")) for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [
        [(int(left), int(right)) for left, right in row]
        for row in encoded.pop("offset_mapping").tolist()
    ]
    inputs = {name: value.to(device) for name, value in encoded.items()}
    valid = torch.tensor(
        [
            [right > left for left, right in row_offsets]
            for row_offsets in offsets
        ],
        dtype=torch.bool,
        device=device,
    )
    return inputs, offsets, valid


def _targets(
    rows: Sequence[dict[str, Any]],
    offsets: Sequence[Sequence[tuple[int, int]]],
    predicate2id: dict[str, int],
    sequence: int,
    device: Any,
) -> tuple[Any, Any, Any, int, list[Any]]:
    import torch

    from dki.nlp.cmeie_onerel import build_onerel_labels, dense_onerel_targets

    link_rows = []
    relation_rows = []
    boundary_rows = []
    label_rows = []
    dropped = 0
    for row, row_offsets in zip(rows, offsets):
        labels = build_onerel_labels(row, row_offsets, predicate2id)
        link, relation, boundary = dense_onerel_targets(
            labels,
            sequence,
            len(predicate2id),
        )
        link_rows.append(link)
        relation_rows.append(relation)
        boundary_rows.append(boundary)
        label_rows.append(labels)
        dropped += labels.dropped_spo
    return (
        torch.stack(link_rows).to(device),
        torch.stack(relation_rows).to(device),
        torch.stack(boundary_rows).to(device),
        dropped,
        label_rows,
    )


def _verifier_batch(
    rows: Sequence[dict[str, Any]],
    offsets: Sequence[Sequence[tuple[int, int]]],
    labels: Sequence[Any],
    predicates: int,
    device: Any,
    *,
    seed: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, list[set[tuple[int, int, int, int, int]]]]:
    import torch

    from dki.nlp.cmeie_onerel import structured_triple_candidates

    parents: list[int] = []
    relation_ids: list[int] = []
    subject_starts: list[int] = []
    subject_ends: list[int] = []
    object_starts: list[int] = []
    object_ends: list[int] = []
    targets: list[float] = []
    row_candidate_sets: list[set[tuple[int, int, int, int, int]]] = []
    for parent, (row, row_offsets, row_labels) in enumerate(
        zip(rows, offsets, labels)
    ):
        valid = [
            index for index, (left, right) in enumerate(row_offsets) if right > left
        ]
        row_seed = int.from_bytes(
            hashlib.sha256(str(row.get("text", "")).encode("utf-8")).digest()[:8],
            "big",
        ) ^ int(seed)
        positives, negatives = structured_triple_candidates(
            row_labels,
            valid,
            predicates,
            seed=row_seed,
        )
        row_candidate_sets.append(set(positives) | set(negatives))
        for candidate, target in (
            *((candidate, 1.0) for candidate in positives),
            *((candidate, 0.0) for candidate in negatives),
        ):
            relation, ss, se, os_, oe = candidate
            parents.append(parent)
            relation_ids.append(relation)
            subject_starts.append(ss)
            subject_ends.append(se)
            object_starts.append(os_)
            object_ends.append(oe)
            targets.append(target)
    tensors = tuple(
        torch.tensor(values, dtype=dtype, device=device)
        for values, dtype in (
            (parents, torch.long),
            (relation_ids, torch.long),
            (subject_starts, torch.long),
            (subject_ends, torch.long),
            (object_starts, torch.long),
            (object_ends, torch.long),
            (targets, torch.float32),
        )
    )
    return (*tensors, row_candidate_sets)


def _online_verifier_batch(
    labels: Sequence[Any],
    static_candidates: Sequence[set[tuple[int, int, int, int, int]]],
    relation_logits: Any,
    boundary_logits: Any,
    link_logits: Any,
    device: Any,
    *,
    maximum_candidates_per_row: int = 4096,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, dict[str, int]]:
    import torch

    from dki.nlp.cmeie_onerel import (
        OneRelDensityError,
        enumerate_onerel_span_candidates,
        sparsify_onerel_logits,
    )

    sparse = sparsify_onerel_logits(link_logits.detach(), ONLINE_MINING_THRESHOLD)
    relation_cpu = relation_logits.detach().float().cpu().tolist()
    boundary_cpu = boundary_logits.detach().float().cpu().tolist()
    parents: list[int] = []
    relation_ids: list[int] = []
    subject_starts: list[int] = []
    subject_ends: list[int] = []
    object_starts: list[int] = []
    object_ends: list[int] = []
    targets: list[float] = []
    maximum_seen = 0
    for parent, (row_labels, excluded) in enumerate(
        zip(labels, static_candidates)
    ):
        proposed = enumerate_onerel_span_candidates(
            relation_cpu[parent],
            boundary_cpu[parent],
            sparse[parent],
            int(relation_logits.shape[1]),
            (
                ONLINE_MINING_THRESHOLD,
                ONLINE_MINING_THRESHOLD,
                ONLINE_MINING_THRESHOLD,
            ),
            maximum_span_candidates=maximum_candidates_per_row,
        )
        maximum_seen = max(maximum_seen, len(proposed))
        if len(proposed) > maximum_candidates_per_row:
            raise OneRelDensityError("online verifier candidate cap exceeded")
        gold = set(row_labels.triples)
        for item in proposed:
            coordinate = (
                item.predicate_id,
                item.subject_start,
                item.subject_end,
                item.object_start,
                item.object_end,
            )
            if coordinate in excluded:
                continue
            parents.append(parent)
            relation_ids.append(item.predicate_id)
            subject_starts.append(item.subject_start)
            subject_ends.append(item.subject_end)
            object_starts.append(item.object_start)
            object_ends.append(item.object_end)
            targets.append(float(coordinate in gold))
    tensors = tuple(
        torch.tensor(values, dtype=dtype, device=device)
        for values, dtype in (
            (parents, torch.long),
            (relation_ids, torch.long),
            (subject_starts, torch.long),
            (subject_ends, torch.long),
            (object_starts, torch.long),
            (object_ends, torch.long),
            (targets, torch.float32),
        )
    )
    return (
        *tensors,
        {
            "online_candidates": len(targets),
            "online_positive_candidates": int(sum(targets)),
            "maximum_proposals_per_row": maximum_seen,
        },
    )


def _fixed_proposal_verifier_batch(
    labels: Sequence[Any],
    relation_logits: Any,
    boundary_logits: Any,
    link_logits: Any,
    device: Any,
    *,
    threshold: float = FIXED_PROPOSAL_REFINEMENT_THRESHOLD,
    hard_negative_ratio: int = REFINEMENT_HARD_NEGATIVE_RATIO,
    maximum_raw_candidates_per_row: int = REFINEMENT_RAW_CANDIDATE_CAP,
    maximum_selected_candidates_per_row: int = REFINEMENT_SELECTED_CANDIDATE_CAP,
    retain_unproposed_gold: bool = True,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, dict[str, int]]:
    """Bind verifier supervision to the frozen proposal distribution.

    The complete proposal set is first enumerated under the declared inference
    threshold.  All exact gold coordinates and at most eight deterministic
    proposal-hard negatives per positive are then retained for training.
    Raw and selected densities are both recorded; inference is never truncated.
    """

    import torch

    from dki.nlp.cmeie_onerel import (
        OneRelDensityError,
        enumerate_onerel_span_candidates,
        select_fixed_proposal_candidates,
        sparsify_onerel_logits,
    )

    sparse = sparsify_onerel_logits(link_logits.detach(), float(threshold))
    relation_cpu = relation_logits.detach().float().cpu().tolist()
    boundary_cpu = boundary_logits.detach().float().cpu().tolist()
    columns: list[list[int]] = [[] for _ in range(6)]
    targets: list[float] = []
    maximum_raw_seen = 0
    maximum_selected_seen = 0
    raw_observations = 0
    raw_negative_observations = 0
    selected_negative_observations = 0
    excluded_unproposed_gold = 0
    gold_inserted = 0
    for parent, row_labels in enumerate(labels):
        proposed = enumerate_onerel_span_candidates(
            relation_cpu[parent],
            boundary_cpu[parent],
            sparse[parent],
            int(relation_logits.shape[1]),
            (float(threshold), float(threshold), float(threshold)),
            maximum_span_candidates=maximum_raw_candidates_per_row,
        )
        gold = set(row_labels.triples)
        coordinates, selection = select_fixed_proposal_candidates(
            proposed,
            gold,
            negative_ratio=hard_negative_ratio,
            maximum_selected_candidates=maximum_selected_candidates_per_row,
            retain_unproposed_gold=retain_unproposed_gold,
        )
        maximum_raw_seen = max(
            maximum_raw_seen, int(selection["raw_proposal_candidates"])
        )
        maximum_selected_seen = max(
            maximum_selected_seen, int(selection["selected_candidates"])
        )
        raw_observations += int(selection["raw_proposal_candidates"])
        raw_negative_observations += int(selection["raw_negative_candidates"])
        selected_negative_observations += int(
            selection["selected_negative_candidates"]
        )
        gold_inserted += int(selection["gold_inserted_outside_proposals"])
        excluded_unproposed_gold += int(
            selection["excluded_unproposed_gold_candidates"]
        )
        for relation, ss, se, os_, oe in coordinates:
            for column, value in zip(
                columns, (parent, relation, ss, se, os_, oe)
            ):
                column.append(value)
            targets.append(float((relation, ss, se, os_, oe) in gold))
    tensors = tuple(
        torch.tensor(values, dtype=torch.long, device=device) for values in columns
    )
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    return (
        *tensors,
        target_tensor,
        {
            "candidate_observations": len(targets),
            "positive_observations": int(sum(targets)),
            "raw_proposal_observations": raw_observations,
            "raw_negative_observations": raw_negative_observations,
            "selected_negative_observations": selected_negative_observations,
            "gold_inserted_outside_proposals": gold_inserted,
            "excluded_unproposed_gold_observations": excluded_unproposed_gold,
            "retain_unproposed_gold": bool(retain_unproposed_gold),
            "maximum_raw_candidates_per_row": maximum_raw_seen,
            "maximum_candidates_per_row": maximum_selected_seen,
            "raw_candidate_cap_per_row": maximum_raw_candidates_per_row,
            "candidate_cap_per_row": maximum_selected_candidates_per_row,
            "negative_ratio": int(hard_negative_ratio),
        },
    )


def _proposal_score_features(
    parents: Any,
    relations: Any,
    subject_starts: Any,
    subject_ends: Any,
    object_starts: Any,
    object_ends: Any,
    relation_logits: Any,
    boundary_logits: Any,
    link_logits: Any,
) -> Any:
    """Return continuous relation/link/boundary scores for exact candidates."""

    import torch

    from dki.nlp.cmeie_onerel import HB_TB, HB_TE, HE_TE

    if not int(parents.numel()):
        return relation_logits.new_empty((0, 3), dtype=torch.float32)
    relation_score = relation_logits[parents, relations].float()
    link_score = torch.stack(
        (
            link_logits[parents, relations, HB_TB, subject_starts, object_starts],
            link_logits[parents, relations, HB_TE, subject_starts, object_ends],
            link_logits[parents, relations, HE_TE, subject_ends, object_ends],
        ),
        dim=-1,
    ).float().amin(dim=-1)
    boundary_score = torch.stack(
        (
            boundary_logits[parents, subject_starts, 0],
            boundary_logits[parents, subject_ends, 1],
            boundary_logits[parents, object_starts, 2],
            boundary_logits[parents, object_ends, 3],
        ),
        dim=-1,
    ).float().amin(dim=-1)
    return torch.stack(
        (relation_score, link_score, boundary_score), dim=-1
    ).detach()


def _metric_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def _grid() -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        (relation, link, boundary, verifier)
        for relation in THRESHOLDS
        for link in THRESHOLDS
        for boundary in THRESHOLDS
        for verifier in THRESHOLDS
    )


def _evaluate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    predicates: list[str],
    device: Any,
    *,
    batch_size: int,
    max_length: int,
    fp16: bool,
) -> dict[str, Any]:
    import torch

    from dki.nlp.cmeie_onerel import (
        OneRelDensityError,
        decode_verified_onerel_candidates,
        enumerate_onerel_span_candidates,
        sparsify_onerel_logits,
    )

    combinations = _grid()
    counts = {combination: [0, 0, 0] for combination in combinations}
    density_rejections = {combination: 0 for combination in combinations}
    link_candidate_counts: list[int] = []
    span_candidate_counts: list[int] = []
    model.eval()
    model.encoder.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, offsets, _valid = _tokenize(
            batch,
            tokenizer,
            device,
            max_length=max_length,
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            hidden = model.encode(**inputs)
            relation_logits, boundary_logits = model.auxiliary_logits(
                hidden,
                inputs["attention_mask"],
            )
            link_logits = model.score_dense(hidden)
        sparse = sparsify_onerel_logits(link_logits, min(THRESHOLDS))
        relation_cpu = relation_logits.float().cpu().tolist()
        boundary_cpu = boundary_logits.float().cpu().tolist()
        for index, row in enumerate(batch):
            gold = _gold(row)
            row_candidates = sparse[index]
            link_candidate_counts.append(len(row_candidates))
            try:
                span_candidates = enumerate_onerel_span_candidates(
                    relation_cpu[index],
                    boundary_cpu[index],
                    row_candidates,
                    len(predicates),
                    (
                        min(THRESHOLDS),
                        min(THRESHOLDS),
                        min(THRESHOLDS),
                    ),
                )
            except OneRelDensityError:
                span_candidate_counts.append(100_001)
                for combination in combinations:
                    density_rejections[combination] += 1
                    counts[combination][2] += len(gold)
                continue
            span_candidate_counts.append(len(span_candidates))
            verifier_scores: list[float] = []
            for candidate_start in range(0, len(span_candidates), 8192):
                chunk = span_candidates[candidate_start : candidate_start + 8192]
                if not chunk:
                    continue
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=fp16,
                ):
                    scores = model.score_triples(
                        hidden[index : index + 1],
                        torch.zeros(len(chunk), dtype=torch.long, device=device),
                        torch.tensor(
                            [candidate.predicate_id for candidate in chunk],
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.tensor(
                            [candidate.subject_start for candidate in chunk],
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.tensor(
                            [candidate.subject_end for candidate in chunk],
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.tensor(
                            [candidate.object_start for candidate in chunk],
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.tensor(
                            [candidate.object_end for candidate in chunk],
                            dtype=torch.long,
                            device=device,
                        ),
                        proposal_scores=torch.tensor(
                            [
                                [
                                    candidate.relation_score,
                                    candidate.link_score,
                                    candidate.boundary_score,
                                ]
                                for candidate in chunk
                            ],
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                verifier_scores.extend(scores.cpu().tolist())
            for combination in combinations:
                try:
                    predicted = decode_verified_onerel_candidates(
                        str(row.get("text", "")),
                        offsets[index],
                        span_candidates,
                        verifier_scores,
                        predicates,
                        combination,
                    )
                except OneRelDensityError:
                    density_rejections[combination] += 1
                    predicted = set()
                counts[combination][0] += len(predicted & gold)
                counts[combination][1] += len(predicted - gold)
                counts[combination][2] += len(gold - predicted)
    metrics = {
        combination: _metric_from_counts(*counts[combination])
        for combination in combinations
    }
    valid_combinations = [
        combination for combination in combinations if density_rejections[combination] == 0
    ]
    if not valid_combinations:
        selected = combinations[-1]
    else:
        selected = max(
            valid_combinations,
            key=lambda combination: (
                metrics[combination]["f1"],
                metrics[combination]["precision"],
                sum(combination),
            ),
        )
    def density(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        if not ordered:
            return {"mean": 0.0, "p95": 0, "p99": 0, "max": 0}

        def percentile(fraction: float) -> int:
            return ordered[
                min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
            ]

        return {
            "mean": sum(ordered) / len(ordered),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(ordered),
        }

    return {
        "metric": metrics[selected],
        "thresholds": {
            "relation": selected[0],
            "link": selected[1],
            "boundary": selected[2],
            "verifier": selected[3],
        },
        "grid_size": len(combinations),
        "density_rejections": density_rejections[selected],
        "link_candidate_density_at_minimum_threshold": density(
            link_candidate_counts
        ),
        "span_candidate_density_at_minimum_thresholds": density(
            span_candidate_counts
        ),
    }


def _length_bucketed_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    maximum_optimizer_steps: int | None,
    accumulation: int,
) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    if maximum_optimizer_steps is not None:
        required = maximum_optimizer_steps * accumulation * batch_size
        repeats = math.ceil(required / max(1, len(ordered)))
        ordered = (ordered * repeats)[:required]
    window = max(batch_size, batch_size * 32)
    bucketed: list[dict[str, Any]] = []
    for start in range(0, len(ordered), window):
        bucket = ordered[start : start + window]
        bucket.sort(key=lambda row: len(str(row.get("text", ""))), reverse=True)
        bucketed.extend(bucket)
    return [
        bucketed[start : start + batch_size]
        for start in range(0, len(bucketed), batch_size)
    ]


def _gradient_norm(parameters: Iterable[Any], torch: Any) -> float:
    values = [parameter for parameter in parameters if parameter.requires_grad and parameter.grad is not None]
    if not values:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(values, 1.0).detach().cpu())


def _train(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    predicate2id: dict[str, int],
    device: Any,
    optimizer: Any,
    scaler: Any,
    torch: Any,
    *,
    batch_size: int,
    accumulation: int,
    max_length: int,
    hard_negative_ratio: int,
    fp16: bool,
    seed: int,
    lora_parameters: list[Any],
    head_parameters: list[Any],
    verifier_parameters: list[Any],
    online_mining_enabled: bool = True,
    online_warmup_optimizer_steps: int = 0,
    maximum_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    from dki.nlp.cmeie_onerel import onerel_loss

    batches = _length_bucketed_batches(
        rows,
        batch_size=batch_size,
        seed=seed,
        maximum_optimizer_steps=maximum_optimizer_steps,
        accumulation=accumulation,
    )
    model.train()
    # Dropout in the frozen backbone stays disabled.  LoRA parameters still
    # receive gradients because eval mode does not disable autograd.
    model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    parts_total = {
        "link": 0.0,
        "relation": 0.0,
        "boundary": 0.0,
        "verifier": 0.0,
    }
    dropped = 0
    attempts = optimizer_steps = skipped = nonfinite = 0
    maximum_norms = {"lora": 0.0, "head": 0.0, "verifier": 0.0}
    online_candidates = online_positive_candidates = 0
    maximum_online_proposals_per_row = 0
    for index, batch in enumerate(batches, start=1):
        group_start = ((index - 1) // accumulation) * accumulation + 1
        divisor = min(accumulation, len(batches) - group_start + 1)
        inputs, offsets, valid = _tokenize(
            batch,
            tokenizer,
            device,
            max_length=max_length,
        )
        sequence = int(inputs["input_ids"].shape[1])
        (
            link_targets,
            relation_targets,
            boundary_targets,
            lost,
            label_rows,
        ) = _targets(
            batch,
            offsets,
            predicate2id,
            sequence,
            device,
        )
        (
            verifier_parents,
            verifier_relations,
            verifier_subject_starts,
            verifier_subject_ends,
            verifier_object_starts,
            verifier_object_ends,
            verifier_targets,
            static_verifier_candidates,
        ) = _verifier_batch(
            batch,
            offsets,
            label_rows,
            len(predicate2id),
            device,
            seed=seed,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            hidden = model.encode(**inputs)
            relation_logits, boundary_logits = model.auxiliary_logits(
                hidden,
                inputs["attention_mask"],
            )
            link_logits = model.score_dense(hidden)
            if (
                not online_mining_enabled
                or optimizer_steps < online_warmup_optimizer_steps
            ):
                online_parents = verifier_parents.new_empty((0,))
                online_relations = verifier_relations.new_empty((0,))
                online_subject_starts = verifier_subject_starts.new_empty((0,))
                online_subject_ends = verifier_subject_ends.new_empty((0,))
                online_object_starts = verifier_object_starts.new_empty((0,))
                online_object_ends = verifier_object_ends.new_empty((0,))
                online_targets = verifier_targets.new_empty((0,))
                online_statistics = {
                    "online_candidates": 0,
                    "online_positive_candidates": 0,
                    "maximum_proposals_per_row": 0,
                }
            else:
                (
                    online_parents,
                    online_relations,
                    online_subject_starts,
                    online_subject_ends,
                    online_object_starts,
                    online_object_ends,
                    online_targets,
                    online_statistics,
                ) = _online_verifier_batch(
                    label_rows,
                    static_verifier_candidates,
                    relation_logits,
                    boundary_logits,
                    link_logits,
                    device,
                )
            verifier_parents = torch.cat((verifier_parents, online_parents))
            verifier_relations = torch.cat((verifier_relations, online_relations))
            verifier_subject_starts = torch.cat(
                (verifier_subject_starts, online_subject_starts)
            )
            verifier_subject_ends = torch.cat(
                (verifier_subject_ends, online_subject_ends)
            )
            verifier_object_starts = torch.cat(
                (verifier_object_starts, online_object_starts)
            )
            verifier_object_ends = torch.cat(
                (verifier_object_ends, online_object_ends)
            )
            verifier_targets = torch.cat((verifier_targets, online_targets))
            verifier_logits = model.score_triples(
                hidden.detach(),
                verifier_parents,
                verifier_relations,
                verifier_subject_starts,
                verifier_subject_ends,
                verifier_object_starts,
                verifier_object_ends,
            )
            total, parts = onerel_loss(
                link_logits,
                relation_logits,
                boundary_logits,
                verifier_logits,
                link_targets,
                relation_targets,
                boundary_targets,
                verifier_targets,
                verifier_parents,
                valid,
                hard_negative_ratio=hard_negative_ratio,
            )
        online_candidates += int(online_statistics["online_candidates"])
        online_positive_candidates += int(
            online_statistics["online_positive_candidates"]
        )
        maximum_online_proposals_per_row = max(
            maximum_online_proposals_per_row,
            int(online_statistics["maximum_proposals_per_row"]),
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError("non-finite OneRel loss")
        scaler.scale(total / divisor).backward()
        losses.append(float(total.detach().cpu()))
        dropped += lost
        for name, value in parts.items():
            parts_total[name] += float(value.cpu())
        if index % accumulation != 0 and index != len(batches):
            continue
        scaler.unscale_(optimizer)
        lora_norm = _gradient_norm(lora_parameters, torch)
        head_norm = _gradient_norm(head_parameters, torch)
        verifier_norm = _gradient_norm(verifier_parameters, torch)
        maximum_norms["lora"] = max(maximum_norms["lora"], lora_norm)
        maximum_norms["head"] = max(maximum_norms["head"], head_norm)
        maximum_norms["verifier"] = max(
            maximum_norms["verifier"], verifier_norm
        )
        bad = any(
            not math.isfinite(value)
            for value in (lora_norm, head_norm, verifier_norm)
        )
        if bad:
            nonfinite += 1
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)
        attempts += 1
        if scale_after >= scale_before:
            optimizer_steps += 1
        else:
            skipped += 1
        if bad and scale_after >= scale_before:
            raise RuntimeError("GradScaler accepted non-finite OneRel gradients")
        if maximum_optimizer_steps is not None and optimizer_steps >= maximum_optimizer_steps:
            break
    return {
        "mean_loss": sum(losses) / max(1, len(losses)),
        "mean_loss_parts": {
            name: value / max(1, len(losses)) for name, value in parts_total.items()
        },
        "micro_batches": len(losses),
        "optimizer_step_attempts": attempts,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped,
        "nonfinite_gradient_steps": nonfinite,
        "maximum_pre_clip_gradient_norms": maximum_norms,
        "dropped_spo_observations": dropped,
        "amp_scale": float(scaler.get_scale()),
        "online_hard_negative_mining": {
            "enabled": bool(online_mining_enabled),
            "candidate_observations": online_candidates,
            "unexpected_positive_observations": online_positive_candidates,
            "maximum_proposals_per_row": maximum_online_proposals_per_row,
            "candidate_cap_per_row": 4096,
            "warmup_optimizer_steps": online_warmup_optimizer_steps,
        },
    }


def _refine_verifier_on_fixed_proposals(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    predicate2id: dict[str, int],
    device: Any,
    torch: Any,
    *,
    verifier_parameters: list[Any],
    batch_size: int,
    accumulation: int,
    max_length: int,
    hard_negative_ratio: int,
    fp16: bool,
    seed: int,
    learning_rate: float,
    proposal_threshold: float = FIXED_PROPOSAL_REFINEMENT_THRESHOLD,
    maximum_optimizer_steps: int | None = None,
    retain_unproposed_gold: bool = True,
    train_verifier_dropout: bool = True,
    optimizer_weight_decay: float = 0.01,
) -> dict[str, Any]:
    """Freeze the proposal distribution and fit its exact candidate verifier."""

    from dki.nlp.cmeie_onerel import (
        build_onerel_labels,
        grouped_balanced_hard_negative_loss,
        grouped_margin_separation_loss,
        grouped_tail_anchor_loss,
    )

    fixed_distinct_overfit = (
        maximum_optimizer_steps is not None
        and batch_size == len(rows)
        and accumulation == 1
    )
    if fixed_distinct_overfit:
        # Overfit refinement must see every distinct row together on every
        # step. Repeating then length-bucketing would create batches filled by
        # duplicate copies of one row and induce catastrophic row forgetting.
        batches = [list(rows) for _ in range(maximum_optimizer_steps)]
    else:
        batches = _length_bucketed_batches(
            rows,
            batch_size=batch_size,
            seed=seed,
            maximum_optimizer_steps=maximum_optimizer_steps,
            accumulation=accumulation,
        )
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad)
        for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in verifier_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        verifier_parameters,
        lr=float(learning_rate),
        eps=1e-8,
        weight_decay=float(optimizer_weight_decay),
    )
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=fp16)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    maximum_norm = 0.0
    attempts = optimizer_steps = skipped = nonfinite = 0
    candidate_observations = positive_observations = 0
    raw_proposal_observations = raw_negative_observations = 0
    selected_negative_observations = 0
    gold_inserted = excluded_unproposed_gold = 0
    maximum_candidates = maximum_raw_candidates = 0
    cached_fixed_payload = None
    try:
        for index, batch in enumerate(batches, start=1):
            group_start = ((index - 1) // accumulation) * accumulation + 1
            divisor = min(accumulation, len(batches) - group_start + 1)
            if cached_fixed_payload is None:
                inputs, offsets, _valid = _tokenize(
                    batch,
                    tokenizer,
                    device,
                    max_length=max_length,
                )
                label_rows = [
                    build_onerel_labels(row, row_offsets, predicate2id)
                    for row, row_offsets in zip(batch, offsets)
                ]
                model.eval()
                model.encoder.eval()
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=fp16,
                ):
                    hidden = model.encode(**inputs).detach()
                    relation_logits, boundary_logits = model.auxiliary_logits(
                        hidden,
                        inputs["attention_mask"],
                    )
                    link_logits = model.score_dense(hidden)
                (
                    parents,
                    relations,
                    subject_starts,
                    subject_ends,
                    object_starts,
                    object_ends,
                    targets,
                    statistics,
                ) = _fixed_proposal_verifier_batch(
                    label_rows,
                    relation_logits,
                    boundary_logits,
                    link_logits,
                    device,
                    threshold=proposal_threshold,
                    hard_negative_ratio=hard_negative_ratio,
                    retain_unproposed_gold=retain_unproposed_gold,
                )
                proposal_features = _proposal_score_features(
                    parents,
                    relations,
                    subject_starts,
                    subject_ends,
                    object_starts,
                    object_ends,
                    relation_logits,
                    boundary_logits,
                    link_logits,
                )
                payload = (
                    hidden,
                    parents,
                    relations,
                    subject_starts,
                    subject_ends,
                    object_starts,
                    object_ends,
                    targets,
                    proposal_features,
                    statistics,
                )
                if fixed_distinct_overfit:
                    cached_fixed_payload = payload
                del inputs, relation_logits, boundary_logits, link_logits
            else:
                (
                    hidden,
                    parents,
                    relations,
                    subject_starts,
                    subject_ends,
                    object_starts,
                    object_ends,
                    targets,
                    proposal_features,
                    statistics,
                ) = cached_fixed_payload
            # Only the verifier dropout is stochastic; encoder and proposal
            # heads remain in the exact deterministic evaluation state.
            model.eval()
            model.verifier_dropout.train(bool(train_verifier_dropout))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=fp16,
            ):
                scores = model.score_triples(
                    hidden,
                    parents,
                    relations,
                    subject_starts,
                    subject_ends,
                    object_starts,
                    object_ends,
                    proposal_scores=proposal_features,
                )
                bce = grouped_balanced_hard_negative_loss(
                    scores,
                    targets,
                    parents,
                    negative_ratio=hard_negative_ratio,
                    minimum_negatives=32,
                )
                margin = grouped_margin_separation_loss(
                    scores,
                    targets,
                    parents,
                    margin=1.0,
                    negative_ratio=hard_negative_ratio,
                )
                anchor = grouped_tail_anchor_loss(
                    scores,
                    targets,
                    parents,
                    positive_floor=1.0,
                    negative_ceiling=-1.0,
                    tail_count=8,
                )
                loss = bce + margin + anchor
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite fixed proposal verifier loss")
            scaler.scale(loss / divisor).backward()
            losses.append(float(loss.detach().cpu()))
            candidate_observations += int(statistics["candidate_observations"])
            positive_observations += int(statistics["positive_observations"])
            raw_proposal_observations += int(
                statistics["raw_proposal_observations"]
            )
            raw_negative_observations += int(
                statistics["raw_negative_observations"]
            )
            selected_negative_observations += int(
                statistics["selected_negative_observations"]
            )
            gold_inserted += int(statistics["gold_inserted_outside_proposals"])
            excluded_unproposed_gold += int(
                statistics["excluded_unproposed_gold_observations"]
            )
            maximum_candidates = max(
                maximum_candidates,
                int(statistics["maximum_candidates_per_row"]),
            )
            maximum_raw_candidates = max(
                maximum_raw_candidates,
                int(statistics["maximum_raw_candidates_per_row"]),
            )
            if index % accumulation != 0 and index != len(batches):
                continue
            scaler.unscale_(optimizer)
            norm = _gradient_norm(verifier_parameters, torch)
            maximum_norm = max(maximum_norm, norm)
            if not math.isfinite(norm):
                nonfinite += 1
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            attempts += 1
            if scale_after >= scale_before:
                optimizer_steps += 1
            else:
                skipped += 1
            if not math.isfinite(norm) and scale_after >= scale_before:
                raise RuntimeError(
                    "GradScaler accepted non-finite fixed proposal gradients"
                )
            if (
                maximum_optimizer_steps is not None
                and optimizer_steps >= maximum_optimizer_steps
            ):
                break
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
        model.eval()
        model.encoder.eval()
    return {
        "mean_loss": sum(losses) / max(1, len(losses)),
        "micro_batches": len(losses),
        "optimizer_step_attempts": attempts,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped,
        "nonfinite_gradient_steps": nonfinite,
        "maximum_pre_clip_gradient_norm": maximum_norm,
        "amp_scale": float(scaler.get_scale()),
        "proposal_frozen": True,
        "encoder_frozen": True,
        "proposal_threshold": float(proposal_threshold),
        "retain_unproposed_gold": bool(retain_unproposed_gold),
        "train_verifier_dropout": bool(train_verifier_dropout),
        "optimizer_weight_decay": float(optimizer_weight_decay),
        "raw_candidate_cap_per_row": REFINEMENT_RAW_CANDIDATE_CAP,
        "candidate_cap_per_row": REFINEMENT_SELECTED_CANDIDATE_CAP,
        "negative_ratio": int(hard_negative_ratio),
        "candidate_observations": candidate_observations,
        "positive_observations": positive_observations,
        "raw_proposal_observations": raw_proposal_observations,
        "raw_negative_observations": raw_negative_observations,
        "selected_negative_observations": selected_negative_observations,
        "gold_inserted_outside_proposals": gold_inserted,
        "excluded_unproposed_gold_observations": excluded_unproposed_gold,
        "maximum_raw_candidates_per_row": maximum_raw_candidates,
        "maximum_candidates_per_row": maximum_candidates,
    }


def _clear_optimizer_state(optimizer: Any, parameters: Sequence[Any]) -> int:
    """Discard stale Adam moments after a separate verifier optimizer moved weights."""

    removed = 0
    for parameter in parameters:
        if parameter in optimizer.state:
            del optimizer.state[parameter]
            removed += 1
    return removed


def _delta_state_dict(model: Any) -> dict[str, Any]:
    result = {}
    for name, value in model.state_dict().items():
        if not name.startswith("encoder.") or ".a.weight" in name or ".b.weight" in name:
            result[name] = value.detach().cpu()
    return result


def _alignment_audit(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    predicate2id: dict[str, int],
    *,
    max_length: int,
) -> dict[str, int]:
    from dki.nlp.cmeie_onerel import build_onerel_labels

    dropped = triples = links = 0
    for row in rows:
        encoded = tokenizer(
            str(row.get("text", "")),
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
        labels = build_onerel_labels(row, offsets, predicate2id)
        dropped += labels.dropped_spo
        triples += len(labels.triples)
        links += len(labels.links)
    return {
        "rows": len(rows),
        "gold_spo": sum(len(row.get("spo_list", [])) for row in rows),
        "aligned_mention_triples": triples,
        "unique_links": links,
        "dropped_spo": dropped,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("overfit", "train", "oof-train", "refine"),
        required=True,
    )
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--overfit-receipt", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--refinement-passes", type=int, default=2)
    parser.add_argument(
        "--refine-scope",
        choices=("verifier", "calibrator", "candidate-verifier"),
        default="verifier",
    )
    parser.add_argument("--calibrator-lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--head-only-epochs", type=int, default=FORMAL_HEAD_ONLY_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--calibration-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=FORMAL_MAX_LENGTH)
    parser.add_argument("--pair-rank", type=int, default=FORMAL_PAIR_RANK)
    parser.add_argument("--lora-rank", type=int, default=FORMAL_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=FORMAL_LORA_ALPHA)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=5e-5)
    parser.add_argument("--hard-negative-ratio", type=int, default=FORMAL_HARD_NEGATIVE_RATIO)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--overfit-rows", type=int, default=8)
    parser.add_argument("--overfit-head-only-steps", type=int, default=1200)
    parser.add_argument("--overfit-joint-steps", type=int, default=100)
    parser.add_argument("--minimum-calibration-f1", type=float, default=0.45)
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_tree_sha256):
        raise SystemExit("source tree hash must be a lowercase SHA-256")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.mode == "train" and (
        args.calibration_data is None or args.overfit_receipt is None
    ):
        raise SystemExit("formal training requires calibration and overfit receipt")
    if args.mode == "oof-train" and args.calibration_data is None:
        raise SystemExit("OOF proposal training requires fit-internal validation data")
    if args.mode == "oof-train" and args.overfit_receipt is not None:
        raise SystemExit(
            "OOF proposal training must not depend on an exact-SPO overfit receipt"
        )
    if args.mode == "overfit" and args.calibration_data is not None:
        raise SystemExit("overfit mode must not receive calibration data")
    if args.mode == "refine" and (
        args.calibration_data is None or args.initial_checkpoint is None
    ):
        raise SystemExit("refine mode requires calibration data and an initial checkpoint")
    if args.mode != "refine" and args.initial_checkpoint is not None:
        raise SystemExit("only refine mode may receive an initial checkpoint")
    if args.refinement_passes != 2:
        raise SystemExit("formal low-threshold verifier refinement is fixed at two passes")
    if args.calibrator_lr != 1e-2:
        raise SystemExit("candidate-conditioned calibrator LR is fixed at 1e-2")
    if args.mode != "refine" and args.refine_scope != "verifier":
        raise SystemExit("only refine mode may select calibrator scope")
    if args.max_length != FORMAL_MAX_LENGTH or args.pair_rank != FORMAL_PAIR_RANK:
        raise SystemExit("formal OneRel dimensions are fixed at max_length=320 and pair_rank=192")
    if args.lora_rank != FORMAL_LORA_RANK or args.lora_alpha != FORMAL_LORA_ALPHA:
        raise SystemExit("formal OneRel LoRA is fixed at rank=8 alpha=16")
    if args.head_only_epochs != FORMAL_HEAD_ONLY_EPOCHS or args.epochs != FORMAL_EPOCHS:
        raise SystemExit("formal OneRel schedule is fixed at two head-only and six total epochs")
    if args.hard_negative_ratio != FORMAL_HARD_NEGATIVE_RATIO:
        raise SystemExit("formal OneRel hard-negative ratio is fixed at 16")

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_onerel import (
        CMeIEOneRel,
        ONEREL_ARCHITECTURE_VERSION,
        inject_bert_attention_lora,
    )

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("OneRel formal training requires an Ascend NPU")
    if not hasattr(torch.npu, "amp") or not hasattr(torch.npu.amp, "GradScaler"):
        raise SystemExit("OneRel requires torch.npu.amp.GradScaler")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    _seed_everything(args.seed, torch)
    fit = _jsonl(args.fit_data)
    predicates = _predicates(fit)
    if len(predicates) != 44:
        raise SystemExit(f"expected 44 fit predicates, received {len(predicates)}")
    predicate2id = {predicate: index for index, predicate in enumerate(predicates)}
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise SystemExit("OneRel requires a fast tokenizer with source offsets")
    encoder = AutoModel.from_pretrained(args.base_model, local_files_only=True)
    lora_parameters = inject_bert_attention_lora(
        encoder,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    model = CMeIEOneRel(
        encoder,
        int(encoder.config.hidden_size),
        len(predicates),
        pair_rank=args.pair_rank,
    ).to(device)
    lora_ids = {id(parameter) for parameter in lora_parameters}
    semantic_verifier_parameters = [
        parameter
        for module in (
            model.subject_span_projection,
            model.object_span_projection,
            model.predicate_embedding,
            model.verifier_norm,
            model.verifier_hidden,
            model.verifier_output,
            model.verifier_context_projection,
            model.verifier_context_norm,
            model.verifier_context_hidden,
            model.verifier_context_output,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    calibrator_parameters = list(model.proposal_calibrator.parameters())
    verifier_parameters = semantic_verifier_parameters + calibrator_parameters
    verifier_ids = {id(parameter) for parameter in verifier_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in lora_ids
        and id(parameter) not in verifier_ids
        and parameter.requires_grad
    ]
    for parameter in lora_parameters:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": 0.0,
                "eps": 1e-4,
                "weight_decay": 0.0,
            },
            {
                "params": head_parameters,
                "lr": args.head_lr,
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
            {
                "params": verifier_parameters,
                "lr": args.head_lr,
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ]
    )
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=args.fp16)
    args.output.mkdir(parents=True, exist_ok=False)
    base_model_tree_sha, base_model_entries = _tree_sha256(args.base_model)
    alignment = _alignment_audit(
        fit
        if args.mode in {"train", "oof-train", "refine"}
        else fit[: args.overfit_rows],
        tokenizer,
        predicate2id,
        max_length=args.max_length,
    )
    common = {
        "schema_version": 1,
        "model_family": "onerel",
        "architecture_version": ONEREL_ARCHITECTURE_VERSION,
        "source_tree_sha256": args.source_tree_sha256,
        "fit_sha256": _sha256(args.fit_data),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "base_model": str(args.base_model),
        "base_model_tree_sha256": base_model_tree_sha,
        "base_model_files": base_model_entries,
        "predicates": predicates,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "lora_parameter_count": sum(parameter.numel() for parameter in lora_parameters),
        "alignment": alignment,
        "device": "npu:0",
        "device_name": torch.npu.get_device_name(0),
        "fp16": bool(args.fp16),
        "optimizer_recipe": {
            "encoder_base_frozen": True,
            "encoder_eval_mode": True,
            "lora_targets": "all_bert_query_and_value",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_lr": args.lora_lr,
            "lora_adam_eps": 1e-4,
            "head_lr": args.head_lr,
            "hard_negative_ratio": args.hard_negative_ratio,
            "online_verifier_hard_negatives": True,
            "online_candidate_cap_per_row": 4096,
            "online_mining_threshold": ONLINE_MINING_THRESHOLD,
            "online_mining_initial_warmup_optimizer_steps": 256,
            "formal_online_mining": "disabled_after_density_ablation",
            "verifier_separation_margin": 1.0,
            "verifier_separation": "all_positive_x_top_hard_negative_pairwise",
            "verifier_tail_anchor": {
                "positive_floor": 1.0,
                "negative_ceiling": -1.0,
                "tail_count_per_row": 8,
            },
            "verifier_gradient_to_shared_hidden": False,
            "verifier_dropout": 0.1,
            "link_hard_negative_scope": "per_source_row",
            "fixed_proposal_verifier_refinement": {
                "proposal_threshold": FIXED_PROPOSAL_REFINEMENT_THRESHOLD,
                "raw_candidate_cap_per_row": REFINEMENT_RAW_CANDIDATE_CAP,
                "candidate_cap_per_row": REFINEMENT_SELECTED_CANDIDATE_CAP,
                "negative_ratio": REFINEMENT_HARD_NEGATIVE_RATIO,
                "negative_selection": "proposal_score_hardest_then_coordinate",
                "proposal_score_features": ["relation", "link_min", "boundary_min"],
                "proposal_calibrator": "zero_initialized_linear_tanh_scale_4",
                "all_gold_retained": args.refine_scope == "verifier",
                "candidate_conditioned_scope": args.refine_scope in {
                    "calibrator",
                    "candidate-verifier",
                },
                "inference_truncation": False,
                "proposal_and_encoder_frozen": True,
                "overfit_steps_per_stage": OVERFIT_REFINEMENT_STEPS,
                "formal_refinement_epochs": [
                    FORMAL_EPOCHS,
                ],
                "formal_proposal_threshold": FORMAL_REFINEMENT_THRESHOLD,
                "verifier_lr": args.head_lr,
                "loss": "grouped_bce_plus_pairwise_margin_plus_tail_anchor",
            },
            "gradient_clip": {
                "lora": 1.0,
                "proposal_head": 1.0,
                "verifier": 1.0,
            },
        },
        "threshold_grid": {
            "relation": THRESHOLDS,
            "link": THRESHOLDS,
            "boundary": THRESHOLDS,
            "verifier": THRESHOLDS,
            "size": len(_grid()),
        },
        "outer_accessed": False,
        "official_dev_accessed": False,
        "script_sha256": _sha256(Path(__file__)),
    }

    if args.mode == "refine":
        assert args.calibration_data is not None and args.initial_checkpoint is not None
        initial = torch.load(
            args.initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        initial_architecture = int(initial.get("architecture_version", -1))
        if (
            initial.get("model_family") != "onerel"
            or initial_architecture not in {2, 3, ONEREL_ARCHITECTURE_VERSION}
        ):
            raise SystemExit("initial checkpoint is not a migratable OneRel architecture")
        if list(initial.get("predicates", [])) != predicates:
            raise SystemExit("initial checkpoint predicate table differs from fit")
        if initial.get("base_model_tree_sha256") != base_model_tree_sha:
            raise SystemExit("initial checkpoint is bound to a different base model")
        missing, unexpected = model.load_state_dict(
            initial.get("model_delta", {}), strict=False
        )
        migration_missing: set[str] = set()
        if initial_architecture == 2:
            migration_missing.update(
                {"proposal_calibrator.weight", "proposal_calibrator.bias"}
            )
        if initial_architecture in {2, 3}:
            migration_missing.update(
                name
                for name in model.state_dict()
                if name.startswith("verifier_context_")
            )
        illegal_missing = [
            name
            for name in missing
            if (
                not name.startswith("encoder.")
                or name.endswith(".a.weight")
                or name.endswith(".b.weight")
            )
            and name not in migration_missing
        ]
        if illegal_missing or unexpected:
            raise SystemExit(
                "initial checkpoint delta mismatch: "
                f"missing={illegal_missing}, unexpected={unexpected}"
            )
        if args.refine_scope == "calibrator":
            torch.nn.init.zeros_(model.proposal_calibrator.weight)
            torch.nn.init.zeros_(model.proposal_calibrator.bias)
            active_refinement_parameters = calibrator_parameters
            refinement_learning_rate = args.calibrator_lr
            retain_unproposed_gold = False
            train_verifier_dropout = False
            refinement_weight_decay = 0.0
        elif args.refine_scope == "candidate-verifier":
            active_refinement_parameters = semantic_verifier_parameters
            refinement_learning_rate = args.head_lr
            retain_unproposed_gold = False
            train_verifier_dropout = True
            refinement_weight_decay = 0.01
        else:
            active_refinement_parameters = verifier_parameters
            refinement_learning_rate = args.head_lr
            retain_unproposed_gold = True
            train_verifier_dropout = True
            refinement_weight_decay = 0.01
        calibration = _jsonl(args.calibration_data)
        history: list[dict[str, Any]] = []
        for pass_index in range(1, args.refinement_passes + 1):
            refinement = _refine_verifier_on_fixed_proposals(
                model,
                tokenizer,
                fit,
                predicate2id,
                device,
                torch,
                verifier_parameters=active_refinement_parameters,
                batch_size=args.batch_size,
                accumulation=args.gradient_accumulation,
                max_length=args.max_length,
                hard_negative_ratio=REFINEMENT_HARD_NEGATIVE_RATIO,
                fp16=args.fp16,
                seed=args.seed * 1000 + pass_index,
                learning_rate=refinement_learning_rate,
                proposal_threshold=FORMAL_REFINEMENT_THRESHOLD,
                retain_unproposed_gold=retain_unproposed_gold,
                train_verifier_dropout=train_verifier_dropout,
                optimizer_weight_decay=refinement_weight_decay,
            )
            evaluation = _evaluate(
                model,
                tokenizer,
                calibration,
                predicates,
                device,
                batch_size=args.calibration_batch_size,
                max_length=args.max_length,
                fp16=args.fp16,
            )
            pass_dir = args.output / f"pass-{pass_index:02d}"
            pass_dir.mkdir(exist_ok=False)
            checkpoint_sha = _save_torch_once(
                pass_dir / "model.delta.pt",
                {
                    "model_delta": _delta_state_dict(model),
                    "model_family": "onerel",
                    "architecture_version": ONEREL_ARCHITECTURE_VERSION,
                    "predicates": predicates,
                    "base_model_tree_sha256": base_model_tree_sha,
                    "refinement_pass": pass_index,
                    "initial_checkpoint_sha256": _sha256(args.initial_checkpoint),
                },
                torch,
            )
            record = {
                "pass": pass_index,
                "refinement": refinement,
                "calibration": evaluation,
                "checkpoint": str(pass_dir / "model.delta.pt"),
                "checkpoint_sha256": checkpoint_sha,
            }
            _write_json_once(pass_dir / "metrics.json", record)
            history.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        best = max(
            history,
            key=lambda item: (
                item["calibration"]["metric"]["f1"],
                item["calibration"]["metric"]["precision"],
                sum(item["calibration"]["thresholds"].values()),
            ),
        )
        selection = {
            **common,
            "kind": "cmeie_onerel_low_threshold_refinement_selection",
            "mode": "refine",
            "refine_scope": args.refine_scope,
            "calibrator_reset_to_zero": args.refine_scope == "calibrator",
            "refinement_learning_rate": refinement_learning_rate,
            "seed": args.seed,
            "initial_checkpoint": str(args.initial_checkpoint),
            "initial_checkpoint_sha256": _sha256(args.initial_checkpoint),
            "calibration_sha256": _sha256(args.calibration_data),
            "selected_pass": best["pass"],
            "checkpoint": best["checkpoint"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "calibration_metric": best["calibration"]["metric"],
            "thresholds": best["calibration"]["thresholds"],
            "history": history,
            "dataset_role": "fit_refinement_calibration_selection",
            "selection_used": True,
            "calibration_gate_f1": args.minimum_calibration_f1,
            "calibration_gate_passed": (
                float(best["calibration"]["metric"]["f1"])
                >= args.minimum_calibration_f1
            ),
        }
        _write_json_once(args.output / "selection.json", selection)
        if not selection["calibration_gate_passed"]:
            _write_json_once(
                args.output / "FAILED_EVIDENCE.json",
                {
                    "status": "BLOCKED_MODEL_QUALITY",
                    "model_family": "onerel",
                    "phase": "low_threshold_verifier_refinement",
                    "seed": args.seed,
                    "calibration_metric": selection["calibration_metric"],
                    "minimum_f1": args.minimum_calibration_f1,
                    "outer_accessed": False,
                    "official_dev_accessed": False,
                },
            )
        print(json.dumps(selection, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if selection["calibration_gate_passed"] else 2

    if args.mode == "overfit":
        rows = fit[: args.overfit_rows]
        if len(rows) != 8 or sum(len(row.get("spo_list", [])) for row in rows) != 44:
            raise SystemExit("OneRel overfit requires the fixed first 8 fit rows / 44 SPOs")
        head_training = _train(
            model,
            tokenizer,
            rows,
            predicate2id,
            device,
            optimizer,
            scaler,
            torch,
            batch_size=min(args.batch_size, len(rows)),
            accumulation=1,
            max_length=args.max_length,
            hard_negative_ratio=args.hard_negative_ratio,
            fp16=args.fp16,
            seed=args.seed,
            lora_parameters=lora_parameters,
            head_parameters=head_parameters,
            verifier_parameters=verifier_parameters,
            online_warmup_optimizer_steps=256,
            maximum_optimizer_steps=args.overfit_head_only_steps,
        )
        head_pre_refinement_evaluation = _evaluate(
            model,
            tokenizer,
            rows,
            predicates,
            device,
            batch_size=1,
            max_length=args.max_length,
            fp16=args.fp16,
        )
        head_refinement = _refine_verifier_on_fixed_proposals(
            model,
            tokenizer,
            rows,
            predicate2id,
            device,
            torch,
            verifier_parameters=verifier_parameters,
            batch_size=len(rows),
            accumulation=1,
            max_length=args.max_length,
            hard_negative_ratio=REFINEMENT_HARD_NEGATIVE_RATIO,
            fp16=args.fp16,
            seed=args.seed + 10_001,
            learning_rate=args.head_lr,
            maximum_optimizer_steps=OVERFIT_REFINEMENT_STEPS,
        )
        head_refinement["main_optimizer_states_cleared"] = _clear_optimizer_state(
            optimizer, verifier_parameters
        )
        head_evaluation = _evaluate(
            model,
            tokenizer,
            rows,
            predicates,
            device,
            batch_size=1,
            max_length=args.max_length,
            fp16=args.fp16,
        )
        for parameter in lora_parameters:
            parameter.requires_grad_(True)
        optimizer.param_groups[0]["lr"] = args.lora_lr
        joint_training = _train(
            model,
            tokenizer,
            rows,
            predicate2id,
            device,
            optimizer,
            scaler,
            torch,
            batch_size=min(args.batch_size, len(rows)),
            accumulation=1,
            max_length=args.max_length,
            hard_negative_ratio=args.hard_negative_ratio,
            fp16=args.fp16,
            seed=args.seed + 1,
            lora_parameters=lora_parameters,
            head_parameters=head_parameters,
            verifier_parameters=verifier_parameters,
            online_warmup_optimizer_steps=0,
            maximum_optimizer_steps=args.overfit_joint_steps,
        )
        joint_pre_refinement_evaluation = _evaluate(
            model,
            tokenizer,
            rows,
            predicates,
            device,
            batch_size=1,
            max_length=args.max_length,
            fp16=args.fp16,
        )
        joint_refinement = _refine_verifier_on_fixed_proposals(
            model,
            tokenizer,
            rows,
            predicate2id,
            device,
            torch,
            verifier_parameters=verifier_parameters,
            batch_size=len(rows),
            accumulation=1,
            max_length=args.max_length,
            hard_negative_ratio=REFINEMENT_HARD_NEGATIVE_RATIO,
            fp16=args.fp16,
            seed=args.seed + 20_001,
            learning_rate=args.head_lr,
            maximum_optimizer_steps=OVERFIT_REFINEMENT_STEPS,
        )
        joint_refinement["main_optimizer_states_cleared"] = _clear_optimizer_state(
            optimizer, verifier_parameters
        )
        joint_evaluation = _evaluate(
            model,
            tokenizer,
            rows,
            predicates,
            device,
            batch_size=1,
            max_length=args.max_length,
            fp16=args.fp16,
        )
        checkpoint_sha = _save_torch_once(
            args.output / "model.delta.pt",
            {
                "model_delta": _delta_state_dict(model),
                "model_family": "onerel",
                "architecture_version": ONEREL_ARCHITECTURE_VERSION,
                "predicates": predicates,
                "base_model_tree_sha256": base_model_tree_sha,
            },
            torch,
        )
        lora_norm = float(joint_training["maximum_pre_clip_gradient_norms"]["lora"])
        passed = (
            float(head_evaluation["metric"]["f1"]) >= 0.98
            and float(joint_evaluation["metric"]["f1"]) >= 0.95
            and alignment["dropped_spo"] == 0
            and head_training["nonfinite_gradient_steps"] == 0
            and joint_training["nonfinite_gradient_steps"] == 0
            and head_training["skipped_optimizer_steps"] == 0
            and joint_training["skipped_optimizer_steps"] == 0
            and head_refinement["nonfinite_gradient_steps"] == 0
            and joint_refinement["nonfinite_gradient_steps"] == 0
            and head_refinement["skipped_optimizer_steps"] == 0
            and joint_refinement["skipped_optimizer_steps"] == 0
            and head_training["online_hard_negative_mining"][
                "unexpected_positive_observations"
            ]
            == 0
            and joint_training["online_hard_negative_mining"][
                "unexpected_positive_observations"
            ]
            == 0
            and math.isfinite(lora_norm)
            and lora_norm <= 10_000.0
        )
        receipt = {
            **common,
            "kind": "cmeie_onerel_overfit_preflight",
            "mode": "overfit",
            "seed": args.seed,
            "rows": len(rows),
            "spo_count": 44,
            "training": {
                "head_only": head_training,
                "head_fixed_proposal_refinement": head_refinement,
                "joint_lora": joint_training,
                "joint_fixed_proposal_refinement": joint_refinement,
            },
            "evaluation": {
                "head_only_pre_refinement": head_pre_refinement_evaluation,
                "head_only": head_evaluation,
                "joint_lora_pre_refinement": joint_pre_refinement_evaluation,
                "joint_lora": joint_evaluation,
            },
            "checkpoint_sha256": checkpoint_sha,
            "calibration_accessed": False,
            "passed": passed,
        }
        _write_json_once(args.output / "overfit_receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if passed else 2

    assert args.calibration_data is not None
    overfit_receipt_sha256 = None
    if args.mode == "train":
        assert args.overfit_receipt is not None
        overfit = json.loads(args.overfit_receipt.read_text(encoding="utf-8"))
        if not overfit.get("passed") or overfit.get("model_family") != "onerel":
            raise SystemExit("formal training requires a passing OneRel overfit receipt")
        if overfit.get("base_model_tree_sha256") != base_model_tree_sha:
            raise SystemExit("overfit receipt is bound to a different base model")
        overfit_receipt_sha256 = _sha256(args.overfit_receipt)
    calibration = _jsonl(args.calibration_data)
    history: list[dict[str, Any]] = []
    stale = 0
    best_rank: tuple[float, float, float] | None = None
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.head_only_epochs:
            for parameter in lora_parameters:
                parameter.requires_grad_(False)
            optimizer.param_groups[0]["lr"] = 0.0
            phase = "head_only"
        else:
            for parameter in lora_parameters:
                parameter.requires_grad_(True)
            optimizer.param_groups[0]["lr"] = args.lora_lr
            phase = "joint_lora"
        training = _train(
            model,
            tokenizer,
            fit,
            predicate2id,
            device,
            optimizer,
            scaler,
            torch,
            batch_size=args.batch_size,
            accumulation=args.gradient_accumulation,
            max_length=args.max_length,
            hard_negative_ratio=args.hard_negative_ratio,
            fp16=args.fp16,
            seed=args.seed * 1000 + epoch,
            lora_parameters=lora_parameters,
            head_parameters=head_parameters,
            verifier_parameters=verifier_parameters,
            online_mining_enabled=False,
            online_warmup_optimizer_steps=0,
        )
        refinement = None
        if epoch == args.epochs:
            refinement = _refine_verifier_on_fixed_proposals(
                model,
                tokenizer,
                fit,
                predicate2id,
                device,
                torch,
                verifier_parameters=verifier_parameters,
                batch_size=args.batch_size,
                accumulation=args.gradient_accumulation,
                max_length=args.max_length,
                hard_negative_ratio=REFINEMENT_HARD_NEGATIVE_RATIO,
                fp16=args.fp16,
                seed=args.seed * 1000 + epoch + 500,
                learning_rate=args.head_lr,
                proposal_threshold=FORMAL_REFINEMENT_THRESHOLD,
            )
            refinement["main_optimizer_states_cleared"] = _clear_optimizer_state(
                optimizer, verifier_parameters
            )
        evaluation = _evaluate(
            model,
            tokenizer,
            calibration,
            predicates,
            device,
            batch_size=args.calibration_batch_size,
            max_length=args.max_length,
            fp16=args.fp16,
        )
        epoch_dir = args.output / f"epoch-{epoch:02d}"
        epoch_dir.mkdir(exist_ok=False)
        checkpoint_sha = _save_torch_once(
            epoch_dir / "model.delta.pt",
            {
                "model_delta": _delta_state_dict(model),
                "model_family": "onerel",
                "architecture_version": ONEREL_ARCHITECTURE_VERSION,
                "predicates": predicates,
                "base_model_tree_sha256": base_model_tree_sha,
                "epoch": epoch,
            },
            torch,
        )
        record = {
            "epoch": epoch,
            "phase": phase,
            "training": training,
            "fixed_proposal_refinement": refinement,
            "calibration": evaluation,
            "checkpoint": str(epoch_dir / "model.delta.pt"),
            "checkpoint_sha256": checkpoint_sha,
        }
        _write_json_once(epoch_dir / "metrics.json", record)
        history.append(record)
        thresholds = evaluation["thresholds"]
        rank = (
            float(evaluation["metric"]["f1"]),
            float(evaluation["metric"]["precision"]),
            float(
                thresholds["relation"]
                + thresholds["link"]
                + thresholds["boundary"]
                + thresholds["verifier"]
            ),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            stale = 0
        elif epoch > args.head_only_epochs:
            stale += 1
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        if (
            epoch > args.head_only_epochs
            and stale >= args.early_stopping_patience
            and refinement is not None
        ):
            break
    best = max(
        history,
        key=lambda item: (
            item["calibration"]["metric"]["f1"],
            item["calibration"]["metric"]["precision"],
            sum(item["calibration"]["thresholds"].values()),
        ),
    )
    is_oof_training = args.mode == "oof-train"
    selection = {
        **common,
        "kind": (
            "cmeie_onerel_fit_internal_oof_fold_selection"
            if is_oof_training
            else "cmeie_onerel_calibration_selection"
        ),
        "mode": args.mode,
        "seed": args.seed,
        (
            "oof_validation_sha256"
            if is_oof_training
            else "calibration_sha256"
        ): _sha256(args.calibration_data),
        "overfit_receipt_sha256": overfit_receipt_sha256,
        "selected_epoch": best["epoch"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        (
            "oof_validation_metric"
            if is_oof_training
            else "calibration_metric"
        ): best["calibration"]["metric"],
        "thresholds": best["calibration"]["thresholds"],
        "history": history,
        "dataset_role": (
            "fit_internal_oof_fold_selection"
            if is_oof_training
            else "fit_training_calibration_selection"
        ),
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "oof_candidate_generator_completed": is_oof_training,
    }
    if not is_oof_training:
        selection.update(
            {
                "calibration_gate_f1": args.minimum_calibration_f1,
                "calibration_gate_passed": (
                    float(best["calibration"]["metric"]["f1"])
                    >= args.minimum_calibration_f1
                ),
            }
        )
    _write_json_once(args.output / "selection.json", selection)
    if not is_oof_training and not selection["calibration_gate_passed"]:
        _write_json_once(
            args.output / "FAILED_EVIDENCE.json",
            {
                "status": "BLOCKED_MODEL_QUALITY",
                "model_family": "onerel",
                "seed": args.seed,
                "calibration_metric": selection["calibration_metric"],
                "minimum_f1": args.minimum_calibration_f1,
                "outer_accessed": False,
                "official_dev_accessed": False,
            },
        )
    print(json.dumps(selection, ensure_ascii=False, sort_keys=True), flush=True)
    if is_oof_training:
        return 0
    return 0 if selection["calibration_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
