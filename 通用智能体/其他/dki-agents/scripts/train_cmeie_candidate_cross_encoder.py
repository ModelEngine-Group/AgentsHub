"""Train an exact-triple binary cross-encoder on real OneRel proposals.

The extractor remains frozen.  Training labels only the fit partition's exact
surface SPOs, samples at most eight proposal-hard negatives per reachable
positive, and selects one global candidate threshold on calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence import sha256_file
from dki.nlp.cmeie_candidate_reranker import (
    PairRerankCandidate,
    build_candidate_hard_negative_groups,
    decode_pair_rerank_candidates,
    micro_spo_metric,
)
from dki.nlp.cmeie_pair import MARKERS
from scripts.eval_cmeie_onerel_pair_reranker import (
    _base_model_tree_sha256,
    _generate_candidates,
    _gold,
    _jsonl,
    _load_onerel,
)


MAX_LENGTH = 320
NEGATIVE_RATIO = 8
THRESHOLDS = tuple(round(value / 20.0, 2) for value in range(1, 20))
PROPOSAL_RELATION_THRESHOLDS = (-math.inf, -0.5, 0.0, 0.5)
PROPOSAL_LINK_THRESHOLDS = (-0.5, 0.0, 0.5)
PROPOSAL_BOUNDARY_THRESHOLDS = (-math.inf, -0.5, 0.0, 0.5)
SURFACE_CANDIDATE_CAP = 4096
HEAD_LR = 5e-5
ENCODER_LR = 1e-6
OVERFIT_HEAD_LR = 1e-3
OVERFIT_JOINT_HEAD_LR = 1e-4
OVERFIT_ENCODER_LR = 1e-6
OVERFIT_HEAD_STEPS = 1200
OVERFIT_JOINT_STEPS = 100
CANDIDATE_HIDDEN_SIZE = 256
PREDICATE_EMBEDDING_SIZE = 64
PROPOSAL_FEATURE_SIZE = 3
MARGIN = 1.0
MARGIN_WEIGHT = 0.5
LISTWISE_WEIGHT = 0.5
PAIR_MARGIN_WEIGHT = 0.5
RELATION_MARGIN_WEIGHT = 0.5
RELATION_AUX_WEIGHT = 0.5
PAIR_AUX_WEIGHT = 0.5
SUBJECT_AUX_WEIGHT = 0.5
OBJECT_AUX_WEIGHT = 0.5
PREDICATE_PRESENCE_AUX_WEIGHT = 0.5
FACTOR_HEAD_VERSION = 2


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _save_torch_once(path: Path, payload: dict[str, Any], torch: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
        return sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


def _encoder_texts(
    candidates: Sequence[PairRerankCandidate], indices: Sequence[int]
) -> list[str]:
    """Return predicate-independent sentence views for the shared encoder.

    The proposed predicate is supplied separately through ``predicate_ids``.
    Putting its text in BERT's second sequence allowed both the supposedly
    predicate-agnostic pair head and the 44-way relation head to see the answer
    proposed by the noisy extractor.  In particular, a wrong-predicate
    candidate changed the entity-pair representation itself.  Keeping the
    contextual span representation invariant across proposed predicates makes
    the pair and relation auxiliary tasks match their declared contracts.
    """

    return [candidates[index].marked_text for index in indices]


def _restore_unmarked_sentence(marked_text: str) -> str:
    """Remove exactly one copy of every candidate marker, fail-closed."""

    restored = str(marked_text)
    for marker in MARKERS:
        if restored.count(marker) != 1:
            raise ValueError(
                "candidate marked text must contain every marker exactly once"
            )
        restored = restored.replace(marker, "", 1)
    if not restored:
        raise ValueError("candidate marked text restores to an empty sentence")
    return restored


def _proposal_feature_row(
    candidate: PairRerankCandidate,
) -> tuple[float, float, float]:
    values = (
        float(candidate.relation_score),
        float(candidate.link_score),
        float(candidate.boundary_score),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("candidate proposal features must be finite")
    return tuple(math.tanh(value) for value in values)


def _encode_candidates(
    tokenizer: Any,
    candidates: Sequence[PairRerankCandidate],
    indices: Sequence[int],
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    marked_texts = _encoder_texts(candidates, indices)
    encoded = tokenizer(
        marked_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    result = {name: value.to(device) for name, value in encoded.items()}
    unique_sentences: list[str] = []
    sentence_to_index: dict[str, int] = {}
    sentence_parent_indices: list[int] = []
    for marked_text in marked_texts:
        sentence = _restore_unmarked_sentence(marked_text)
        parent = sentence_to_index.get(sentence)
        if parent is None:
            parent = len(unique_sentences)
            sentence_to_index[sentence] = parent
            unique_sentences.append(sentence)
        sentence_parent_indices.append(parent)
    sentence_encoded = tokenizer(
        unique_sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    for name, value in sentence_encoded.items():
        result[f"sentence_{name}"] = value.to(device)
    result["sentence_parent_indices"] = torch.tensor(
        sentence_parent_indices,
        dtype=torch.long,
        device=device,
    )
    result["predicate_ids"] = torch.tensor(
        [candidates[index].predicate_id for index in indices],
        dtype=torch.long,
        device=device,
    )
    result["proposal_features"] = torch.tensor(
        [_proposal_feature_row(candidates[index]) for index in indices],
        dtype=torch.float32,
        device=device,
    )
    return result


def _layers_and_embeddings(model: Any) -> tuple[list[Any], Any]:
    encoder = getattr(model, model.base_model_prefix)
    layers = list(encoder.encoder.layer)
    if len(layers) < 2:
        raise ValueError("cross-encoder base model has fewer than two layers")
    return layers[-2:], encoder.embeddings


def _install_candidate_classifier(
    model: Any,
    torch: Any,
    *,
    input_multiplier: int = 1,
    extra_size: int = 0,
) -> None:
    """Install the declared nonlinear reranker head."""

    hidden_size = int(model.config.hidden_size)
    input_size = hidden_size * input_multiplier + int(extra_size)
    model.classifier = torch.nn.Sequential(
        torch.nn.LayerNorm(input_size),
        torch.nn.Linear(input_size, CANDIDATE_HIDDEN_SIZE),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(CANDIDATE_HIDDEN_SIZE, 1),
    )


def _pool_marked_span_features(
    *,
    hidden: Any,
    input_ids: Any,
    marker_token_ids: Sequence[int],
    torch: Any,
) -> tuple[list[Any], Any, Any]:
    """Return marker vectors plus the subject/object token-content means.

    Marker-only pooling makes the formal classifier depend heavily on four
    newly-added token embeddings.  Those embeddings are deliberately frozen
    during the head-only phase, so the old representation could memorize the
    eight-row feasibility set without learning a useful semantic comparison.
    Pooling the tokens *inside* each marked span exposes the pretrained entity
    content directly.  The strict marker and non-empty-span checks keep this
    fail-closed when truncation or malformed inputs invalidate a candidate.
    """

    if len(marker_token_ids) != 4 or len(set(marker_token_ids)) != 4:
        raise ValueError("span-pair pooling requires four distinct marker token ids")
    if hidden.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("span-pair pooling received invalid tensor ranks")
    if hidden.shape[:2] != input_ids.shape:
        raise ValueError("span-pair hidden states do not align with input ids")

    batch = torch.arange(hidden.shape[0], device=hidden.device)
    positions: list[Any] = []
    marker_vectors: list[Any] = []
    for marker_id in marker_token_ids:
        marker_mask = input_ids.eq(int(marker_id))
        counts = marker_mask.sum(dim=1)
        if not bool(torch.all(counts == 1)):
            raise ValueError(
                "candidate tokenization must preserve each span marker exactly once"
            )
        marker_position = marker_mask.to(dtype=torch.int64).argmax(dim=1)
        positions.append(marker_position)
        marker_vectors.append(hidden[batch, marker_position])

    subject_open, subject_close, object_open, object_close = positions
    if not bool(torch.all(subject_close > subject_open + 1)):
        raise ValueError("candidate subject marker span contains no tokens")
    if not bool(torch.all(object_close > object_open + 1)):
        raise ValueError("candidate object marker span contains no tokens")

    token_positions = torch.arange(
        hidden.shape[1], device=hidden.device
    ).unsqueeze(0)

    def _mean_between(left: Any, right: Any) -> Any:
        mask = (token_positions > left.unsqueeze(1)) & (
            token_positions < right.unsqueeze(1)
        )
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / denominator

    return (
        marker_vectors,
        _mean_between(subject_open, subject_close),
        _mean_between(object_open, object_close),
    )


def _fuse_auxiliary_logits(
    *,
    verifier_logits: Any,
    scaled_logits: Sequence[tuple[Any, Any]],
    torch: Any,
) -> Any:
    """Fuse necessary-condition logits with independently learned gates.

    Every auxiliary head starts with a zero-initialized final layer, so this
    function preserves the exact verifier at step zero.  A single learned
    scalar per factor lets training use subject role, object role, pair
    existence, sentence-level predicate presence, and pair-level relation
    choice without introducing per-predicate decoding thresholds.
    """

    fused = verifier_logits
    for scale_logit, factor_logits in scaled_logits:
        if factor_logits.shape != verifier_logits.shape:
            raise ValueError("auxiliary factor logits differ from verifier logits")
        fused = fused + torch.sigmoid(scale_logit) * factor_logits
    return fused


def _build_span_pair_model(
    *,
    base_model: Path,
    marker_token_ids: Sequence[int],
    n_predicates: int,
    torch: Any,
    auto_model: Any,
    factor_head_version: int = FACTOR_HEAD_VERSION,
) -> Any:
    """Build a candidate model that explicitly pools all four span markers."""

    if len(marker_token_ids) != 4 or len(set(marker_token_ids)) != 4:
        raise ValueError("span-pair model requires four distinct marker token ids")
    if n_predicates < 1:
        raise ValueError("span-pair model requires at least one predicate")
    if factor_head_version not in (1, FACTOR_HEAD_VERSION):
        raise ValueError("unsupported candidate factor-head version")

    class SpanPairCrossEncoder(torch.nn.Module):
        base_model_prefix = "encoder"

        def __init__(self) -> None:
            super().__init__()
            self.encoder = auto_model.from_pretrained(
                base_model,
                local_files_only=True,
            )
            self.config = self.encoder.config
            self.factor_head_version = int(factor_head_version)
            self.marker_token_ids = tuple(int(value) for value in marker_token_ids)
            self.predicate_embedding = torch.nn.Embedding(
                n_predicates, PREDICATE_EMBEDDING_SIZE
            )
            hidden_size = int(self.config.hidden_size)
            self.relation_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size * 4),
                torch.nn.Linear(hidden_size * 4, CANDIDATE_HIDDEN_SIZE),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(CANDIDATE_HIDDEN_SIZE, n_predicates),
            )
            torch.nn.init.zeros_(self.relation_head[-1].weight)
            torch.nn.init.zeros_(self.relation_head[-1].bias)
            self.pair_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size * 4 + 2),
                torch.nn.Linear(
                    hidden_size * 4 + 2, CANDIDATE_HIDDEN_SIZE
                ),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(CANDIDATE_HIDDEN_SIZE, 1),
            )
            torch.nn.init.zeros_(self.pair_head[-1].weight)
            torch.nn.init.zeros_(self.pair_head[-1].bias)
            self.relation_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
            self.pair_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
            if self.factor_head_version >= 2:
                def binary_role_head() -> Any:
                    head = torch.nn.Sequential(
                        torch.nn.LayerNorm(hidden_size * 3),
                        torch.nn.Linear(
                            hidden_size * 3, CANDIDATE_HIDDEN_SIZE
                        ),
                        torch.nn.GELU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(CANDIDATE_HIDDEN_SIZE, 1),
                    )
                    torch.nn.init.zeros_(head[-1].weight)
                    torch.nn.init.zeros_(head[-1].bias)
                    return head

                self.subject_head = binary_role_head()
                self.object_head = binary_role_head()
                self.predicate_presence_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, CANDIDATE_HIDDEN_SIZE),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(CANDIDATE_HIDDEN_SIZE, n_predicates),
                )
                torch.nn.init.zeros_(self.predicate_presence_head[-1].weight)
                torch.nn.init.zeros_(self.predicate_presence_head[-1].bias)
                self.subject_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
                self.object_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
                self.predicate_presence_scale_logit = torch.nn.Parameter(
                    torch.tensor(0.0)
                )
            self.classifier = torch.nn.Identity()
            _install_candidate_classifier(
                self,
                torch,
                input_multiplier=9,
                extra_size=PREDICATE_EMBEDDING_SIZE + PROPOSAL_FEATURE_SIZE,
            )

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any | None = None,
            token_type_ids: Any | None = None,
            predicate_ids: Any | None = None,
            proposal_features: Any | None = None,
            sentence_input_ids: Any | None = None,
            sentence_attention_mask: Any | None = None,
            sentence_token_type_ids: Any | None = None,
            sentence_parent_indices: Any | None = None,
            **_unused: Any,
        ) -> Any:
            if predicate_ids is None or proposal_features is None:
                raise ValueError("candidate auxiliary features are required")
            if proposal_features.ndim != 2 or proposal_features.shape[1] != PROPOSAL_FEATURE_SIZE:
                raise ValueError("candidate proposal features have invalid shape")
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            marker_vectors, subject_vector, object_vector = (
                _pool_marked_span_features(
                    hidden=hidden,
                    input_ids=input_ids,
                    marker_token_ids=self.marker_token_ids,
                    torch=torch,
                )
            )
            pooled = [
                hidden[:, 0],
                *marker_vectors,
                subject_vector,
                object_vector,
                subject_vector * object_vector,
                torch.abs(subject_vector - object_vector),
            ]
            predicate_vectors = self.predicate_embedding(predicate_ids).to(hidden.dtype)
            auxiliary = proposal_features.to(hidden.dtype)
            verifier_logits = self.classifier(
                torch.cat([*pooled, predicate_vectors, auxiliary], dim=-1)
            )
            pair_features = torch.cat(
                [
                    subject_vector,
                    object_vector,
                    subject_vector * object_vector,
                    torch.abs(subject_vector - object_vector),
                ],
                dim=-1,
            )
            relation_logits = self.relation_head(pair_features)
            # Pair existence is deliberately predicate-agnostic at its output:
            # it receives only the contextual pair representation plus the
            # OneRel link/boundary evidence.  The predicate-specific proposal
            # score remains in the exact verifier, avoiding leakage of the
            # proposed relation into the auxiliary pair label.
            pair_logits = self.pair_head(
                torch.cat([pair_features, auxiliary[:, 1:]], dim=-1)
            )
            selected_relation_logits = relation_logits.gather(
                1, predicate_ids.unsqueeze(1)
            )
            subject_logits = object_logits = predicate_presence_logits = None
            factors = [
                (self.pair_scale_logit, pair_logits),
                (self.relation_scale_logit, selected_relation_logits),
            ]
            if self.factor_head_version >= 2:
                if sentence_input_ids is None or sentence_parent_indices is None:
                    raise ValueError("factor-head v2 requires unmarked sentence inputs")
                if (
                    sentence_parent_indices.ndim != 1
                    or sentence_parent_indices.shape[0] != input_ids.shape[0]
                ):
                    raise ValueError("sentence parent indices differ from candidates")
                if not bool(
                    torch.all(
                        (sentence_parent_indices >= 0)
                        & (sentence_parent_indices < sentence_input_ids.shape[0])
                    )
                ):
                    raise ValueError("sentence parent index is out of range")
                sentence_outputs = self.encoder(
                    input_ids=sentence_input_ids,
                    attention_mask=sentence_attention_mask,
                    token_type_ids=sentence_token_type_ids,
                    return_dict=True,
                )
                sentence_cls = sentence_outputs.last_hidden_state[
                    :, 0
                ][sentence_parent_indices]
                subject_logits = self.subject_head(
                    torch.cat(
                        [marker_vectors[0], marker_vectors[1], subject_vector],
                        dim=-1,
                    )
                )
                object_logits = self.object_head(
                    torch.cat(
                        [marker_vectors[2], marker_vectors[3], object_vector],
                        dim=-1,
                    )
                )
                predicate_presence_logits = self.predicate_presence_head(
                    sentence_cls
                )
                selected_presence_logits = predicate_presence_logits.gather(
                    1, predicate_ids.unsqueeze(1)
                )
                factors.extend(
                    [
                        (self.subject_scale_logit, subject_logits),
                        (self.object_scale_logit, object_logits),
                        (
                            self.predicate_presence_scale_logit,
                            selected_presence_logits,
                        ),
                    ]
                )
            logits = _fuse_auxiliary_logits(
                verifier_logits=verifier_logits,
                scaled_logits=factors,
                torch=torch,
            )
            return type(
                "CandidateOutput",
                (),
                {
                    "logits": logits,
                    "pair_logits": pair_logits,
                    "relation_logits": relation_logits,
                    "subject_logits": subject_logits,
                    "object_logits": object_logits,
                    "predicate_presence_logits": predicate_presence_logits,
                },
            )()

    return SpanPairCrossEncoder()


def _set_trainable(model: Any, *, joint: bool) -> tuple[list[Any], list[Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = list(model.classifier.parameters())
    head.extend(model.predicate_embedding.parameters())
    head.extend(model.pair_head.parameters())
    head.extend(model.relation_head.parameters())
    head.append(model.pair_scale_logit)
    head.append(model.relation_scale_logit)
    if getattr(model, "factor_head_version", 1) >= 2:
        head.extend(model.subject_head.parameters())
        head.extend(model.object_head.parameters())
        head.extend(model.predicate_presence_head.parameters())
        head.append(model.subject_scale_logit)
        head.append(model.object_scale_logit)
        head.append(model.predicate_presence_scale_logit)
    for parameter in head:
        parameter.requires_grad_(True)
    encoder_parameters: list[Any] = []
    if joint:
        layers, embeddings = _layers_and_embeddings(model)
        for module in [embeddings, *layers]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                encoder_parameters.append(parameter)
    return head, encoder_parameters


def _optimizer(model: Any, torch: Any, *, joint: bool, overfit: bool = False):
    head, encoder = _set_trainable(model, joint=joint)
    groups = [
        {
            "params": head,
            "lr": (
                OVERFIT_JOINT_HEAD_LR
                if overfit and joint
                else OVERFIT_HEAD_LR if overfit else HEAD_LR
            ),
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
    ]
    if encoder:
        groups.append(
            {
                "params": encoder,
                "lr": OVERFIT_ENCODER_LR if overfit else ENCODER_LR,
                "eps": 1e-4,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(groups), head, encoder


def _phase_schedule(
    *,
    mode: str,
    n_groups: int,
    groups_per_batch: int,
    formal_epochs: int,
    formal_head_only_epochs: int,
) -> tuple[str, ...]:
    """Return an immutable phase schedule without resetting Adam each epoch.

    The feasibility gate deliberately uses optimizer-step counts instead of a
    nominal epoch count.  This prevents a tiny overfit set from receiving far
    fewer updates than the declared 1200 head-only + 100 joint recipe.
    """

    if n_groups < 1 or groups_per_batch < 1:
        raise ValueError("phase schedule requires positive groups and batch size")
    if mode == "train":
        return tuple(
            "head_only" if epoch <= formal_head_only_epochs else "top2_joint"
            for epoch in range(1, formal_epochs + 1)
        )
    if mode != "overfit":
        raise ValueError(f"unsupported training mode: {mode}")
    batches_per_epoch = math.ceil(n_groups / groups_per_batch)
    head_epochs = math.ceil(OVERFIT_HEAD_STEPS / batches_per_epoch)
    joint_epochs = math.ceil(OVERFIT_JOINT_STEPS / batches_per_epoch)
    return ("head_only",) * head_epochs + ("top2_joint",) * joint_epochs


def _train_mode(model: Any, *, joint: bool) -> None:
    model.eval()
    model.classifier.train()
    model.predicate_embedding.train()
    model.pair_head.train()
    model.relation_head.train()
    if getattr(model, "factor_head_version", 1) >= 2:
        model.subject_head.train()
        model.object_head.train()
        model.predicate_presence_head.train()
    if joint:
        layers, _embeddings = _layers_and_embeddings(model)
        for layer in layers:
            layer.train()


def _group_batches(
    groups: Sequence[tuple[int, ...]],
    *,
    groups_per_batch: int,
    seed: int,
) -> list[list[tuple[int, ...]]]:
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    return [
        ordered[start : start + groups_per_batch]
        for start in range(0, len(ordered), groups_per_batch)
    ]


def _candidate_relation_targets(
    *,
    candidates: Sequence[PairRerankCandidate],
    gold_rows: Sequence[set[tuple[str, str, str]]],
    predicates: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Return fit-only multi-label predicates for each exact entity pair."""

    predicate2id = {str(value): index for index, value in enumerate(predicates)}
    if len(predicate2id) != len(predicates):
        raise ValueError("predicate labels must be unique")
    pair_targets: list[dict[tuple[str, str], tuple[int, ...]]] = []
    for row in gold_rows:
        grouped: dict[tuple[str, str], set[int]] = {}
        for subject, predicate, obj in row:
            if predicate not in predicate2id:
                raise ValueError("gold predicate is absent from predicate vocabulary")
            grouped.setdefault((subject, obj), set()).add(predicate2id[predicate])
        pair_targets.append(
            {key: tuple(sorted(values)) for key, values in grouped.items()}
        )
    targets: list[tuple[int, ...]] = []
    for candidate in candidates:
        if not 0 <= candidate.row_index < len(pair_targets):
            raise ValueError("candidate row index is out of range")
        targets.append(
            pair_targets[candidate.row_index].get(
                (candidate.subject, candidate.object), ()
            )
        )
    return tuple(targets)


def _candidate_factor_targets(
    *,
    candidates: Sequence[PairRerankCandidate],
    gold_rows: Sequence[set[tuple[str, str, str]]],
    predicates: Sequence[str],
) -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[tuple[int, ...], ...]]:
    """Return fit-only role and sentence-predicate targets per candidate.

    These are necessary conditions for an exact SPO, not alternate decoders:
    the proposed subject must occur in a gold subject role, the proposed object
    must occur in a gold object role, and its predicate must be present
    somewhere in the source row.  Surface identity is correct for the official
    metric, which scores ``(subject, predicate, object)`` rather than mention
    coordinates.  No calibration labels participate in these targets.
    """

    predicate2id = {str(value): index for index, value in enumerate(predicates)}
    if len(predicate2id) != len(predicates):
        raise ValueError("predicate labels must be unique")
    row_subjects: list[set[str]] = []
    row_objects: list[set[str]] = []
    row_predicates: list[tuple[int, ...]] = []
    for row in gold_rows:
        subjects: set[str] = set()
        objects: set[str] = set()
        present: set[int] = set()
        for subject, predicate, obj in row:
            if predicate not in predicate2id:
                raise ValueError("gold predicate is absent from predicate vocabulary")
            subjects.add(subject)
            objects.add(obj)
            present.add(predicate2id[predicate])
        row_subjects.append(subjects)
        row_objects.append(objects)
        row_predicates.append(tuple(sorted(present)))

    subject_targets: list[bool] = []
    object_targets: list[bool] = []
    predicate_targets: list[tuple[int, ...]] = []
    for candidate in candidates:
        row_index = candidate.row_index
        if not 0 <= row_index < len(gold_rows):
            raise ValueError("candidate row index is out of range")
        subject_targets.append(candidate.subject in row_subjects[row_index])
        object_targets.append(candidate.object in row_objects[row_index])
        predicate_targets.append(row_predicates[row_index])
    return (
        tuple(subject_targets),
        tuple(object_targets),
        tuple(predicate_targets),
    )


def _factor_target_audit(
    *,
    candidates: Sequence[PairRerankCandidate],
    subject_targets: Sequence[bool],
    object_targets: Sequence[bool],
    predicate_presence_targets: Sequence[tuple[int, ...]],
    indices: Iterable[int],
) -> dict[str, Any]:
    """Aggregate factor labels without exposing fit row text or triples."""

    selected = tuple(int(index) for index in indices)
    if any(not 0 <= index < len(candidates) for index in selected):
        raise ValueError("factor target audit index is out of range")

    def binary(values: Sequence[bool]) -> dict[str, float | int]:
        positive = sum(int(values[index]) for index in selected)
        total = len(selected)
        return {
            "positive": positive,
            "negative": total - positive,
            "positive_rate": positive / max(1, total),
        }

    predicate_values = tuple(
        candidates[index].predicate_id in predicate_presence_targets[index]
        for index in range(len(candidates))
    )
    return {
        "n_occurrences": len(selected),
        "subject_valid": binary(subject_targets),
        "object_valid": binary(object_targets),
        "proposed_predicate_present": binary(predicate_values),
    }


def _decomposed_group_margins(
    *,
    candidates: Sequence[PairRerankCandidate],
    flat_candidate_indices: Sequence[int],
    spans: Sequence[tuple[int, int]],
    pair_logits: Any,
    selected_relation_logits: Any,
    torch: Any,
) -> tuple[Any, Any]:
    """Rank exact pairs and within-pair predicates with separate margins."""

    import torch.nn.functional as F

    if pair_logits.ndim != 1 or selected_relation_logits.ndim != 1:
        raise ValueError("decomposed margin logits must be vectors")
    if len(flat_candidate_indices) != pair_logits.shape[0] or len(
        flat_candidate_indices
    ) != selected_relation_logits.shape[0]:
        raise ValueError("decomposed margin candidate/logit counts differ")
    pair_terms = []
    relation_terms = []
    for start, end in spans:
        positive = candidates[flat_candidate_indices[start]]
        positive_pair = (positive.subject, positive.object)
        for position in range(start + 1, end):
            negative = candidates[flat_candidate_indices[position]]
            negative_pair = (negative.subject, negative.object)
            if negative_pair == positive_pair:
                relation_terms.append(
                    F.relu(
                        MARGIN
                        - selected_relation_logits[start]
                        + selected_relation_logits[position]
                    )
                )
            else:
                pair_terms.append(
                    F.relu(
                        MARGIN
                        - pair_logits[start]
                        + pair_logits[position]
                    )
                )
    zero = pair_logits.sum() * 0.0
    pair_margin = torch.stack(pair_terms).mean() if pair_terms else zero
    relation_margin = (
        torch.stack(relation_terms).mean() if relation_terms else zero
    )
    return pair_margin, relation_margin


def _group_listwise_loss(*, logits: Any, spans: Sequence[tuple[int, int]], torch: Any) -> Any:
    """Rank the exact positive first within every hard-negative group."""

    import torch.nn.functional as F

    if logits.ndim != 1:
        raise ValueError("listwise candidate logits must be a vector")
    terms = []
    for start, end in spans:
        if not 0 <= start < end <= logits.shape[0]:
            raise ValueError("listwise candidate span is out of range")
        terms.append(-F.log_softmax(logits[start:end], dim=0)[0])
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _soft_relation_cross_entropy(
    *, relation_logits: Any, relation_labels: Any, active: Any, torch: Any
) -> Any:
    """Train relation choice as normalized-label softmax cross entropy.

    Nearly every exact entity pair in fit has one predicate.  The previous
    positive-weighted 44-way BCE treated the other 43 labels as independent
    decisions and encouraged broad activation.  A normalized target makes the
    dominant single-label case an actual relation-choice task while retaining
    deterministic support for the rare multi-predicate pair.
    """

    import torch.nn.functional as F

    if relation_logits.shape != relation_labels.shape:
        raise ValueError("relation logits and labels differ")
    if active.ndim != 1 or active.shape[0] != relation_logits.shape[0]:
        raise ValueError("relation active mask differs")
    if not bool(torch.any(active)):
        return relation_logits.sum() * 0.0
    active_logits = relation_logits[active]
    active_labels = relation_labels[active]
    target_mass = active_labels.sum(dim=1, keepdim=True)
    if not bool(torch.all(target_mass > 0)):
        raise ValueError("active relation target has no positive label")
    targets = active_labels / target_mass
    return -(targets * F.log_softmax(active_logits, dim=1)).sum(dim=1).mean()


def _balanced_binary_aux_bce(*, logits: Any, labels: Any, torch: Any) -> Any:
    """Balanced BCE that keeps ordinary gradients for single-class batches."""

    import torch.nn.functional as F

    if logits.shape != labels.shape:
        raise ValueError("binary auxiliary logits and labels differ")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("binary auxiliary logits are non-finite")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("binary auxiliary labels must be zero or one")
    positives = labels.sum()
    negatives = labels.numel() - positives
    if float(positives) == 0.0 or float(negatives) == 0.0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=(negatives / positives).detach(),
    )


def _batch_loss(
    *,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[PairRerankCandidate],
    relation_targets: Sequence[tuple[int, ...]],
    subject_targets: Sequence[bool],
    object_targets: Sequence[bool],
    predicate_presence_targets: Sequence[tuple[int, ...]],
    batch_groups: Sequence[tuple[int, ...]],
    device: Any,
    torch: Any,
    fp16: bool,
):
    import torch.nn.functional as F

    flat: list[int] = []
    spans: list[tuple[int, int]] = []
    for group in batch_groups:
        start = len(flat)
        flat.extend(group)
        spans.append((start, len(flat)))
    encoded = _encode_candidates(tokenizer, candidates, flat, device, torch)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=fp16,
    ):
        outputs = model(**encoded)
        logits = outputs.logits.reshape(-1).float()
        pair_logits = outputs.pair_logits.reshape(-1).float()
        relation_logits = outputs.relation_logits.float()
        if (
            outputs.subject_logits is None
            or outputs.object_logits is None
            or outputs.predicate_presence_logits is None
        ):
            raise ValueError("formal factorized training requires factor-head v2")
        subject_logits = outputs.subject_logits.reshape(-1).float()
        object_logits = outputs.object_logits.reshape(-1).float()
        predicate_presence_logits = outputs.predicate_presence_logits.float()
        selected_relation_logits = relation_logits.gather(
            1, encoded["predicate_ids"].reshape(-1, 1)
        ).reshape(-1)
        targets = torch.zeros_like(logits)
        for start, _end in spans:
            targets[start] = 1.0
        positives = targets.sum().clamp_min(1.0)
        negatives = (1.0 - targets).sum().clamp_min(1.0)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=(negatives / positives).detach(),
        )
        margins = []
        for start, end in spans:
            if end > start + 1:
                margins.append(
                    F.relu(MARGIN - logits[start] + logits[start + 1 : end]).mean()
                )
        margin = torch.stack(margins).mean() if margins else logits.sum() * 0.0
        listwise = _group_listwise_loss(logits=logits, spans=spans, torch=torch)
        relation_labels = torch.zeros_like(relation_logits)
        for batch_position, candidate_index in enumerate(flat):
            for predicate_id in relation_targets[candidate_index]:
                relation_labels[batch_position, predicate_id] = 1.0
        relation_active = relation_labels.sum(dim=1) > 0
        pair_margin, relation_margin = _decomposed_group_margins(
            candidates=candidates,
            flat_candidate_indices=flat,
            spans=spans,
            pair_logits=pair_logits,
            selected_relation_logits=selected_relation_logits,
            torch=torch,
        )
        pair_labels = relation_active.to(dtype=pair_logits.dtype)
        pair_positives = pair_labels.sum().clamp_min(1.0)
        pair_negatives = (1.0 - pair_labels).sum().clamp_min(1.0)
        pair_bce = F.binary_cross_entropy_with_logits(
            pair_logits,
            pair_labels,
            pos_weight=(pair_negatives / pair_positives).detach(),
        )
        relation_ce = _soft_relation_cross_entropy(
            relation_logits=relation_logits,
            relation_labels=relation_labels,
            active=relation_active,
            torch=torch,
        )
        subject_labels = torch.tensor(
            [float(subject_targets[index]) for index in flat],
            dtype=subject_logits.dtype,
            device=subject_logits.device,
        )
        object_labels = torch.tensor(
            [float(object_targets[index]) for index in flat],
            dtype=object_logits.dtype,
            device=object_logits.device,
        )

        subject_bce = _balanced_binary_aux_bce(
            logits=subject_logits,
            labels=subject_labels,
            torch=torch,
        )
        object_bce = _balanced_binary_aux_bce(
            logits=object_logits,
            labels=object_labels,
            torch=torch,
        )
        predicate_presence_labels = torch.zeros_like(predicate_presence_logits)
        for batch_position, candidate_index in enumerate(flat):
            for predicate_id in predicate_presence_targets[candidate_index]:
                predicate_presence_labels[batch_position, predicate_id] = 1.0
        predicate_presence_bce = _balanced_binary_aux_bce(
            logits=predicate_presence_logits,
            labels=predicate_presence_labels,
            torch=torch,
        )
        loss = (
            bce
            + MARGIN_WEIGHT * margin
            + LISTWISE_WEIGHT * listwise
            + PAIR_MARGIN_WEIGHT * pair_margin
            + RELATION_MARGIN_WEIGHT * relation_margin
            + PAIR_AUX_WEIGHT * pair_bce
            + RELATION_AUX_WEIGHT * relation_ce
            + SUBJECT_AUX_WEIGHT * subject_bce
            + OBJECT_AUX_WEIGHT * object_bce
            + PREDICATE_PRESENCE_AUX_WEIGHT * predicate_presence_bce
        )
    return (
        loss,
        bce.detach(),
        margin.detach(),
        listwise.detach(),
        pair_margin.detach(),
        relation_margin.detach(),
        pair_bce.detach(),
        relation_ce.detach(),
        subject_bce.detach(),
        object_bce.detach(),
        predicate_presence_bce.detach(),
    )


def _train_epoch(
    *,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[PairRerankCandidate],
    relation_targets: Sequence[tuple[int, ...]],
    subject_targets: Sequence[bool],
    object_targets: Sequence[bool],
    predicate_presence_targets: Sequence[tuple[int, ...]],
    groups: Sequence[tuple[int, ...]],
    optimizer: Any,
    active_parameters: Iterable[Any],
    device: Any,
    torch: Any,
    fp16: bool,
    groups_per_batch: int,
    accumulation: int,
    seed: int,
) -> dict[str, float | int]:
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=fp16)
    batches = _group_batches(groups, groups_per_batch=groups_per_batch, seed=seed)
    optimizer.zero_grad(set_to_none=True)
    loss_total = bce_total = margin_total = listwise_total = 0.0
    pair_margin_total = relation_margin_total = 0.0
    pair_total = relation_total = 0.0
    subject_total = object_total = predicate_presence_total = 0.0
    optimizer_steps = skipped_steps = 0
    parameters = [parameter for parameter in active_parameters if parameter.requires_grad]
    for batch_index, batch_groups in enumerate(batches):
        (
            loss,
            bce,
            margin,
            listwise,
            pair_margin,
            relation_margin,
            pair_bce,
            relation_ce,
            subject_bce,
            object_bce,
            predicate_presence_bce,
        ) = _batch_loss(
            model=model,
            tokenizer=tokenizer,
            candidates=candidates,
            relation_targets=relation_targets,
            subject_targets=subject_targets,
            object_targets=object_targets,
            predicate_presence_targets=predicate_presence_targets,
            batch_groups=batch_groups,
            device=device,
            torch=torch,
            fp16=fp16,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("cross-encoder loss became non-finite")
        scaler.scale(loss / accumulation).backward()
        loss_total += float(loss.detach().cpu())
        bce_total += float(bce.cpu())
        margin_total += float(margin.cpu())
        listwise_total += float(listwise.cpu())
        pair_margin_total += float(pair_margin.cpu())
        relation_margin_total += float(relation_margin.cpu())
        pair_total += float(pair_bce.cpu())
        relation_total += float(relation_ce.cpu())
        subject_total += float(subject_bce.cpu())
        object_total += float(object_bce.cpu())
        predicate_presence_total += float(predicate_presence_bce.cpu())
        boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
        if not boundary:
            continue
        before = float(scaler.get_scale())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        skipped_steps += int(float(scaler.get_scale()) < before)
    return {
        "mean_loss": loss_total / max(1, len(batches)),
        "mean_bce": bce_total / max(1, len(batches)),
        "mean_margin": margin_total / max(1, len(batches)),
        "mean_listwise": listwise_total / max(1, len(batches)),
        "mean_pair_margin": pair_margin_total / max(1, len(batches)),
        "mean_relation_margin": relation_margin_total / max(1, len(batches)),
        "mean_pair_bce": pair_total / max(1, len(batches)),
        "mean_relation_softmax_ce": relation_total / max(1, len(batches)),
        "mean_subject_role_bce": subject_total / max(1, len(batches)),
        "mean_object_role_bce": object_total / max(1, len(batches)),
        "mean_predicate_presence_bce": (
            predicate_presence_total / max(1, len(batches))
        ),
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
    }


def _score_candidates(
    *,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[PairRerankCandidate],
    device: Any,
    torch: Any,
    fp16: bool,
    batch_size: int,
) -> tuple[float, ...]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(candidates), batch_size):
        indices = list(range(start, min(start + batch_size, len(candidates))))
        encoded = _encode_candidates(
            tokenizer, candidates, indices, device, torch
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            probabilities = model(**encoded).logits.reshape(-1).float().sigmoid()
        scores.extend(probabilities.cpu().tolist())
    return tuple(scores)


def _build_model_hard_negative_groups(
    *,
    candidates: Sequence[PairRerankCandidate],
    gold_rows: Sequence[set[tuple[str, str, str]]],
    scores: Sequence[float],
    negative_ratio: int = NEGATIVE_RATIO,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, int | float]]:
    """Mine the current model's highest-scoring non-gold row candidates.

    Each reachable positive is paired with the same deterministic top-k false
    positives from its source row.  Reuse is intentional: the goal is to train
    every positive against the exact candidates the current model is most likely
    to emit, rather than distributing easy proposal negatives across positives.
    """

    if len(candidates) != len(scores):
        raise ValueError("candidate and model-score counts differ")
    if negative_ratio < 1:
        raise ValueError("negative ratio must be positive")
    by_row: list[list[int]] = [[] for _ in gold_rows]
    for index, (candidate, score) in enumerate(zip(candidates, scores)):
        if not 0 <= candidate.row_index < len(gold_rows):
            raise ValueError("candidate row index is out of range")
        if not math.isfinite(float(score)):
            raise ValueError("model hard-negative score is non-finite")
        by_row[candidate.row_index].append(index)
    groups: list[tuple[int, ...]] = []
    selected_negatives: set[int] = set()
    reachable_gold: set[tuple[int, tuple[str, str, str]]] = set()
    for row_index, indices in enumerate(by_row):
        positives = sorted(
            index
            for index in indices
            if candidates[index].triple in gold_rows[row_index]
        )
        negatives = sorted(
            (
                index
                for index in indices
                if candidates[index].triple not in gold_rows[row_index]
            ),
            key=lambda index: (
                -float(scores[index]),
                -float(candidates[index].link_score),
                -float(candidates[index].relation_score),
                -float(candidates[index].boundary_score),
                index,
            ),
        )
        hardest = tuple(negatives[:negative_ratio])
        selected_negatives.update(hardest)
        for positive in positives:
            reachable_gold.add((row_index, candidates[positive].triple))
            groups.append((positive, *hardest))
    total_gold = sum(len(row) for row in gold_rows)
    return tuple(groups), {
        "strategy": "current_model_top_false_positives_per_row",
        "n_groups": len(groups),
        "n_positive_samples": len(groups),
        "n_negative_samples": sum(len(group) - 1 for group in groups),
        "n_unique_negative_samples": len(selected_negatives),
        "negative_ratio": negative_ratio,
        "maximum_negatives_per_positive": max(
            (len(group) - 1 for group in groups), default=0
        ),
        "reachable_gold": len(reachable_gold),
        "total_gold": total_gold,
        "reachable_gold_recall": len(reachable_gold) / max(1, total_gold),
    }


def _hard_negative_refresh_strategy(
    *,
    mode: str,
    epoch: int,
    phase: str,
    previous_phase: str | None,
) -> str | None:
    """Declare the single leak-safe mining refresh for each training mode."""

    if mode == "train" and phase == "top2_joint":
        return "mixed_proposal_and_model"
    if mode == "overfit" and phase == "top2_joint" and previous_phase != phase:
        return "model_false_positives"
    return None


def _negative_error_category(
    candidate: PairRerankCandidate,
    gold_row: set[tuple[str, str, str]],
) -> str:
    """Classify one fit-only false proposal by its structural error.

    Categories are computed only from the current fit row.  They are used to
    spend the four model-hard slots on different failure modes instead of four
    near-identical boundary proposals.  No category changes the HNR or the
    calibration decoder.
    """

    subject, predicate, obj = candidate.triple
    if candidate.triple in gold_row:
        raise ValueError("negative error category received a gold triple")
    if subject == obj:
        return "self_pair"
    if (obj, predicate, subject) in gold_row:
        return "direction_swap"
    if any(g_subject == subject and g_object == obj for g_subject, _, g_object in gold_row):
        return "wrong_predicate"
    if any(g_subject == subject and g_predicate == predicate for g_subject, g_predicate, _ in gold_row):
        return "wrong_object_or_boundary"
    if any(g_predicate == predicate and g_object == obj for _, g_predicate, g_object in gold_row):
        return "wrong_subject_or_boundary"
    gold_subjects = {g_subject for g_subject, _, _ in gold_row}
    gold_objects = {g_object for _, _, g_object in gold_row}
    if subject in gold_subjects and obj in gold_objects:
        return "cross_spo_pair"
    return "other_oof_false_positive"


def _overfit_gate_payload(
    *,
    head_best: dict[str, Any],
    joint_best: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the declared exact-SPO and post-refinement overfit gates."""

    exact_best = max(
        (head_best, joint_best),
        key=lambda item: (
            item["metric"]["f1"],
            item["metric"]["precision"],
            item["threshold"],
        ),
    )
    gates = {
        "head_only_f1_at_least_0_98": head_best["metric"]["f1"] >= 0.98,
        "exact_spo_f1_at_least_0_98": exact_best["metric"]["f1"] >= 0.98,
        "joint_f1_at_least_0_95": joint_best["metric"]["f1"] >= 0.95,
        "head_only_selection": head_best,
        "joint_selection": joint_best,
        "exact_spo_selection": exact_best,
    }
    return gates, exact_best


def _build_mixed_hard_negative_groups(
    *,
    candidates: Sequence[PairRerankCandidate],
    gold_rows: Sequence[set[tuple[str, str, str]]],
    scores: Sequence[float],
    proposal_negative_count: int = 4,
    model_negative_count: int = 4,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, int | float]]:
    """Mix broad proposal-hard coverage with current-model false positives."""

    if len(candidates) != len(scores):
        raise ValueError("candidate and model-score counts differ")
    if proposal_negative_count < 1 or model_negative_count < 1:
        raise ValueError("mixed negative counts must be positive")
    if proposal_negative_count + model_negative_count > NEGATIVE_RATIO:
        raise ValueError("mixed hard negatives exceed the declared HNR")
    by_row: list[list[int]] = [[] for _ in gold_rows]
    for index, (candidate, score) in enumerate(zip(candidates, scores)):
        if not 0 <= candidate.row_index < len(gold_rows):
            raise ValueError("candidate row index is out of range")
        if not math.isfinite(float(score)):
            raise ValueError("mixed hard-negative score is non-finite")
        by_row[candidate.row_index].append(index)

    def proposal_key(index: int) -> tuple[float, float, float, int]:
        candidate = candidates[index]
        return (
            -float(candidate.link_score),
            -float(candidate.relation_score),
            -float(candidate.boundary_score),
            index,
        )

    def model_key(index: int) -> tuple[float, float, float, float, int]:
        candidate = candidates[index]
        return (
            -float(scores[index]),
            -float(candidate.link_score),
            -float(candidate.relation_score),
            -float(candidate.boundary_score),
            index,
        )

    groups: list[tuple[int, ...]] = []
    proposal_selected: set[int] = set()
    model_selected: set[int] = set()
    selected_model_categories: dict[str, int] = {}
    reachable_gold: set[tuple[int, tuple[str, str, str]]] = set()
    for row_index, indices in enumerate(by_row):
        positives = sorted(
            index
            for index in indices
            if candidates[index].triple in gold_rows[row_index]
        )
        negatives = [
            index
            for index in indices
            if candidates[index].triple not in gold_rows[row_index]
        ]
        proposal_order = sorted(negatives, key=proposal_key)
        model_order = sorted(negatives, key=model_key)
        for positive_offset, positive in enumerate(positives):
            start = positive_offset * proposal_negative_count
            proposal = list(
                proposal_order[start : start + proposal_negative_count]
            )
            if len(proposal) < min(proposal_negative_count, len(proposal_order)):
                missing = min(proposal_negative_count, len(proposal_order)) - len(
                    proposal
                )
                proposal.extend(proposal_order[:missing])
            proposal_set = set(proposal)
            model_pool = [
                index for index in model_order if index not in proposal_set
            ]
            model: list[int] = []
            represented: set[str] = set()
            # model_pool is already ordered by current false-positive score.
            # Its first unseen category is therefore the hardest example of
            # that structural error.  Fill any unused slots with the remaining
            # globally hardest examples so rows with one error type still use
            # the complete HNR budget.
            for index in model_pool:
                category = _negative_error_category(
                    candidates[index], gold_rows[row_index]
                )
                if category in represented:
                    continue
                model.append(index)
                represented.add(category)
                if len(model) == model_negative_count:
                    break
            if len(model) < model_negative_count:
                model.extend(
                    index
                    for index in model_pool
                    if index not in model
                )
                del model[model_negative_count:]
            selected = tuple((*proposal, *model))
            if len(selected) > NEGATIVE_RATIO:
                raise AssertionError("mixed hard-negative group exceeds HNR")
            proposal_selected.update(proposal)
            model_selected.update(model)
            for index in model:
                category = _negative_error_category(
                    candidates[index], gold_rows[row_index]
                )
                selected_model_categories[category] = (
                    selected_model_categories.get(category, 0) + 1
                )
            reachable_gold.add((row_index, candidates[positive].triple))
            groups.append((positive, *selected))
    total_gold = sum(len(row) for row in gold_rows)
    return tuple(groups), {
        "strategy": (
            "four_proposal_hard_plus_four_category_balanced_"
            "model_hard_per_positive"
        ),
        "n_groups": len(groups),
        "n_positive_samples": len(groups),
        "n_negative_samples": sum(len(group) - 1 for group in groups),
        "n_unique_proposal_negatives": len(proposal_selected),
        "n_unique_model_negatives": len(model_selected),
        "n_unique_negative_samples": len(proposal_selected | model_selected),
        "selected_model_error_categories": dict(
            sorted(selected_model_categories.items())
        ),
        "proposal_negative_count": proposal_negative_count,
        "model_negative_count": model_negative_count,
        "maximum_negatives_per_positive": max(
            (len(group) - 1 for group in groups), default=0
        ),
        "reachable_gold": len(reachable_gold),
        "total_gold": total_gold,
        "reachable_gold_recall": len(reachable_gold) / max(1, total_gold),
    }


def _select_threshold(
    *,
    gold_rows: Sequence[set[tuple[str, str, str]]],
    candidates: Sequence[PairRerankCandidate],
    scores: Sequence[float],
) -> dict[str, Any]:
    sweep = []
    for threshold in THRESHOLDS:
        predicted = decode_pair_rerank_candidates(
            n_rows=len(gold_rows),
            candidates=candidates,
            scores=scores,
            threshold=threshold,
        )
        sweep.append(
            {
                "threshold": threshold,
                "metric": micro_spo_metric(gold_rows, predicted),
            }
        )
    selected = max(
        sweep,
        key=lambda item: (
            item["metric"]["f1"],
            item["metric"]["precision"],
            item["threshold"],
        ),
    )
    return {**selected, "grid_size": len(sweep), "sweep": sweep}


def _proposal_threshold_value(value: float) -> str | float:
    return "-inf" if math.isinf(value) and value < 0 else float(value)


def _filter_proposal_candidates(
    candidates: Sequence[PairRerankCandidate],
    *,
    relation_threshold: float,
    link_threshold: float,
    boundary_threshold: float,
) -> list[PairRerankCandidate]:
    """Apply one global, label-independent proposal policy without truncation."""

    return [
        candidate
        for candidate in candidates
        if candidate.relation_score >= relation_threshold
        and candidate.link_score >= link_threshold
        and candidate.boundary_score >= boundary_threshold
    ]


def _candidate_density(
    candidates: Sequence[PairRerankCandidate], *, n_rows: int
) -> dict[str, float | int]:
    counts = [0] * n_rows
    for candidate in candidates:
        if not 0 <= candidate.row_index < n_rows:
            raise ValueError("candidate row index is out of range")
        counts[candidate.row_index] += 1
    ordered = sorted(counts)
    if not ordered:
        return {"mean": 0.0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
        return int(ordered[max(0, index)])

    return {
        "mean": sum(ordered) / len(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": int(ordered[-1]),
    }


def _adaptive_filter_proposal_candidates(
    candidates: Sequence[PairRerankCandidate],
    *,
    n_rows: int,
    relation_threshold: float,
    boundary_threshold: float,
    link_threshold_ladder: Sequence[float] = PROPOSAL_LINK_THRESHOLDS,
    candidate_cap: int = SURFACE_CANDIDATE_CAP,
) -> tuple[list[PairRerankCandidate], dict[str, Any]]:
    """Raise a row's link threshold only when needed to satisfy the hard cap.

    This is score filtering, not top-k truncation: every candidate at or above
    the selected row threshold is retained, and every candidate below it is
    rejected.  Threshold choice uses candidate count only and never row gold.
    """

    ladder = tuple(float(value) for value in link_threshold_ladder)
    if not ladder or tuple(sorted(set(ladder))) != ladder:
        raise ValueError("link threshold ladder must be unique and increasing")
    if candidate_cap < 1:
        raise ValueError("candidate cap must be positive")
    by_row: list[list[PairRerankCandidate]] = [[] for _ in range(n_rows)]
    for candidate in candidates:
        if not 0 <= candidate.row_index < n_rows:
            raise ValueError("candidate row index is out of range")
        if (
            candidate.relation_score >= relation_threshold
            and candidate.boundary_score >= boundary_threshold
        ):
            by_row[candidate.row_index].append(candidate)
    selected: list[PairRerankCandidate] = []
    threshold_counts = {str(value): 0 for value in ladder}
    for row in by_row:
        accepted: list[PairRerankCandidate] | None = None
        chosen: float | None = None
        for link_threshold in ladder:
            filtered = [
                candidate
                for candidate in row
                if candidate.link_score >= link_threshold
            ]
            if len(filtered) <= candidate_cap:
                accepted = filtered
                chosen = link_threshold
                break
        if accepted is None or chosen is None:
            raise ValueError(
                "density-adaptive link threshold ladder cannot satisfy candidate cap"
            )
        threshold_counts[str(chosen)] += 1
        selected.extend(accepted)
    raw_density = _candidate_density(
        [
            candidate
            for row in by_row
            for candidate in row
        ],
        n_rows=n_rows,
    )
    selected_density = _candidate_density(selected, n_rows=n_rows)
    if selected_density["max"] > candidate_cap:
        raise AssertionError("adaptive proposal policy violated candidate cap")
    return selected, {
        "mode": "density_adaptive_link_ladder",
        "relation_threshold": _proposal_threshold_value(relation_threshold),
        "boundary_threshold": _proposal_threshold_value(boundary_threshold),
        "link_threshold_ladder": list(ladder),
        "candidate_cap_per_row": candidate_cap,
        "threshold_row_counts": threshold_counts,
        "raw_density": raw_density,
        "selected_density": selected_density,
        "inference_truncation": False,
        "gold_used_for_row_threshold": False,
    }


def _apply_selected_proposal_policy(
    candidates: Sequence[PairRerankCandidate],
    *,
    n_rows: int,
    policy: dict[str, Any],
) -> tuple[list[PairRerankCandidate], dict[str, Any]]:
    mode = policy.get("mode")

    def value(name: str) -> float:
        raw = policy["thresholds"][name]
        return -math.inf if raw == "-inf" else float(raw)

    if mode == "global":
        selected = _filter_proposal_candidates(
            candidates,
            relation_threshold=value("relation"),
            link_threshold=value("link"),
            boundary_threshold=value("boundary"),
        )
        density = _candidate_density(selected, n_rows=n_rows)
        if density["max"] > SURFACE_CANDIDATE_CAP:
            raise ValueError("selected global proposal policy violates candidate cap")
        return selected, {
            "mode": "global",
            "thresholds": policy["thresholds"],
            "selected_density": density,
            "inference_truncation": False,
        }
    if mode == "density_adaptive_link_ladder":
        return _adaptive_filter_proposal_candidates(
            candidates,
            n_rows=n_rows,
            relation_threshold=value("relation"),
            boundary_threshold=value("boundary"),
            link_threshold_ladder=tuple(
                float(item) for item in policy["thresholds"]["link_ladder"]
            ),
        )
    raise ValueError("unknown proposal policy mode")


def _select_proposal_policy(
    *,
    fit_candidates: Sequence[PairRerankCandidate],
    fit_gold: Sequence[set[tuple[str, str, str]]],
    guard_candidates: Sequence[PairRerankCandidate],
    n_guard_rows: int,
) -> dict[str, Any]:
    """Select one fit-only global proposal policy under strict density guards.

    Labels are read only from fit OOF rows.  Calibration candidates participate
    solely as an unlabeled density guard, so calibration gold cannot influence
    proposal thresholds.  Candidates are filtered by score; they are never
    ranked or truncated to satisfy the cap.
    """

    sweep: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for relation_threshold in PROPOSAL_RELATION_THRESHOLDS:
        for link_threshold in PROPOSAL_LINK_THRESHOLDS:
            for boundary_threshold in PROPOSAL_BOUNDARY_THRESHOLDS:
                selected_fit = _filter_proposal_candidates(
                    fit_candidates,
                    relation_threshold=relation_threshold,
                    link_threshold=link_threshold,
                    boundary_threshold=boundary_threshold,
                )
                selected_guard = _filter_proposal_candidates(
                    guard_candidates,
                    relation_threshold=relation_threshold,
                    link_threshold=link_threshold,
                    boundary_threshold=boundary_threshold,
                )
                fit_rows = [set() for _ in fit_gold]
                for candidate in selected_fit:
                    fit_rows[candidate.row_index].add(candidate.triple)
                metric = micro_spo_metric(fit_gold, fit_rows)
                fit_density = _candidate_density(selected_fit, n_rows=len(fit_gold))
                guard_density = _candidate_density(
                    selected_guard, n_rows=n_guard_rows
                )
                item = {
                    "mode": "global",
                    "thresholds": {
                        "relation": _proposal_threshold_value(relation_threshold),
                        "link": _proposal_threshold_value(link_threshold),
                        "boundary": _proposal_threshold_value(boundary_threshold),
                    },
                    "candidate_oracle": metric,
                    "fit_density": fit_density,
                    "calibration_unlabeled_density": guard_density,
                    "density_safe": (
                        fit_density["max"] <= SURFACE_CANDIDATE_CAP
                        and guard_density["max"] <= SURFACE_CANDIDATE_CAP
                    ),
                }
                sweep.append(item)
                if item["density_safe"] and metric["recall"] >= 0.60:
                    feasible.append(item)
    for relation_threshold in PROPOSAL_RELATION_THRESHOLDS:
        for boundary_threshold in PROPOSAL_BOUNDARY_THRESHOLDS:
            selected_fit, fit_adaptive = _adaptive_filter_proposal_candidates(
                fit_candidates,
                n_rows=len(fit_gold),
                relation_threshold=relation_threshold,
                boundary_threshold=boundary_threshold,
            )
            selected_guard, guard_adaptive = _adaptive_filter_proposal_candidates(
                guard_candidates,
                n_rows=n_guard_rows,
                relation_threshold=relation_threshold,
                boundary_threshold=boundary_threshold,
            )
            fit_rows = [set() for _ in fit_gold]
            for candidate in selected_fit:
                fit_rows[candidate.row_index].add(candidate.triple)
            metric = micro_spo_metric(fit_gold, fit_rows)
            item = {
                "mode": "density_adaptive_link_ladder",
                "thresholds": {
                    "relation": _proposal_threshold_value(relation_threshold),
                    "link_ladder": list(PROPOSAL_LINK_THRESHOLDS),
                    "boundary": _proposal_threshold_value(boundary_threshold),
                },
                "candidate_oracle": metric,
                "fit_density": fit_adaptive["selected_density"],
                "calibration_unlabeled_density": guard_adaptive[
                    "selected_density"
                ],
                "fit_adaptive_audit": fit_adaptive,
                "calibration_adaptive_audit": guard_adaptive,
                "density_safe": True,
            }
            sweep.append(item)
            if metric["recall"] >= 0.60:
                feasible.append(item)
    if not feasible:
        return {
            "status": "rejected_no_density_safe_policy",
            "minimum_candidate_recall": 0.60,
            "candidate_cap_per_row": SURFACE_CANDIDATE_CAP,
            "grid_size": len(sweep),
            "sweep": sweep,
        }
    selected = max(
        feasible,
        key=lambda item: (
            item["candidate_oracle"]["f1"],
            item["candidate_oracle"]["precision"],
            item["candidate_oracle"]["recall"],
            -int(item["mode"] == "density_adaptive_link_ladder"),
        ),
    )
    return {
        "status": "selected",
        "minimum_candidate_recall": 0.60,
        "candidate_cap_per_row": SURFACE_CANDIDATE_CAP,
        "grid_size": len(sweep),
        "selection_basis": "fit_oof_labels_plus_unlabeled_calibration_density",
        "inference_truncation": False,
        "selected": selected,
        "sweep": sweep,
    }


def _snapshot_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _canonical_row_bytes(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _fold_shard_sha256(
    source_rows: Sequence[dict[str, Any]], indices: Sequence[int]
) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(_canonical_row_bytes(source_rows[index]))
    return digest.hexdigest()


def _validated_oof_partitions(
    *,
    source_train: Path,
    fit_rows: Sequence[dict[str, Any]],
    split_manifest_path: Path,
    oof_manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    """Bind the exact fit shard to five source-index OOF partitions."""

    source_rows = _jsonl(source_train)
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    oof = json.loads(oof_manifest_path.read_text(encoding="utf-8"))
    source_sha = sha256_file(source_train)
    if split.get("source_train_sha256") != source_sha:
        raise ValueError("split manifest is bound to different source training bytes")
    if oof.get("source_train_sha256") != source_sha:
        raise ValueError("OOF manifest is bound to different source training bytes")
    if oof.get("base_split_manifest_sha256") != sha256_file(split_manifest_path):
        raise ValueError("OOF manifest is bound to a different split manifest")
    fit_indices = split.get("splits", {}).get("fit", {}).get("indices")
    if not isinstance(fit_indices, list) or len(fit_indices) != len(fit_rows):
        raise ValueError("fit shard length differs from split manifest")
    fit_indices = [int(value) for value in fit_indices]
    if len(set(fit_indices)) != len(fit_indices) or any(
        index < 0 or index >= len(source_rows) for index in fit_indices
    ):
        raise ValueError("split fit indices are invalid")
    for fit_position, source_index in enumerate(fit_indices):
        if _canonical_row_bytes(fit_rows[fit_position]) != _canonical_row_bytes(
            source_rows[source_index]
        ):
            raise ValueError("fit shard row order differs from split source indices")
    fit_set = set(fit_indices)
    partitions = oof.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 5:
        raise ValueError("OOF manifest must contain five partitions")
    normalized: list[dict[str, Any]] = []
    validation_seen: set[int] = set()
    for expected_fold, raw in enumerate(partitions):
        if not isinstance(raw, dict) or raw.get("fold") != expected_fold:
            raise ValueError("OOF folds must be ordered 0..4")
        train = sorted(int(value) for value in raw.get("train_indices", []))
        validation = sorted(
            int(value) for value in raw.get("validation_indices", [])
        )
        train_set, validation_set = set(train), set(validation)
        if (
            len(train_set) != len(train)
            or len(validation_set) != len(validation)
            or not train_set
            or not validation_set
            or train_set & validation_set
            or train_set | validation_set != fit_set
            or validation_seen & validation_set
            or raw.get("family_intersection") not in ([], ())
        ):
            raise ValueError("OOF fold violates fit-internal disjointness")
        validation_seen.update(validation_set)
        normalized.append(
            {**raw, "train_indices": train, "validation_indices": validation}
        )
    if validation_seen != fit_set:
        raise ValueError("OOF validation does not cover fit exactly once")
    return source_rows, normalized, {
        source_index: position for position, source_index in enumerate(fit_indices)
    }


def _remap_oof_candidates(
    candidates: Sequence[PairRerankCandidate],
    *,
    validation_indices: Sequence[int],
    fit_positions: dict[int, int],
) -> list[PairRerankCandidate]:
    remapped: list[PairRerankCandidate] = []
    for candidate in candidates:
        if not 0 <= candidate.row_index < len(validation_indices):
            raise ValueError("fold candidate row index is outside validation shard")
        source_index = int(validation_indices[candidate.row_index])
        if source_index not in fit_positions:
            raise ValueError("fold candidate is not a member of base fit")
        remapped.append(
            replace(candidate, row_index=int(fit_positions[source_index]))
        )
    return remapped


def _fold_checkpoint(
    *,
    model_root: Path,
    fold: int,
    source_tree_sha256: str,
    source_rows: Sequence[dict[str, Any]],
    partition: dict[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    fold_root = (model_root / f"fold-{fold:02d}").resolve(strict=True)
    selection_path = (fold_root / "selection.json").resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("mode") != "oof-train"
        or selection.get("dataset_role") != "fit_internal_oof_fold_selection"
        or not selection.get("oof_candidate_generator_completed")
        or selection.get("source_tree_sha256") != source_tree_sha256
        or selection.get("fit_sha256")
        != _fold_shard_sha256(source_rows, partition["train_indices"])
        or selection.get("oof_validation_sha256")
        != _fold_shard_sha256(source_rows, partition["validation_indices"])
    ):
        raise ValueError(f"OOF fold {fold} selection binding is invalid")
    checkpoint = Path(str(selection.get("checkpoint", ""))).resolve(strict=True)
    try:
        checkpoint.relative_to(fold_root)
    except ValueError as error:
        raise ValueError(f"OOF fold {fold} checkpoint escapes its fold root") from error
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != selection.get("checkpoint_sha256"):
        raise ValueError(f"OOF fold {fold} checkpoint hash differs")
    return checkpoint, selection, sha256_file(selection_path)


def _generate_oof_fit_candidates(
    *,
    source_train: Path,
    fit_rows: Sequence[dict[str, Any]],
    split_manifest_path: Path,
    oof_manifest_path: Path,
    model_root: Path,
    base_model: Path,
    expected_predicates: Sequence[str],
    oof_source_tree_sha256: str,
    device: Any,
    torch: Any,
    batch_size: int,
    fp16: bool,
) -> tuple[list[PairRerankCandidate], dict[str, Any]]:
    source_rows, partitions, fit_positions = _validated_oof_partitions(
        source_train=source_train,
        fit_rows=fit_rows,
        split_manifest_path=split_manifest_path,
        oof_manifest_path=oof_manifest_path,
    )
    combined: list[PairRerankCandidate] = []
    fold_receipts: list[dict[str, Any]] = []
    for partition in partitions:
        fold = int(partition["fold"])
        checkpoint, selection, selection_sha = _fold_checkpoint(
            model_root=model_root,
            fold=fold,
            source_tree_sha256=oof_source_tree_sha256,
            source_rows=source_rows,
            partition=partition,
        )
        model, tokenizer, predicates, _, base_sha = _load_onerel(
            checkpoint_path=checkpoint,
            base_model=base_model,
            device=device,
            torch=torch,
        )
        if list(predicates) != list(expected_predicates):
            raise ValueError(f"OOF fold {fold} predicate table differs")
        validation_rows = [
            source_rows[index] for index in partition["validation_indices"]
        ]
        local_candidates, audit = _generate_candidates(
            model=model,
            tokenizer=tokenizer,
            rows=validation_rows,
            predicates=predicates,
            device=device,
            torch=torch,
            batch_size=batch_size,
            fp16=fp16,
            enforce_surface_cap=False,
        )
        combined.extend(
            _remap_oof_candidates(
                local_candidates,
                validation_indices=partition["validation_indices"],
                fit_positions=fit_positions,
            )
        )
        fold_receipts.append(
            {
                "fold": fold,
                "n_validation": len(validation_rows),
                "n_candidates": len(local_candidates),
                "selection_sha256": selection_sha,
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "base_model_tree_sha256": base_sha,
                "proposal": audit,
            }
        )
        del model
        torch.npu.empty_cache()
    return combined, {
        "candidate_source": "five_fold_article_family_disjoint_oof",
        "oof_manifest_sha256": sha256_file(oof_manifest_path),
        "n_folds": len(partitions),
        "validation_coverage_exactly_once": True,
        "folds": fold_receipts,
        "raw_coordinate_safety_cap_per_row": 32768,
        "candidate_cap_per_row": 4096,
        "oof_source_tree_sha256": oof_source_tree_sha256,
        "raw_candidates_retained_for_global_policy_search": True,
        "inference_truncation": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overfit", "train"), required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--onerel-base-model", type=Path, required=True)
    parser.add_argument("--onerel-checkpoint", type=Path, required=True)
    parser.add_argument("--source-train", type=Path)
    parser.add_argument("--oof-manifest", type=Path)
    parser.add_argument("--oof-model-root", type=Path)
    parser.add_argument("--oof-source-tree-sha256")
    parser.add_argument("--cross-encoder-base-model", type=Path, required=True)
    parser.add_argument("--overfit-receipt", type=Path)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proposal-batch-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head-only-epochs", type=int, default=1)
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=None,
        help="deprecated compatibility flag; the overfit schedule is step-bound",
    )
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_tree_sha256):
        raise SystemExit("source tree hash must be lowercase SHA-256")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.mode == "train" and (
        args.calibration_data is None or args.overfit_receipt is None
    ):
        raise SystemExit("formal training requires calibration and overfit receipt")
    if args.mode == "overfit" and args.calibration_data is not None:
        raise SystemExit("overfit must not receive calibration")
    oof_inputs = (
        args.source_train,
        args.oof_manifest,
        args.oof_model_root,
        args.oof_source_tree_sha256,
    )
    if args.mode == "train" and any(path is None for path in oof_inputs):
        raise SystemExit(
            "formal training requires source-train, oof-manifest, and "
            "oof-model-root; in-sample fit proposals are forbidden"
        )
    if args.mode == "overfit" and any(path is not None for path in oof_inputs):
        raise SystemExit("overfit mode must not receive OOF inputs")
    if args.oof_source_tree_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.oof_source_tree_sha256
    ):
        raise SystemExit("OOF source tree hash must be lowercase SHA-256")
    if args.epochs != 3 or args.head_only_epochs != 1:
        raise SystemExit("formal cross-encoder schedule is fixed at 1+2 epochs")
    if args.groups_per_batch < 1 or args.accumulation < 1:
        raise SystemExit("batch controls must be positive")

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    if not torch.npu.is_available():
        raise SystemExit("cross-encoder training requires an Ascend NPU")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    _seed(args.seed, torch)
    started = time.time()
    fit_rows = _jsonl(args.fit_data)
    if args.mode == "overfit":
        fit_rows = fit_rows[:8]
    calibration_rows = (
        _jsonl(args.calibration_data) if args.calibration_data is not None else fit_rows
    )
    fit_gold = [_gold(row) for row in fit_rows]
    calibration_gold = [_gold(row) for row in calibration_rows]

    onerel, onerel_tokenizer, predicates, onerel_payload, onerel_base_sha = _load_onerel(
        checkpoint_path=args.onerel_checkpoint,
        base_model=args.onerel_base_model,
        device=device,
        torch=torch,
    )
    if args.mode == "overfit":
        fit_candidates, fit_proposal = _generate_candidates(
            model=onerel,
            tokenizer=onerel_tokenizer,
            rows=fit_rows,
            predicates=predicates,
            device=device,
            torch=torch,
            batch_size=args.proposal_batch_size,
            fp16=args.fp16,
        )
        calibration_candidates = fit_candidates
        calibration_proposal = fit_proposal
    else:
        calibration_candidates, calibration_proposal = _generate_candidates(
            model=onerel,
            tokenizer=onerel_tokenizer,
            rows=calibration_rows,
            predicates=predicates,
            device=device,
            torch=torch,
            batch_size=args.proposal_batch_size,
            fp16=args.fp16,
            enforce_surface_cap=False,
        )
        del onerel
        torch.npu.empty_cache()
        assert args.source_train is not None
        assert args.oof_manifest is not None
        assert args.oof_model_root is not None
        assert args.oof_source_tree_sha256 is not None
        fit_candidates, fit_proposal = _generate_oof_fit_candidates(
            source_train=args.source_train,
            fit_rows=fit_rows,
            split_manifest_path=args.split_manifest,
            oof_manifest_path=args.oof_manifest,
            model_root=args.oof_model_root,
            base_model=args.onerel_base_model,
            expected_predicates=predicates,
            oof_source_tree_sha256=args.oof_source_tree_sha256,
            device=device,
            torch=torch,
            batch_size=args.proposal_batch_size,
            fp16=args.fp16,
        )
        proposal_policy = _select_proposal_policy(
            fit_candidates=fit_candidates,
            fit_gold=fit_gold,
            guard_candidates=calibration_candidates,
            n_guard_rows=len(calibration_rows),
        )
        if proposal_policy["status"] != "selected":
            args.output.mkdir(parents=True, exist_ok=False)
            rejection = {
                "schema_version": 1,
                "status": "rejected_no_density_safe_policy",
                "dataset_role": "fit_internal_oof_candidate_audit",
                "source_tree_sha256": args.source_tree_sha256,
                "oof_source_tree_sha256": args.oof_source_tree_sha256,
                "oof_manifest_sha256": sha256_file(args.oof_manifest),
                "proposal_policy": proposal_policy,
                "raw_fit_proposal": fit_proposal,
                "inference_truncation": False,
                "calibration_gold_used_for_proposal_selection": False,
                "outer_accessed": False,
                "official_dev_accessed": False,
            }
            _write_json_once(args.output / "FAILED_EVIDENCE.json", rejection)
            print(json.dumps(rejection, ensure_ascii=False, sort_keys=True), flush=True)
            return 2
        fit_candidates, fit_policy_application = _apply_selected_proposal_policy(
            fit_candidates,
            n_rows=len(fit_rows),
            policy=proposal_policy["selected"],
        )
        calibration_candidates, calibration_policy_application = (
            _apply_selected_proposal_policy(
                calibration_candidates,
                n_rows=len(calibration_rows),
                policy=proposal_policy["selected"],
            )
        )
        fit_proposal = {
            **fit_proposal,
            "global_proposal_policy": proposal_policy,
            "selected_policy_application": fit_policy_application,
        }
        calibration_proposal = {
            **calibration_proposal,
            "global_proposal_policy": proposal_policy["selected"],
            "selected_policy_application": calibration_policy_application,
            "calibration_gold_used_for_proposal_selection": False,
        }
    groups, group_audit = build_candidate_hard_negative_groups(
        candidates=fit_candidates,
        gold_rows=fit_gold,
        negative_ratio=NEGATIVE_RATIO,
    )
    relation_targets = _candidate_relation_targets(
        candidates=fit_candidates,
        gold_rows=fit_gold,
        predicates=predicates,
    )
    (
        subject_targets,
        object_targets,
        predicate_presence_targets,
    ) = _candidate_factor_targets(
        candidates=fit_candidates,
        gold_rows=fit_gold,
        predicates=predicates,
    )
    initial_group_audit = dict(group_audit)
    fit_candidate_rows = [set() for _ in fit_rows]
    for candidate in fit_candidates:
        fit_candidate_rows[candidate.row_index].add(candidate.triple)
    fit_candidate_oracle = micro_spo_metric(fit_gold, fit_candidate_rows)
    if not groups:
        raise SystemExit("OneRel proposals contain no reachable fit positives")
    if args.mode == "train" and float(fit_candidate_oracle["recall"]) < 0.60:
        args.output.mkdir(parents=True, exist_ok=False)
        rejection = {
            "schema_version": 1,
            "status": "rejected_oof_candidate_recall",
            "dataset_role": "fit_internal_oof_candidate_audit",
            "source_tree_sha256": args.source_tree_sha256,
            "oof_manifest_sha256": sha256_file(args.oof_manifest),
            "candidate_oracle": fit_candidate_oracle,
            "minimum_candidate_recall": 0.60,
            "proposal": fit_proposal,
            "inference_truncation": False,
            "calibration_accessed": False,
            "outer_accessed": False,
            "official_dev_accessed": False,
        }
        _write_json_once(args.output / "FAILED_EVIDENCE.json", rejection)
        print(json.dumps(rejection, ensure_ascii=False, sort_keys=True), flush=True)
        return 2
    if args.mode == "overfit" and group_audit["reachable_gold_recall"] < 1.0:
        raise SystemExit(
            "fixed 8-row proposal oracle recall is below 1.0; overfit is impossible"
        )
    if args.mode == "overfit":
        del onerel
    torch.npu.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        args.cross_encoder_base_model,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": list(MARKERS)})
    marker_token_ids = tuple(
        int(tokenizer.convert_tokens_to_ids(marker)) for marker in MARKERS
    )
    model = _build_span_pair_model(
        base_model=args.cross_encoder_base_model,
        marker_token_ids=marker_token_ids,
        n_predicates=len(predicates),
        torch=torch,
        auto_model=AutoModel,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= 1_000_000_000:
        raise SystemExit("candidate model parameter count must remain below 1B")
    cross_base_sha = _base_model_tree_sha256(args.cross_encoder_base_model)

    if args.mode == "train":
        overfit = json.loads(args.overfit_receipt.read_text(encoding="utf-8"))
        if overfit.get("status") != "passed" or overfit.get("metric", {}).get("f1", 0) < 0.98:
            raise SystemExit("overfit receipt did not pass F1 >= 0.98")
        gates = overfit.get("overfit_gates") or {}
        exact_gate = gates.get("exact_spo_f1_at_least_0_98")
        if exact_gate is None:
            # Read-only compatibility with receipts created before the
            # explicit proposal-refinement gate was introduced.
            exact_gate = gates.get("head_only_f1_at_least_0_98")
        if not exact_gate or not gates.get("joint_f1_at_least_0_95"):
            raise SystemExit("overfit receipt did not pass both phase gates")
        if overfit.get("source_tree_sha256") != args.source_tree_sha256:
            raise SystemExit("overfit receipt is bound to a different source tree")
        if overfit.get("cross_encoder_base_model_tree_sha256") != cross_base_sha:
            raise SystemExit("overfit receipt is bound to a different cross-encoder base")

    history: list[dict[str, Any]] = []
    best_by_phase: dict[str, dict[str, Any]] = {}
    best_selection: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    if args.overfit_epochs is not None:
        raise SystemExit(
            "--overfit-epochs is no longer accepted; overfit is fixed at "
            f"{OVERFIT_HEAD_STEPS} head-only + {OVERFIT_JOINT_STEPS} joint steps"
        )
    phases = _phase_schedule(
        mode=args.mode,
        n_groups=len(groups),
        groups_per_batch=args.groups_per_batch,
        formal_epochs=args.epochs,
        formal_head_only_epochs=args.head_only_epochs,
    )
    epochs = len(phases)
    optimizer = None
    head: list[Any] = []
    encoder: list[Any] = []
    previous_phase = None
    online_mining_records: list[dict[str, Any]] = []
    for epoch, phase in enumerate(phases, start=1):
        joint = phase == "top2_joint"
        refresh_strategy = _hard_negative_refresh_strategy(
            mode=args.mode,
            epoch=epoch,
            phase=phase,
            previous_phase=previous_phase,
        )
        if refresh_strategy is not None:
            fit_scores = _score_candidates(
                model=model,
                tokenizer=tokenizer,
                candidates=fit_candidates,
                device=device,
                torch=torch,
                fp16=args.fp16,
                batch_size=args.eval_batch_size,
            )
            if refresh_strategy == "model_false_positives":
                groups, group_audit = _build_model_hard_negative_groups(
                    candidates=fit_candidates,
                    gold_rows=fit_gold,
                    scores=fit_scores,
                )
            else:
                groups, group_audit = _build_mixed_hard_negative_groups(
                    candidates=fit_candidates,
                    gold_rows=fit_gold,
                    scores=fit_scores,
                )
            mining_record = {
                "before_epoch": epoch,
                "refresh_strategy": refresh_strategy,
                "score_min": min(fit_scores, default=0.0),
                "score_max": max(fit_scores, default=0.0),
                "score_mean": sum(fit_scores) / max(1, len(fit_scores)),
                "n_scored_candidates": len(fit_scores),
                "groups": group_audit,
                "gold_used_only_to_exclude_true_spo": True,
                "outer_accessed": False,
                "official_dev_accessed": False,
            }
            online_mining_records.append(mining_record)
            print(
                json.dumps(
                    {"stage": "online_hard_negative_mining", **mining_record},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del fit_scores
        if phase != previous_phase:
            optimizer, head, encoder = _optimizer(
                model,
                torch,
                joint=joint,
                overfit=args.mode == "overfit",
            )
            previous_phase = phase
        assert optimizer is not None
        _train_mode(model, joint=joint)
        train_record = _train_epoch(
            model=model,
            tokenizer=tokenizer,
            candidates=fit_candidates,
            relation_targets=relation_targets,
            subject_targets=subject_targets,
            object_targets=object_targets,
            predicate_presence_targets=predicate_presence_targets,
            groups=groups,
            optimizer=optimizer,
            active_parameters=[*head, *encoder],
            device=device,
            torch=torch,
            fp16=args.fp16,
            groups_per_batch=args.groups_per_batch,
            accumulation=1 if args.mode == "overfit" else args.accumulation,
            seed=args.seed * 1000 + epoch,
        )
        should_evaluate = (
            args.mode == "train"
            or epoch % 5 == 0
            or epoch == epochs
            or (epoch < epochs and phases[epoch] != phase)
        )
        if not should_evaluate:
            continue
        scores = _score_candidates(
            model=model,
            tokenizer=tokenizer,
            candidates=calibration_candidates,
            device=device,
            torch=torch,
            fp16=args.fp16,
            batch_size=args.eval_batch_size,
        )
        selection = _select_threshold(
            gold_rows=calibration_gold,
            candidates=calibration_candidates,
            scores=scores,
        )
        record = {
            "epoch": epoch,
            "phase": "top2_joint" if joint else "head_only",
            "train": train_record,
            "selection": selection,
        }
        history.append(record)
        phase_best = best_by_phase.get(phase)
        if phase_best is None or (
            selection["metric"]["f1"],
            selection["metric"]["precision"],
            selection["threshold"],
        ) > (
            phase_best["metric"]["f1"],
            phase_best["metric"]["precision"],
            phase_best["threshold"],
        ):
            best_by_phase[phase] = selection
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "phase": record["phase"],
                    "loss": train_record["mean_loss"],
                    "metric": selection["metric"],
                    "threshold": selection["threshold"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        eligible_for_selection = args.mode == "train" or joint
        if eligible_for_selection and (best_selection is None or (
            selection["metric"]["f1"],
            selection["metric"]["precision"],
            selection["threshold"],
        ) > (
            best_selection["metric"]["f1"],
            best_selection["metric"]["precision"],
            best_selection["threshold"],
        )):
            best_selection = selection
            best_state = _snapshot_state(model)
        if (
            args.mode == "overfit"
            and joint
            and selection["metric"]["f1"] >= 0.999
        ):
            break

    assert best_selection is not None and best_state is not None
    if args.mode == "overfit":
        head_best = best_by_phase["head_only"]
        joint_best = best_by_phase["top2_joint"]
        overfit_gates, exact_best = _overfit_gate_payload(
            head_best=head_best,
            joint_best=joint_best,
        )
        status = "passed" if (
            overfit_gates["exact_spo_f1_at_least_0_98"]
            and overfit_gates["joint_f1_at_least_0_95"]
        ) else "failed"
        reported_selection = exact_best
    else:
        overfit_gates = None
        status = "passed" if best_selection["metric"]["f1"] >= 0.45 else "failed"
        reported_selection = best_selection
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = None
    if args.mode == "train":
        checkpoint_sha = _save_torch_once(
            args.output / "model.pt",
            {
                "model_family": "candidate_cross_encoder",
                "architecture_version": FACTOR_HEAD_VERSION,
                "model_state": best_state,
                "predicates": predicates,
                "source_tree_sha256": args.source_tree_sha256,
                "cross_encoder_base_model_tree_sha256": cross_base_sha,
                "onerel_checkpoint_sha256": sha256_file(args.onerel_checkpoint),
                "fit_candidate_source": fit_proposal.get(
                    "candidate_source", "fixed_eight_row_full_fit_checkpoint"
                ),
                "oof_manifest_sha256": (
                    sha256_file(args.oof_manifest) if args.oof_manifest else None
                ),
                "oof_source_tree_sha256": args.oof_source_tree_sha256,
                "max_length": MAX_LENGTH,
                "candidate_hidden_size": CANDIDATE_HIDDEN_SIZE,
                "predicate_embedding_size": PREDICATE_EMBEDDING_SIZE,
                "proposal_feature_size": PROPOSAL_FEATURE_SIZE,
                "marker_token_ids": list(marker_token_ids),
                "markers": list(MARKERS),
                "selection": best_selection,
                "online_hard_negative_mining": online_mining_records,
                "seed": args.seed,
            },
            torch,
        )
        tokenizer.save_pretrained(args.output / "tokenizer")
    factor_target_audit = {
        "all_fit_candidates": _factor_target_audit(
            candidates=fit_candidates,
            subject_targets=subject_targets,
            object_targets=object_targets,
            predicate_presence_targets=predicate_presence_targets,
            indices=range(len(fit_candidates)),
        ),
        "final_sampled_group_occurrences": _factor_target_audit(
            candidates=fit_candidates,
            subject_targets=subject_targets,
            object_targets=object_targets,
            predicate_presence_targets=predicate_presence_targets,
            indices=(index for group in groups for index in group),
        ),
    }
    payload = {
        "schema_version": 1,
        "model_family": "candidate_cross_encoder",
        "architecture_version": FACTOR_HEAD_VERSION,
        "mode": args.mode,
        "status": status,
        "dataset_role": "fit_overfit" if args.mode == "overfit" else "calibration_selection_used",
        "outer_accessed": False,
        "official_dev_accessed": False,
        "source_tree_sha256": args.source_tree_sha256,
        "fit_data_sha256": sha256_file(args.fit_data),
        "calibration_data_sha256": (
            sha256_file(args.calibration_data) if args.calibration_data else None
        ),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "onerel_checkpoint_sha256": sha256_file(args.onerel_checkpoint),
        "fit_candidate_source": fit_proposal.get(
            "candidate_source", "fixed_eight_row_full_fit_checkpoint"
        ),
        "source_train_sha256": (
            sha256_file(args.source_train) if args.source_train else None
        ),
        "oof_manifest_sha256": (
            sha256_file(args.oof_manifest) if args.oof_manifest else None
        ),
        "oof_source_tree_sha256": args.oof_source_tree_sha256,
        "oof_model_root": (
            str(args.oof_model_root.resolve()) if args.oof_model_root else None
        ),
        "onerel_architecture_version": onerel_payload.get("architecture_version"),
        "onerel_base_model_tree_sha256": onerel_base_sha,
        "cross_encoder_base_model_tree_sha256": cross_base_sha,
        "script_sha256": sha256_file(Path(__file__)),
        "seed": args.seed,
        "device": "npu:0",
        "device_name": torch.npu.get_device_name(0),
        "fp16": bool(args.fp16),
        "parameter_count": parameter_count,
        "parameter_limit_exclusive": 1_000_000_000,
        "schedule": {
            "max_length": MAX_LENGTH,
            "candidate_head": (
                "cls-four-marker-subject-object-content-"
                "pair-interactions-plus-factorized-pair-existence-and-"
                "44-label-relation-plus-subject-role-object-role-and-"
                "sentence-predicate-presence-auxiliary-heads-without-"
                "proposed-predicate-text-leakage"
            ),
            "factor_head_version": FACTOR_HEAD_VERSION,
            "factor_head_final_layers_zero_initialized": True,
            "factor_fusion": (
                "verifier + sigmoid(pair_scale)*pair + "
                "sigmoid(relation_scale)*selected_relation + "
                "sigmoid(subject_scale)*subject_valid + "
                "sigmoid(object_scale)*object_valid + "
                "sigmoid(predicate_presence_scale)*selected_sentence_predicate"
            ),
            "candidate_hidden_size": CANDIDATE_HIDDEN_SIZE,
            "predicate_embedding_size": PREDICATE_EMBEDDING_SIZE,
            "proposal_features": [
                "tanh_relation_score",
                "tanh_link_score",
                "tanh_boundary_score",
            ],
            "marker_token_ids": list(marker_token_ids),
            "epochs": epochs,
            "formal_head_only_epochs": 1,
            "formal_top2_joint_epochs": 2,
            "head_lr": HEAD_LR,
            "encoder_lr": ENCODER_LR,
            "overfit_head_lr": OVERFIT_HEAD_LR,
            "overfit_joint_head_lr": OVERFIT_JOINT_HEAD_LR,
            "overfit_encoder_lr": OVERFIT_ENCODER_LR,
            "overfit_head_steps_floor": OVERFIT_HEAD_STEPS,
            "overfit_joint_steps_floor": OVERFIT_JOINT_STEPS,
            "optimizer_state_preserved_within_phase": True,
            "encoder_adam_eps": 1e-4,
            "negative_ratio": NEGATIVE_RATIO,
            "margin": MARGIN,
            "margin_weight": MARGIN_WEIGHT,
            "listwise_weight": LISTWISE_WEIGHT,
            "pair_margin_weight": PAIR_MARGIN_WEIGHT,
            "relation_margin_weight": RELATION_MARGIN_WEIGHT,
            "relation_auxiliary_weight": RELATION_AUX_WEIGHT,
            "pair_auxiliary_weight": PAIR_AUX_WEIGHT,
            "subject_role_auxiliary_weight": SUBJECT_AUX_WEIGHT,
            "object_role_auxiliary_weight": OBJECT_AUX_WEIGHT,
            "predicate_presence_auxiliary_weight": (
                PREDICATE_PRESENCE_AUX_WEIGHT
            ),
            "pair_auxiliary_labels": "fit_exact_subject_object_existence",
            "pair_auxiliary_scope": "all_sampled_fit_candidates",
            "relation_auxiliary_labels": "fit_exact_subject_object_multi_predicate",
            "relation_auxiliary_scope": "gold_entity_pairs_only",
            "relation_auxiliary_loss": (
                "normalized_multi_target_softmax_cross_entropy"
            ),
            "subject_role_auxiliary_labels": (
                "fit_row_surface_appears_in_any_gold_subject_role"
            ),
            "object_role_auxiliary_labels": (
                "fit_row_surface_appears_in_any_gold_object_role"
            ),
            "predicate_presence_auxiliary_labels": (
                "fit_row_gold_predicate_multilabel"
            ),
            "factor_auxiliary_scope": "all_sampled_fit_candidates",
            "predicate_presence_input": (
                "unmarked_source_sentence_unique_first_occurrence_batch_order"
            ),
            "encoder_predicate_conditioning": (
                "predicate_id_embedding_only_no_second_sequence_text"
            ),
            "pair_fusion_scale": float(
                torch.sigmoid(best_state["pair_scale_logit"]).cpu()
            ),
            "relation_fusion_scale": float(
                torch.sigmoid(best_state["relation_scale_logit"]).cpu()
            ),
            "subject_role_fusion_scale": float(
                torch.sigmoid(best_state["subject_scale_logit"]).cpu()
            ),
            "object_role_fusion_scale": float(
                torch.sigmoid(best_state["object_scale_logit"]).cpu()
            ),
            "predicate_presence_fusion_scale": float(
                torch.sigmoid(
                    best_state["predicate_presence_scale_logit"]
                ).cpu()
            ),
            "global_thresholds": list(THRESHOLDS),
            "per_predicate_thresholds": False,
            "per_factor_thresholds": False,
            "online_hard_negative_mining": {
                "enabled_for_formal_training": args.mode == "train",
                "enabled_for_overfit_refinement": args.mode == "overfit",
                "refresh_before_epochs": (
                    [2, 3]
                    if args.mode == "train"
                    else [
                        next(
                            index
                            for index, phase in enumerate(phases, start=1)
                            if phase == "top2_joint"
                        )
                    ]
                ),
                "scoring_scope": "all_density_safe_fit_candidates",
                "maximum_negatives_per_positive": NEGATIVE_RATIO,
                "formal_proposal_hard_per_positive": 4,
                "formal_model_hard_per_positive": 4,
                "overfit_model_hard_per_positive": 8,
                "joint_epochs_reuse_same_groups": False,
            },
        },
        "fit_proposal": fit_proposal,
        "fit_candidate_oracle": fit_candidate_oracle,
        "calibration_proposal": calibration_proposal,
        "initial_hard_negative_groups": initial_group_audit,
        "hard_negative_groups": group_audit,
        "factor_target_audit": factor_target_audit,
        "online_hard_negative_mining": online_mining_records,
        "metric": reported_selection["metric"],
        "selected_threshold": reported_selection["threshold"],
        "selection": reported_selection,
        "overfit_gates": overfit_gates,
        "history": history,
        "checkpoint_sha256": checkpoint_sha,
        "elapsed_seconds": time.time() - started,
    }
    receipt_name = "overfit_receipt.json" if args.mode == "overfit" else "selection.json"
    _write_json_once(args.output / receipt_name, payload)
    print(
        json.dumps(
            {
                "status": status,
                "metric": payload["metric"],
                "threshold": payload["selected_threshold"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
