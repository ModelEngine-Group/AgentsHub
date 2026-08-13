"""OneRel-style single-stage triple extraction primitives for CMeIE.

Each exact ``(subject, predicate, object)`` mention is represented by three
relation-specific links:

* ``HB-TB``: subject begin -> object begin
* ``HB-TE``: subject begin -> object end
* ``HE-TE``: subject end   -> object end

Unlike the older GPLinker and CasRel paths, a decoded triple does not depend on
an independently predicted entity span or a teacher-forced subject.  The three
links are scored from one shared token-pair representation.  The implementation
uses multi-label links rather than OneRel's mutually-exclusive four-way cell so
that rare, but legal, link collisions are not silently discarded.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dki.nlp.cmeie_spo import char_span_to_tokens, select_mention_pairs


ONEREL_ARCHITECTURE_VERSION = 4
LINK_NAMES = ("HB-TB", "HB-TE", "HE-TE")
HB_TB = 0
HB_TE = 1
HE_TE = 2


class OneRelDensityError(RuntimeError):
    """Raised instead of silently truncating a dense OneRel prediction."""


@dataclass(frozen=True)
class OneRelLabels:
    """Sparse, exact labels for one source row."""

    links: tuple[tuple[int, int, int, int], ...]
    triples: tuple[tuple[int, int, int, int, int], ...]
    dropped_spo: int


@dataclass(frozen=True)
class OneRelSpanCandidate:
    """One exact span quadruple proposed by the three OneRel links."""

    predicate_id: int
    subject_start: int
    subject_end: int
    object_start: int
    object_end: int
    relation_score: float
    link_score: float
    boundary_score: float


def build_onerel_labels(
    row: dict[str, Any],
    offsets: Sequence[tuple[int, int]],
    predicate2id: dict[str, int],
    *,
    allow_cross_sentence_fallback: bool = True,
) -> OneRelLabels:
    """Align surface-form SPO labels to relation-specific token-pair links."""

    text = str(row.get("text", ""))
    sequence = len(offsets)
    links: set[tuple[int, int, int, int]] = set()
    triples: set[tuple[int, int, int, int, int]] = set()
    dropped = 0
    offset_list = list(offsets)
    for spo in row.get("spo_list", []):
        predicate = str(spo.get("predicate", ""))
        if predicate not in predicate2id:
            dropped += 1
            continue
        mention_pairs = select_mention_pairs(
            text,
            str(spo.get("subject", "")),
            str(spo.get("object", "")),
            allow_cross_sentence_fallback=allow_cross_sentence_fallback,
        )
        aligned = 0
        for subject_chars, object_chars in mention_pairs:
            subject = char_span_to_tokens(offset_list, subject_chars)
            obj = char_span_to_tokens(offset_list, object_chars)
            if subject is None or obj is None:
                continue
            ss, se = subject
            os_, oe = obj
            if not (
                0 <= ss <= se < sequence
                and 0 <= os_ <= oe < sequence
                and offsets[ss][1] > offsets[ss][0]
                and offsets[se][1] > offsets[se][0]
                and offsets[os_][1] > offsets[os_][0]
                and offsets[oe][1] > offsets[oe][0]
            ):
                continue
            relation = predicate2id[predicate]
            triples.add((relation, ss, se, os_, oe))
            links.add((relation, HB_TB, ss, os_))
            links.add((relation, HB_TE, ss, oe))
            links.add((relation, HE_TE, se, oe))
            aligned += 1
        if not aligned:
            dropped += 1
    return OneRelLabels(
        links=tuple(sorted(links)),
        triples=tuple(sorted(triples)),
        dropped_spo=dropped,
    )


def dense_onerel_targets(
    labels: OneRelLabels,
    sequence: int,
    predicates: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize link, relation-presence and role-boundary targets."""

    if sequence < 1 or predicates < 1:
        raise ValueError("sequence and predicate counts must be positive")
    links = torch.zeros(
        (predicates, len(LINK_NAMES), sequence, sequence),
        dtype=torch.float32,
        device=device,
    )
    relation = torch.zeros((predicates,), dtype=torch.float32, device=device)
    boundary = torch.zeros((sequence, 4), dtype=torch.float32, device=device)
    for predicate, tag, left, right in labels.links:
        if not (
            0 <= predicate < predicates
            and 0 <= tag < len(LINK_NAMES)
            and 0 <= left < sequence
            and 0 <= right < sequence
        ):
            raise ValueError("OneRel link is out of bounds")
        links[predicate, tag, left, right] = 1.0
    for predicate, ss, se, os_, oe in labels.triples:
        if not (
            0 <= predicate < predicates
            and 0 <= ss <= se < sequence
            and 0 <= os_ <= oe < sequence
        ):
            raise ValueError("OneRel triple is out of bounds")
        relation[predicate] = 1.0
        boundary[ss, 0] = 1.0
        boundary[se, 1] = 1.0
        boundary[os_, 2] = 1.0
        boundary[oe, 3] = 1.0
    return links, relation, boundary


def structured_negative_links(
    labels: OneRelLabels,
    valid_token_indices: Sequence[int],
    predicates: int,
    *,
    seed: int,
    maximum_pool: int = 4096,
    random_per_positive: int = 16,
) -> tuple[tuple[int, int, int, int], ...]:
    """Build a deterministic, gold-free pool of structured link negatives.

    The pool contains wrong predicates at the exact positive coordinates,
    wrong corner tags, one-token boundary perturbations, crossed gold
    coordinates and deterministic random background cells.  It is a training
    pool only; inference never truncates candidates.
    """

    if predicates < 2:
        raise ValueError("at least two predicates are required")
    if maximum_pool < 1 or random_per_positive < 0:
        raise ValueError("negative-pool limits are invalid")
    valid = tuple(sorted(set(int(index) for index in valid_token_indices)))
    if not valid:
        raise ValueError("at least one valid token is required")
    valid_set = set(valid)
    positives = set(labels.links)
    negatives: set[tuple[int, int, int, int]] = set()

    def add(candidate: tuple[int, int, int, int]) -> None:
        relation, tag, left, right = candidate
        if (
            0 <= relation < predicates
            and 0 <= tag < len(LINK_NAMES)
            and left in valid_set
            and right in valid_set
            and candidate not in positives
        ):
            negatives.add(candidate)

    position = {token: index for index, token in enumerate(valid)}
    links_by_tag: dict[int, list[tuple[int, int, int]]] = {
        tag: [] for tag in range(len(LINK_NAMES))
    }
    for relation, tag, left, right in labels.links:
        links_by_tag[tag].append((relation, left, right))
        for wrong_relation in range(predicates):
            if wrong_relation != relation:
                add((wrong_relation, tag, left, right))
        for wrong_tag in range(len(LINK_NAMES)):
            if wrong_tag != tag:
                add((relation, wrong_tag, left, right))
        for token, replace_left in ((left, True), (right, False)):
            index = position.get(token)
            if index is None:
                continue
            for neighbour in (index - 1, index + 1):
                if not 0 <= neighbour < len(valid):
                    continue
                if replace_left:
                    add((relation, tag, valid[neighbour], right))
                else:
                    add((relation, tag, left, valid[neighbour]))
        add((relation, tag, left, left))
        add((relation, tag, right, right))

    for tag, tagged in links_by_tag.items():
        lefts = sorted({left for _relation, left, _right in tagged})
        rights = sorted({right for _relation, _left, right in tagged})
        relations = sorted({relation for relation, _left, _right in tagged})
        for relation in relations:
            for left in lefts:
                for right in rights:
                    add((relation, tag, left, right))

    rng = random.Random(int(seed))
    random_target = max(
        len(negatives),
        min(maximum_pool, max(1, len(positives)) * random_per_positive),
    )
    attempts = 0
    maximum_attempts = max(100, random_target * 20)
    while len(negatives) < random_target and attempts < maximum_attempts:
        attempts += 1
        add(
            (
                rng.randrange(predicates),
                rng.randrange(len(LINK_NAMES)),
                valid[rng.randrange(len(valid))],
                valid[rng.randrange(len(valid))],
            )
        )
    return tuple(sorted(negatives)[:maximum_pool])


def structured_triple_candidates(
    labels: OneRelLabels,
    valid_token_indices: Sequence[int],
    predicates: int,
    *,
    seed: int,
    maximum_negative_pool: int = 4096,
    random_per_positive: int = 16,
) -> tuple[
    tuple[tuple[int, int, int, int, int], ...],
    tuple[tuple[int, int, int, int, int], ...],
]:
    """Return exact positives and deterministic verifier hard negatives.

    In addition to wrong predicates and perturbed boundaries, the pool
    explicitly contains every ghost quadruple that the row's *gold* OneRel
    links can compose.  This closes the structural supervision gap exposed by
    repeated mentions and rows containing several triples for one relation.
    """

    if predicates < 2:
        raise ValueError("at least two predicates are required")
    if maximum_negative_pool < 1 or random_per_positive < 0:
        raise ValueError("verifier negative-pool limits are invalid")
    valid = tuple(sorted(set(int(index) for index in valid_token_indices)))
    if not valid:
        raise ValueError("at least one valid token is required")
    valid_set = set(valid)
    gold = set(labels.triples)
    negatives: set[tuple[int, int, int, int, int]] = set()

    def add(candidate: tuple[int, int, int, int, int]) -> None:
        relation, ss, se, os_, oe = candidate
        if (
            0 <= relation < predicates
            and ss in valid_set
            and se in valid_set
            and os_ in valid_set
            and oe in valid_set
            and ss <= se
            and os_ <= oe
            and candidate not in gold
        ):
            negatives.add(candidate)

    position = {token: index for index, token in enumerate(valid)}
    subjects = sorted({(ss, se) for _r, ss, se, _os, _oe in gold})
    objects = sorted({(os_, oe) for _r, _ss, _se, os_, oe in gold})
    for relation, ss, se, os_, oe in sorted(gold):
        for wrong_relation in range(predicates):
            if wrong_relation != relation:
                add((wrong_relation, ss, se, os_, oe))
        for index, value in enumerate((ss, se, os_, oe), start=1):
            location = position.get(value)
            if location is None:
                continue
            for neighbour in (location - 1, location + 1):
                if not 0 <= neighbour < len(valid):
                    continue
                changed = [relation, ss, se, os_, oe]
                changed[index] = valid[neighbour]
                add(tuple(changed))  # type: ignore[arg-type]
        add((relation, os_, oe, ss, se))
        add((relation, ss, se, ss, se))
        for subject_start, subject_end in subjects:
            for object_start, object_end in objects:
                add(
                    (
                        relation,
                        subject_start,
                        subject_end,
                        object_start,
                        object_end,
                    )
                )

    # Reproduce every collision obtainable from the row's perfect gold links.
    for relation in range(predicates):
        begin: dict[int, set[int]] = {}
        head_to_object_end: dict[int, set[int]] = {}
        subject_end_by_object_end: dict[int, set[int]] = {}
        for link_relation, tag, left, right in labels.links:
            if link_relation != relation:
                continue
            if tag == HB_TB:
                begin.setdefault(left, set()).add(right)
            elif tag == HB_TE:
                head_to_object_end.setdefault(left, set()).add(right)
            elif tag == HE_TE:
                subject_end_by_object_end.setdefault(right, set()).add(left)
        for ss, object_starts in begin.items():
            for oe in head_to_object_end.get(ss, ()):
                for se in subject_end_by_object_end.get(oe, ()):
                    for os_ in object_starts:
                        add((relation, ss, se, os_, oe))

    rng = random.Random(int(seed))
    random_target = max(
        len(negatives),
        min(
            maximum_negative_pool,
            max(1, len(gold)) * int(random_per_positive),
        ),
    )
    attempts = 0
    maximum_attempts = max(100, random_target * 30)
    while len(negatives) < random_target and attempts < maximum_attempts:
        attempts += 1
        left_a = rng.randrange(len(valid))
        right_a = rng.randrange(left_a, min(len(valid), left_a + 16))
        left_b = rng.randrange(len(valid))
        right_b = rng.randrange(left_b, min(len(valid), left_b + 16))
        add(
            (
                rng.randrange(predicates),
                valid[left_a],
                valid[right_a],
                valid[left_b],
                valid[right_b],
            )
        )
    return tuple(sorted(gold)), tuple(sorted(negatives)[:maximum_negative_pool])


class OneRelContextAdapter(nn.Module):
    """Zero-initialized residual adapter over a frozen contextual encoder."""

    def __init__(self, hidden_size: int, bottleneck: int = 256) -> None:
        super().__init__()
        width = min(int(bottleneck), max(16, hidden_size // 2))
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, width)
        self.up = nn.Linear(width, hidden_size)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(hidden))))
        return hidden + residual * self.scale.float().clamp(0.0, 1.0)


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable, zero-start low-rank update."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.b(self.a(values)) * self.scaling


def inject_bert_attention_lora(
    encoder: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
) -> list[nn.Parameter]:
    """Inject LoRA into every BERT query/value projection.

    Base encoder parameters remain frozen.  The returned parameters are the
    only trainable encoder parameters and can be placed in their own optimizer
    group with a conservative Adam epsilon.
    """

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError("encoder does not expose BERT encoder.layer modules")
    injected: list[nn.Parameter] = []
    for layer in layers:
        attention = layer.attention.self
        for name in ("query", "value"):
            base = getattr(attention, name)
            if isinstance(base, LoRALinear):
                raise ValueError("LoRA has already been injected")
            if not isinstance(base, nn.Linear):
                raise ValueError(f"BERT attention {name} is not linear")
            wrapped = LoRALinear(base, rank=rank, alpha=alpha)
            setattr(attention, name, wrapped)
            injected.extend([wrapped.a.weight, wrapped.b.weight])
    return injected


class CMeIEOneRel(nn.Module):
    """RoBERTa/BERT encoder with one shared relation-aware pair table."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        predicates: int,
        *,
        pair_rank: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if predicates < 1 or pair_rank < 8:
            raise ValueError("invalid OneRel dimensions")
        self.encoder = encoder
        self.predicates = int(predicates)
        self.pair_rank = int(pair_rank)
        self.context_adapter = OneRelContextAdapter(hidden_size)
        self.left_projection = nn.Linear(hidden_size, self.pair_rank)
        self.right_projection = nn.Linear(hidden_size, self.pair_rank)
        self.pair_norm = nn.LayerNorm(self.pair_rank * 4)
        self.pair_hidden = nn.Linear(self.pair_rank * 4, self.pair_rank)
        self.pair_dropout = nn.Dropout(float(dropout))
        self.link_weight = nn.Parameter(
            torch.empty(self.predicates, len(LINK_NAMES), self.pair_rank)
        )
        self.link_bias = nn.Parameter(
            torch.full((self.predicates, len(LINK_NAMES)), -4.0)
        )
        self.relation_presence = nn.Linear(hidden_size, self.predicates)
        self.boundary = nn.Linear(hidden_size, 4)
        self.subject_span_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, self.pair_rank),
            nn.GELU(),
        )
        self.object_span_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, self.pair_rank),
            nn.GELU(),
        )
        self.predicate_embedding = nn.Embedding(self.predicates, self.pair_rank)
        self.verifier_norm = nn.LayerNorm(self.pair_rank * 7)
        self.verifier_hidden = nn.Linear(self.pair_rank * 7, self.pair_rank)
        self.verifier_dropout = nn.Dropout(float(dropout))
        self.verifier_output = nn.Linear(self.pair_rank, 1)
        self.proposal_calibrator = nn.Linear(3, 1)
        # Endpoint-only candidate scoring cannot see the lexical trigger between
        # a subject and object. The context branch pools both spans, their
        # covering window and the row-level CLS representation, then conditions
        # that evidence on the predicate. Its final projection is zero-start so
        # architecture-v3 checkpoints migrate without changing one logit.
        self.verifier_context_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, self.pair_rank),
            nn.GELU(),
        )
        self.verifier_context_norm = nn.LayerNorm(self.pair_rank * 4)
        self.verifier_context_hidden = nn.Linear(
            self.pair_rank * 4, self.pair_rank
        )
        self.verifier_context_output = nn.Linear(self.pair_rank, 1)
        nn.init.xavier_uniform_(self.link_weight)
        nn.init.constant_(self.relation_presence.bias, -2.0)
        nn.init.constant_(self.boundary.bias, -2.0)
        nn.init.constant_(self.verifier_output.bias, -2.0)
        nn.init.zeros_(self.proposal_calibrator.weight)
        nn.init.zeros_(self.proposal_calibrator.bias)
        nn.init.zeros_(self.verifier_context_output.weight)
        nn.init.zeros_(self.verifier_context_output.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, torch.Tensor] = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        encoded = self.encoder(input_ids, **kwargs)
        hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
        return self.context_adapter(hidden)

    def auxiliary_logits(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.relation_presence(pooled).float(), self.boundary(hidden).float()

    def _pair_features(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [left, right, left * right, torch.abs(left - right)],
            dim=-1,
        )
        return self.pair_dropout(
            F.gelu(self.pair_hidden(self.pair_norm(features)))
        )

    def score_sparse(
        self,
        hidden: torch.Tensor,
        parent_indices: torch.Tensor,
        predicate_ids: torch.Tensor,
        tag_ids: torch.Tensor,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
    ) -> torch.Tensor:
        sizes = {
            int(parent_indices.numel()),
            int(predicate_ids.numel()),
            int(tag_ids.numel()),
            int(left_indices.numel()),
            int(right_indices.numel()),
        }
        if len(sizes) != 1:
            raise ValueError("sparse OneRel index tensors must have equal length")
        if not sizes or next(iter(sizes)) == 0:
            return hidden.new_empty((0,), dtype=torch.float32)
        parent_indices = parent_indices.long()
        predicate_ids = predicate_ids.long()
        tag_ids = tag_ids.long()
        left_indices = left_indices.long()
        right_indices = right_indices.long()
        if (
            bool((parent_indices < 0).any())
            or bool((parent_indices >= hidden.shape[0]).any())
            or bool((predicate_ids < 0).any())
            or bool((predicate_ids >= self.predicates).any())
            or bool((tag_ids < 0).any())
            or bool((tag_ids >= len(LINK_NAMES)).any())
            or bool((left_indices < 0).any())
            or bool((left_indices >= hidden.shape[1]).any())
            or bool((right_indices < 0).any())
            or bool((right_indices >= hidden.shape[1]).any())
        ):
            raise ValueError("sparse OneRel index is out of range")
        left_all = self.left_projection(hidden)
        right_all = self.right_projection(hidden)
        left = left_all[parent_indices, left_indices]
        right = right_all[parent_indices, right_indices]
        pair = self._pair_features(left, right)
        weight = self.link_weight[predicate_ids, tag_ids]
        bias = self.link_bias[predicate_ids, tag_ids]
        return (pair.float() * weight.float()).sum(dim=-1) + bias.float()

    def score_dense(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, relation, tag, left, right]`` raw link logits."""

        left = self.left_projection(hidden)
        right = self.right_projection(hidden)
        pair = self._pair_features(
            left.unsqueeze(2).expand(-1, -1, hidden.shape[1], -1),
            right.unsqueeze(1).expand(-1, hidden.shape[1], -1, -1),
        )
        logits = torch.einsum(
            "bijh,rth->brtij",
            pair,
            self.link_weight.to(dtype=pair.dtype),
        ).float()
        return logits + self.link_bias.float().view(1, self.predicates, len(LINK_NAMES), 1, 1)

    def score_triples(
        self,
        hidden: torch.Tensor,
        parent_indices: torch.Tensor,
        predicate_ids: torch.Tensor,
        subject_starts: torch.Tensor,
        subject_ends: torch.Tensor,
        object_starts: torch.Tensor,
        object_ends: torch.Tensor,
        proposal_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score exact ``(predicate, ss, se, os, oe)`` candidates."""

        tensors = (
            parent_indices,
            predicate_ids,
            subject_starts,
            subject_ends,
            object_starts,
            object_ends,
        )
        sizes = {int(value.numel()) for value in tensors}
        if len(sizes) != 1:
            raise ValueError("verifier index tensors must have equal length")
        if next(iter(sizes)) == 0:
            return hidden.new_empty((0,), dtype=torch.float32)
        parent_indices, predicate_ids, subject_starts, subject_ends, object_starts, object_ends = (
            value.long() for value in tensors
        )
        sequence = int(hidden.shape[1])
        if (
            bool((parent_indices < 0).any())
            or bool((parent_indices >= hidden.shape[0]).any())
            or bool((predicate_ids < 0).any())
            or bool((predicate_ids >= self.predicates).any())
            or bool((subject_starts < 0).any())
            or bool((subject_starts >= sequence).any())
            or bool((subject_ends < subject_starts).any())
            or bool((subject_ends >= sequence).any())
            or bool((object_starts < 0).any())
            or bool((object_starts >= sequence).any())
            or bool((object_ends < object_starts).any())
            or bool((object_ends >= sequence).any())
        ):
            raise ValueError("verifier index is out of range")
        subject = self.subject_span_projection(
            torch.cat(
                [
                    hidden[parent_indices, subject_starts],
                    hidden[parent_indices, subject_ends],
                ],
                dim=-1,
            )
        )
        obj = self.object_span_projection(
            torch.cat(
                [
                    hidden[parent_indices, object_starts],
                    hidden[parent_indices, object_ends],
                ],
                dim=-1,
            )
        )
        predicate = self.predicate_embedding(predicate_ids)
        features = torch.cat(
            [
                subject,
                obj,
                subject * obj,
                torch.abs(subject - obj),
                predicate,
                subject * predicate,
                obj * predicate,
            ],
            dim=-1,
        )
        features = self.verifier_dropout(
            F.gelu(self.verifier_hidden(self.verifier_norm(features)))
        )
        semantic_score = self.verifier_output(features).squeeze(-1).float()

        # Prefix sums avoid materializing one [candidate, sequence, hidden]
        # tensor. Candidate scoring therefore stays linear in exact proposals
        # even when a row contains many repeated mentions.
        prefix = torch.cat(
            (
                hidden.new_zeros((hidden.shape[0], 1, hidden.shape[2])),
                hidden.float().cumsum(dim=1).to(dtype=hidden.dtype),
            ),
            dim=1,
        )

        def span_mean(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
            total = (
                prefix[parent_indices, ends + 1].float()
                - prefix[parent_indices, starts].float()
            )
            width = (ends - starts + 1).clamp_min(1).float().unsqueeze(-1)
            return total / width

        subject_mean = span_mean(subject_starts, subject_ends)
        object_mean = span_mean(object_starts, object_ends)
        window_starts = torch.minimum(subject_starts, object_starts)
        window_ends = torch.maximum(subject_ends, object_ends)
        window_mean = span_mean(window_starts, window_ends)
        row_context = hidden[parent_indices, 0].float()
        context = self.verifier_context_projection(
            torch.cat(
                (subject_mean, object_mean, window_mean, row_context), dim=-1
            )
        )
        context_features = torch.cat(
            (
                context,
                predicate,
                context * predicate,
                torch.abs(context - predicate),
            ),
            dim=-1,
        )
        context_features = self.verifier_dropout(
            F.gelu(
                self.verifier_context_hidden(
                    self.verifier_context_norm(context_features)
                )
            )
        )
        context_score = self.verifier_context_output(context_features).squeeze(-1)
        semantic_score = semantic_score + context_score.float()
        if proposal_scores is None:
            return semantic_score
        if proposal_scores.shape != (semantic_score.numel(), 3):
            raise ValueError("proposal scores must have shape [candidates, 3]")
        # Bounded monotonic features prevent an unusually large raw proposal
        # logit from dominating the verifier gradient.  The zero-initialized
        # calibrator makes architecture-v2 checkpoint migration behaviorally
        # exact before refinement.
        calibrated = self.proposal_calibrator(
            torch.tanh(proposal_scores.float() / 4.0)
        ).squeeze(-1)
        return semantic_score + calibrated.float()


def balanced_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    negative_ratio: int = 16,
    minimum_negatives: int = 32,
) -> torch.Tensor:
    """Balanced BCE with exact positives and the hardest valid negatives."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    try:
        mask = torch.broadcast_to(valid_mask.bool(), logits.shape)
    except RuntimeError as exc:
        raise ValueError("valid mask is not broadcastable to logits") from exc
    if negative_ratio < 1 or minimum_negatives < 0:
        raise ValueError("hard-negative settings are invalid")
    positive = mask & (targets > 0.5)
    negative = mask & ~positive
    positive_logits = logits[positive]
    negative_logits = logits[negative]
    if positive_logits.numel():
        positive_loss = F.softplus(-positive_logits.float()).mean()
        wanted = max(minimum_negatives, int(positive_logits.numel()) * negative_ratio)
    else:
        positive_loss = logits.float().sum() * 0.0
        wanted = minimum_negatives
    if negative_logits.numel() and wanted:
        count = min(wanted, int(negative_logits.numel()))
        hardest = torch.topk(negative_logits.float(), count).values
        negative_loss = F.softplus(hardest).mean()
    else:
        negative_loss = logits.float().sum() * 0.0
    return positive_loss + negative_loss


def grouped_balanced_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    negative_ratio: int = 16,
    minimum_negatives: int = 32,
) -> torch.Tensor:
    """Apply hard-negative mining independently to every source row."""

    if logits.ndim != 1 or targets.shape != logits.shape:
        raise ValueError("grouped verifier logits and targets must be one-dimensional")
    if group_indices.shape != logits.shape:
        raise ValueError("verifier groups and logits differ")
    if not logits.numel():
        return logits.float().sum() * 0.0
    groups = sorted(set(int(value) for value in group_indices.detach().cpu().tolist()))
    losses = []
    for group in groups:
        selected = group_indices == group
        losses.append(
            balanced_hard_negative_loss(
                logits[selected],
                targets[selected],
                torch.ones_like(targets[selected], dtype=torch.bool),
                negative_ratio=negative_ratio,
                minimum_negatives=minimum_negatives,
            )
        )
    return torch.stack(losses).mean()


def grouped_margin_separation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    margin: float = 1.0,
    negative_ratio: int = 16,
) -> torch.Tensor:
    """Smoothly rank every positive above each row's hardest negatives."""

    if logits.ndim != 1 or targets.shape != logits.shape:
        raise ValueError("margin verifier logits and targets must be one-dimensional")
    if group_indices.shape != logits.shape:
        raise ValueError("margin verifier groups and logits differ")
    losses = []
    groups = sorted(set(int(value) for value in group_indices.detach().cpu().tolist()))
    for group in groups:
        selected = group_indices == group
        positive = logits[selected & (targets > 0.5)].float()
        negative = logits[selected & (targets <= 0.5)].float()
        if positive.numel() and negative.numel():
            count = min(
                int(negative.numel()),
                max(32, int(positive.numel()) * int(negative_ratio)),
            )
            hardest = torch.topk(negative, count).values
            losses.append(
                F.softplus(
                    hardest[:, None] - positive[None, :] + float(margin)
                ).mean()
            )
    if not losses:
        return logits.float().sum() * 0.0
    return torch.stack(losses).mean()


def grouped_tail_anchor_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    positive_floor: float = 1.0,
    negative_ceiling: float = -1.0,
    tail_count: int = 8,
) -> torch.Tensor:
    """Anchor weak positives and hard negatives around the zero threshold."""

    if logits.ndim != 1 or targets.shape != logits.shape:
        raise ValueError("tail verifier logits and targets must be one-dimensional")
    if group_indices.shape != logits.shape or tail_count < 1:
        raise ValueError("tail verifier groups or count are invalid")
    losses = []
    groups = sorted(set(int(value) for value in group_indices.detach().cpu().tolist()))
    for group in groups:
        selected = group_indices == group
        positive = logits[selected & (targets > 0.5)].float()
        negative = logits[selected & (targets <= 0.5)].float()
        terms = []
        if positive.numel():
            weakest = torch.topk(
                positive,
                min(int(positive.numel()), int(tail_count)),
                largest=False,
            ).values
            terms.append(F.softplus(float(positive_floor) - weakest).mean())
        if negative.numel():
            hardest = torch.topk(
                negative,
                min(int(negative.numel()), int(tail_count)),
            ).values
            terms.append(F.softplus(hardest - float(negative_ceiling)).mean())
        if terms:
            losses.append(torch.stack(terms).sum())
    if not losses:
        return logits.float().sum() * 0.0
    return torch.stack(losses).mean()


def onerel_loss(
    link_logits: torch.Tensor,
    relation_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    verifier_logits: torch.Tensor,
    link_targets: torch.Tensor,
    relation_targets: torch.Tensor,
    boundary_targets: torch.Tensor,
    verifier_targets: torch.Tensor,
    verifier_groups: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    hard_negative_ratio: int = 16,
    relation_weight: float = 0.25,
    boundary_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint OneRel link, predicate-presence and role-boundary objective."""

    token_mask = valid_tokens.bool()
    pair_mask = token_mask[:, None, None, :, None] & token_mask[:, None, None, None, :]
    link = torch.stack(
        [
            balanced_hard_negative_loss(
                link_logits[row],
                link_targets[row],
                pair_mask[row],
                negative_ratio=hard_negative_ratio,
                minimum_negatives=128,
            )
            for row in range(int(link_logits.shape[0]))
        ]
    ).mean()
    relation = balanced_hard_negative_loss(
        relation_logits,
        relation_targets,
        torch.ones_like(relation_targets, dtype=torch.bool),
        negative_ratio=4,
        minimum_negatives=8,
    )
    boundary = balanced_hard_negative_loss(
        boundary_logits,
        boundary_targets,
        token_mask.unsqueeze(-1),
        negative_ratio=hard_negative_ratio,
        minimum_negatives=32,
    )
    verifier_bce = grouped_balanced_hard_negative_loss(
        verifier_logits,
        verifier_targets,
        verifier_groups,
        negative_ratio=hard_negative_ratio,
        minimum_negatives=32,
    )
    verifier_margin = grouped_margin_separation_loss(
        verifier_logits,
        verifier_targets,
        verifier_groups,
        margin=1.0,
        negative_ratio=hard_negative_ratio,
    )
    verifier_anchor = grouped_tail_anchor_loss(
        verifier_logits,
        verifier_targets,
        verifier_groups,
        positive_floor=1.0,
        negative_ceiling=-1.0,
        tail_count=8,
    )
    verifier = verifier_bce + verifier_margin + verifier_anchor
    total = (
        link
        + verifier
        + relation * float(relation_weight)
        + boundary * float(boundary_weight)
    )
    return total, {
        "link": link.detach(),
        "relation": relation.detach(),
        "boundary": boundary.detach(),
        "verifier": verifier.detach(),
    }


def _candidate_map(
    candidates: Iterable[tuple[int, int, int, int, float]],
    threshold: float,
) -> dict[tuple[int, int], dict[int, set[int]]]:
    result: dict[tuple[int, int], dict[int, set[int]]] = {}
    for relation, tag, left, right, score in candidates:
        if score <= threshold:
            continue
        result.setdefault((relation, tag), {}).setdefault(left, set()).add(right)
    return result


def _scored_candidate_map(
    candidates: Iterable[tuple[int, int, int, int, float]],
    threshold: float,
) -> dict[tuple[int, int], dict[int, dict[int, float]]]:
    result: dict[tuple[int, int], dict[int, dict[int, float]]] = {}
    for relation, tag, left, right, raw_score in candidates:
        score = float(raw_score)
        if score <= threshold:
            continue
        rights = result.setdefault((relation, tag), {}).setdefault(left, {})
        rights[right] = max(score, rights.get(right, -math.inf))
    return result


def enumerate_onerel_span_candidates(
    relation_logits: Sequence[float],
    boundary_logits: Sequence[Sequence[float]],
    candidates: Iterable[tuple[int, int, int, int, float]],
    predicates: int,
    thresholds: tuple[float, float, float],
    *,
    maximum_link_candidates: int = 200_000,
    maximum_span_candidates: int = 100_000,
) -> tuple[OneRelSpanCandidate, ...]:
    """Compose a scored verifier-candidate superset from OneRel links."""

    relation_threshold, link_threshold, boundary_threshold = thresholds
    candidate_list = [
        candidate for candidate in candidates if float(candidate[4]) > link_threshold
    ]
    if len(candidate_list) > maximum_link_candidates:
        raise OneRelDensityError(
            f"OneRel links exceed cap ({len(candidate_list)} > {maximum_link_candidates})"
        )
    if len(relation_logits) != predicates:
        raise ValueError("relation logits and predicate table differ")
    links = _scored_candidate_map(candidate_list, link_threshold)
    proposed: dict[
        tuple[int, int, int, int, int], OneRelSpanCandidate
    ] = {}
    for relation in range(predicates):
        relation_score = float(relation_logits[relation])
        if relation_score <= relation_threshold:
            continue
        begin_pairs = links.get((relation, HB_TB), {})
        begin_to_object_end = links.get((relation, HB_TE), {})
        subject_end_to_object_end = links.get((relation, HE_TE), {})
        if not begin_pairs or not begin_to_object_end or not subject_end_to_object_end:
            continue
        subject_ends_by_object_end: dict[int, dict[int, float]] = {}
        for subject_end, object_ends in subject_end_to_object_end.items():
            for object_end, score in object_ends.items():
                ends = subject_ends_by_object_end.setdefault(object_end, {})
                ends[subject_end] = max(score, ends.get(subject_end, -math.inf))
        for subject_start, object_starts in begin_pairs.items():
            for object_end, head_end_score in begin_to_object_end.get(
                subject_start, {}
            ).items():
                for subject_end, tail_score in subject_ends_by_object_end.get(
                    object_end, {}
                ).items():
                    if subject_end < subject_start:
                        continue
                    for object_start, head_score in object_starts.items():
                        if object_end < object_start:
                            continue
                        boundary_score = min(
                            float(boundary_logits[subject_start][0]),
                            float(boundary_logits[subject_end][1]),
                            float(boundary_logits[object_start][2]),
                            float(boundary_logits[object_end][3]),
                        )
                        if boundary_score <= boundary_threshold:
                            continue
                        key = (
                            relation,
                            subject_start,
                            subject_end,
                            object_start,
                            object_end,
                        )
                        value = OneRelSpanCandidate(
                            predicate_id=relation,
                            subject_start=subject_start,
                            subject_end=subject_end,
                            object_start=object_start,
                            object_end=object_end,
                            relation_score=relation_score,
                            link_score=min(head_score, head_end_score, tail_score),
                            boundary_score=boundary_score,
                        )
                        previous = proposed.get(key)
                        if previous is None or value.link_score > previous.link_score:
                            proposed[key] = value
                        if len(proposed) > maximum_span_candidates:
                            raise OneRelDensityError(
                                "OneRel span candidates exceed cap "
                                f"({len(proposed)} > {maximum_span_candidates})"
                            )
    return tuple(proposed[key] for key in sorted(proposed))


def select_fixed_proposal_candidates(
    proposed: Sequence[OneRelSpanCandidate],
    gold: set[tuple[int, int, int, int, int]],
    *,
    negative_ratio: int = 8,
    maximum_selected_candidates: int = 4096,
    retain_unproposed_gold: bool = True,
) -> tuple[tuple[tuple[int, int, int, int, int], ...], dict[str, int]]:
    """Keep all gold and the deterministic hardest proposal negatives.

    ``proposed`` is the complete fail-closed proposal set produced at the
    declared inference thresholds.  Negative sampling is an explicit training
    operation rather than an inference truncation.  Semantic-verifier training
    can retain every gold coordinate; candidate-conditioned calibration can
    instead retain only reachable gold so impossible low-score positives do
    not reverse the proposal-score supervision.  At most ``negative_ratio``
    proposal negatives are retained per selected positive.  Proposal
    confidence supplies the hardness ordering; exact coordinates provide a
    deterministic final tie-breaker.
    """

    if negative_ratio < 1 or maximum_selected_candidates < 1:
        raise ValueError("fixed proposal sampling limits are invalid")
    proposed_by_coordinate = {
        (
            item.predicate_id,
            item.subject_start,
            item.subject_end,
            item.object_start,
            item.object_end,
        ): item
        for item in proposed
    }
    if len(proposed_by_coordinate) != len(proposed):
        raise ValueError("fixed proposals contain duplicate coordinates")

    def hardness_key(
        coordinate: tuple[int, int, int, int, int],
    ) -> tuple[float, float, float, float, tuple[int, int, int, int, int]]:
        item = proposed_by_coordinate[coordinate]
        values = (
            float(item.link_score),
            float(item.relation_score),
            float(item.boundary_score),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("fixed proposal contains a non-finite score")
        return (
            -min(values),
            -values[0],
            -values[1],
            -values[2],
            coordinate,
        )

    proposed_coordinates = set(proposed_by_coordinate)
    reachable_gold = gold & proposed_coordinates
    selected_gold = gold if retain_unproposed_gold else reachable_gold
    negative_coordinates = proposed_coordinates - gold
    maximum_negatives = max(1, len(selected_gold)) * int(negative_ratio)
    selected_negatives = tuple(
        sorted(negative_coordinates, key=hardness_key)[:maximum_negatives]
    )
    selected = tuple(sorted(selected_gold | set(selected_negatives)))
    if len(selected) > maximum_selected_candidates:
        raise OneRelDensityError(
            "sampled fixed proposal verifier candidates exceed cap "
            f"({len(selected)} > {maximum_selected_candidates})"
        )
    return selected, {
        "raw_proposal_candidates": len(proposed_coordinates),
        "raw_negative_candidates": len(negative_coordinates),
        "selected_negative_candidates": len(selected_negatives),
        "selected_candidates": len(selected),
        "gold_candidates": len(selected_gold),
        "reachable_gold_candidates": len(reachable_gold),
        "excluded_unproposed_gold_candidates": len(gold - selected_gold),
        "gold_inserted_outside_proposals": len(gold - proposed_coordinates),
        "retain_unproposed_gold": int(bool(retain_unproposed_gold)),
        "negative_ratio": int(negative_ratio),
        "candidate_cap_per_row": int(maximum_selected_candidates),
    }


def decode_verified_onerel_candidates(
    text: str,
    offsets: Sequence[tuple[int, int]],
    candidates: Sequence[OneRelSpanCandidate],
    verifier_scores: Sequence[float],
    predicates: Sequence[str],
    thresholds: tuple[float, float, float, float],
    *,
    maximum_triples: int = 4096,
) -> set[tuple[str, str, str]]:
    """Filter exact span candidates with one global quadruple verifier."""

    if len(candidates) != len(verifier_scores):
        raise ValueError("verifier scores and candidates differ")
    relation_threshold, link_threshold, boundary_threshold, verifier_threshold = (
        thresholds
    )
    predicted: set[tuple[str, str, str]] = set()
    for candidate, verifier_score in zip(candidates, verifier_scores):
        if (
            candidate.relation_score <= relation_threshold
            or candidate.link_score <= link_threshold
            or candidate.boundary_score <= boundary_threshold
            or float(verifier_score) <= verifier_threshold
        ):
            continue
        relation = candidate.predicate_id
        if not 0 <= relation < len(predicates):
            raise ValueError("candidate predicate is out of range")
        ss, se = candidate.subject_start, candidate.subject_end
        os_, oe = candidate.object_start, candidate.object_end
        if (
            not 0 <= ss <= se < len(offsets)
            or not 0 <= os_ <= oe < len(offsets)
            or offsets[ss][1] <= offsets[ss][0]
            or offsets[se][1] <= offsets[se][0]
            or offsets[os_][1] <= offsets[os_][0]
            or offsets[oe][1] <= offsets[oe][0]
        ):
            continue
        subject = text[offsets[ss][0] : offsets[se][1]]
        obj = text[offsets[os_][0] : offsets[oe][1]]
        if not subject or not obj:
            continue
        predicted.add((subject, str(predicates[relation]), obj))
        if len(predicted) > maximum_triples:
            raise OneRelDensityError(
                f"OneRel triples exceed cap ({len(predicted)} > {maximum_triples})"
            )
    return predicted


def decode_onerel_candidates(
    text: str,
    offsets: Sequence[tuple[int, int]],
    relation_logits: Sequence[float],
    boundary_logits: Sequence[Sequence[float]],
    candidates: Iterable[tuple[int, int, int, int, float]],
    predicates: Sequence[str],
    thresholds: tuple[float, float, float],
    *,
    maximum_link_candidates: int = 200_000,
    maximum_triples: int = 4096,
) -> set[tuple[str, str, str]]:
    """Decode exact SPOs from sparse scored links without truncation."""

    relation_threshold, link_threshold, boundary_threshold = thresholds
    candidate_list = [
        candidate for candidate in candidates if float(candidate[4]) > link_threshold
    ]
    if len(candidate_list) > maximum_link_candidates:
        raise OneRelDensityError(
            f"OneRel links exceed cap ({len(candidate_list)} > {maximum_link_candidates})"
        )
    if len(relation_logits) != len(predicates):
        raise ValueError("relation logits and predicate table differ")
    if len(boundary_logits) != len(offsets):
        raise ValueError("boundary logits and offsets differ")
    links = _candidate_map(candidate_list, link_threshold)
    predicted: set[tuple[str, str, str]] = set()
    for relation, predicate in enumerate(predicates):
        if float(relation_logits[relation]) <= relation_threshold:
            continue
        begin_pairs = links.get((relation, HB_TB), {})
        begin_to_object_end = links.get((relation, HB_TE), {})
        subject_end_to_object_end = links.get((relation, HE_TE), {})
        if not begin_pairs or not begin_to_object_end or not subject_end_to_object_end:
            continue
        subject_ends_by_object_end: dict[int, set[int]] = {}
        for subject_end, object_ends in subject_end_to_object_end.items():
            for object_end in object_ends:
                subject_ends_by_object_end.setdefault(object_end, set()).add(subject_end)
        for subject_start, object_starts in begin_pairs.items():
            if float(boundary_logits[subject_start][0]) <= boundary_threshold:
                continue
            for object_end in begin_to_object_end.get(subject_start, ()):
                if float(boundary_logits[object_end][3]) <= boundary_threshold:
                    continue
                for subject_end in subject_ends_by_object_end.get(object_end, ()):
                    if subject_end < subject_start:
                        continue
                    if float(boundary_logits[subject_end][1]) <= boundary_threshold:
                        continue
                    for object_start in object_starts:
                        if object_end < object_start:
                            continue
                        if float(boundary_logits[object_start][2]) <= boundary_threshold:
                            continue
                        if (
                            offsets[subject_start][1] <= offsets[subject_start][0]
                            or offsets[subject_end][1] <= offsets[subject_end][0]
                            or offsets[object_start][1] <= offsets[object_start][0]
                            or offsets[object_end][1] <= offsets[object_end][0]
                        ):
                            continue
                        subject = text[
                            offsets[subject_start][0] : offsets[subject_end][1]
                        ]
                        obj = text[offsets[object_start][0] : offsets[object_end][1]]
                        if not subject or not obj:
                            continue
                        predicted.add((subject, str(predicate), obj))
                        if len(predicted) > maximum_triples:
                            raise OneRelDensityError(
                                f"OneRel triples exceed cap ({len(predicted)} > {maximum_triples})"
                            )
    return predicted


def sparsify_onerel_logits(
    logits: torch.Tensor,
    minimum_threshold: float,
) -> list[list[tuple[int, int, int, int, float]]]:
    """Transfer only links above the lowest declared global threshold."""

    if logits.ndim != 5 or logits.shape[2] != len(LINK_NAMES):
        raise ValueError("expected [batch, relation, tag, left, right] logits")
    result: list[list[tuple[int, int, int, int, float]]] = []
    for row in logits:
        indices = (row > float(minimum_threshold)).nonzero(as_tuple=False)
        if indices.numel():
            scores = row[
                indices[:, 0],
                indices[:, 1],
                indices[:, 2],
                indices[:, 3],
            ].float()
            values = [
                (int(r), int(t), int(left), int(right), float(score))
                for (r, t, left, right), score in zip(
                    indices.cpu().tolist(), scores.cpu().tolist()
                )
            ]
        else:
            values = []
        result.append(values)
    return result
