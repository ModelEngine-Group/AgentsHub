from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dki.nlp.cmeie_candidate_reranker import (
    PairRerankCandidate,
    aggregate_candidate_probabilities,
    apply_fit_surface_schema,
    build_fit_surface_schema,
    build_candidate_hard_negative_groups,
    decode_pair_rerank_candidates,
    expand_pair_rerank_inputs,
    predicate_signature_map,
    select_global_rerank_policy,
    surface_schema_decision,
)
from scripts.train_cmeie_candidate_cross_encoder import (
    CANDIDATE_HIDDEN_SIZE,
    OVERFIT_HEAD_STEPS,
    OVERFIT_JOINT_STEPS,
    SURFACE_CANDIDATE_CAP,
    _adaptive_filter_proposal_candidates,
    _balanced_binary_aux_bce,
    _build_model_hard_negative_groups,
    _build_mixed_hard_negative_groups,
    _candidate_factor_targets,
    _candidate_relation_targets,
    _decomposed_group_margins,
    _encode_candidates,
    _encoder_texts,
    _filter_proposal_candidates,
    _factor_target_audit,
    _fuse_auxiliary_logits,
    _group_listwise_loss,
    _hard_negative_refresh_strategy,
    _install_candidate_classifier,
    _negative_error_category,
    _overfit_gate_payload,
    _phase_schedule,
    _pool_marked_span_features,
    _proposal_feature_row,
    _remap_oof_candidates,
    _restore_unmarked_sentence,
    _select_proposal_policy,
    _set_trainable,
    _soft_relation_cross_entropy,
    _validated_oof_partitions,
)
from scripts.eval_cmeie_candidate_schema import (
    _apply_group_cardinality,
    _factorized_probabilities,
    _fit_cardinality_contract,
)
from scripts.eval_cmeie_onerel_pair_reranker import (
    CANDIDATE_CAP,
    COORDINATE_CANDIDATE_CAP,
    _mask_padded_link_logits,
)


def _candidate(row: int, predicate_id: int, predicate: str, obj: str):
    return PairRerankCandidate(
        row_index=row,
        predicate_id=predicate_id,
        subject="甲",
        predicate=predicate,
        object=obj,
        marked_text="[SUBJ]甲[/SUBJ]与[OBJ]乙[/OBJ]",
    )


def test_schema_inputs_are_fit_only_and_deterministic():
    contract = {
        "schema_constraints": {
            "疾病\t症状": [1, 0],
            "疾病\t药物": [1],
        }
    }
    signatures = predicate_signature_map(contract, ["p0", "p1"])
    assert len(signatures[0]) == 1
    assert len(signatures[1]) == 2
    candidates = (_candidate(0, 1, "p1", "乙"),)
    enabled = expand_pair_rerank_inputs(
        candidates, signatures, schema_on=True
    )
    disabled = expand_pair_rerank_inputs(
        candidates, signatures, schema_on=False
    )
    assert tuple(item.signature for item in enabled) == signatures[1]
    assert tuple(item.signature for item in disabled) == ("",)


def test_fit_surface_schema_is_role_specific_ambiguous_and_fail_open():
    rows = [
        {
            "spo_list": [
                {
                    "subject": "s",
                    "subject_type": "disease",
                    "predicate": "treats",
                    "object": "o",
                    "object_type": "drug",
                },
                {
                    "subject": "s",
                    "subject_type": "symptom",
                    "predicate": "indicates",
                    "object": "o2",
                    "object_type": "disease",
                },
            ]
        }
    ]
    contract = build_fit_surface_schema(rows)
    admissible = PairRerankCandidate(0, 0, "s", "treats", "o", "m")
    rejected = PairRerankCandidate(0, 0, "s", "treats", "o2", "m")
    unknown = PairRerankCandidate(0, 0, "unseen", "treats", "o", "m")
    assert surface_schema_decision(admissible, contract) == (
        True,
        "known_schema_admissible",
    )
    assert surface_schema_decision(rejected, contract) == (
        False,
        "known_schema_rejected",
    )
    assert surface_schema_decision(unknown, contract) == (
        True,
        "unknown_surface_fail_open",
    )
    retained, audit = apply_fit_surface_schema(
        (admissible, rejected, unknown), contract
    )
    assert retained == (0, 2)
    assert audit == {
        "known_schema_admissible": 1,
        "known_schema_rejected": 1,
        "unknown_surface_fail_open": 1,
    }


def test_fit_cardinality_contract_and_group_caps_are_fit_fixed():
    rows = [
        {
            "spo_list": [
                {"subject": "s", "predicate": "p1", "object": "o"},
                {"subject": "s", "predicate": "p2", "object": "o"},
                {"subject": "s", "predicate": "p1", "object": "o2"},
            ]
        }
    ]
    contract = _fit_cardinality_contract(rows, target=0.5)
    assert contract["pair_predicate"]["cap"] == 1
    assert contract["subject_predicate_object"]["cap"] == 1
    candidates = (
        PairRerankCandidate(0, 0, "s", "p1", "o", "m", 0.2, 0.2, 0.2),
        PairRerankCandidate(0, 1, "s", "p2", "o", "m", 0.9, 0.2, 0.2),
        PairRerankCandidate(0, 0, "s", "p1", "o2", "m", 0.8, 0.2, 0.2),
    )
    retained, audit = _apply_group_cardinality(
        candidates=candidates,
        scores=(0.4, 0.9, 0.8),
        eligible_indices=(0, 1, 2),
        pair_predicate_cap=1,
        subject_predicate_object_cap=1,
    )
    assert retained == (1, 2)
    assert audit["rejected_by_pair_predicate_cap"] == 1
    assert audit["rejected_by_subject_predicate_object_cap"] == 0


def test_factorized_probability_requires_all_three_heads():
    import torch

    outputs = type(
        "Output",
        (),
        {
            "logits": torch.tensor([[0.0]]),
            "pair_logits": torch.tensor([[0.0]]),
            "relation_logits": torch.tensor([[0.0, 8.0]]),
        },
    )()
    model = type(
        "Model",
        (),
        {
            "pair_scale_logit": torch.tensor(0.0),
            "relation_scale_logit": torch.tensor(0.0),
        },
    )()
    probability = _factorized_probabilities(
        outputs=outputs,
        predicate_ids=torch.tensor([0]),
        model=model,
        torch=torch,
    )
    assert torch.allclose(probability, torch.tensor([0.125]))


def test_decomposed_margins_separate_pair_and_predicate_errors():
    import torch

    candidates = (
        PairRerankCandidate(0, 0, "s", "p", "o", "m"),
        PairRerankCandidate(0, 1, "s", "wrong", "o", "m"),
        PairRerankCandidate(0, 0, "s", "p", "other", "m"),
    )
    pair_margin, relation_margin = _decomposed_group_margins(
        candidates=candidates,
        flat_candidate_indices=(0, 1, 2),
        spans=((0, 3),),
        pair_logits=torch.tensor([1.0, 1.0, 0.5]),
        selected_relation_logits=torch.tensor([1.0, 0.25, 1.0]),
        torch=torch,
    )
    assert torch.allclose(pair_margin, torch.tensor(0.5))
    assert torch.allclose(relation_margin, torch.tensor(0.25))


def test_probability_aggregation_uses_candidate_predicate_and_max_signature():
    candidates = (
        _candidate(0, 1, "p1", "乙"),
        _candidate(0, 0, "p0", "丙"),
    )
    signatures = {0: ("s0",), 1: ("s1", "s2")}
    inputs = expand_pair_rerank_inputs(candidates, signatures, schema_on=True)
    scores = aggregate_candidate_probabilities(
        candidates=candidates,
        inputs=inputs,
        probability_rows=((0.9, 0.2), (0.1, 0.8), (0.7, 0.4)),
        n_predicates=2,
    )
    assert scores == pytest.approx((0.8, 0.7))


def test_global_policy_tie_breaks_by_precision_threshold_then_schema_on():
    candidates = (
        _candidate(0, 0, "p0", "乙"),
        _candidate(0, 0, "p0", "丙"),
    )
    gold = [{("甲", "p0", "乙")}]
    selected = select_global_rerank_policy(
        gold_rows=gold,
        candidates=candidates,
        score_sets={False: (0.8, 0.4), True: (0.8, 0.4)},
        thresholds=(0.3, 0.5, 0.7),
    )
    assert selected["metric"]["f1"] == 1.0
    assert selected["threshold"] == 0.7
    assert selected["schema_on"] is True
    assert selected["grid_size"] == 6


def test_decode_deduplicates_surface_triples_and_rejects_shape_mismatch():
    candidates = (
        _candidate(0, 0, "p0", "乙"),
        _candidate(0, 0, "p0", "乙"),
    )
    decoded = decode_pair_rerank_candidates(
        n_rows=1, candidates=candidates, scores=(0.9, 0.8), threshold=0.5
    )
    assert decoded == [{("甲", "p0", "乙")}]
    with pytest.raises(ValueError, match="differ"):
        decode_pair_rerank_candidates(
            n_rows=1, candidates=candidates, scores=(0.9,), threshold=0.5
        )


def test_hard_negative_groups_are_bounded_gold_free_and_deterministic():
    positive = _candidate(0, 0, "p0", "乙")
    negatives = tuple(
        PairRerankCandidate(
            row_index=0,
            predicate_id=0,
            subject="甲",
            predicate="p0",
            object=f"负{i}",
            marked_text=f"[SUBJ]甲[/SUBJ][OBJ]负{i}[/OBJ]",
            link_score=float(i),
            relation_score=0.0,
            boundary_score=0.0,
        )
        for i in range(12)
    )
    candidates = (positive, *negatives)
    gold = [{positive.triple}]
    first, stats = build_candidate_hard_negative_groups(
        candidates=candidates, gold_rows=gold, negative_ratio=8
    )
    second, _ = build_candidate_hard_negative_groups(
        candidates=candidates, gold_rows=gold, negative_ratio=8
    )
    assert first == second
    assert len(first) == 1 and len(first[0]) == 9
    assert first[0][0] == 0
    assert all(candidates[index].triple not in gold[0] for index in first[0][1:])
    assert [candidates[index].link_score for index in first[0][1:]] == list(
        reversed([float(value) for value in range(4, 12)])
    )
    assert stats["reachable_gold_recall"] == 1.0


def test_hard_negative_groups_use_disjoint_capacity_before_repeating():
    positives = (
        _candidate(0, 0, "p0", "gold0"),
        _candidate(0, 1, "p1", "gold1"),
    )
    negatives = tuple(
        _candidate(0, 0, "p0", f"negative-{index}") for index in range(16)
    )
    candidates = (*positives, *negatives)
    groups, stats = build_candidate_hard_negative_groups(
        candidates=candidates,
        gold_rows=[{candidate.triple for candidate in positives}],
        negative_ratio=8,
    )
    assert len(groups) == 2
    assert set(groups[0][1:]).isdisjoint(groups[1][1:])
    assert stats["n_unique_negative_samples"] == 16
    assert stats["negative_candidate_coverage"] == 1.0


def test_cross_encoder_overfit_schedule_is_step_bound_and_phase_ordered():
    phases = _phase_schedule(
        mode="overfit",
        n_groups=44,
        groups_per_batch=4,
        formal_epochs=3,
        formal_head_only_epochs=1,
    )
    batches_per_epoch = 11
    head_epochs = phases.count("head_only")
    joint_epochs = phases.count("top2_joint")
    assert head_epochs * batches_per_epoch >= OVERFIT_HEAD_STEPS
    assert joint_epochs * batches_per_epoch >= OVERFIT_JOINT_STEPS
    assert phases == tuple(
        sorted(phases, key={"head_only": 0, "top2_joint": 1}.get)
    )


def test_cross_encoder_formal_schedule_is_one_plus_two():
    assert _phase_schedule(
        mode="train",
        n_groups=44,
        groups_per_batch=4,
        formal_epochs=3,
        formal_head_only_epochs=1,
    ) == ("head_only", "top2_joint", "top2_joint")


def test_cross_encoder_formal_route_has_oof_candidate_recall_gate() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_cmeie_candidate_cross_encoder.py"
    ).read_text(encoding="utf-8")
    assert 'float(fit_candidate_oracle["recall"]) < 0.60' in source
    assert '"status": "rejected_oof_candidate_recall"' in source
    assert '"inference_truncation": False' in source


def test_global_proposal_policy_is_fit_label_selected_and_density_safe() -> None:
    fit_candidates = (
        PairRerankCandidate(0, 0, "s0", "p", "o0", "m", 1.0, 1.0, 1.0),
        PairRerankCandidate(1, 0, "s1", "p", "o1", "m", 1.0, 1.0, 1.0),
    )
    guard_candidates = tuple(
        PairRerankCandidate(0, 0, "s", "p", f"noise-{index}", "m", -2.0, 0.0, 0.0)
        for index in range(SURFACE_CANDIDATE_CAP + 1)
    )
    policy = _select_proposal_policy(
        fit_candidates=fit_candidates,
        fit_gold=({("s0", "p", "o0")}, {("s1", "p", "o1")}),
        guard_candidates=guard_candidates,
        n_guard_rows=1,
    )
    assert policy["status"] == "selected"
    assert policy["selected"]["candidate_oracle"]["recall"] == 1.0
    assert policy["selected"]["fit_density"]["max"] <= SURFACE_CANDIDATE_CAP
    assert (
        policy["selected"]["calibration_unlabeled_density"]["max"]
        <= SURFACE_CANDIDATE_CAP
    )
    assert policy["selection_basis"].startswith("fit_oof_labels")
    assert policy["inference_truncation"] is False


def test_global_proposal_policy_rejects_when_recall_floor_is_unreachable() -> None:
    candidate = PairRerankCandidate(
        0, 0, "s", "p", "o", "m", relation_score=-1.0, link_score=-1.0,
        boundary_score=-1.0,
    )
    policy = _select_proposal_policy(
        fit_candidates=(candidate,),
        fit_gold=({("s", "p", "o")},),
        guard_candidates=(),
        n_guard_rows=1,
    )
    assert policy["status"] == "rejected_no_density_safe_policy"
    assert policy["minimum_candidate_recall"] == 0.60


def test_proposal_filter_never_silently_truncates() -> None:
    candidates = tuple(
        PairRerankCandidate(0, 0, "s", "p", str(index), "m", 1.0, 1.0, 1.0)
        for index in range(SURFACE_CANDIDATE_CAP + 1)
    )
    filtered = _filter_proposal_candidates(
        candidates,
        relation_threshold=0.5,
        link_threshold=0.5,
        boundary_threshold=0.5,
    )
    assert len(filtered) == SURFACE_CANDIDATE_CAP + 1


def test_density_adaptive_link_ladder_filters_by_threshold_not_top_k() -> None:
    dense = [
        PairRerankCandidate(
            0, 0, "s", "p", f"low-{index}", "m", 1.0, -0.25, 1.0
        )
        for index in range(SURFACE_CANDIDATE_CAP + 1)
    ]
    high = [
        PairRerankCandidate(0, 0, "s", "p", f"high-{index}", "m", 1.0, 0.25, 1.0)
        for index in range(2)
    ]
    selected, audit = _adaptive_filter_proposal_candidates(
        (*dense, *high),
        n_rows=1,
        relation_threshold=-0.5,
        boundary_threshold=-0.5,
    )
    assert [candidate.object for candidate in selected] == ["high-0", "high-1"]
    assert audit["threshold_row_counts"]["0.0"] == 1
    assert audit["raw_density"]["max"] == SURFACE_CANDIDATE_CAP + 3
    assert audit["selected_density"]["max"] == 2
    assert audit["inference_truncation"] is False
    assert audit["gold_used_for_row_threshold"] is False


def test_online_model_hard_negative_groups_use_top8_and_exclude_gold() -> None:
    candidates = [
        PairRerankCandidate(0, 0, "s", "p", "gold-a", "m"),
        PairRerankCandidate(0, 0, "s", "p", "gold-b", "m"),
    ]
    candidates.extend(
        PairRerankCandidate(0, 0, "s", "p", f"neg-{index}", "m")
        for index in range(12)
    )
    scores = [0.01, 0.02, *[index / 20 for index in range(12)]]
    gold = ({("s", "p", "gold-a"), ("s", "p", "gold-b")},)
    first, audit = _build_model_hard_negative_groups(
        candidates=candidates,
        gold_rows=gold,
        scores=scores,
    )
    second, _ = _build_model_hard_negative_groups(
        candidates=candidates,
        gold_rows=gold,
        scores=scores,
    )
    assert first == second
    assert len(first) == 2
    assert all(len(group) == 9 for group in first)
    assert first[0][1:] == first[1][1:]
    assert {candidates[index].object for index in first[0][1:]} == {
        f"neg-{index}" for index in range(4, 12)
    }
    assert audit["maximum_negatives_per_positive"] == 8
    assert audit["reachable_gold_recall"] == 1.0


def test_mixed_hard_negative_groups_are_4_plus_4_and_deterministic() -> None:
    candidates = [
        PairRerankCandidate(0, 0, "s", "p", "gold-a", "m"),
        PairRerankCandidate(0, 0, "s", "p", "gold-b", "m"),
    ]
    candidates.extend(
        PairRerankCandidate(
            0,
            0,
            "s",
            "p",
            f"neg-{index}",
            "m",
            relation_score=float(index),
            link_score=float(index),
            boundary_score=float(index),
        )
        for index in range(20)
    )
    scores = [0.0, 0.0, *[float(20 - index) for index in range(20)]]
    gold = ({("s", "p", "gold-a"), ("s", "p", "gold-b")},)
    first, audit = _build_mixed_hard_negative_groups(
        candidates=candidates,
        gold_rows=gold,
        scores=scores,
    )
    second, _ = _build_mixed_hard_negative_groups(
        candidates=candidates,
        gold_rows=gold,
        scores=scores,
    )
    assert first == second
    assert len(first) == 2
    assert all(len(group) == 9 for group in first)
    assert audit["proposal_negative_count"] == 4
    assert audit["model_negative_count"] == 4
    assert audit["maximum_negatives_per_positive"] == 8
    assert audit["n_unique_proposal_negatives"] == 8
    assert audit["n_unique_model_negatives"] == 4
    assert all(index not in {0, 1} for group in first for index in group[1:])


def test_hard_negative_refresh_tracks_each_joint_epoch_and_mode() -> None:
    assert _hard_negative_refresh_strategy(
        mode="train", epoch=2, phase="top2_joint", previous_phase="head_only"
    ) == "mixed_proposal_and_model"
    assert _hard_negative_refresh_strategy(
        mode="train", epoch=3, phase="top2_joint", previous_phase="top2_joint"
    ) == "mixed_proposal_and_model"
    assert _hard_negative_refresh_strategy(
        mode="overfit", epoch=111, phase="top2_joint", previous_phase="head_only"
    ) == "model_false_positives"
    assert _hard_negative_refresh_strategy(
        mode="overfit", epoch=112, phase="top2_joint", previous_phase="top2_joint"
    ) is None


def test_negative_error_categories_cover_relation_boundary_and_cross_spo() -> None:
    gold = {("s1", "p1", "o1"), ("s2", "p2", "o2")}
    cases = {
        ("s1", "p2", "o1"): "wrong_predicate",
        ("s1", "p1", "noise"): "wrong_object_or_boundary",
        ("noise", "p1", "o1"): "wrong_subject_or_boundary",
        ("o1", "p1", "s1"): "direction_swap",
        ("s1", "p1", "s1"): "self_pair",
        ("s1", "p1", "o2"): "wrong_object_or_boundary",
        ("s2", "p1", "o1"): "wrong_subject_or_boundary",
        ("s1", "p3", "o2"): "cross_spo_pair",
        ("noise", "p2", "other"): "other_oof_false_positive",
    }
    for triple, expected in cases.items():
        candidate = PairRerankCandidate(
            0, 0, triple[0], triple[1], triple[2], "marked"
        )
        assert _negative_error_category(candidate, gold) == expected


def test_overfit_exact_gate_can_be_satisfied_after_joint_refinement() -> None:
    head = {
        "metric": {"f1": 0.9777, "precision": 0.9565},
        "threshold": 0.95,
    }
    joint = {
        "metric": {"f1": 0.9885, "precision": 1.0},
        "threshold": 0.30,
    }
    gates, selected = _overfit_gate_payload(head_best=head, joint_best=joint)
    assert not gates["head_only_f1_at_least_0_98"]
    assert gates["exact_spo_f1_at_least_0_98"]
    assert gates["joint_f1_at_least_0_95"]
    assert selected is joint


def test_cross_encoder_installs_nonlinear_candidate_head():
    import torch

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 32})()
            self.classifier = torch.nn.Linear(32, 1)

    model = Stub()
    _install_candidate_classifier(model, torch)
    layers = list(model.classifier)
    assert isinstance(layers[0], torch.nn.LayerNorm)
    assert isinstance(layers[1], torch.nn.Linear)
    assert layers[1].out_features == CANDIDATE_HIDDEN_SIZE
    assert isinstance(layers[-1], torch.nn.Linear)
    assert layers[-1].out_features == 1


def test_factor_heads_and_scales_are_in_the_head_optimizer_group():
    import torch

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.factor_head_version = 2
            self.classifier = torch.nn.Linear(2, 1)
            self.predicate_embedding = torch.nn.Embedding(3, 2)
            self.pair_head = torch.nn.Linear(2, 1)
            self.relation_head = torch.nn.Linear(2, 3)
            self.subject_head = torch.nn.Linear(2, 1)
            self.object_head = torch.nn.Linear(2, 1)
            self.predicate_presence_head = torch.nn.Linear(2, 3)
            self.pair_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
            self.relation_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
            self.subject_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
            self.object_scale_logit = torch.nn.Parameter(torch.tensor(0.0))
            self.predicate_presence_scale_logit = torch.nn.Parameter(
                torch.tensor(0.0)
            )

    model = Stub()
    head, encoder = _set_trainable(model, joint=False)
    assert not encoder
    head_ids = {id(parameter) for parameter in head}
    for module in (
        model.subject_head,
        model.object_head,
        model.predicate_presence_head,
    ):
        assert {id(parameter) for parameter in module.parameters()} <= head_ids
    for parameter in (
        model.subject_scale_logit,
        model.object_scale_logit,
        model.predicate_presence_scale_logit,
    ):
        assert id(parameter) in head_ids


def test_cross_encoder_pools_subject_and_object_token_content():
    import torch

    input_ids = torch.tensor([[101, 11, 201, 202, 12, 13, 301, 302, 14]])
    hidden = torch.arange(18, dtype=torch.float32).reshape(1, 9, 2)
    markers, subject, obj = _pool_marked_span_features(
        hidden=hidden,
        input_ids=input_ids,
        marker_token_ids=(11, 12, 13, 14),
        torch=torch,
    )
    assert len(markers) == 4
    assert torch.equal(markers[0], hidden[:, 1])
    assert torch.equal(markers[3], hidden[:, 8])
    assert torch.equal(subject, hidden[:, 2:4].mean(dim=1))
    assert torch.equal(obj, hidden[:, 6:8].mean(dim=1))


def test_cross_encoder_span_pooling_fails_closed_on_empty_content():
    import torch

    input_ids = torch.tensor([[101, 11, 12, 13, 301, 14]])
    hidden = torch.zeros((1, 6, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="subject marker span contains no tokens"):
        _pool_marked_span_features(
            hidden=hidden,
            input_ids=input_ids,
            marker_token_ids=(11, 12, 13, 14),
            torch=torch,
        )


def test_relation_auxiliary_targets_exact_pair_and_all_gold_predicates():
    candidates = (
        _candidate(0, 0, "相关疾病", "乙"),
        _candidate(0, 1, "治疗", "乙"),
        _candidate(0, 1, "治疗", "丙"),
    )
    gold = [
        {
            ("甲", "相关疾病", "乙"),
            ("甲", "治疗", "乙"),
        }
    ]
    targets = _candidate_relation_targets(
        candidates=candidates,
        gold_rows=gold,
        predicates=("相关疾病", "治疗"),
    )
    assert targets == ((0, 1), (0, 1), ())


def test_pair_existence_target_is_independent_of_proposed_predicate():
    candidates = (
        _candidate(0, 0, "p0", "o"),
        _candidate(0, 1, "p1", "o"),
        _candidate(0, 1, "p1", "noise"),
    )
    targets = _candidate_relation_targets(
        candidates=candidates,
        gold_rows=[{(candidates[0].subject, "p0", "o")}],
        predicates=("p0", "p1"),
    )
    # The first two proposals share the same exact surface entity pair.  The
    # pair auxiliary target is therefore positive for both; the relation head,
    # rather than pair existence, must reject p1.
    assert tuple(bool(values) for values in targets) == (True, True, False)


def test_factor_targets_separate_roles_and_sentence_predicate_presence():
    candidates = (
        PairRerankCandidate(0, 0, "subject-a", "p0", "object-a", "m"),
        PairRerankCandidate(0, 1, "object-a", "p1", "subject-a", "m"),
        PairRerankCandidate(0, 1, "noise", "p1", "object-a", "m"),
    )
    subject, obj, predicates = _candidate_factor_targets(
        candidates=candidates,
        gold_rows=[
            {
                ("subject-a", "p0", "object-a"),
                ("subject-b", "p1", "object-b"),
            }
        ],
        predicates=("p0", "p1", "p2"),
    )
    assert subject == (True, False, False)
    assert obj == (True, False, True)
    assert predicates == ((0, 1), (0, 1), (0, 1))
    audit = _factor_target_audit(
        candidates=candidates,
        subject_targets=subject,
        object_targets=obj,
        predicate_presence_targets=predicates,
        indices=(0, 1, 2),
    )
    assert audit["subject_valid"]["positive"] == 1
    assert audit["object_valid"]["positive"] == 2
    assert audit["proposed_predicate_present"]["positive"] == 3


def test_factor_fusion_is_sensitive_to_every_necessary_condition():
    import torch

    base = torch.tensor([[0.0]])
    scale = torch.tensor(0.0)
    neutral = [(scale, torch.tensor([[0.0]])) for _ in range(5)]
    reference = _fuse_auxiliary_logits(
        verifier_logits=base,
        scaled_logits=neutral,
        torch=torch,
    )
    for index in range(len(neutral)):
        factors = list(neutral)
        factors[index] = (scale, torch.tensor([[2.0]]))
        changed = _fuse_auxiliary_logits(
            verifier_logits=base,
            scaled_logits=factors,
            torch=torch,
        )
        assert changed > reference


def test_unmarked_sentence_round_trip_and_marker_validation():
    marked = "[SUBJ]subject[/SUBJ] treats [OBJ]object[/OBJ]"
    assert _restore_unmarked_sentence(marked) == "subject treats object"
    with pytest.raises(ValueError, match="every marker exactly once"):
        _restore_unmarked_sentence("[SUBJ]subject[/SUBJ] object")
    with pytest.raises(ValueError, match="every marker exactly once"):
        _restore_unmarked_sentence(
            "[SUBJ]subject[/SUBJ] [OBJ]object[/OBJ] [OBJ]again[/OBJ]"
        )


def test_candidate_batch_deduplicates_unmarked_sentence_views():
    import torch

    class StubTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, texts, **_kwargs):
            rows = list(texts)
            self.calls.append(rows)
            return {
                "input_ids": torch.ones((len(rows), 3), dtype=torch.long),
                "attention_mask": torch.ones((len(rows), 3), dtype=torch.long),
            }

    candidates = (
        PairRerankCandidate(
            0, 0, "a", "p0", "b", "[SUBJ]a[/SUBJ] x [OBJ]b[/OBJ]"
        ),
        PairRerankCandidate(
            0, 1, "b", "p1", "a", "[OBJ]a[/OBJ] x [SUBJ]b[/SUBJ]"
        ),
    )
    tokenizer = StubTokenizer()
    encoded = _encode_candidates(
        tokenizer,
        candidates,
        (0, 1),
        torch.device("cpu"),
        torch,
    )
    assert tokenizer.calls[1] == ["a x b"]
    assert encoded["sentence_input_ids"].shape[0] == 1
    assert encoded["sentence_parent_indices"].tolist() == [0, 0]


@pytest.mark.parametrize("labels", ((1.0, 1.0), (0.0, 0.0), (1.0, 0.0)))
def test_balanced_binary_aux_bce_is_finite_for_single_and_mixed_classes(labels):
    import torch

    logits = torch.tensor((0.2, -0.1), requires_grad=True)
    loss = _balanced_binary_aux_bce(
        logits=logits,
        labels=torch.tensor(labels),
        torch=torch,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_encoder_context_is_invariant_to_proposed_predicate_text():
    first = _candidate(0, 0, "predicate-a", "o")
    second = PairRerankCandidate(
        row_index=first.row_index,
        predicate_id=1,
        subject=first.subject,
        predicate="predicate-b",
        object=first.object,
        marked_text=first.marked_text,
    )
    assert _encoder_texts((first, second), (0, 1)) == [
        first.marked_text,
        first.marked_text,
    ]


def test_group_listwise_loss_prefers_exact_positive_at_group_start():
    import torch

    good = _group_listwise_loss(
        logits=torch.tensor([4.0, 0.0, -1.0, 3.0, 0.0]),
        spans=((0, 3), (3, 5)),
        torch=torch,
    )
    bad = _group_listwise_loss(
        logits=torch.tensor([0.0, 4.0, -1.0, 0.0, 3.0]),
        spans=((0, 3), (3, 5)),
        torch=torch,
    )
    assert good < bad


def test_relation_softmax_ce_prefers_gold_and_supports_multi_label():
    import torch

    labels = torch.tensor(((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
    active = torch.tensor((True, True))
    good = _soft_relation_cross_entropy(
        relation_logits=torch.tensor(((5.0, 0.0, -1.0), (4.0, 4.0, -2.0))),
        relation_labels=labels,
        active=active,
        torch=torch,
    )
    bad = _soft_relation_cross_entropy(
        relation_logits=torch.tensor(((0.0, 5.0, -1.0), (-2.0, -2.0, 4.0))),
        relation_labels=labels,
        active=active,
        torch=torch,
    )
    assert good < bad


def test_onerel_candidate_batch_masks_both_padded_link_endpoints():
    import torch

    logits = torch.zeros((2, 1, 4, 3, 3))
    attention = torch.tensor(((1, 1, 0), (1, 1, 1)))
    masked = _mask_padded_link_logits(logits, attention)
    assert torch.isfinite(masked[0, :, :, :2, :2]).all()
    assert torch.isneginf(masked[0, :, :, 2, :]).all()
    assert torch.isneginf(masked[0, :, :, :, 2]).all()
    assert torch.isfinite(masked[1]).all()


def test_raw_coordinate_safety_cap_precedes_strict_model_candidate_cap():
    assert COORDINATE_CANDIDATE_CAP > CANDIDATE_CAP
    assert CANDIDATE_CAP == 4096


def test_cross_encoder_receives_bounded_proposal_scores():
    candidate = PairRerankCandidate(
        row_index=0,
        predicate_id=1,
        subject="s",
        predicate="p",
        object="o",
        marked_text="[SUBJ]s[/SUBJ][OBJ]o[/OBJ]",
        relation_score=10.0,
        link_score=0.0,
        boundary_score=-10.0,
    )
    relation, link, boundary = _proposal_feature_row(candidate)
    assert 0.99 < relation < 1.0
    assert link == 0.0
    assert -1.0 < boundary < -0.99


def test_oof_candidate_rows_remap_to_base_fit_order() -> None:
    local = (
        _candidate(0, 0, "p0", "乙"),
        _candidate(1, 0, "p0", "丙"),
    )
    remapped = _remap_oof_candidates(
        local,
        validation_indices=(9, 3),
        fit_positions={3: 0, 9: 1},
    )
    assert [candidate.row_index for candidate in remapped] == [1, 0]


def test_oof_contract_binds_source_rows_and_exact_once_validation(tmp_path) -> None:
    rows = [
        {"text": f"f{i}@body", "spo_list": [{"predicate": "p"}]}
        for i in range(10)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    split = {
        "source_train_sha256": source_sha,
        "splits": {"fit": {"indices": list(range(10))}},
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    partitions = []
    for fold in range(5):
        validation = [2 * fold, 2 * fold + 1]
        partitions.append(
            {
                "fold": fold,
                "train_indices": [i for i in range(10) if i not in validation],
                "validation_indices": validation,
                "family_intersection": [],
            }
        )
    oof = {
        "source_train_sha256": source_sha,
        "base_split_manifest_sha256": hashlib.sha256(
            split_path.read_bytes()
        ).hexdigest(),
        "partitions": partitions,
    }
    oof_path = tmp_path / "oof.json"
    oof_path.write_text(json.dumps(oof), encoding="utf-8")
    loaded, normalized, fit_positions = _validated_oof_partitions(
        source_train=source,
        fit_rows=rows,
        split_manifest_path=split_path,
        oof_manifest_path=oof_path,
    )
    assert loaded == rows
    assert len(normalized) == 5
    assert fit_positions == {index: index for index in range(10)}
