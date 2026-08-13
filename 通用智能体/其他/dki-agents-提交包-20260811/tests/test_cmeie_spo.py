import pytest
from itertools import product

torch = pytest.importorskip("torch")

from dki.nlp.cmeie_spo import (
    all_occurrences,
    build_labels,
    char_span_to_tokens,
    decode_spo,
    decode_spo_grid,
    decode_spo_grid_sparse,
    decode_spo_grid_sparse_axes,
    exact_micro_f1,
    labels_from_fit,
    same_sentence,
    schema_constraints_from_fit,
    select_mention_pair,
    select_mention_pairs,
    sparse_join_match_lower_bound,
    sparsify_pointer_logits,
)


def test_role_only_entity_labels_schema_and_alignment():
    rows = [{
        "text": "alpha causes beta",
        "spo_list": [{
            "subject": "alpha",
            "subject_type": "disease",
            "predicate": "causes",
            "object": "beta",
            "object_type": "symptom",
        }],
    }]
    entity_labels, predicates = labels_from_fit(
        rows,
        entity_label_mode="role_only",
    )
    assert entity_labels == ["object:*", "subject:*"]
    assert predicates == ["causes"]
    assert schema_constraints_from_fit(
        rows,
        entity_label_mode="role_only",
    ) == {"causes": [("subject:*", "object:*")]}

    offsets = [(0, 0)] + [
        (index, index + 1) for index in range(len(rows[0]["text"]))
    ] + [(0, 0)]
    entity2id = {
        label: index for index, label in enumerate(entity_labels)
    }
    labels, dropped = build_labels(
        rows[0],
        offsets,
        entity2id,
        {"causes": 0},
    )
    assert dropped == 0
    assert labels["entity"].sum() == 2
    assert labels["relation_head"].sum() == 1
    assert labels["relation_tail"].sum() == 1


def test_entity_label_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="entity_label_mode"):
        labels_from_fit([], entity_label_mode="unknown")


def test_occurrences_offsets_and_sentence_boundary():
    assert all_occurrences("甲乙甲", "甲") == [(0, 1), (2, 3)]
    assert char_span_to_tokens([(0, 0), (0, 1), (1, 2), (0, 0)], (0, 2)) == (1, 2)
    assert char_span_to_tokens([(0, 0), (1, 2), (2, 3), (0, 0)], (0, 3)) == (1, 2)
    assert same_sentence("甲和乙。甲和乙", (0, 1), (2, 3))
    assert not same_sentence("甲。乙", (0, 1), (2, 3))


def test_labels_keep_nearest_links_for_all_repeated_same_sentence_mentions():
    row = {"text": "甲导致乙，甲导致乙", "spo_list": [{"subject": "甲", "subject_type": "疾病", "predicate": "临床表现", "object": "乙", "object_type": "症状"}]}
    offsets = [(0, 0)] + [(index, index + 1) for index in range(len(row["text"]))] + [(0, 0)]
    labels, dropped = build_labels(row, offsets, {"subject:疾病": 0, "object:症状": 1}, {"临床表现": 0})
    assert dropped == 0
    assert select_mention_pairs(row["text"], "甲", "乙") == [
        ((0, 1), (3, 4)),
        ((5, 6), (8, 9)),
    ]
    assert labels["entity"].sum() == 4
    assert labels["relation_head"].sum() == 2
    assert labels["relation_tail"].sum() == 2
    metric = exact_micro_f1([{("甲", "临床表现", "乙")}], [{("甲", "临床表现", "乙")}])
    assert metric["f1"] == 1.0


def test_cross_sentence_gold_requires_explicit_joint_model_fallback():
    row = {
        "text": "甲出现。随后乙出现。",
        "spo_list": [{
            "subject": "甲",
            "subject_type": "疾病",
            "predicate": "临床表现",
            "object": "乙",
            "object_type": "症状",
        }],
    }
    offsets = [(0, 0)] + [
        (index, index + 1) for index in range(len(row["text"]))
    ] + [(0, 0)]
    entity2id = {"subject:疾病": 0, "object:症状": 1}
    predicate2id = {"临床表现": 0}
    strict, strict_dropped = build_labels(
        row, offsets, entity2id, predicate2id
    )
    joint, joint_dropped = build_labels(
        row,
        offsets,
        entity2id,
        predicate2id,
        allow_cross_sentence_fallback=True,
    )
    assert select_mention_pair(row["text"], "甲", "乙") is None
    assert select_mention_pair(
        row["text"], "甲", "乙", allow_cross_sentence_fallback=True
    ) == ((0, 1), (6, 7))
    assert select_mention_pairs(
        row["text"], "甲", "乙", allow_cross_sentence_fallback=True
    ) == [((0, 1), (6, 7))]
    assert strict_dropped == 1
    assert strict["relation_head"].sum() == 0
    assert joint_dropped == 0
    assert joint["relation_head"].sum() == 1


def test_schema_constraint_filters_disallowed_type_pair():
    entity = torch.full((2, 5, 5), -1.0)
    entity[0, 1, 1] = entity[1, 2, 2] = 1.0
    links = torch.full((1, 5, 5), -1.0)
    links[0, 1, 2] = 1.0
    offsets = [(0, 0), (0, 1), (1, 2), (0, 0), (0, 0)]
    unrestricted = decode_spo("甲乙", offsets, entity, links, links, ["subject:疾病", "object:症状"], ["P"], (0, 0, 0))
    restricted = decode_spo("甲乙", offsets, entity, links, links, ["subject:疾病", "object:症状"], ["P"], (0, 0, 0), {"P": [("subject:疾病", "object:药物")]})
    assert unrestricted == {("甲", "P", "乙")}
    assert restricted == set()


def test_declared_grid_decode_matches_all_single_decodes():
    entity = torch.full((2, 5, 5), -1.0)
    entity[0, 1, 1] = 0.25
    entity[1, 2, 2] = 0.75
    links = torch.full((1, 5, 5), -1.0)
    links[0, 1, 2] = 0.25
    offsets = [(0, 0), (0, 1), (1, 2), (0, 0), (0, 0)]
    values = (-0.5, 0.0, 0.5)
    constraints = {"P": [("subject:疾病", "object:症状")]}
    grid = decode_spo_grid(
        "甲乙",
        offsets,
        entity,
        links,
        links,
        ["subject:疾病", "object:症状"],
        ["P"],
        values,
        constraints,
    )
    sparse_grid = decode_spo_grid_sparse(
        "甲乙",
        offsets,
        sparsify_pointer_logits(entity, min(values)),
        sparsify_pointer_logits(links, min(values)),
        sparsify_pointer_logits(links, min(values)),
        ["subject:疾病", "object:症状"],
        ["P"],
        values,
        constraints,
    )
    assert sparse_grid == grid
    nonnegative_tail_grid = decode_spo_grid_sparse_axes(
        "甲乙",
        offsets,
        sparsify_pointer_logits(entity, min(values)),
        sparsify_pointer_logits(links, min(values)),
        sparsify_pointer_logits(links, 0.0),
        ["subject:疾病", "object:症状"],
        ["P"],
        values,
        values,
        (0.0, 0.5),
        constraints,
    )
    assert nonnegative_tail_grid == {
        candidate: predictions
        for candidate, predictions in grid.items()
        if candidate[0][2] >= 0.0
    }
    for thresholds in product(values, repeat=3):
        for schema_on in (False, True):
            assert grid[(thresholds, schema_on)] == decode_spo(
                "甲乙",
                offsets,
                entity,
                links,
                links,
                ["subject:疾病", "object:症状"],
                ["P"],
                thresholds,
                constraints if schema_on else None,
            )


def test_sparse_join_lower_bound_caps_without_materialising_predictions():
    entity = torch.full((2, 5, 5), -1.0)
    entity[0, 1, 1] = entity[0, 1, 2] = 1.0
    entity[1, 2, 2] = entity[1, 2, 3] = 1.0
    links = torch.full((1, 5, 5), -1.0)
    links[0, 1, 2] = 1.0
    links[0, 1, 3] = 1.0
    links[0, 2, 2] = 1.0
    links[0, 2, 3] = 1.0
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (0, 0)]
    estimate = sparse_join_match_lower_bound(
        offsets,
        sparsify_pointer_logits(entity, -0.5),
        sparsify_pointer_logits(links, -0.5),
        sparsify_pointer_logits(links, -0.5),
        ["subject:疾病", "object:症状"],
        (-0.5, -0.5, -0.5),
        cap=2,
    )
    assert estimate["capped"] is True
    assert estimate["match_lower_bound"] >= 2
