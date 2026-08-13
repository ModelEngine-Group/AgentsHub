import hashlib

from dki.nlp.softlexicon import (
    argument_surface_lexicon_payload,
    build_cmeie_argument_surface_lexicon,
    canonical_surface_lexicon_bytes,
    char_bmes_features,
    surface_lexicon_sha256,
)


def test_cmeie_softlexicon_is_fit_only_native_typed_and_deterministic():
    labels = ["disease", "symptom"]
    type2id = {label: index for index, label in enumerate(labels)}
    rows = [
        {
            "text": "甲病伴乙症，甲病需复查。",
            "spo_list": [
                {
                    "subject": "甲病",
                    "subject_type": "disease",
                    "predicate": "临床表现",
                    "object": "乙症",
                    "object_type": "symptom",
                },
                {
                    "subject": "不在原文",
                    "subject_type": "disease",
                    "predicate": "相关",
                    "object": "乙症",
                    "object_type": "symptom",
                },
            ],
        }
    ]
    lexicon = build_cmeie_argument_surface_lexicon(
        rows,
        type2id,
        min_chars=2,
    )
    assert lexicon == {"甲病": {0}, "乙症": {1}}
    payload = argument_surface_lexicon_payload(
        lexicon,
        labels,
        min_chars=2,
    )
    encoded = canonical_surface_lexicon_bytes(payload)
    assert surface_lexicon_sha256(payload) == hashlib.sha256(encoded).hexdigest()
    assert payload["feature_dim"] == 8
    assert payload["surface_count"] == 2
    assert payload["type_assignment_count"] == 2


def test_cmeie_softlexicon_bmes_marks_repeated_fit_surfaces():
    features = char_bmes_features("甲病和甲病", {"甲病": {0}}, 1)
    assert features.shape == (5, 4)
    assert features[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert features[1].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert features[3].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert features[4].tolist() == [0.0, 0.0, 1.0, 0.0]
