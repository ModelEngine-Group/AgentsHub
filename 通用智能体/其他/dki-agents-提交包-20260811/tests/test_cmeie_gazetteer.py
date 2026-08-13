from dki.nlp.cmeie_gazetteer import CmeieTrainGazetteer


def test_train_gazetteer_returns_typed_literal_mentions():
    gazetteer = CmeieTrainGazetteer([
        {"spo_list": [{
            "subject": "高血压", "subject_type": "疾病",
            "predicate": "临床表现", "object": "头痛", "object_type": "症状",
        }]},
    ])
    assert gazetteer.extract("高血压患者出现头痛") == [
        {"name": "高血压", "type": "疾病", "span": [0, 3]},
        {"name": "头痛", "type": "症状", "span": [7, 9]},
    ]


def test_train_gazetteer_keeps_ambiguous_surface_types_and_minimum_length():
    gazetteer = CmeieTrainGazetteer([
        {"spo_list": [
            {"subject": "甲", "subject_type": "疾病", "object": "乙病", "object_type": "症状"},
            {"subject": "乙病", "subject_type": "疾病", "object": "甲", "object_type": "症状"},
        ]},
    ], min_chars=2)
    assert gazetteer.extract("乙病") == [
        {"name": "乙病", "type": "疾病", "span": [0, 2]},
        {"name": "乙病", "type": "症状", "span": [0, 2]},
    ]
    assert gazetteer.provenance()["source_rows"] == 1


def test_train_gazetteer_only_augments_matching_upstream_spans():
    gazetteer = CmeieTrainGazetteer([
        {"spo_list": [{
            "subject": "高血压", "subject_type": "疾病",
            "object": "头痛", "object_type": "症状",
        }]},
    ])
    assert gazetteer.augment_types("高血压伴头痛", [
        {"name": "高血压", "type": "dis", "span": [0, 3]},
        {"name": "头痛", "type": "sym", "span": [4, 6]},
        {"name": "不存在", "type": "dis", "span": [7, 10]},
    ]) == [
        {"name": "高血压", "type": "疾病", "span": [0, 3]},
        {"name": "头痛", "type": "症状", "span": [4, 6]},
        {"name": "不存在", "type": "疾病", "span": [7, 10]},
    ]
