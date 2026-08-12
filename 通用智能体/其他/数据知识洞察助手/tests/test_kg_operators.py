from operators import (
    EntityRecognitionOperator,
    GraphBuildOperator,
    RelationExtractionOperator,
    TripleGenerationOperator,
    TripleValidationOperator,
)


SAMPLE_DISEASE_TEXT = """
## 肺炎
肺炎是一种常见的呼吸道感染性疾病。常见症状包括发热、咳嗽、咳痰、胸痛和呼吸困难。
常用治疗药物包括阿莫西林、头孢类抗生素、左氧氟沙星等。诊断主要依靠胸部X线或CT检查。
"""


def test_entity_recognition_operator_extracts_medical_entities():
    entities = EntityRecognitionOperator().execute(SAMPLE_DISEASE_TEXT)

    assert {"name": "肺炎", "type": "Disease"} in entities
    assert {"name": "发热", "type": "Symptom"} in entities
    assert {"name": "咳嗽", "type": "Symptom"} in entities
    assert {"name": "阿莫西林", "type": "Drug"} in entities
    assert {"name": "胸部X线", "type": "Examination"} in entities


def test_relation_extraction_operator_extracts_typed_relations():
    relations = RelationExtractionOperator().execute(SAMPLE_DISEASE_TEXT)

    assert {"subject": "肺炎", "relation": "has_symptom", "object": "发热", "object_type": "Symptom"} in relations
    assert {"subject": "肺炎", "relation": "treated_by", "object": "阿莫西林", "object_type": "Drug"} in relations
    assert {"subject": "肺炎", "relation": "requires_exam", "object": "胸部X线", "object_type": "Examination"} in relations


def test_triple_generation_operator_creates_medical_triples():
    triples = TripleGenerationOperator().execute(SAMPLE_DISEASE_TEXT)

    assert {"subject": "肺炎", "predicate": "has_symptom", "object": "发热"} in triples
    assert {"subject": "肺炎", "predicate": "has_symptom", "object": "咳嗽"} in triples
    assert {"subject": "肺炎", "predicate": "treated_by", "object": "阿莫西林"} in triples
    assert {"subject": "肺炎", "predicate": "requires_exam", "object": "胸部X线"} in triples


def test_triple_validation_operator_filters_invalid_triples():
    triples = [
        {"subject": "肺炎", "predicate": "has_symptom", "object": "发热"},
        {"subject": "", "predicate": "has_symptom", "object": "咳嗽"},
        {"subject": "肺炎", "predicate": "unknown_relation", "object": "发热"},
    ]

    valid = TripleValidationOperator().execute(triples)

    assert valid == [{"subject": "肺炎", "predicate": "has_symptom", "object": "发热"}]


def test_graph_build_operator_builds_nodes_and_edges_from_triples():
    triples = [
        {"subject": "肺炎", "predicate": "has_symptom", "object": "发热"},
        {"subject": "肺炎", "predicate": "treated_by", "object": "阿莫西林"},
    ]
    entities = [
        {"name": "肺炎", "type": "Disease"},
        {"name": "发热", "type": "Symptom"},
        {"name": "阿莫西林", "type": "Drug"},
    ]

    graph = GraphBuildOperator().execute(triples=triples, entities=entities)

    assert {"id": "肺炎", "label": "肺炎", "type": "Disease"} in graph["nodes"]
    assert {"id": "发热", "label": "发热", "type": "Symptom"} in graph["nodes"]
    assert {"source": "肺炎", "target": "发热", "relation": "has_symptom"} in graph["edges"]
    assert {"source": "肺炎", "target": "阿莫西林", "relation": "treated_by"} in graph["edges"]
