from operators import (
    CentralityAnalysisOperator,
    DiseaseSymptomCountOperator,
    DrugDiseaseCountOperator,
    GraphStatisticsOperator,
    NL2AnalysisOperator,
)


SAMPLE_GRAPH = {
    "nodes": [
        {"id": "肺炎", "label": "肺炎", "type": "Disease"},
        {"id": "冠心病", "label": "冠心病", "type": "Disease"},
        {"id": "发热", "label": "发热", "type": "Symptom"},
        {"id": "胸痛", "label": "胸痛", "type": "Symptom"},
        {"id": "阿莫西林", "label": "阿莫西林", "type": "Drug"},
    ],
    "edges": [
        {"source": "肺炎", "target": "发热", "relation": "has_symptom"},
        {"source": "肺炎", "target": "胸痛", "relation": "has_symptom"},
        {"source": "冠心病", "target": "胸痛", "relation": "has_symptom"},
        {"source": "肺炎", "target": "阿莫西林", "relation": "treated_by"},
    ],
}


def test_graph_statistics_operator_counts_nodes_edges_types_and_relations():
    result = GraphStatisticsOperator().execute(SAMPLE_GRAPH)

    assert result["node_count"] == 5
    assert result["edge_count"] == 4
    assert result["entity_type_count"]["Disease"] == 2
    assert result["entity_type_count"]["Symptom"] == 2
    assert result["relation_type_count"]["has_symptom"] == 3
    assert result["relation_type_count"]["treated_by"] == 1


def test_disease_symptom_count_operator_counts_symptoms_per_disease():
    result = DiseaseSymptomCountOperator().execute(SAMPLE_GRAPH)

    assert result[0] == {"disease": "肺炎", "symptom_count": 2}
    assert {"disease": "冠心病", "symptom_count": 1} in result


def test_drug_disease_count_operator_counts_diseases_per_drug():
    result = DrugDiseaseCountOperator().execute(SAMPLE_GRAPH)

    assert result == [{"drug": "阿莫西林", "disease_count": 1, "diseases": ["肺炎"]}]


def test_centrality_analysis_operator_returns_top_nodes():
    result = CentralityAnalysisOperator().execute(SAMPLE_GRAPH, top_k=2)

    assert len(result) == 2
    assert result[0]["node"] in {"肺炎", "胸痛"}
    assert "degree_centrality" in result[0]


def test_nl2analysis_operator_maps_question_to_analysis_task():
    operator = NL2AnalysisOperator()

    assert operator.execute("统计每类实体数量") == "graph_statistics"
    assert operator.execute("哪个疾病的症状最多？") == "disease_symptom_count"
    assert operator.execute("药物关联疾病数量") == "drug_disease_count"
    assert operator.execute("中心性最高的节点") == "centrality_analysis"
