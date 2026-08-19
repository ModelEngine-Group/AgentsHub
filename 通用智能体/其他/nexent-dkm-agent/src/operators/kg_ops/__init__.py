"""Task 2 knowledge graph operators."""

from src.operators.kg_ops.entity_extractor import (
    ENTITY_DICTIONARY,
    extract_medical_entities,
    infer_entity_type,
)
from src.operators.kg_ops.extraction_eval import (
    entity_in_vocabulary,
    evaluate_extraction_quality,
    evaluate_extraction_vocabulary_split,
    evaluate_relation_quality,
)
from src.operators.kg_ops.graph_builder import build_medical_graph
from src.operators.kg_ops.llm_extractor import (
    extract_entities_with_llm,
    extract_relations_with_llm,
)
from src.operators.kg_ops.multi_hop_qa import (
    answer_with_evidence_chain,
    build_evidence_chain,
    multi_hop_query,
)
from src.operators.kg_ops.neo4j_query import (
    neo4j_answer_question,
    neo4j_find_entities,
    neo4j_multi_hop,
    neo4j_query_neighbors,
)
from src.operators.kg_ops.neo4j_store import (
    check_neo4j_connection,
    clear_neo4j_graph,
    graph_to_neo4j,
    neo4j_to_graph,
)
from src.operators.kg_ops.qa import answer_graph_question
from src.operators.kg_ops.query import find_graph_entities, query_graph_neighbors
from src.operators.kg_ops.relation_extractor import (
    RELATION_SCHEMA,
    extract_relations,
    extract_relations_tensorized,
)
from src.operators.kg_ops.relation_features import (
    build_relation_candidates,
    build_scoring_inputs,
    encode_relation_candidates,
    generate_relation_projection_weights,
)
from src.operators.kg_ops.reporting import build_kg_quality_report
from src.operators.kg_ops.triple_validator import validate_triples

__all__ = [
    "ENTITY_DICTIONARY",
    "RELATION_SCHEMA",
    "answer_graph_question",
    "answer_with_evidence_chain",
    "build_evidence_chain",
    "build_kg_quality_report",
    "build_medical_graph",
    "build_relation_candidates",
    "build_scoring_inputs",
    "check_neo4j_connection",
    "clear_neo4j_graph",
    "encode_relation_candidates",
    "entity_in_vocabulary",
    "evaluate_extraction_quality",
    "evaluate_extraction_vocabulary_split",
    "evaluate_relation_quality",
    "extract_entities_with_llm",
    "extract_medical_entities",
    "extract_relations",
    "extract_relations_tensorized",
    "extract_relations_with_llm",
    "generate_relation_projection_weights",
    "find_graph_entities",
    "graph_to_neo4j",
    "infer_entity_type",
    "multi_hop_query",
    "neo4j_answer_question",
    "neo4j_find_entities",
    "neo4j_multi_hop",
    "neo4j_query_neighbors",
    "neo4j_to_graph",
    "query_graph_neighbors",
    "validate_triples",
]
