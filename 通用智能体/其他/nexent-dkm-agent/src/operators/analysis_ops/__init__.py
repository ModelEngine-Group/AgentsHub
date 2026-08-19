"""Task 3 graph-driven analysis operators."""

from src.operators.analysis_ops.association import generate_association_analysis
from src.operators.analysis_ops.dashboard import export_analysis_dashboard
from src.operators.analysis_ops.echarts_dashboard import export_echarts_dashboard
from src.operators.analysis_ops.graph_analytics import (
    compute_centrality,
    compute_shortest_paths,
    detect_communities,
    prepare_graph_degree_tensor_cache,
)
from src.operators.analysis_ops.graph_loader import load_graph
from src.operators.analysis_ops.hybrid_planner import (
    REGISTERED_ANALYSIS_OPERATORS,
    AnalysisHybridPlanner,
    plan_analysis_task,
)
from src.operators.analysis_ops.insight_report import (
    export_insight_report,
    generate_graph_insights,
)
from src.operators.analysis_ops.llm_nl2sql import (
    translate_question_to_sql_llm_only,
    translate_question_to_sql_local_only,
    translate_question_to_sql_with_llm,
    translate_question_with_fallbacks,
)
from src.operators.analysis_ops.nl2sql import (
    INTENT_SQL,
    build_graph_sqlite,
    classify_question_intent,
    disease_names_from_connection,
    disease_names_from_graph,
    drug_names_from_connection,
    drug_names_from_graph,
    evaluate_nl2sql_accuracy,
    evaluate_nl2sql_execution_accuracy,
    execute_read_only_sql,
    execute_sql,
    symptom_names_from_connection,
    symptom_names_from_graph,
    translate_question_to_sql,
    treatment_names_from_connection,
    treatment_names_from_graph,
    validate_read_only_sql,
)
from src.operators.analysis_ops.reporting import build_analysis_report
from src.operators.analysis_ops.statistics import generate_statistical_summary
from src.operators.analysis_ops.trend import generate_trend_analysis
from src.operators.analysis_ops.visualization import build_analysis_visualizations

__all__ = [
    "INTENT_SQL",
    "REGISTERED_ANALYSIS_OPERATORS",
    "AnalysisHybridPlanner",
    "build_analysis_report",
    "build_analysis_visualizations",
    "build_graph_sqlite",
    "classify_question_intent",
    "compute_centrality",
    "compute_shortest_paths",
    "detect_communities",
    "disease_names_from_connection",
    "disease_names_from_graph",
    "drug_names_from_connection",
    "drug_names_from_graph",
    "evaluate_nl2sql_accuracy",
    "evaluate_nl2sql_execution_accuracy",
    "execute_read_only_sql",
    "execute_sql",
    "export_analysis_dashboard",
    "export_echarts_dashboard",
    "export_insight_report",
    "generate_association_analysis",
    "generate_graph_insights",
    "generate_statistical_summary",
    "generate_trend_analysis",
    "load_graph",
    "plan_analysis_task",
    "prepare_graph_degree_tensor_cache",
    "symptom_names_from_connection",
    "symptom_names_from_graph",
    "translate_question_to_sql",
    "translate_question_to_sql_llm_only",
    "translate_question_to_sql_local_only",
    "translate_question_to_sql_with_llm",
    "translate_question_with_fallbacks",
    "treatment_names_from_connection",
    "treatment_names_from_graph",
    "validate_read_only_sql",
]
