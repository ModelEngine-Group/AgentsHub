"""Task 3 graph-driven analysis agent.

Loads a task-2 knowledge graph, plans analysis operators with a rule-based,
LLM-assisted, or local-model planner, and produces graph analytics, NL2SQL
results, and BI/insight visualizations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.common.results import PipelineResult
from src.operators.analysis_ops import (
    build_analysis_report,
    build_analysis_visualizations,
    build_graph_sqlite,
    compute_centrality,
    compute_shortest_paths,
    detect_communities,
    export_echarts_dashboard,
    export_insight_report,
    generate_association_analysis,
    generate_statistical_summary,
    generate_trend_analysis,
    load_graph,
    prepare_graph_degree_tensor_cache,
    translate_question_with_fallbacks,
)
from src.common.integration import summarize_graph_for_planning
from src.common.run_tracker import AgentRunTracker
from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner
from src.pipelines.task2_kg_pipeline import run_task2_pipeline

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_FILE = PROJECT_ROOT / "outputs" / "task2" / "medical_kg.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"
DEFAULT_QUESTION = "哪些疾病关联最多症状？"


class GraphAnalysisAgent:
    """Plan and execute graph-driven analysis over a task-2 KG artifact."""

    task_name = "task3_analysis_agent"

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        local_model_path: str | None = None,
    ) -> None:
        self._llm_config = llm_config
        self._local_model_path = local_model_path

    def run(
        self,
        graph_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        question: str | None = DEFAULT_QUESTION,
        task_request: str | None = None,
    ) -> PipelineResult:
        tracker = AgentRunTracker()
        tracker.start()
        source = Path(graph_file) if graph_file else DEFAULT_GRAPH_FILE
        target_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        input_source = "graph_file"

        planner = AnalysisHybridPlanner(
            llm_config=self._llm_config,
            local_model_path=self._local_model_path,
        )

        try:
            if not source.exists():
                input_source = "bootstrapped_task2"
                source = _bootstrap_graph(
                    tracker, source.parent, llm_config=self._llm_config
                )

            graph = tracker.run_step(
                "load_graph",
                lambda: load_graph(source),
                "Loaded task-2 graph artifact.",
            )
            graph_summary = summarize_graph_for_planning(graph)

            # Planning: graph-aware hybrid planner
            plan = tracker.run_step(
                "plan_analysis",
                lambda: planner.plan(
                    task_request,
                    question=question,
                    graph_summary=graph_summary,
                ),
                "Planned graph analysis operators using graph context.",
            )

            statistics = tracker.run_step(
                "generate_statistics",
                lambda: generate_statistical_summary(graph),
                "Generated graph statistics.",
            )
            associations = tracker.run_step(
                "generate_associations",
                lambda: generate_association_analysis(graph),
                "Generated disease-centered associations.",
            )
            trends = tracker.run_step(
                "generate_trends",
                lambda: generate_trend_analysis(graph),
                "Generated record-sequence trends.",
            )
            degree_tensor_cache = tracker.run_step(
                "prepare_degree_tensor_cache",
                lambda: prepare_graph_degree_tensor_cache(graph),
                "Prepared reusable graph-degree tensor cache for NPU top-hubs when available.",
            )

            # Graph analytics (centrality, paths, communities)
            centrality = tracker.run_step(
                "compute_centrality",
                lambda: compute_centrality(graph, degree_tensor_cache=degree_tensor_cache),
                "Computed node degree centrality.",
            )

            # Plan-driven extended analytics: run shortest-path and community
            # detection only when the planner asks for them (see
            # _should_run_extended_analytics for the rule-vs-LLM contract).
            if _should_run_extended_analytics(plan):
                graph_analytics = tracker.run_step(
                    "extended_graph_analytics",
                    lambda: _run_extended_graph_analytics(graph, centrality),
                    "Computed communities and hub-anchored shortest paths.",
                )
            else:
                graph_analytics = {
                    "status": "skipped",
                    "reason": "graph_analytics intent not selected by planner",
                }

            # Plan-driven NL2SQL: honour the same rule-vs-LLM gating contract as
            # Task 2's QA step. The rule planner lists every operator (so NL2SQL
            # runs), while an LLM planner that omits the SQL operators skips it.
            nl2sql_selected = _operator_selected(plan, "translate_question_to_sql")
            if nl2sql_selected:
                conn = tracker.run_step(
                    "build_sqlite",
                    lambda: build_graph_sqlite(graph),
                    "Built in-memory SQLite analytics schema.",
                )
                nl2sql = tracker.run_step(
                    "translate_and_execute_sql",
                    lambda: translate_question_with_fallbacks(
                        question, conn,
                        llm_config=self._llm_config,
                        local_model_path=self._local_model_path,
                    ),
                    "Translated question to SQL and executed.",
                )
            else:
                nl2sql = {
                    "status": "skipped",
                    "reason": "NL2SQL operators not selected by plan",
                }

            visualizations = tracker.run_step(
                "build_visualizations",
                lambda: build_analysis_visualizations(statistics, associations, trends),
                "Built BI visualization specifications.",
            )
            export = tracker.run_step(
                "export_analysis",
                lambda: _export_analysis(
                    target_dir=target_dir,
                    payload={
                        "graph_file": str(source),
                        "statistics": statistics,
                        "associations": associations,
                        "trends": trends,
                        "centrality": centrality,
                        "graph_analytics": graph_analytics,
                        "nl2sql": nl2sql,
                        "visualizations": visualizations,
                    },
                ),
                "Exported task-3 analysis report.",
            )
            insight_report = tracker.run_step(
                "export_insight_report",
                lambda: export_insight_report(
                    target_dir=target_dir,
                    graph=graph,
                    statistics=statistics,
                    associations=associations,
                    trends=trends,
                    nl2sql=nl2sql,
                    visualizations=visualizations,
                    centrality=centrality,
                    graph_analytics=graph_analytics,
                ),
                "Exported human-readable task-3 insight report.",
            )
            # ECharts interactive dashboard
            echarts_dashboard = tracker.run_step(
                "export_echarts_dashboard",
                lambda: export_echarts_dashboard(
                    target_dir=target_dir,
                    statistics=statistics,
                    associations=associations,
                    trends=trends,
                    nl2sql=nl2sql,
                    visualizations=visualizations,
                    centrality=centrality,
                ),
                "Exported ECharts interactive dashboard.",
            )
            quality_report = tracker.run_step(
                "build_quality_report",
                lambda: build_analysis_report(
                    graph=graph,
                    statistics=statistics,
                    associations=associations,
                    trends=trends,
                    nl2sql=nl2sql,
                    visualizations=visualizations,
                    insight_report=insight_report,
                ),
                "Built task-3 quality report.",
            )
        except Exception as exc:
            tracker.fail()
            return PipelineResult(
                task=self.task_name,
                status="failed",
                message=f"Task 3 analysis failed: {exc}",
                artifacts={
                    "input": {"path": str(source), "source": input_source},
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "run_state": tracker.to_dict(),
                },
            )

        tracker.complete("completed")
        extended_ran = graph_analytics.get("status") != "skipped"
        nl2sql_ran = nl2sql.get("status") != "skipped"
        executed_optional = ["compute_centrality"]
        if extended_ran:
            executed_optional += ["compute_shortest_paths", "detect_communities"]
        if nl2sql_ran:
            executed_optional += ["translate_question_to_sql", "execute_sql"]
        planned_operators = plan.get("operators", [])
        executed_operators = _build_executed_analysis_operators(
            planned_operators, extended_ran, nl2sql_ran
        )
        artifacts = {
            "input": {"path": str(source), "source": input_source},
            "graph": graph.get("statistics", {}),
            "plan": plan,
            "plan_execution": {
                "selected_intents": plan.get("intent_keywords", []),
                "extended_analytics": extended_ran,
                "nl2sql_executed": nl2sql_ran,
                "executed_optional_operators": executed_optional,
                "planned_operators": planned_operators,
                "executed_operators": executed_operators,
            },
            "statistics": statistics,
            "associations": associations,
            "trends": trends,
            "centrality": centrality,
            "graph_analytics": graph_analytics,
            "nl2sql": nl2sql,
            "visualizations": visualizations,
            "export": export,
            "insight_report": insight_report,
            "echarts_dashboard": echarts_dashboard,
            "quality_report": quality_report,
            "run_state": tracker.to_dict(),
        }
        return PipelineResult(
            task=self.task_name,
            status="completed",
            message=(
                f"Task 3 analyzed {artifacts['graph'].get('node_count', 0)} nodes "
                f"and {artifacts['graph'].get('edge_count', 0)} edges."
            ),
            artifacts=artifacts,
        )


_EXTENDED_GRAPH_OPERATORS = {"compute_shortest_paths", "detect_communities"}
_NL2SQL_OPERATORS = {"translate_question_to_sql", "execute_sql"}


def _operator_selected(plan: dict[str, Any], operator: str, *, rule_default: bool = True) -> bool:
    """Uniform rule-vs-LLM operator gating contract (shared across optional steps).

    Mirrors Task 2's gating semantics:
    - LLM planner emits a meaningful operator subset, so an operator is selected
      only when present in ``plan["operators"]``.
    - The rule planner lists every operator, so its list is not a selection
      signal; optional steps default to running (``rule_default``).
    """

    if plan.get("planner_mode") == "llm":
        return operator in set(plan.get("operators", []))
    return rule_default


def _build_executed_analysis_operators(
    planned_operators: list[str],
    extended_ran: bool,
    nl2sql_ran: bool = True,
) -> list[str]:
    """Map planned analysis operators to those actually executed (plan order).

    The foundational operators (load/statistics/association/trend/centrality/
    visualization/report) always run; the extended graph operators and the
    NL2SQL operators run only when the plan selects them (rule-vs-LLM contract).
    """

    executed: set[str] = {
        "load_graph",
        "generate_statistical_summary",
        "generate_association_analysis",
        "generate_trend_analysis",
        "compute_centrality",
        "build_analysis_visualizations",
        "build_analysis_report",
    }
    if extended_ran:
        executed |= _EXTENDED_GRAPH_OPERATORS
    if nl2sql_ran:
        executed |= _NL2SQL_OPERATORS
    ordered = [op for op in planned_operators if op in executed]
    # Include any executed operators not present in the planned list (defensive).
    ordered += [op for op in executed if op not in planned_operators]
    return ordered


def _should_run_extended_analytics(plan: dict[str, Any]) -> bool:
    """Decide whether to run extended graph analytics for this plan.

    Both planner styles must be able to drive execution:

    - The rule planner emits a ``graph_analytics`` intent keyword when the
      request mentions hubs/communities/paths, but always lists *every*
      operator, so its operator list is not a meaningful selection signal.
    - The LLM planner selects a real operator subset, so its choice of the
      extended operators is an explicit, honored signal.
    """

    if "graph_analytics" in set(plan.get("intent_keywords", [])):
        return True
    if plan.get("planner_mode") == "llm":
        return bool(set(plan.get("operators", [])) & _EXTENDED_GRAPH_OPERATORS)
    return False


def _run_extended_graph_analytics(
    graph: dict[str, Any],
    centrality: dict[str, Any],
) -> dict[str, Any]:
    """Run community detection and hub-anchored shortest-path reachability.

    The shortest-path start node is selected autonomously as the highest-degree
    hub from the centrality result, so no manual entity input is required.
    """

    communities = detect_communities(graph)

    hubs = centrality.get("top_hubs", [])
    start_id = hubs[0]["id"] if hubs else None
    if start_id:
        paths = compute_shortest_paths(graph, start_entity=start_id)
    else:
        paths = {"status": "skipped", "reason": "no hub node available"}

    return {
        "status": "completed",
        "start_hub": start_id,
        "communities": communities,
        "shortest_paths": paths,
    }


def _bootstrap_graph(
    tracker: AgentRunTracker,
    output_dir: Path,
    llm_config: dict[str, Any] | None,
) -> Path:
    """Bootstrap a task-2 graph by chaining Task 1 -> Task 2.

    When no graph artifact is supplied, task 3 reuses the upstream tasks
    end-to-end: the task-1 agent cleans the raw medical notes and the task-2
    agent builds the KG from that cleaned text.  This makes the default run a
    genuine "data -> knowledge -> insight" reuse, not a task-2-only shortcut.
    If task-1 cleaning is unavailable for any reason, the build degrades
    gracefully to a direct task-2 graph from the raw sample.
    """

    from src.agents.data_processing_agent.agent import (
        DEFAULT_SAMPLE_TEXT,
        DataProcessingAgent,
    )

    cleaned_text: str | None = None
    try:
        task1 = tracker.run_step(
            "bootstrap_task1_clean",
            lambda: DataProcessingAgent(llm_config=llm_config).run(
                task_request="清洗并标准化医疗文本数据",
                input_path=DEFAULT_SAMPLE_TEXT,
                output_dir=output_dir / "task1",
                datamate_base_url=None,
            ),
            "Cleaned raw medical text with the task-1 agent for reuse.",
        )
        if task1.status in {"completed", "completed_with_warnings"}:
            cleaned_text = task1.artifacts.get("processing", {}).get("output_path")
    except Exception:
        logger.warning(
            "Task-1 bootstrap cleaning failed; building task-2 graph from raw sample.",
            exc_info=True,
        )

    task2_result = tracker.run_step(
        "bootstrap_task2_graph",
        lambda: run_task2_pipeline(input_path=cleaned_text, output_dir=output_dir),
        "Generated a task-2 graph artifact for task-3 analysis.",
    )
    if task2_result.status != "completed":
        raise RuntimeError(task2_result.message)
    return Path(task2_result.artifacts["graph"]["output_path"])


def _export_analysis(target_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "task3_analysis_report.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "completed", "output_path": str(output_path)}
