"""Task 2 medical knowledge graph agent.

Supports rule-based, LLM-assisted, local model planning,
and Neo4j graph database backend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agents.kg_agent.planner import KGHybridPlanner
from src.common.results import PipelineResult
from src.common.run_tracker import AgentRunTracker
from src.operators.kg_ops import (
    answer_graph_question,
    build_kg_quality_report,
    build_medical_graph,
    extract_medical_entities,
    extract_relations,
    extract_relations_tensorized,
    find_graph_entities,
    graph_to_neo4j,
    query_graph_neighbors,
    validate_triples,
)
from src.operators.kg_ops.llm_extractor import (
    extract_entities_with_llm,
    extract_relations_with_llm,
)
from src.operators.kg_ops.local_model_ner import predict_kg_entities
from src.operators.analysis_ops.graph_loader import load_graph

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "samples" / "task2_medical_notes.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task2"
DEFAULT_QUESTION = "高血压有哪些症状和用药？"


class MedicalKGAgent:
    """Plan and execute a medical KG generation and QA workflow."""

    task_name = "task2_kg_agent"

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        local_model_path: str | None = None,
        neo4j_config: dict[str, str] | None = None,
        relation_backend: str = "rule",
    ) -> None:
        self._llm_config = llm_config
        self._local_model_path = local_model_path
        self._neo4j_config = neo4j_config
        self._relation_backend = (relation_backend or "rule").lower()

    @property
    def _effective_local_model_path(self) -> str | None:
        """Return local_model_path only if it points to an existing directory."""
        if self._local_model_path and Path(self._local_model_path).is_dir():
            return self._local_model_path
        if self._local_model_path:
            logger.info(
                "local_model_path '%s' does not exist or is not a directory; skipping local model NER.",
                self._local_model_path,
            )
        return None

    def run(
        self,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        question: str | None = DEFAULT_QUESTION,
        task_request: str | None = None,
        graph_file: str | Path | None = None,
    ) -> PipelineResult:
        tracker = AgentRunTracker()
        tracker.start()
        source = Path(input_path) if input_path else DEFAULT_INPUT
        target_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

        planner = KGHybridPlanner(
            llm_config=self._llm_config,
            local_model_path=self._effective_local_model_path,
        )

        # Plan FIRST (without loading the corpus yet) so query-only plans can
        # skip the expensive text -> graph rebuild and reuse a cached graph.
        plan = planner.plan(task_request, question=question)
        plan_dict = plan.to_dict()
        plan_operators = plan_dict.get("operators", [])
        task_type = plan_dict.get("understanding", {}).get("task_type", "full_pipeline")
        logger.info(
            "KG plan: task_type=%s, planner_mode=%s, operators=%s",
            task_type, plan.planner_mode, plan_operators,
        )

        build_operators = {"extract_medical_entities", "extract_relations", "build_medical_graph"}
        plan_wants_build = bool(build_operators & set(plan_operators))
        resolved_graph_path = (
            Path(graph_file) if graph_file else (target_dir / "medical_kg.json")
        )
        # Reuse a cached graph for query-only plans (kg_query), or whenever an
        # explicit graph_file is supplied for a plan that doesn't request build.
        reuse_graph = resolved_graph_path.exists() and (
            task_type == "kg_query" or (graph_file is not None and not plan_wants_build)
        )

        if reuse_graph:
            return self._run_query_only(
                tracker=tracker,
                plan_dict=plan_dict,
                plan_operators=plan_operators,
                graph_path=resolved_graph_path,
                question=question,
                task_request=task_request,
            )

        try:
            text = tracker.run_step(
                "load_input",
                lambda: _read_input(source),
                "Loaded medical text corpus.",
            )

            # Use LLM-enhanced extraction if configured, else rule-based
            if self._llm_config:
                extraction = tracker.run_step(
                    "extract_entities",
                    lambda: extract_entities_with_llm(text, self._llm_config),
                    "Extracted entities using LLM.",
                )
                # Also run rule-based extraction for fallback triples
                rule_extraction = extract_medical_entities(text)
            else:
                extraction = tracker.run_step(
                    "extract_entities",
                    lambda: extract_medical_entities(text),
                    "Extracted disease, symptom, drug, examination, and treatment entities.",
                )
                rule_extraction = None

            # Local model NER: merge predicted entities BEFORE relation extraction
            # so they participate in graph construction
            local_model_entities = None
            effective_lmp = self._effective_local_model_path
            if effective_lmp:
                local_model_entities = predict_kg_entities(effective_lmp, text)
                if local_model_entities:
                    extraction = _merge_local_model_entities(extraction, local_model_entities)
                    if rule_extraction is not None:
                        rule_extraction = _merge_local_model_entities(rule_extraction, local_model_entities)
                    logger.info("Local model NER merged %d entity types.", len(local_model_entities))

            # Relation extraction uses the enriched records
            relation_scoring: dict[str, Any] = {
                "backend": self._relation_backend,
                "mode": "rule",
            }
            if self._llm_config:
                rule_triples = extract_relations(rule_extraction["records"])
                # Reuse LLM relations cached during entity extraction to avoid
                # calling the LLM a second time for the same chunks.
                cached_llm_relations = extraction.get("_cached_llm_relations", [])
                if cached_llm_relations:
                    triples = tracker.run_step(
                        "extract_relations",
                        lambda: _merge_triples(cached_llm_relations, rule_triples),
                        "Generated triples using cached LLM results + rule-based merge.",
                    )
                else:
                    triples = tracker.run_step(
                        "extract_relations",
                        lambda: extract_relations_with_llm(text, self._llm_config, rule_triples),
                        "Generated triples using LLM + rule-based merge.",
                    )
            elif self._relation_backend in {"cpu", "npu"}:
                tensor_result = tracker.run_step(
                    "extract_relations",
                    lambda: extract_relations_tensorized(
                        extraction["records"], backend=self._relation_backend
                    ),
                    f"Generated triples via tensorized relation scoring ({self._relation_backend}).",
                )
                triples = tensor_result["triples"]
                scoring_backend = tensor_result.get("scoring_backend", "rule")
                scoring_device = tensor_result.get("scoring_device", "cpu")
                relation_scoring = {
                    "backend": self._relation_backend,
                    "mode": "tensorized",
                    "scoring_backend": scoring_backend,
                    "scoring_device": scoring_device,
                    "candidate_count": tensor_result.get("candidate_count", 0),
                    "status": tensor_result.get("status"),
                }
                if scoring_backend != "rule":
                    logger.info(
                        "Tensorized relation scoring used backend=%s device=%s candidates=%d status=%s",
                        scoring_backend,
                        scoring_device,
                        tensor_result.get("candidate_count", 0),
                        tensor_result.get("status"),
                    )
                else:
                    logger.info(
                        "Tensorized relation scoring fell back to rule path (status=%s).",
                        tensor_result.get("status"),
                    )
            else:
                triples = tracker.run_step(
                    "extract_relations",
                    lambda: extract_relations(extraction["records"]),
                    "Generated candidate medical KG triples.",
                )

            validation = tracker.run_step(
                "validate_triples",
                lambda: validate_triples(triples),
                "Validated triples against the task-2 schema.",
            )
            graph = tracker.run_step(
                "build_graph",
                lambda: build_medical_graph(validation["triples"], extraction["records"]),
                "Built a deduplicated medical knowledge graph.",
            )
            export = tracker.run_step(
                "export_graph",
                lambda: _export_graph(graph, target_dir),
                "Exported graph JSON artifact.",
            )

            # Optionally persist to Neo4j
            neo4j_result = None
            if self._neo4j_config:
                neo4j_result = tracker.run_step(
                    "persist_neo4j",
                    lambda: graph_to_neo4j(graph, **self._neo4j_config),
                    "Persisted graph to Neo4j.",
                )
                if neo4j_result.get("status") != "completed":
                    logger.warning("Neo4j persist failed: %s", neo4j_result.get("message"))

            # Plan-driven retrieval: when the planner selects the graph query
            # operators (kg_query tasks), actually run them against the built
            # graph so the planned operators are executed, not just advertised.
            retrieval: dict[str, Any] = {"status": "skipped"}
            query_term = question or (task_request or "")
            if query_term and "find_graph_entities" in plan_operators:
                entities_found = tracker.run_step(
                    "find_graph_entities",
                    lambda: find_graph_entities(query_term, graph),
                    "Searched the graph for entities matching the query.",
                )
                neighbors = None
                if entities_found.get("matches") and "query_graph_neighbors" in plan_operators:
                    top_entity = entities_found["matches"][0]["name"]
                    neighbors = tracker.run_step(
                        "query_graph_neighbors",
                        lambda: query_graph_neighbors(top_entity, graph, direction="both"),
                        "Retrieved neighbor relationships for the top matched entity.",
                    )
                retrieval = {
                    "status": "completed",
                    "query": query_term,
                    "entities": entities_found,
                    "neighbors": neighbors,
                }

            # Plan-driven QA: only answer a question when the planner selected
            # the QA operator. For full_pipeline, QA is always included.
            wants_qa = bool(question) and "answer_graph_question" in plan_operators
            if wants_qa:
                qa = tracker.run_step(
                    "answer_question",
                    lambda: answer_graph_question(question, graph),
                    "Answered the user question from graph evidence.",
                )
            else:
                qa = {
                    "status": "skipped",
                    "reason": "answer_graph_question not selected by plan",
                }
            quality_report = tracker.run_step(
                "build_quality_report",
                lambda: build_kg_quality_report(
                    extraction=extraction,
                    validation=validation,
                    graph=graph,
                    qa=qa,
                    export=export,
                ),
                "Built a reproducible KG quality report.",
            )
        except Exception as exc:
            tracker.fail()
            return PipelineResult(
                task=self.task_name,
                status="failed",
                message=f"Task 2 pipeline failed for {source.name}: {exc}",
                artifacts={
                    "input": {"path": str(source)},
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "run_state": tracker.to_dict(),
                },
            )

        tracker.complete("completed")
        graph_artifact = {
            **graph["statistics"],
            "output_path": export["output_path"],
        }
        artifacts = {
            "input": {"path": str(source)},
            "extraction": extraction,
            "triples": validation["triples"],
            "validation": validation,
            "graph": graph_artifact,
            "relation_scoring": relation_scoring,
            "qa": qa,
            "retrieval": retrieval,
            "plan": plan_dict,
            "plan_execution": {
                "task_type": plan_dict.get("understanding", {}).get("task_type"),
                "planner_mode": plan_dict.get("planner_mode"),
                "selected_operators": plan_operators,
                "executed_operators": _build_executed_kg_operators(
                    plan_operators, retrieval, wants_qa
                ),
                "qa_executed": wants_qa,
            },
            "quality_report": quality_report,
            "run_state": tracker.to_dict(),
        }
        if neo4j_result:
            artifacts["neo4j"] = neo4j_result

        return PipelineResult(
            task=self.task_name,
            status="completed",
            message=(
                f"Task 2 pipeline built {graph['statistics']['triple_count']} triples "
                f"from {extraction['record_count']} records."
            ),
            artifacts=artifacts,
        )


    def _run_query_only(
        self,
        tracker: AgentRunTracker,
        plan_dict: dict[str, Any],
        plan_operators: list[str],
        graph_path: Path,
        question: str | None,
        task_request: str | None,
    ) -> PipelineResult:
        """Query an existing graph without rebuilding it from text.

        Used for ``kg_query`` plans (and explicit ``graph_file`` reuse): loads
        the cached graph artifact and runs only the planned retrieval/QA
        operators. The expensive ``extract -> relations -> validate -> build ->
        export`` chain is skipped entirely.
        """

        try:
            graph = tracker.run_step(
                "load_graph_artifact",
                lambda: load_graph(graph_path),
                f"Loaded cached medical knowledge graph from {graph_path.name}.",
            )
            extraction = _reconstruct_extraction_from_graph(graph)
            validation = {
                "status": "skipped",
                "triples": graph.get("triples", []),
                "valid_count": len(graph.get("triples", [])),
                "invalid_count": 0,
                "reason": "graph reused; triples not re-validated",
            }
            export = {
                "status": "completed",
                "output_path": str(graph_path),
                "source": "loaded",
            }

            neo4j_result = None
            if self._neo4j_config:
                neo4j_result = tracker.run_step(
                    "persist_neo4j",
                    lambda: graph_to_neo4j(graph, **self._neo4j_config),
                    "Persisted reused graph to Neo4j.",
                )

            retrieval: dict[str, Any] = {"status": "skipped"}
            query_term = question or (task_request or "")
            if query_term and "find_graph_entities" in plan_operators:
                entities_found = tracker.run_step(
                    "find_graph_entities",
                    lambda: find_graph_entities(query_term, graph),
                    "Searched the reused graph for entities matching the query.",
                )
                neighbors = None
                if entities_found.get("matches") and "query_graph_neighbors" in plan_operators:
                    top_entity = entities_found["matches"][0]["name"]
                    neighbors = tracker.run_step(
                        "query_graph_neighbors",
                        lambda: query_graph_neighbors(top_entity, graph, direction="both"),
                        "Retrieved neighbor relationships for the top matched entity.",
                    )
                retrieval = {
                    "status": "completed",
                    "query": query_term,
                    "entities": entities_found,
                    "neighbors": neighbors,
                }

            wants_qa = bool(question) and "answer_graph_question" in plan_operators
            if wants_qa:
                qa = tracker.run_step(
                    "answer_question",
                    lambda: answer_graph_question(question, graph),
                    "Answered the user question from reused graph evidence.",
                )
            else:
                qa = {
                    "status": "skipped",
                    "reason": "answer_graph_question not selected by plan",
                }

            quality_report = tracker.run_step(
                "build_quality_report",
                lambda: build_kg_quality_report(
                    extraction=extraction,
                    validation=validation,
                    graph=graph,
                    qa=qa,
                    export=export,
                ),
                "Built a reproducible KG quality report from the reused graph.",
            )
        except Exception as exc:
            tracker.fail()
            return PipelineResult(
                task=self.task_name,
                status="failed",
                message=f"Task 2 query-only run failed for {graph_path.name}: {exc}",
                artifacts={
                    "input": {"graph_file": str(graph_path)},
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "run_state": tracker.to_dict(),
                },
            )

        tracker.complete("completed")
        graph_artifact = {
            **graph["statistics"],
            "output_path": str(graph_path),
        }
        artifacts = {
            "input": {"graph_file": str(graph_path)},
            "extraction": extraction,
            "triples": graph.get("triples", []),
            "validation": validation,
            "graph": graph_artifact,
            "relation_scoring": {
                "backend": self._relation_backend,
                "mode": "skipped_reused_graph",
            },
            "qa": qa,
            "retrieval": retrieval,
            "plan": plan_dict,
            "plan_execution": {
                "task_type": plan_dict.get("understanding", {}).get("task_type"),
                "planner_mode": plan_dict.get("planner_mode"),
                "selected_operators": plan_operators,
                "executed_operators": _build_executed_kg_operators(
                    plan_operators, retrieval, wants_qa, build_executed=False
                ),
                "qa_executed": wants_qa,
                "graph_reused": True,
                "graph_source": str(graph_path),
            },
            "quality_report": quality_report,
            "run_state": tracker.to_dict(),
        }
        if neo4j_result:
            artifacts["neo4j"] = neo4j_result

        return PipelineResult(
            task=self.task_name,
            status="completed",
            message=(
                f"Task 2 query-only run reused {graph['statistics'].get('triple_count', 0)} "
                f"triples from {graph_path.name} (no rebuild)."
            ),
            artifacts=artifacts,
        )


def _reconstruct_extraction_from_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Derive extraction-style summary fields from a loaded graph artifact.

    The query-only path has no fresh extraction result; reconstruct the few
    fields the quality report and artifacts consume (record_count,
    entity_counts) from the graph's nodes/records.
    """

    entity_counts: dict[str, int] = {}
    for node in graph.get("nodes", []):
        node_type = node.get("type", "Unknown")
        entity_counts[node_type] = entity_counts.get(node_type, 0) + 1
    statistics = graph.get("statistics", {})
    record_count = statistics.get("record_count")
    if record_count is None:
        record_count = len(graph.get("records", []))
    return {
        "record_count": record_count,
        "entity_counts": entity_counts,
        "records": graph.get("records", []),
        "source": "loaded_graph",
    }


def _build_executed_kg_operators(
    plan_operators: list[str],
    retrieval: dict[str, Any],
    wants_qa: bool,
    build_executed: bool = True,
) -> list[str]:
    """Report which planned operators actually executed, preserving plan order."""

    executed: set[str] = set()
    # The graph build operators run only when the graph was (re)built from text.
    if build_executed:
        for op in ("extract_medical_entities", "extract_relations", "validate_triples",
                   "build_medical_graph", "build_kg_quality_report"):
            if op in plan_operators:
                executed.add(op)
    elif "build_kg_quality_report" in plan_operators:
        executed.add("build_kg_quality_report")
    if retrieval.get("status") == "completed":
        executed.add("find_graph_entities")
        if retrieval.get("neighbors") is not None:
            executed.add("query_graph_neighbors")
    if wants_qa:
        executed.add("answer_graph_question")
    return [op for op in plan_operators if op in executed]


def _read_input(source: Path) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if source.suffix.lower() not in {".txt", ".text", ".md"}:
        raise ValueError(f"Unsupported input format: {source.suffix}")
    return source.read_text(encoding="utf-8")


def _export_graph(graph: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "medical_kg.json"
    target.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"status": "completed", "output_path": str(target)}


def _merge_triples(
    llm_triples: list[dict[str, Any]],
    rule_triples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge LLM and rule-based triples, deduplicating by (subject, predicate, object)."""
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict[str, Any]] = []
    for triple in llm_triples + rule_triples:
        key = (triple["subject"], triple["predicate"], triple["object"])
        if key not in seen:
            seen.add(key)
            merged.append(triple)
    return merged


def _merge_local_model_entities(
    extraction: dict[str, Any],
    local_model_entities: dict[str, list[str]],
) -> dict[str, Any]:
    """Merge local model NER predictions into the extraction result.

    Adds new entities discovered by the local model that weren't found
    by the rule-based or LLM extractor.  Existing entities are preserved.
    """
    for record in extraction.get("records", []):
        existing = record.get("entities", {})
        for etype, names in local_model_entities.items():
            if etype not in existing:
                existing[etype] = []
            for name in names:
                if name not in existing[etype]:
                    existing[etype].append(name)

    # Update aggregate counts
    entity_counts = extraction.get("entity_counts", {})
    for etype in local_model_entities:
        all_names = set()
        for record in extraction.get("records", []):
            all_names.update(record.get("entities", {}).get(etype, []))
        entity_counts[etype] = len(all_names)
    extraction["entity_counts"] = entity_counts
    return extraction
