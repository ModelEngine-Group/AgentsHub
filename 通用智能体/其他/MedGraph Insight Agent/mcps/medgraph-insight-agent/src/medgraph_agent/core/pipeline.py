from __future__ import annotations

from pathlib import Path
from typing import Any

from medgraph_agent.core.models import GraphSnapshot, PipelineRun, stable_id, utc_now
from medgraph_agent.core.planner import Planner
from medgraph_agent.core.storage import GraphStore, write_json
from medgraph_agent.operators.base import Operator
from medgraph_agent.operators.medical_extraction import (
    EntityRecognitionOperator,
    RelationExtractionOperator,
    TripleValidationOperator,
)
from medgraph_agent.operators.processing import DataIngestionOperator, TextCleaningOperator


class PipelineRunner:
    def __init__(self, output_dir: str | Path = "outputs/latest") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.planner = Planner()
        self.operators: dict[str, Operator] = {
            "data_ingestion": DataIngestionOperator(),
            "text_cleaning": TextCleaningOperator(),
            "entity_recognition": EntityRecognitionOperator(),
            "relation_extraction": RelationExtractionOperator(),
            "triple_validation": TripleValidationOperator(),
        }

    @property
    def db_path(self) -> Path:
        return self.output_dir / "medgraph.db"

    def run(self, task: str, source: str | Path) -> PipelineRun:
        run = PipelineRun(
            id=stable_id("run", task, str(source), utc_now()),
            task=task,
            source=str(source),
            status="running",
            started_at=utc_now(),
        )
        context: dict[str, Any] = {"task": task, "source": str(source), "output_dir": str(self.output_dir)}
        run.plan = self.planner.plan(task)
        try:
            for step in run.plan.steps:
                result = self.operators[step.operator].execute(context)
                run.operator_results.append(result)
                if result.status != "succeeded":
                    raise RuntimeError(result.error or f"operator failed: {step.operator}")
            graph: GraphSnapshot = context["graph"]
            store = GraphStore(self.db_path)
            store.save_graph(graph)
            run.status = "succeeded"
            run.finished_at = utc_now()
            store.save_run(run)
            self._write_artifacts(run, context)
            return run
        except Exception as exc:
            run.status = "failed"
            run.finished_at = utc_now()
            run.error = f"{type(exc).__name__}: {exc}"
            GraphStore(self.db_path).save_run(run)
            self._write_artifacts(run, context)
            return run

    def _write_artifacts(self, run: PipelineRun, context: dict[str, Any]) -> None:
        write_json(self.output_dir / "run.json", run)
        if "records" in context:
            write_json(self.output_dir / "records.json", [record.__dict__ for record in context["records"]])
        if "graph" in context:
            graph = context["graph"]
            write_json(self.output_dir / "graph.json", graph)
            write_json(self.output_dir / "graph_stats.json", graph.stats())
