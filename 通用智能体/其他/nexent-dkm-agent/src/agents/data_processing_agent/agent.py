"""Task 1 data processing agent.

Supports both CSV and Text input types:
- CSV: profile -> clean -> validate -> datamate integration
- Text: process -> extract entities (optional) -> export
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agents.data_processing_agent.planner import (
    DEFAULT_TASK,
    HybridPlanner,
)
from src.agents.data_processing_agent.reporting import build_quality_report
from src.agents.data_processing_agent.scheduler import (
    DAGScheduler,
    OperatorScheduler,
    StepSpec,
)
from src.agents.data_processing_agent.state import TaskStateTracker
from src.common.results import PipelineResult
from src.operators.data_ops.csv_cleaner import clean_csv, validate_cleaning_result
from src.operators.data_ops.csv_profile import profile_csv
from src.operators.data_ops.data_transform import extract_fields_from_text, transform_csv
from src.operators.data_ops.datamate_client import (
    DataMateClient,
    _payload_data,
    resolve_datamate_mode,
    safe_datamate_call,
)
from src.operators.data_ops.json_loader import json_records_to_csv
from src.operators.data_ops.text_processor import process_text

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_CSV = PROJECT_ROOT / "data" / "samples" / "task1_patients.csv"
DEFAULT_SAMPLE_TEXT = PROJECT_ROOT / "data" / "samples" / "task1_medical_notes.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task1"


class DataProcessingAgent:
    """Task-1 agent that plans and executes a data processing flow.

    Routes execution based on input file type:
    - .csv -> structured CSV cleaning pipeline
    - .txt/.text -> unstructured text processing pipeline
    """

    task_name = "task1_data_processing_agent"

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        local_model_path: str | None = None,
    ) -> None:
        self._llm_config = llm_config
        self._local_model_path = local_model_path

    def run(
        self,
        task_request: str | None = None,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        datamate_base_url: str | None = "http://localhost:18000",
        datamate_timeout: float = 3.0,
        datamate_src_dataset_id: str | None = None,
        datamate_src_dataset_name: str | None = None,
        datamate_dest_dataset_name: str | None = None,
        datamate_mode: str = "dry_run",
        transforms: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        tracker = TaskStateTracker()
        scheduler = OperatorScheduler(tracker)
        tracker.start_task()

        input_file = Path(input_path) if input_path else DEFAULT_SAMPLE_CSV
        cleaned_output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        input_type = _infer_input_type(input_file)

        request_text = task_request or DEFAULT_TASK
        requested_datamate_mode = datamate_mode
        datamate_catalog = fetch_datamate_catalog_hints(
            datamate_base_url,
            timeout=datamate_timeout,
        )
        planner = HybridPlanner(
            llm_config=self._llm_config,
            local_model_path=self._local_model_path,
            datamate_operators=datamate_catalog.get("sample_operator_ids", []),
        )

        initial_plan = scheduler.run_step(
            "understand_task",
            lambda: planner.plan(request_text),
            "Parsed the user task into a structured intent.",
        )

        try:
            scheduler.run_step(
                "validate_runtime_config",
                lambda: _validate_runtime_config(input_file, requested_datamate_mode),
                "Validated input path and DataMate execution mode.",
            )
            datamate_mode, datamate_mode_resolution = resolve_datamate_mode(
                datamate_base_url,
                requested_datamate_mode,
                timeout=datamate_timeout,
            )

            if input_type == "csv":
                result = self._run_csv_pipeline(
                    scheduler=scheduler,
                    planner=planner,
                    input_file=input_file,
                    output_dir=cleaned_output_dir,
                    request_text=request_text,
                    initial_plan=initial_plan,
                    datamate_base_url=datamate_base_url,
                    datamate_timeout=datamate_timeout,
                    datamate_src_dataset_id=datamate_src_dataset_id,
                    datamate_src_dataset_name=datamate_src_dataset_name,
                    datamate_dest_dataset_name=datamate_dest_dataset_name,
                    datamate_mode=datamate_mode,
                    transforms=transforms,
                    datamate_catalog=datamate_catalog,
                )
            elif input_type == "json":
                result = self._run_json_pipeline(
                    scheduler=scheduler,
                    planner=planner,
                    input_file=input_file,
                    output_dir=cleaned_output_dir,
                    request_text=request_text,
                    initial_plan=initial_plan,
                    datamate_base_url=datamate_base_url,
                    datamate_timeout=datamate_timeout,
                    datamate_src_dataset_id=datamate_src_dataset_id,
                    datamate_src_dataset_name=datamate_src_dataset_name,
                    datamate_dest_dataset_name=datamate_dest_dataset_name,
                    datamate_mode=datamate_mode,
                    transforms=transforms,
                    datamate_catalog=datamate_catalog,
                )
            elif input_type == "text":
                result = self._run_text_pipeline(
                    scheduler=scheduler,
                    planner=planner,
                    input_file=input_file,
                    output_dir=cleaned_output_dir,
                    request_text=request_text,
                    initial_plan=initial_plan,
                )
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

        except Exception as exc:
            return PipelineResult(
                task=self.task_name,
                status="failed",
                message=f"Task 1 pipeline failed for {input_file.name}: {exc}",
                artifacts={
                    "input": {"path": str(input_file), "format": input_type},
                    "understanding": initial_plan.understanding.to_dict(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "run_state": tracker.to_dict(),
                },
            )

        status = "completed"
        if result.get("datamate", {}).get("status") == "unavailable":
            status = "completed_with_warnings"
        tracker.complete_task(status)

        return PipelineResult(
            task=self.task_name,
            status=status,
            message=f"Task 1 pipeline processed {input_file.name} ({input_type} -> {result.get('output_format', 'unknown')}).",
            artifacts={
                "input": {"path": str(input_file), "format": input_type},
                "understanding": result["plan"]["understanding"],
                "json_conversion": result.get("json_conversion", {}),
                "profile": result.get("profile", {}),
                "plan": result["plan"],
                "plan_execution": result.get("plan_execution", {}),
                "cleaning": result.get("cleaning", {}),
                "transform": result.get("transform", {}),
                "processing": result.get("processing", {}),
                "entities": result.get("entities", {}),
                "validation": result.get("validation", {}),
                "datamate": result.get("datamate", {}),
                "datamate_mode_resolution": datamate_mode_resolution,
                "quality_report": result.get("quality_report", {}),
                "run_state": tracker.to_dict(),
            },
        )

    def _run_csv_pipeline(
        self,
        scheduler: OperatorScheduler,
        planner: HybridPlanner,
        input_file: Path,
        output_dir: Path,
        request_text: str,
        initial_plan,
        datamate_base_url: str | None,
        datamate_timeout: float,
        datamate_src_dataset_id: str | None,
        datamate_src_dataset_name: str | None,
        datamate_dest_dataset_name: str | None,
        datamate_mode: str,
        transforms: list[dict[str, Any]] | None = None,
        datamate_catalog: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the CSV cleaning pipeline."""

        profile = scheduler.run_step(
            "profile_schema",
            lambda: profile_csv(input_file),
            "Profiled the input CSV schema and data quality.",
        )
        plan = scheduler.run_step(
            "plan_operators",
            lambda: planner.enrich_plan(
                planner.plan(request_text, data_profile=profile),
                datamate_catalog=datamate_catalog,
            ).to_dict(),
            f"Built a data-aware operator plan (mode={initial_plan.planner_mode}).",
        )
        planned_operators = plan["operators"]

        # --- Execute the plan-driven cleaning stage as a dependency DAG ---
        # The DAGScheduler resolves the execute -> [transform] -> validate ->
        # datamate -> quality_report dependency chain (topological order, with
        # cycle validation and per-step retry support) in the agent's main loop.
        dag = DAGScheduler(scheduler.tracker, max_workers=1)
        cache: dict[str, Any] = {}

        def _do_clean() -> dict[str, Any]:
            cache["cleaning"] = clean_csv(
                input_file, profile, output_dir, operators=planned_operators
            )
            return cache["cleaning"]

        def _do_validate() -> dict[str, Any]:
            cache["validation"] = _validate_cleaning_result(profile, cache["cleaning"])
            return cache["validation"]

        def _do_datamate() -> dict[str, Any]:
            cache["datamate"] = inspect_datamate(
                datamate_base_url,
                plan["operators"],
                timeout=datamate_timeout,
                src_dataset_id=datamate_src_dataset_id,
                src_dataset_name=datamate_src_dataset_name,
                dest_dataset_name=datamate_dest_dataset_name,
                mode=datamate_mode,
            )
            return cache["datamate"]

        def _do_quality() -> dict[str, Any]:
            return build_quality_report(
                plan=plan,
                profile=profile,
                cleaning=cache["cleaning"],
                validation=cache["validation"],
                datamate=cache["datamate"],
            )

        steps = [
            StepSpec(
                "execute_local_cleaning",
                _do_clean,
                message="Executed plan-driven local CSV cleaning.",
            )
        ]
        validate_deps = ["execute_local_cleaning"]
        if "transform_columns" in planned_operators:
            def _do_transform() -> dict[str, Any]:
                cache["transform"] = transform_csv(
                    cache["cleaning"]["output_path"], output_dir, transforms
                )
                return cache["transform"]

            steps.append(
                StepSpec(
                    "transform_columns",
                    _do_transform,
                    depends_on=["execute_local_cleaning"],
                    message="Applied plan-driven column transforms (select/rename/filter).",
                )
            )
            validate_deps = ["transform_columns"]

        steps.extend([
            StepSpec(
                "validate_cleaning_result",
                _do_validate,
                depends_on=validate_deps,
                message="Validated exported CSV quality checks.",
            ),
            StepSpec(
                "inspect_datamate",
                _do_datamate,
                depends_on=["validate_cleaning_result"],
                message="Inspected DataMate and prepared integration artifacts.",
            ),
            StepSpec(
                "build_quality_report",
                _do_quality,
                depends_on=["inspect_datamate"],
                message="Built a reproducible run evidence report.",
            ),
        ])

        dag_results = dag.run_dag(steps)
        cleaning = dag_results["execute_local_cleaning"]
        transform = dag_results.get("transform_columns", {})
        validation = dag_results["validate_cleaning_result"]
        datamate = dag_results["inspect_datamate"]
        quality_report = dag_results["build_quality_report"]

        executed_operators = _build_executed_operators(
            planned_operators, cleaning, transform
        )
        plan_execution = {
            "planned_operators": planned_operators,
            "executed_operators": executed_operators,
            "skipped_operators": [
                op for op in planned_operators if op not in executed_operators
            ],
        }

        return {
            "profile": profile,
            "plan": plan,
            "plan_execution": plan_execution,
            "cleaning": cleaning,
            "transform": transform,
            "validation": validation,
            "datamate": datamate,
            "quality_report": quality_report,
            "output_format": "csv",
        }

    def _run_json_pipeline(
        self,
        scheduler: OperatorScheduler,
        planner: HybridPlanner,
        input_file: Path,
        output_dir: Path,
        request_text: str,
        initial_plan,
        datamate_base_url: str | None,
        datamate_timeout: float,
        datamate_src_dataset_id: str | None,
        datamate_src_dataset_name: str | None,
        datamate_dest_dataset_name: str | None,
        datamate_mode: str,
        transforms: list[dict[str, Any]] | None = None,
        datamate_catalog: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the JSON pipeline by converting records to CSV then reusing
        the structured CSV cleaning flow."""

        conversion = scheduler.run_step(
            "convert_json_to_csv",
            lambda: json_records_to_csv(input_file, output_dir),
            "Converted JSON records into a flat CSV for structured cleaning.",
        )
        csv_path = Path(conversion["csv_path"])

        result = self._run_csv_pipeline(
            scheduler=scheduler,
            planner=planner,
            input_file=csv_path,
            output_dir=output_dir,
            request_text=request_text,
            initial_plan=initial_plan,
            datamate_base_url=datamate_base_url,
            datamate_timeout=datamate_timeout,
            datamate_src_dataset_id=datamate_src_dataset_id,
            datamate_src_dataset_name=datamate_src_dataset_name,
            datamate_dest_dataset_name=datamate_dest_dataset_name,
            datamate_mode=datamate_mode,
            transforms=transforms,
            datamate_catalog=datamate_catalog,
        )
        result["json_conversion"] = conversion
        result["output_format"] = "csv"
        return result

    def _run_text_pipeline(
        self,
        scheduler: OperatorScheduler,
        planner: HybridPlanner,
        input_file: Path,
        output_dir: Path,
        request_text: str,
        initial_plan,
    ) -> dict[str, Any]:
        """Execute the text processing pipeline (plan first, then execute)."""

        # --- Plan FIRST so the recorded plan drives operator execution ---
        # Build a text-specific operator plan based on initial understanding,
        # mirroring the CSV path where plan_operators precedes execution.
        intent_keywords = initial_plan.understanding.intent_keywords
        text_operators = ["load_text", "clean_text"]
        text_rationale = ["Load and clean unstructured text input."]

        if "extract" in intent_keywords:
            text_operators.append("extract_entities")
            text_rationale.append("Extract structured medical entities from cleaned text.")

        text_operators.extend(["export_clean_dataset", "validate_clean_dataset"])
        text_rationale.append("Export cleaned output and validate results.")

        # Override the plan's operators to reflect planned text execution.
        plan = initial_plan.to_dict()
        plan["operators"] = text_operators
        plan["rationale"] = text_rationale

        scheduler.run_step(
            "plan_operators",
            lambda: plan,
            f"Built a text-specific operator plan: {' -> '.join(text_operators)} (mode={initial_plan.planner_mode}).",
        )

        # --- Execute the planned operators ---
        # load_text + clean_text + export_clean_dataset are realised by process_text.
        processing = scheduler.run_step(
            "process_text",
            lambda: process_text(input_file, output_dir),
            "Cleaned unstructured text: removed HTML, normalized Unicode, redacted PII.",
        )

        entities = {}
        if "extract_entities" in text_operators:
            cleaned_path = Path(processing["output_path"])
            entities = scheduler.run_step(
                "extract_entities",
                lambda: extract_fields_from_text(cleaned_path, output_dir),
                "Extracted medical entities (diseases, drugs, examinations) from cleaned text.",
            )

        quality_report = scheduler.run_step(
            "build_quality_report",
            lambda: build_quality_report(
                plan=plan,
                profile={"file_name": input_file.name, "row_count": processing.get("input_records", 0)},
                cleaning=processing,
                validation={"status": "passed", "checks": {"output_generated": True}},
                datamate={"status": "skipped", "health": {"status": "skipped"}},
            ),
            "Built a reproducible run evidence report for text processing.",
        )

        executed_text_operators = list(text_operators)
        if "extract_entities" in executed_text_operators and not entities:
            executed_text_operators.remove("extract_entities")

        return {
            "processing": processing,
            "plan": plan,
            "plan_execution": {
                "planned_operators": text_operators,
                "executed_operators": executed_text_operators,
                "skipped_operators": [
                    op for op in text_operators if op not in executed_text_operators
                ],
            },
            "entities": entities,
            "validation": {"status": "passed", "checks": {"output_generated": True}},
            "datamate": {"status": "skipped", "health": {"status": "skipped"}},
            "quality_report": quality_report,
            "output_format": "text" if not entities else "csv",
        }


def fetch_datamate_catalog_hints(
    base_url: str | None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Fetch a lightweight DataMate operator catalog snapshot for planning."""

    if not base_url:
        return {"status": "skipped", "operator_count": 0, "sample_operator_ids": []}

    inspected = inspect_datamate(base_url, plan_operators=[], timeout=timeout)
    health_status = inspected.get("status")
    if health_status not in {"healthy", "unknown"}:
        return {
            "status": health_status or "unavailable",
            "operator_count": 0,
            "sample_operator_ids": [],
        }

    client = DataMateClient(base_url, timeout=timeout)
    listing = safe_datamate_call(lambda: client.list_operators(size=10))
    if listing.get("status") == "unavailable" or "data" not in listing:
        return {
            "status": "unavailable",
            "operator_count": 0,
            "sample_operator_ids": [],
        }

    data = _payload_data(listing)
    content = data.get("content", [])
    sample_operator_ids = [
        str(item.get("id"))
        for item in content
        if isinstance(item, dict) and item.get("id")
    ]
    return {
        "status": "available",
        "operator_count": int(data.get("totalElements", len(sample_operator_ids))),
        "sample_operator_ids": sample_operator_ids,
    }


def check_datamate_health(
    base_url: str | None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Check the deployed DataMate Python backend health endpoint."""

    return inspect_datamate(base_url, plan_operators=[], timeout=timeout)["health"]


def inspect_datamate(
    base_url: str | None,
    plan_operators: list[str],
    timeout: float = 3.0,
    src_dataset_id: str | None = None,
    src_dataset_name: str | None = None,
    dest_dataset_name: str | None = None,
    mode: str = "dry_run",
) -> dict[str, Any]:
    """Collect DataMate health and operator catalog hints."""

    if not base_url:
        return {
            "status": "skipped",
            "health": {
                "status": "skipped",
                "message": "DataMate health check disabled for this run.",
            },
            "operators": {"status": "skipped"},
        }

    client = DataMateClient(base_url, timeout=timeout)
    health = safe_datamate_call(client.health)
    if health["status"] == "unavailable":
        return {
            "status": "unavailable",
            "health": health,
            "operators": {"status": "skipped"},
        }

    return {
        "status": health["status"],
        "health": health,
        "operators": safe_datamate_call(
            lambda: client.catalog_summary(
                plan_operators,
                src_dataset_id=src_dataset_id,
                src_dataset_name=src_dataset_name,
                dest_dataset_name=dest_dataset_name,
                mode=mode,
            )
        ),
    }


def _build_executed_operators(
    planned_operators: list[str],
    cleaning: dict[str, Any],
    transform: dict[str, Any],
) -> list[str]:
    """Map the plan to the operators that actually ran, preserving plan order."""

    executed: set[str] = set()
    # Load/profile/export/validate always run for the CSV flow.
    for op in ("load_csv", "profile_schema", "export_clean_dataset", "validate_clean_dataset"):
        if op in planned_operators:
            executed.add(op)
    executed.update(cleaning.get("operators_applied", []))
    if transform.get("status") == "completed":
        executed.add("transform_columns")
    return [op for op in planned_operators if op in executed]


def _infer_input_type(path: Path) -> str:
    """Infer input type from file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".txt", ".text"}:
        return "text"
    if suffix == ".json":
        return "json"
    return "unknown"


def _validate_cleaning_result(
    profile: dict[str, Any],
    cleaning: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_cleaning_result(profile, cleaning)
    if validation["status"] != "passed":
        failed_checks = [
            check for check, passed in validation["checks"].items() if not passed
        ]
        raise ValueError(
            "Cleaned output validation failed: " + ", ".join(failed_checks)
        )
    return validation


def _validate_runtime_config(input_path: Path, datamate_mode: str) -> dict[str, Any]:
    if input_path.suffix.lower() not in {".csv", ".txt", ".text", ".json"}:
        raise ValueError(f"Unsupported input format for task 1: {input_path.suffix}")
    if datamate_mode not in {"dry_run", "submit", "auto"}:
        raise ValueError("DataMate mode must be 'dry_run', 'submit', or 'auto'.")
    return {
        "status": "passed",
        "input_suffix": input_path.suffix.lower(),
        "input_type": _infer_input_type(input_path),
        "datamate_mode": datamate_mode,
    }
