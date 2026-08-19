"""CLI demo entrypoint for task 1.

Supports both CSV and Text input types with appropriate output formatting.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.llm_config import load_llm_config
from src.pipelines.task1_data_pipeline import run_task1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run task 1 data processing agent.")
    parser.add_argument(
        "--task",
        default=None,
        help="Free-form data processing request.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input file path (CSV or TXT). Defaults to data/samples/task1_patients.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated output files.",
    )
    parser.add_argument(
        "--datamate-url",
        default="http://localhost:18000",
        help="DataMate Python backend base URL. Use 'none' to disable.",
    )
    parser.add_argument(
        "--datamate-mode",
        choices=["dry_run", "submit"],
        default="dry_run",
        help="dry_run only prepares payloads; submit posts to DataMate.",
    )
    parser.add_argument("--src-dataset-id", default=None)
    parser.add_argument("--src-dataset-name", default=None)
    parser.add_argument("--dest-dataset-name", default=None)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-assisted planning.",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Path to LLM config file (.env or .json) with API credentials.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="(Deprecated) Use --llm-config instead.",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Path to a locally fine-tuned model for planning.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start REST API server instead of running a one-shot pipeline.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API host used with --serve. Use 0.0.0.0 only for trusted networks.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server (used with --serve).",
    )
    parser.add_argument(
        "--allow-api-datamate-write",
        action="store_true",
        help=(
            "Allow API requests to use datamate_mode=submit. "
            "Disabled by default."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.serve:
        from src.pipelines.task1_api_server import serve
        print(f"Starting Task 1 API server on {args.host}:{args.port}...")
        serve(
            host=args.host,
            port=args.port,
            datamate_url=args.datamate_url,
            allow_datamate_write=args.allow_api_datamate_write,
        )
        return 0

    datamate_url = None if args.datamate_url.lower() == "none" else args.datamate_url

    llm_config = None
    if args.llm:
        # Prefer --llm-config, fall back to --env-file for backward compatibility
        config_path = args.llm_config or args.env_file or str(ROOT / ".local" / "llm_config.env")
        llm_config = load_llm_config(config_path)
        if not llm_config:
            print("LLM mode requested but config file does not contain valid credentials.")
            print(f"  config: {config_path}")
            print("  Required: OPENAI_API_KEY + OPENAI_BASE_URL (.env) or api_key + base_url (.json)")
            return 2

    result = run_task1_pipeline(
        task_request=args.task,
        input_path=args.input,
        output_dir=args.output_dir,
        datamate_base_url=datamate_url,
        datamate_src_dataset_id=args.src_dataset_id,
        datamate_src_dataset_name=args.src_dataset_name,
        datamate_dest_dataset_name=args.dest_dataset_name,
        datamate_mode=args.datamate_mode,
        llm_config=llm_config,
        local_model_path=args.local_model,
    )
    print(f"{result.task}: {result.status} - {result.message}")
    print(f"input: {result.artifacts['input']['path']} ({result.artifacts['input']['format']})")

    understanding = result.artifacts.get("understanding", {})
    if understanding:
        print(
            "understanding: "
            f"{understanding.get('task_type')} / "
            f"{understanding.get('data_type')} / "
            f"{', '.join(understanding.get('intent_keywords', []))}"
        )
    plan = result.artifacts.get("plan", {})
    if plan:
        print(f"planner_mode: {plan.get('planner_mode', 'rule')}")

    if result.status == "failed":
        error = result.artifacts.get("error", {})
        print(f"error: {error.get('type')} - {error.get('message')}")
        return 1

    # Branch based on input format
    input_format = result.artifacts["input"]["format"]

    if input_format == "csv":
        _print_csv_results(result)
    elif input_format == "text":
        _print_text_results(result)
    else:
        print(f"output: processed {input_format} file")

    run_state = result.artifacts.get("run_state", {})
    if run_state:
        steps = ", ".join(
            f"{step['name']}:{step['status']}"
            for step in run_state.get("steps", [])
        )
        print(f"run_state: {run_state.get('status')} [{steps}]")
    return 0


def _print_csv_results(result):
    """Print CSV cleaning pipeline results."""
    profile = result.artifacts.get("profile", {})
    if profile:
        print(
            "profile: "
            f"{profile.get('row_count', 0)} rows, "
            f"{profile.get('column_count', 0)} columns, "
            f"{profile.get('duplicate_rows', 0)} duplicate rows"
        )

    plan = result.artifacts.get("plan", {})
    if plan.get("operators"):
        print("operators: " + " -> ".join(plan["operators"]))

    cleaning = result.artifacts.get("cleaning", {})
    if cleaning:
        print(
            "cleaning: "
            f"{cleaning.get('output_rows', 'N/A')} rows exported to "
            f"{cleaning.get('output_path', 'N/A')}"
        )

    validation = result.artifacts.get("validation", {})
    if validation:
        checks = validation.get("checks", {})
        print(
            "validation: "
            f"{validation.get('status')} / "
            f"duplicates_ok={checks.get('duplicate_rows_removed', 'N/A')} / "
            f"missing_ok={checks.get('missing_values_filled', 'N/A')}"
        )

    quality_report = result.artifacts.get("quality_report", {})
    if quality_report:
        metrics = quality_report.get("metrics", {})
        readiness = quality_report.get("readiness", {})
        ready_items = [name for name, ready in readiness.items() if ready]
        print(
            "quality_report: "
            f"{quality_report.get('status')} / "
            f"operators={metrics.get('planned_operator_count', 'N/A')} / "
            f"datamate_ops={metrics.get('datamate_operator_count', 'N/A')}"
        )
        print("  readiness: " + ", ".join(ready_items))

    datamate = result.artifacts.get("datamate", {})
    if datamate:
        print(f"datamate: {datamate.get('status', 'unknown')}")
        operator_catalog = datamate.get("operators", {})
        if operator_catalog.get("status") == "available":
            print(f"datamate operators: {operator_catalog.get('operator_count', 0)} listed")
            for local_operator, mapping in operator_catalog.get("candidate_mappings", {}).items():
                selected = ", ".join(mapping.get("selected_operator_ids", [])) or "local-only"
                print(f"  {local_operator}: {selected} ({mapping.get('support_level', 'unknown')})")
                if mapping.get("note"):
                    print(f"    note: {mapping['note']}")


def _print_text_results(result):
    """Print text processing pipeline results."""
    processing = result.artifacts.get("processing", {})
    if processing:
        print(
            "processing: "
            f"{processing.get('input_records', 0)} records -> "
            f"{processing.get('output_records', 0)} cleaned, "
            f"{processing.get('html_tags_removed', 0)} HTML tags removed, "
            f"{processing.get('pii_redacted', 0)} PII redacted"
        )
        print(f"  output: {processing.get('output_path', 'N/A')}")

    plan = result.artifacts.get("plan", {})
    if plan.get("operators"):
        print("operators: " + " -> ".join(plan["operators"]))

    entities = result.artifacts.get("entities", {})
    if entities:
        print(
            "entities: "
            f"{entities.get('records_processed', 0)} records processed, "
            f"fields: {', '.join(entities.get('fields_extracted', []))}"
        )
        print(f"  output: {entities.get('output_path', 'N/A')}")

    quality_report = result.artifacts.get("quality_report", {})
    if quality_report:
        print(f"quality_report: {quality_report.get('status', 'unknown')}")


if __name__ == "__main__":
    raise SystemExit(main())
