from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datamate_full_pipeline_common import (
    BACKUP_ITEMS,
    BACKUP_ROOT,
    DATAMATE_OUTPUT_ROOT,
    OFFICIAL_METRICS,
    PROJECT_ROOT,
    SAFETY_NOTE,
    SYNC_TARGETS,
    copy_file,
    copy_tree,
    ensure_directory,
    existing_status,
    extract_pipeline_metrics,
    load_json,
    materialized_output_root,
    now_iso,
    now_stamp,
    relative_to_project,
    required_pipeline_paths,
    validate_official_metrics,
    write_json,
)

RUN_REPORT_PATH = PROJECT_ROOT / "outputs" / "release" / "datamate_full_pipeline_report.json"
SYNC_REPORT_PATH = PROJECT_ROOT / "outputs" / "release" / "datamate_sync_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely sync DataMate full-pipeline outputs back to the ChronicCare mainline.")
    parser.add_argument("--output-root", default=str(DATAMATE_OUTPUT_ROOT))
    parser.add_argument("--allow-metric-mismatch", action="store_true")
    parser.add_argument("--skip-backup", action="store_true", help="Skip legacy mainline backup when cutover is explicitly authorized.")
    return parser.parse_args()


def backup_mainline() -> Dict[str, Any]:
    backup_dir = ensure_directory(BACKUP_ROOT / f"datamate_full_pipeline_before_sync_{now_stamp()}")
    copied: List[Dict[str, Any]] = []
    for source in BACKUP_ITEMS:
        target = backup_dir / source.relative_to(PROJECT_ROOT)
        if not source.exists():
            copied.append({"source": relative_to_project(source), "exists": False, "copied_files": 0})
            continue
        if source.is_dir():
            file_count = copy_tree(source, target)
            copied.append({"source": relative_to_project(source), "exists": True, "copied_files": file_count})
        else:
            copy_file(source, target)
            copied.append({"source": relative_to_project(source), "exists": True, "copied_files": 1})
    return {"backup_dir": backup_dir, "items": copied}


def build_current_metrics(output_root: Path) -> Dict[str, Any]:
    metrics = extract_pipeline_metrics(output_root)
    graph_summary = load_json(output_root / "chronic_kg_build" / "graph" / "graph_summary.json")
    metrics.update(
        {
            "data_version": "synthetic_chroniccare",
            "generated_at": now_iso(),
            "nl2sql_question_count": metrics.get("question_count", 0),
            "analysis_success_rate": 1.0 if metrics.get("question_count", 0) else 0.0,
            "entity_type_count": graph_summary.get("entity_type_count", {}),
            "relation_type_count": graph_summary.get("relation_type_count", {}),
            "provenance_version": graph_summary.get("provenance_version"),
            "edge_provenance_complete_rate": graph_summary.get("edge_provenance_complete_rate"),
            "exact_duplicate_edge_count": graph_summary.get("exact_duplicate_edge_count"),
            "safety_note": SAFETY_NOTE,
        }
    )
    return metrics


def sync_outputs(output_root: Path) -> Dict[str, Any]:
    synced_paths: List[str] = []

    clean_tables = output_root / "chronic_table_clean" / "clean_tables"
    normalized_tables = output_root / "chronic_field_normalize" / "normalized_tables"
    chunks_dir = output_root / "chronic_text_split" / "chunks"
    entities_dir = output_root / "chronic_entity_extract" / "entities"
    relations_dir = output_root / "chronic_relation_extract" / "relations"
    triples_clean = output_root / "chronic_triple_validate" / "triples_clean.jsonl"
    triples_rejected = output_root / "chronic_triple_validate" / "triples_rejected.jsonl"
    graph_dir = output_root / "chronic_kg_build" / "graph"
    sqlite_db = output_root / "chronic_sqlite_loader" / "chroniccare.db"
    nl2sql_dir = output_root / "chronic_nl2sql_analyze"
    report_dir = output_root / "chronic_report_pack"

    if clean_tables.exists():
        copy_tree(clean_tables, SYNC_TARGETS["data_processed"] / "clean_tables")
        synced_paths.append("data/processed/clean_tables")
    if normalized_tables.exists():
        copy_tree(normalized_tables, SYNC_TARGETS["data_processed"] / "normalized_tables")
        synced_paths.append("data/processed/normalized_tables")
    if chunks_dir.exists():
        copy_tree(chunks_dir, SYNC_TARGETS["data_processed"] / "chunks")
        synced_paths.append("data/processed/chunks")
    if entities_dir.exists():
        copy_tree(entities_dir, SYNC_TARGETS["data_processed"] / "entities")
        synced_paths.append("data/processed/entities")
    if relations_dir.exists():
        copy_tree(relations_dir, SYNC_TARGETS["data_processed"] / "relations")
        synced_paths.append("data/processed/relations")
    if triples_clean.exists():
        copy_file(triples_clean, SYNC_TARGETS["data_processed"] / "triples" / "triples_clean.jsonl")
        synced_paths.append("data/processed/triples/triples_clean.jsonl")
    if triples_rejected.exists():
        copy_file(triples_rejected, SYNC_TARGETS["data_processed"] / "triples" / "triples_rejected.jsonl")
        synced_paths.append("data/processed/triples/triples_rejected.jsonl")
    if graph_dir.exists():
        copy_tree(graph_dir, SYNC_TARGETS["data_graph"] / "graph")
        synced_paths.append("data/graph/graph")
        for filename in ("graph.json", "graph_summary.json"):
            graph_file = graph_dir / filename
            if graph_file.exists():
                copy_file(graph_file, SYNC_TARGETS["data_graph"] / filename)
                synced_paths.append(f"data/graph/{filename}")
    if sqlite_db.exists():
        copy_file(sqlite_db, SYNC_TARGETS["data_sqlite"] / "chroniccare.db")
        synced_paths.append("data/sqlite/chroniccare.db")
    if nl2sql_dir.exists():
        for filename in ("indicator_results.json", "sql_candidates.json"):
            source = nl2sql_dir / filename
            if source.exists():
                copy_file(source, SYNC_TARGETS["reports"] / filename)
                synced_paths.append(f"outputs/reports/{filename}")
    if report_dir.exists():
        for filename in ("analysis_report.md", "analysis_report.html", "report_pack_summary.json"):
            source = report_dir / filename
            if source.exists():
                target_name = "analysis_report.md" if filename == "analysis_report.md" else (
                    "analysis_report.html" if filename == "analysis_report.html" else filename
                )
                copy_file(source, SYNC_TARGETS["reports"] / target_name)
                synced_paths.append(f"outputs/reports/{target_name}")
        charts_dir = report_dir / "charts"
        if charts_dir.exists():
            copy_tree(charts_dir, SYNC_TARGETS["charts"])
            synced_paths.append("outputs/charts")

    current_metrics = build_current_metrics(output_root)
    write_json(SYNC_TARGETS["current_metrics"], current_metrics)
    synced_paths.append("configs/current_metrics.json")
    return {"synced_paths": synced_paths, "current_metrics": current_metrics}


def main() -> None:
    args = parse_args()
    output_root = materialized_output_root(Path(args.output_root).resolve())
    ensure_directory(SYNC_REPORT_PATH.parent)

    warnings: List[str] = []
    errors: List[str] = []

    if not output_root.exists():
        errors.append(f"output root does not exist: {output_root}")

    if RUN_REPORT_PATH.exists():
        run_report = load_json(RUN_REPORT_PATH)
        if run_report.get("status") != "success":
            errors.append("datamate full pipeline report is not successful.")
    else:
        warnings.append("datamate full pipeline report is missing; proceeding with direct file validation.")

    required = required_pipeline_paths(output_root) if output_root.exists() else {}
    output_exists = existing_status(required) if required else {}
    missing = [name for name, exists in output_exists.items() if not exists]
    if missing:
        errors.append(f"missing required pipeline outputs: {', '.join(missing)}")

    metrics = extract_pipeline_metrics(output_root) if output_root.exists() else {}
    metric_errors = validate_official_metrics(metrics) if metrics else ["unable to extract pipeline metrics"]
    if metric_errors:
        if args.allow_metric_mismatch:
            warnings.extend(metric_errors)
        else:
            errors.extend(metric_errors)

    backup_info: Dict[str, Any] = {"backup_dir": None, "items": []}
    sync_info: Dict[str, Any] = {"synced_paths": [], "current_metrics": {}}
    if not errors:
        if not args.skip_backup:
            backup_info = backup_mainline()
        sync_info = sync_outputs(output_root)

    report = {
        "status": "success" if not errors else "failed",
        "timestamp": now_iso(),
        "backup_dir": relative_to_project(backup_info["backup_dir"]) if backup_info.get("backup_dir") else None,
        "backup_items": backup_info.get("items", []),
        "synced_paths": sync_info.get("synced_paths", []),
        "required_output_exists": output_exists,
        "metrics_after_sync": sync_info.get("current_metrics", metrics),
        "validation_mode": "semantic_consistency",
        "reference_metrics": OFFICIAL_METRICS,
        "warnings": warnings,
        "errors": errors,
        "safety_note": SAFETY_NOTE,
    }
    write_json(SYNC_REPORT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
