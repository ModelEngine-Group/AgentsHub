from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "configs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RELEASE_OUTPUT_ROOT = OUTPUT_ROOT / "release"
DATAMATE_OUTPUT_ROOT = OUTPUT_ROOT / "datamate_full_pipeline"
BACKUP_ROOT = PROJECT_ROOT / "backups"

SAFETY_NOTE = "本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。"
CONTAINER_NAME = "datamate-runtime"
CONTAINER_WORK_ROOT = Path("/tmp/chroniccare_datamate_full_pipeline")
CONTAINER_INPUT_ROOT = CONTAINER_WORK_ROOT / "input"
CONTAINER_OUTPUT_ROOT = CONTAINER_WORK_ROOT / "output"

PIPELINE_STEPS: List[str] = [
    "chronic_file_ingest",
    "chronic_table_clean",
    "chronic_field_normalize",
    "chronic_text_split",
    "chronic_entity_extract",
    "chronic_relation_extract",
    "chronic_triple_validate",
    "chronic_kg_build",
    "chronic_sqlite_loader",
    "chronic_nl2sql_analyze",
    "chronic_report_pack",
]

OFFICIAL_METRICS = {
    "patient_count_min": 1000,
    "visit_count_min": 5000,
    "lab_result_count_min": 78000,
    "quality_score_total_min": 85,
    "question_count_min": 60,
}

SYNC_TARGETS = {
    "data_processed": DATA_ROOT / "processed",
    "data_graph": DATA_ROOT / "graph",
    "data_sqlite": DATA_ROOT / "sqlite",
    "reports": OUTPUT_ROOT / "reports",
    "charts": OUTPUT_ROOT / "charts",
    "current_metrics": CONFIG_ROOT / "current_metrics.json",
}

BACKUP_ITEMS: Sequence[Path] = (
    SYNC_TARGETS["data_processed"],
    SYNC_TARGETS["data_graph"],
    SYNC_TARGETS["data_sqlite"],
    SYNC_TARGETS["reports"],
    SYNC_TARGETS["charts"],
    SYNC_TARGETS["current_metrics"],
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_project(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run_command(args: List[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        text=True,
        input=input_text,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def copy_tree(src: Path, dest: Path) -> int:
    ensure_directory(dest.parent)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return sum(1 for item in dest.rglob("*") if item.is_file())


def copy_file(src: Path, dest: Path) -> None:
    ensure_directory(dest.parent)
    shutil.copy2(src, dest)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return max(sum(1 for _ in file) - 1, 0)


def raw_dataset_counts() -> Dict[str, int]:
    structured_dir = DATA_ROOT / "raw" / "structured"
    mapping = {
        "patient_count": "patient_profile.csv",
        "visit_count": "visit_record.csv",
        "lab_result_count": "lab_result.csv",
        "medication_record_count": "medication_record.csv",
    }
    counts: Dict[str, int] = {}
    for key, filename in mapping.items():
        csv_path = structured_dir / filename
        counts[key] = count_csv_rows(csv_path) if csv_path.exists() else 0
    return counts


def materialized_output_root(output_root: Path) -> Path:
    nested_output = output_root / "output"
    return nested_output if nested_output.exists() else output_root


def extract_pipeline_metrics(output_root: Path) -> Dict[str, Any]:
    output_root = materialized_output_root(output_root)
    graph_summary_path = output_root / "chronic_kg_build" / "graph" / "graph_summary.json"
    indicator_results_path = output_root / "chronic_nl2sql_analyze" / "indicator_results.json"
    metrics = dict(raw_dataset_counts())
    if graph_summary_path.exists():
        graph_summary = load_json(graph_summary_path)
        metrics.update(
            {
                "node_count": int(graph_summary.get("node_count", 0) or 0),
                "edge_count": int(graph_summary.get("edge_count", 0) or 0),
                "quality_score_total": int((graph_summary.get("quality_score") or {}).get("total", 0) or 0),
            }
        )
    else:
        metrics.update({"node_count": 0, "edge_count": 0, "quality_score_total": 0})

    if indicator_results_path.exists():
        indicator_results = load_json(indicator_results_path)
        metrics["question_count"] = len(indicator_results.get("items", []))
    else:
        metrics["question_count"] = 0
    return metrics


def validate_official_metrics(metrics: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    mapped = {
        "patient_count": int(metrics.get("patient_count", 0) or 0),
        "visit_count": int(metrics.get("visit_count", 0) or 0),
        "lab_result_count": int(metrics.get("lab_result_count", 0) or 0),
        "node_count": int(metrics.get("node_count", 0) or 0),
        "edge_count": int(metrics.get("edge_count", 0) or 0),
        "quality_score_total": int(metrics.get("quality_score_total", 0) or 0),
        "question_count": int(metrics.get("question_count", metrics.get("nl2sql_question_count", 0)) or 0),
    }
    if mapped["patient_count"] < OFFICIAL_METRICS["patient_count_min"]:
        errors.append(f"patient_count below minimum: {mapped['patient_count']}")
    if mapped["visit_count"] < OFFICIAL_METRICS["visit_count_min"]:
        errors.append(f"visit_count below minimum: {mapped['visit_count']}")
    if mapped["lab_result_count"] < OFFICIAL_METRICS["lab_result_count_min"]:
        errors.append(f"lab_result_count below minimum: {mapped['lab_result_count']}")
    if mapped["node_count"] <= 0:
        errors.append("node_count must be greater than 0")
    if mapped["edge_count"] <= 0:
        errors.append("edge_count must be greater than 0")
    if mapped["quality_score_total"] < OFFICIAL_METRICS["quality_score_total_min"]:
        errors.append(f"quality_score_total below minimum: {mapped['quality_score_total']}")
    if mapped["question_count"] < OFFICIAL_METRICS["question_count_min"]:
        errors.append(f"question_count below minimum: {mapped['question_count']}")
    return errors


def required_pipeline_paths(output_root: Path) -> Dict[str, Path]:
    output_root = materialized_output_root(output_root)
    return {
        "manifest": output_root / "chronic_file_ingest" / "manifest.json",
        "clean_tables": output_root / "chronic_table_clean" / "clean_tables",
        "normalized_tables": output_root / "chronic_field_normalize" / "normalized_tables",
        "chunks": output_root / "chronic_text_split" / "chunks",
        "entities": output_root / "chronic_entity_extract" / "entities",
        "relations": output_root / "chronic_relation_extract" / "relations",
        "triples_clean": output_root / "chronic_triple_validate" / "triples_clean.jsonl",
        "graph_summary": output_root / "chronic_kg_build" / "graph" / "graph_summary.json",
        "sqlite_db": output_root / "chronic_sqlite_loader" / "chroniccare.db",
        "indicator_results": output_root / "chronic_nl2sql_analyze" / "indicator_results.json",
        "report_html": output_root / "chronic_report_pack" / "analysis_report.html",
        "chart_index": output_root / "chronic_report_pack" / "charts" / "chart_index.html",
    }


def existing_status(paths: Dict[str, Path]) -> Dict[str, bool]:
    return {key: path.exists() for key, path in paths.items()}
