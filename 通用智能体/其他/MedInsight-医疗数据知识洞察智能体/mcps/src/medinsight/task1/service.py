from __future__ import annotations

from pathlib import Path
from typing import Any

from medinsight.common.utils import ensure_dir, read_csv_records, read_jsonl, write_json, write_jsonl
from medinsight.task1.agent import DataProcessingAgent
from medinsight.task1.evolution import (
    build_evolution_experience,
    build_planner_training_sample,
    build_workflow_skill,
)


DEFAULT_INPUT = Path("data/task1/raw/qamedkg_samples.jsonl")
DEFAULT_TABLE_INPUT = Path("data/task1/raw/qamedkg_structured_table.csv")
DEFAULT_OUTPUT = Path("artifacts/task1/api")
DEFAULT_TASK = "清洗医疗数据，完成数据画像、规范化、文本清洗和句子切分。"
ANOMALY_TASK = "清洗并规范化医疗数据，遇到缺字段和空文本时自动恢复或记录异常。"


def build_anomaly_records(source: Path = DEFAULT_INPUT) -> list[dict[str, Any]]:
    base = read_jsonl(source)[:3]
    records = [dict(item) for item in base]

    missing_doc_id = dict(base[0])
    missing_doc_id.pop("doc_id", None)
    missing_doc_id["title"] = ""
    records.append(missing_doc_id)

    rebuildable = dict(base[1])
    rebuildable["doc_id"] = "anomaly-rebuildable-structured-only"
    rebuildable["text"] = ""
    records.append(rebuildable)

    records.append(
        {
            "doc_id": "anomaly-unrecoverable-empty",
            "title": "不可恢复空记录",
            "text": "",
            "source_dataset": "synthetic_anomaly_case",
        }
    )
    return records


def run_task1(
    records: list[dict[str, Any]],
    task: str = DEFAULT_TASK,
    output_dir: Path = DEFAULT_OUTPUT,
    scheduler_enabled: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    result = DataProcessingAgent().run(records, task, scheduler_enabled=scheduler_enabled)
    report = result["report"]
    experience = build_evolution_experience(report)
    workflow_skill = build_workflow_skill(experience)
    planner_sample = build_planner_training_sample(experience)
    evolution_dir = ensure_dir(output_dir / "evolution")
    write_jsonl(evolution_dir / "evolution_experience.jsonl", [experience])
    write_json(evolution_dir / "workflow_skills.json", [workflow_skill])
    write_jsonl(evolution_dir / "evolution_planner_samples.jsonl", [planner_sample])
    report["self_evolution"] = {
        "mode": "offline_auditable_experience",
        "experience_path": str(evolution_dir / "evolution_experience.jsonl"),
        "workflow_skills_path": str(evolution_dir / "workflow_skills.json"),
        "planner_samples_path": str(evolution_dir / "evolution_planner_samples.jsonl"),
        "suggested_update": experience["suggested_update"],
        "reuse": experience["reuse"],
    }
    write_json(output_dir / "data_processing_report.json", report)
    write_json(output_dir / "events.json", {"data_processing": result["events"]})
    write_json(output_dir / "processed_documents.json", [item.to_dict() for item in result["documents"]])
    write_json(output_dir / "segments.json", [item.to_dict() for item in result["segments"]])
    return {
        "status": "ok",
        "quality": report["quality"],
        "report_path": str(output_dir / "data_processing_report.json"),
        "events_path": str(output_dir / "events.json"),
        "documents_path": str(output_dir / "processed_documents.json"),
        "segments_path": str(output_dir / "segments.json"),
        "plan": report["plan"],
        "performance": report["performance"],
        "self_evolution": report["self_evolution"],
    }


def run_task1_from_file(
    input_path: Path = DEFAULT_INPUT,
    task: str = DEFAULT_TASK,
    output_dir: Path = DEFAULT_OUTPUT,
    scheduler_enabled: bool = True,
) -> dict[str, Any]:
    return run_task1(read_jsonl(input_path), task, output_dir, scheduler_enabled=scheduler_enabled)


def run_task1_structured_table_from_csv(
    input_path: Path = DEFAULT_TABLE_INPUT,
    task: str = "处理结构化医疗表格，完成字段画像、规范化、质量检查和文本片段生成。",
    output_dir: Path = Path("artifacts/task1/structured_table"),
    scheduler_enabled: bool = True,
) -> dict[str, Any]:
    records = read_csv_records(input_path)
    result = run_task1(records, task, output_dir, scheduler_enabled=scheduler_enabled)
    result["input_type"] = "structured_table_csv"
    result["input_path"] = str(input_path)
    return result


def run_anomaly_demo(output_dir: Path = Path("artifacts/task1/anomaly_demo")) -> dict[str, Any]:
    records = build_anomaly_records()
    ensure_dir(output_dir)
    write_json(output_dir / "input_records.json", records)
    return run_task1(records, ANOMALY_TASK, output_dir)
