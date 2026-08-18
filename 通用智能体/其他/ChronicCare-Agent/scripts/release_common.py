from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFETY_NOTE = "本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。"
PUBLIC_DAY_LABEL_PATTERN = re.compile(r"\bDay\s*\d+\b|Day\d+\b|day\d+\b")
FAKE_PUBLIC_METRIC_PATTERN = re.compile(r"\b(?:1000|2000|0\.95|1,000|2,000)\b")
OFFICIAL_CURRENT_METRICS = {
    "patient_count_min": 1000,
    "visit_count_min": 5000,
    "lab_result_count_min": 78000,
    "nl2sql_question_count_min": 60,
    "quality_score_total_min": 85,
}
BASELINE_METRICS = {
    "patient_count": 120,
    "visit_count": 480,
    "lab_result_count": 7200,
    "medication_record_count": 1920,
    "node_count": 9680,
    "edge_count": 22440,
    "nl2sql_question_count": 15,
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")


def relative_to_project(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_current_metrics() -> Dict[str, Any]:
    candidates = [
        PROJECT_ROOT / "configs" / "current_metrics.json",
        PROJECT_ROOT / "outputs" / "enhanced" / "current_metrics_snapshot.json",
        PROJECT_ROOT / "outputs" / "evaluation" / "current_metrics_snapshot.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return load_json(path)
            except json.JSONDecodeError:
                continue
    return {}


def load_question_count() -> int:
    questions_path = PROJECT_ROOT / "configs" / "nl2sql_questions.json"
    if not questions_path.exists():
        return 0
    payload = load_json(questions_path)
    return len(payload.get("questions", []))



def current_release_metrics() -> Dict[str, Any]:
    metrics = load_current_metrics()
    return {
        "patient_count": int(metrics.get("patient_count", 0) or 0),
        "visit_count": int(metrics.get("visit_count", 0) or 0),
        "lab_result_count": int(metrics.get("lab_result_count", 0) or 0),
        "medication_record_count": int(metrics.get("medication_record_count", 0) or 0),
        "node_count": int(metrics.get("node_count", 0) or 0),
        "edge_count": int(metrics.get("edge_count", 0) or 0),
        "quality_score_total": int(metrics.get("quality_score_total", 0) or 0),
        "nl2sql_question_count": int(metrics.get("nl2sql_question_count", load_question_count()) or 0),
        "analysis_success_rate": float(metrics.get("analysis_success_rate", 0.0) or 0.0),
    }


def baseline_release_metrics() -> Dict[str, Any]:
    return dict(BASELINE_METRICS)


def validate_official_metrics(metrics: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    mapped = {
        "patient_count": int(metrics.get("patient_count", 0) or 0),
        "visit_count": int(metrics.get("visit_count", 0) or 0),
        "lab_result_count": int(metrics.get("lab_result_count", 0) or 0),
        "node_count": int(metrics.get("node_count", 0) or 0),
        "edge_count": int(metrics.get("edge_count", 0) or 0),
        "quality_score_total": int(metrics.get("quality_score_total", 0) or 0),
        "nl2sql_question_count": int(metrics.get("nl2sql_question_count", metrics.get("question_count", 0)) or 0),
    }
    if mapped["patient_count"] < OFFICIAL_CURRENT_METRICS["patient_count_min"]:
        errors.append(
            f"patient_count below minimum {OFFICIAL_CURRENT_METRICS['patient_count_min']}: {mapped['patient_count']}"
        )
    if mapped["visit_count"] < OFFICIAL_CURRENT_METRICS["visit_count_min"]:
        errors.append(f"visit_count below minimum {OFFICIAL_CURRENT_METRICS['visit_count_min']}: {mapped['visit_count']}")
    if mapped["lab_result_count"] < OFFICIAL_CURRENT_METRICS["lab_result_count_min"]:
        errors.append(
            f"lab_result_count below minimum {OFFICIAL_CURRENT_METRICS['lab_result_count_min']}: {mapped['lab_result_count']}"
        )
    if mapped["node_count"] <= BASELINE_METRICS["node_count"]:
        errors.append(f"node_count did not exceed baseline {BASELINE_METRICS['node_count']}: {mapped['node_count']}")
    if mapped["edge_count"] <= BASELINE_METRICS["edge_count"]:
        errors.append(f"edge_count did not exceed baseline {BASELINE_METRICS['edge_count']}: {mapped['edge_count']}")
    if mapped["quality_score_total"] < OFFICIAL_CURRENT_METRICS["quality_score_total_min"]:
        errors.append(
            f"quality_score_total below minimum {OFFICIAL_CURRENT_METRICS['quality_score_total_min']}: {mapped['quality_score_total']}"
        )
    expected_nl2sql = load_question_count()
    if expected_nl2sql < OFFICIAL_CURRENT_METRICS["nl2sql_question_count_min"]:
        errors.append(
            f"configured nl2sql question count below minimum {OFFICIAL_CURRENT_METRICS['nl2sql_question_count_min']}: {expected_nl2sql}"
        )
    if mapped["nl2sql_question_count"] != expected_nl2sql:
        errors.append(
            f"nl2sql_question_count mismatch: expected {expected_nl2sql} but found {mapped['nl2sql_question_count']}"
        )
    return errors


def copy_file(src: Path, dest: Path) -> None:
    ensure_directory(dest.parent)
    shutil.copy2(src, dest)


def copy_tree(src: Path, dest: Path) -> int:
    ensure_directory(dest.parent)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return sum(1 for item in dest.rglob("*") if item.is_file())


def collect_public_files() -> List[Path]:
    candidates = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "app" / "streamlit_app.py",
        PROJECT_ROOT / "release" / "README.md",
        PROJECT_ROOT / "docker-compose.yml",
    ]
    candidates.extend(sorted((PROJECT_ROOT / "docs").glob("release_*.md")))
    candidates.extend(sorted((PROJECT_ROOT / "integrations" / "nexent").glob("*")))
    return [path for path in candidates if path.is_file()]


def scan_text_patterns(files: Sequence[Path], pattern: re.Pattern[str]) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for path in files:
        matches = pattern.findall(read_text(path))
        if matches:
            hits[relative_to_project(path)] = sorted({str(match) for match in matches})
    return hits


def summarize_release_tree(root: Path) -> Dict[str, Any]:
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return {
        "root": relative_to_project(root),
        "file_count": file_count,
        "size_mb": round(total_bytes / 1024 / 1024, 2),
    }


def markdown_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
