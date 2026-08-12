from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = AGENT_ROOT / "mcps" / "src"
KNOWLEDGE_ROOT = AGENT_ROOT / "knowledge"
_runtime_setting = Path(os.getenv("MEDINSIGHT_RUNTIME_ROOT", "runtime")).expanduser()
RUNTIME_ROOT = _runtime_setting.resolve() if _runtime_setting.is_absolute() else (AGENT_ROOT / _runtime_setting).resolve()

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.chdir(AGENT_ROOT)

from medinsight.common.utils import write_json  # noqa: E402
from medinsight.task1.service import run_task1  # noqa: E402
from medinsight.task2.operators.qa import answer_question  # noqa: E402
from medinsight.task2.service import run_task2_text_kg_build  # noqa: E402
from medinsight.task3.agent import GraphInsightAgent  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_run_id(value: str | None, prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip()).strip("-.")
    return cleaned[:64] or prefix


def _parse_records(input_json: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(input_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input_json is not valid JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    elif isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("input_json must be a JSON object, an object array, or {'records': [...]}")
    if not payload:
        raise ValueError("input_json must contain at least one record")
    return payload


def _resolve_local_path(path_value: str | None, default: Path) -> Path:
    candidate = Path(path_value).expanduser() if path_value else default
    if not candidate.is_absolute():
        candidate = (AGENT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")
    return candidate


def _latest_graph_path() -> Path:
    state_path = RUNTIME_ROOT / "state.json"
    if state_path.exists():
        state = _read_json(state_path)
        path = Path(str(state.get("latest_graph_path") or ""))
        if path.exists():
            return path
    return KNOWLEDGE_ROOT / "empty_graph.json"


def _update_state(**values: Any) -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = RUNTIME_ROOT / "state.json"
    state = _read_json(state_path) if state_path.exists() else {}
    state.update(values)
    write_json(state_path, state)


def process_medical_data(
    input_json: str,
    task: str = "清洗、规范化并切分医疗数据，输出质量报告。",
    run_id: str = "task1-latest",
    require_datamate: bool = False,
) -> dict[str, Any]:
    """Process JSON medical records and emit auditable Task1 artifacts."""
    records = _parse_records(input_json)
    output_dir = RUNTIME_ROOT / "task1" / _safe_run_id(run_id, "task1-latest")
    result = run_task1(records=records, task=task, output_dir=output_dir)
    report = _read_json(Path(result["report_path"]))
    datamate = report.get("datamate_integration") or {}
    fallback = datamate.get("fallback") or {}
    if require_datamate and not datamate.get("connected"):
        raise RuntimeError(
            "DataMate is required for this run but is not connected. "
            f"missing_services={fallback.get('missing_services') or []}"
        )
    result["datamate_integration"] = datamate
    _update_state(
        latest_task1_report=result["report_path"],
        latest_processed_documents=result["documents_path"],
    )
    return result


def build_medical_knowledge_graph(
    processed_data_json: str,
    task: str = "从任务一处理后的医疗文本生成可问答知识图谱。",
    run_id: str = "task2-latest",
    question: str = "",
) -> dict[str, Any]:
    """Extract entities and relations, validate triples, and build a medical graph."""
    output_dir = RUNTIME_ROOT / "task2" / _safe_run_id(run_id, "task2-latest")
    shared_dir = RUNTIME_ROOT / "knowledge"
    result = run_task2_text_kg_build(
        text=processed_data_json,
        output_dir=output_dir,
        shared_output_dir=shared_dir,
        task=task,
        doc_id=_safe_run_id(run_id, "task2-latest"),
        question=question or None,
    )
    _update_state(
        latest_graph_path=result["graph_path"],
        latest_kg_report=result["report_path"],
    )
    return result


def ask_medical_knowledge_graph(
    question: str,
    graph_path: str = "",
    max_facts: int = 10,
) -> dict[str, Any]:
    """Answer a medical question strictly from a generated graph and its evidence paths."""
    path = _resolve_local_path(graph_path or None, _latest_graph_path())
    graph = _read_json(path)
    answer = answer_question(graph, question, max_facts=max(1, min(max_facts, 30)))
    return {"status": "ok", "graph_path": str(path), "result": answer}


def analyze_medical_graph(
    graph_path: str = "",
    task: str = "分析医疗知识图谱并生成指标、NL2SQL 评测、BI 看板和洞察报告。",
    run_id: str = "task3-latest",
) -> dict[str, Any]:
    """Profile a Task2 graph and generate graph-grounded analytical artifacts."""
    path = _resolve_local_path(graph_path or None, _latest_graph_path())
    output_dir = RUNTIME_ROOT / "task3" / _safe_run_id(run_id, "task3-latest")
    state_path = RUNTIME_ROOT / "state.json"
    state = _read_json(state_path) if state_path.exists() else {}
    task1_report = state.get("latest_task1_report")
    return GraphInsightAgent().run_analysis(
        graph_path=path,
        output_dir=output_dir,
        task=task,
        task1_report_path=task1_report,
    )


def query_medical_graph_analysis(
    question: str,
    graph_path: str = "",
    run_id: str = "task3-query-latest",
) -> dict[str, Any]:
    """Map a natural-language analytical request to verified read-only SQL."""
    path = _resolve_local_path(graph_path or None, _latest_graph_path())
    output_dir = RUNTIME_ROOT / "task3" / _safe_run_id(run_id, "task3-query-latest")
    state_path = RUNTIME_ROOT / "state.json"
    state = _read_json(state_path) if state_path.exists() else {}
    return GraphInsightAgent().ask(
        question=question,
        graph_path=path,
        output_dir=output_dir,
        task1_report_path=state.get("latest_task1_report"),
    )


def run_data_knowledge_insight_pipeline(
    input_json: str,
    analysis_question: str = "统计实体分布",
    qa_question: str = "",
    run_id: str = "pipeline-latest",
    require_datamate: bool = False,
) -> dict[str, Any]:
    """Run the complete Data -> Knowledge -> Insight pipeline."""
    safe_id = _safe_run_id(run_id, "pipeline-latest")
    task1 = process_medical_data(
        input_json=input_json,
        run_id=f"{safe_id}-task1",
        require_datamate=require_datamate,
    )
    processed_text = Path(task1["documents_path"]).read_text(encoding="utf-8")
    task2 = build_medical_knowledge_graph(
        processed_data_json=processed_text,
        run_id=f"{safe_id}-task2",
        question=qa_question,
    )
    task3 = query_medical_graph_analysis(
        question=analysis_question,
        graph_path=task2["graph_path"],
        run_id=f"{safe_id}-task3",
    )
    return {
        "status": "ok",
        "pipeline": "data_to_knowledge_to_insight",
        "task1": task1,
        "task2": task2,
        "task3": task3,
    }
