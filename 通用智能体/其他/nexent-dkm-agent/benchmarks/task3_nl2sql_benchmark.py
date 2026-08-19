"""Measure task-3 NL2SQL accuracy at three levels.

1. Intent-classification accuracy over the labeled benchmark
   (``benchmarks/data/nl2sql_benchmark.json``): each canonical intent maps to
   one SQL template, so intent accuracy equals template-correctness.
2. Execution-level accuracy over the execution benchmark
   (``benchmarks/data/nl2sql_execution_benchmark.json``): the translator's SQL
   is executed against an embedded graph and its result rows are compared to a
   hand-written gold query's rows. This validates end-to-end correctness,
   including entity-aware disease filtering.
3. Expanded paraphrase regression accuracy over
   ``benchmarks/data/nl2sql_holdout_benchmark.json``. The filename is retained
   for compatibility, but this is a development regression set rather than an
   untouched generalization claim.

4. Independent per-path execution accuracy (``independent_paths``): template,
   LLM-only, and local-model-only translators are measured separately so
   reviewers can see each NL2SQL path on its own merits (not only the fallback
   chain aggregate).

The task-3 rubric requires NL2SQL accuracy of at least 85%.

Usage:
    python benchmarks/task3_nl2sql_benchmark.py
    python benchmarks/task3_nl2sql_benchmark.py --report benchmarks/reports/task3_nl2sql_report.json
    python benchmarks/task3_nl2sql_benchmark.py --evaluate-paths --llm-config .env --local-model outputs/models/nl2sql
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.analysis_ops import (
    evaluate_nl2sql_accuracy,
    evaluate_nl2sql_execution_accuracy,
)

DEFAULT_BENCHMARK = ROOT / "benchmarks" / "data" / "nl2sql_benchmark.json"
DEFAULT_EXECUTION_BENCHMARK = ROOT / "benchmarks" / "data" / "nl2sql_execution_benchmark.json"
DEFAULT_HOLDOUT_BENCHMARK = ROOT / "benchmarks" / "data" / "nl2sql_holdout_benchmark.json"
THRESHOLD = 0.85
DEFAULT_LLM_CONFIG_CANDIDATES = (
    ROOT / ".local" / "llm_deepseek_v4.env",
    ROOT / ".local" / "llm_config.env",
)
DEFAULT_LOCAL_MODEL_PATH = ROOT / "data" / "training" / "analysis_nl2sql_model_output" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-3 NL2SQL accuracy.")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK), help="Labeled intent benchmark JSON path.")
    parser.add_argument(
        "--execution-benchmark",
        default=str(DEFAULT_EXECUTION_BENCHMARK),
        help="Execution benchmark JSON path (graph + gold_sql cases).",
    )
    parser.add_argument(
        "--holdout-benchmark",
        default=str(DEFAULT_HOLDOUT_BENCHMARK),
        help="Expanded paraphrase regression JSON path (graph + gold_sql cases).",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="LLM config path (.env or .json). Used for LLM-only and fallback-chain paths.",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Local fine-tuned NL2SQL model path. Used for local-model-only path.",
    )
    parser.add_argument(
        "--evaluate-paths",
        action="store_true",
        default=True,
        help="Measure template, LLM-only, and local-model-only paths independently (default: on).",
    )
    parser.add_argument(
        "--no-evaluate-paths",
        action="store_false",
        dest="evaluate_paths",
        help="Skip independent per-path execution measurement.",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Deprecated alias for --evaluate-paths with LLM config.",
    )
    parser.add_argument(
        "--include-local-model",
        action="store_true",
        help="Deprecated alias for --evaluate-paths with local model path.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _discover_llm_config_path(explicit: str | None) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.exists() else None
    for candidate in DEFAULT_LLM_CONFIG_CANDIDATES:
        if candidate.exists():
            from src.common.llm_config import load_llm_config

            if load_llm_config(candidate):
                return candidate
    return None


def _discover_local_model_path(explicit: str | None) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        return None
    if (DEFAULT_LOCAL_MODEL_PATH / "adapter_config.json").exists():
        return DEFAULT_LOCAL_MODEL_PATH
    return None


def _build_template_translator() -> Callable[..., dict[str, Any]]:
    from src.operators.analysis_ops.nl2sql import (
        drug_names_from_connection,
        symptom_names_from_connection,
        translate_question_to_sql,
        treatment_names_from_connection,
    )

    def _translate(question, conn, disease_names):
        symptom_names = symptom_names_from_connection(conn)
        drug_names = drug_names_from_connection(conn)
        treatment_names = treatment_names_from_connection(conn)
        result = translate_question_to_sql(
            question,
            disease_names=disease_names,
            symptom_names=symptom_names,
            drug_names=drug_names,
            treatment_names=treatment_names,
        )
        return {**result, "translator": "template"}

    return _translate


def _build_llm_only_translator(llm_config: dict[str, Any] | None) -> Callable[..., dict[str, Any]]:
    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_llm_only

    def _translate(question, conn, _disease_names):
        return translate_question_to_sql_llm_only(question, conn, llm_config=llm_config)

    return _translate


def _build_local_only_translator(local_model_path: str | None) -> Callable[..., dict[str, Any]]:
    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_local_only

    def _translate(question, conn, _disease_names):
        return translate_question_to_sql_local_only(question, conn, local_model_path=local_model_path)

    return _translate


def _build_fallback_translator(
    llm_config: dict[str, Any] | None,
    local_model_path: str | None,
) -> Callable[..., dict[str, Any]]:
    from src.operators.analysis_ops.llm_nl2sql import translate_question_with_fallbacks

    def _translate(question, conn, _disease_names):
        result = translate_question_with_fallbacks(
            question,
            conn,
            llm_config=llm_config,
            local_model_path=local_model_path,
        )
        result.setdefault("translator", "fallback_chain")
        return result

    return _translate


def _summarize_execution(exec_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "total": exec_report["total"],
        "correct": exec_report["correct"],
        "accuracy": exec_report["accuracy"],
        "passed": exec_report["accuracy"] >= THRESHOLD,
        "metric": exec_report.get("metric", "execution_row_match"),
        "mistakes": exec_report.get("mistakes", []),
    }


def _evaluate_path(
    label: str,
    translator: Callable[..., dict[str, Any]] | None,
    cases: list[dict[str, Any]],
    graph: dict[str, Any],
    *,
    status: str = "evaluated",
) -> dict[str, Any]:
    if translator is None:
        return {
            "path": label,
            "status": status,
            "execution": {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "passed": False,
            },
        }

    exec_report = evaluate_nl2sql_execution_accuracy(cases, graph, translator=translator)
    summary = _summarize_execution(exec_report)
    summary["per_translator"] = exec_report.get("per_translator", {})
    return {
        "path": label,
        "status": status,
        "execution": summary,
    }


def main() -> int:
    args = parse_args()
    evaluate_paths = args.evaluate_paths or args.include_llm or args.include_local_model

    llm_config_path = _discover_llm_config_path(args.llm_config)
    local_model_path = _discover_local_model_path(args.local_model)

    llm_config = None
    if llm_config_path:
        from src.common.llm_config import load_llm_config

        llm_config = load_llm_config(llm_config_path)

    local_model = str(local_model_path) if local_model_path else None

    benchmark_path = Path(args.benchmark)
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = data["cases"]

    intent_report = evaluate_nl2sql_accuracy(cases)
    intent_report["benchmark"] = _relative(benchmark_path)
    intent_report["threshold"] = THRESHOLD
    intent_report["passed"] = intent_report["accuracy"] >= THRESHOLD

    per_translator: dict[str, dict] = {
        "template": {
            "total": intent_report["total"],
            "correct": intent_report["correct"],
            "accuracy": intent_report["accuracy"],
            "passed": intent_report["passed"],
        }
    }

    report: dict = {
        "nl2sql_path": "template",
        "enhancement_config": {
            "llm_config": _relative(llm_config_path) if llm_config_path else None,
            "local_model": _relative(local_model_path) if local_model_path else None,
        },
        "per_translator": per_translator,
        "intent_classification": intent_report,
        "accuracy": intent_report["accuracy"],
        "threshold": THRESHOLD,
        "passed": intent_report["passed"],
    }

    execution_path = Path(args.execution_benchmark)
    exec_cases: list[dict[str, Any]] = []
    exec_graph: dict[str, Any] = {}
    if execution_path.exists():
        exec_data = json.loads(execution_path.read_text(encoding="utf-8"))
        exec_cases = exec_data["cases"]
        exec_graph = exec_data["graph"]
        # Primary rubric metrics always use the template translator.
        exec_report = evaluate_nl2sql_execution_accuracy(exec_cases, exec_graph)
        exec_report["benchmark"] = _relative(execution_path)
        exec_report["threshold"] = THRESHOLD
        exec_report["passed"] = exec_report["accuracy"] >= THRESHOLD
        report["execution"] = exec_report
        report["passed"] = report["passed"] and exec_report["passed"]

    holdout_path = Path(args.holdout_benchmark)
    if holdout_path.exists():
        holdout_data = json.loads(holdout_path.read_text(encoding="utf-8"))
        holdout_report = evaluate_nl2sql_execution_accuracy(
            holdout_data["cases"], holdout_data["graph"]
        )
        holdout_report["benchmark"] = _relative(holdout_path)
        holdout_report["evaluation_role"] = "expanded_paraphrase_regression"
        holdout_report["untouched_holdout"] = False
        holdout_report["threshold"] = THRESHOLD
        holdout_report["passed"] = holdout_report["accuracy"] >= THRESHOLD
        report["holdout_generalization"] = holdout_report
        report["passed"] = report["passed"] and holdout_report["passed"]

    if evaluate_paths and exec_cases:
        independent: dict[str, Any] = {}
        independent["template"] = _evaluate_path(
            "template",
            _build_template_translator(),
            exec_cases,
            exec_graph,
        )
        if llm_config:
            independent["llm"] = _evaluate_path(
                "llm",
                _build_llm_only_translator(llm_config),
                exec_cases,
                exec_graph,
            )
        else:
            independent["llm"] = _evaluate_path(
                "llm", None, exec_cases, exec_graph, status="not_configured"
            )

        if local_model:
            independent["local_model"] = _evaluate_path(
                "local_model",
                _build_local_only_translator(local_model),
                exec_cases,
                exec_graph,
            )
        else:
            independent["local_model"] = _evaluate_path(
                "local_model", None, exec_cases, exec_graph, status="not_configured"
            )

        if llm_config or local_model:
            fallback = _build_fallback_translator(llm_config, local_model)
            independent["fallback_chain"] = _evaluate_path(
                "fallback_chain",
                fallback,
                exec_cases,
                exec_graph,
            )
            report["nl2sql_path"] = "fallback_chain(local_model>llm>template)"

        report["independent_paths"] = independent
        for label, path_report in independent.items():
            if path_report.get("status") != "evaluated":
                continue
            exec_summary = path_report["execution"]
            per_translator[label] = {
                "total": exec_summary["total"],
                "correct": exec_summary["correct"],
                "accuracy": exec_summary["accuracy"],
                "passed": exec_summary["passed"],
                "status": "evaluated",
            }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
