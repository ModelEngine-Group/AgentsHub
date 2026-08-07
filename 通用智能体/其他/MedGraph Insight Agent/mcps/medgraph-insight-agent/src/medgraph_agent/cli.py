from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from medgraph_agent.core.analytics import GraphAnalyzer, evaluate_nl2sql
from medgraph_agent.core.benchmark import run_benchmarks
from medgraph_agent.core.pipeline import PipelineRunner
from medgraph_agent.core.quality import audit_graph
from medgraph_agent.core.qa import answer_question
from medgraph_agent.core.storage import GraphStore, load_graph_json, read_json, write_json
from medgraph_agent.core.models import to_dict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source() -> Path:
    env_source = os.environ.get("MEDGRAPH_SOURCE")
    if env_source:
        return Path(env_source)
    candidates = (
        Path("knowledge/medical_cases.jsonl"),
        Path("data/sample/medical_cases.jsonl"),
        repo_root().parents[1] / "knowledge" / "medical_cases.jsonl",
        repo_root() / "data/sample/medical_cases.jsonl",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("medical_cases.jsonl not found; set MEDGRAPH_SOURCE")


def default_output_dir() -> Path:
    return Path(os.environ.get("MEDGRAPH_OUTPUT_DIR", "outputs/latest"))


def ensure_graph(output_dir: Path, source: Path | None = None) -> Path:
    graph_path = output_dir / "graph.json"
    if graph_path.exists():
        return graph_path
    runner = PipelineRunner(output_dir)
    run = runner.run("构建医疗数据处理、知识图谱问答和图谱分析闭环", source or default_source())
    if run.status != "succeeded":
        raise SystemExit(f"pipeline failed: {run.error}")
    return graph_path


def print_json(payload: Any) -> None:
    print(json.dumps(to_dict(payload), ensure_ascii=False, indent=2))


def format_quality_summary(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    failed_checks = [name for name, passed in report["checks"].items() if not passed]
    failed_text = ", ".join(failed_checks) if failed_checks else "none"
    return "\n".join(
        [
            f"quality={status}",
            f"evidence_coverage={report['evidence_coverage']}",
            f"failed_checks={failed_text}",
        ]
    )


def cmd_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    runner = PipelineRunner(output_dir)
    run = runner.run(args.task, Path(args.source))
    graph = load_graph_json(output_dir / "graph.json")
    analyzer = GraphAnalyzer(output_dir / "medgraph.db")
    answer = answer_question("高血压有哪些症状、治疗和检查证据？", graph)
    analysis = analyzer.analyze("统计关系类型分布")
    benchmark = run_benchmarks(Path(args.source), repeat=args.repeat)
    quality = audit_graph(graph)
    nl2sql_questions = read_json(Path(args.eval_file))
    nl2sql_eval = evaluate_nl2sql(output_dir / "medgraph.db", nl2sql_questions)
    write_json(output_dir / "qa_sample.json", answer)
    write_json(output_dir / "analysis_sample.json", analysis)
    write_json(output_dir / "benchmark.json", benchmark)
    write_json(output_dir / "quality_report.json", quality)
    write_json(output_dir / "nl2sql_eval.json", nl2sql_eval)
    print_json(
        {
            "run": run,
            "graph_stats": graph.stats(),
            "qa_sample": answer,
            "analysis_sample": analysis,
            "benchmark": benchmark,
            "quality": quality,
            "nl2sql_eval": nl2sql_eval,
        }
    )


def cmd_run(args: argparse.Namespace) -> None:
    run = PipelineRunner(args.output_dir).run(args.task, args.source)
    print_json(run)
    if run.status != "succeeded":
        raise SystemExit(1)


def cmd_qa(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    graph = load_graph_json(ensure_graph(output_dir, Path(args.source) if args.source else None))
    print_json(answer_question(args.question, graph))


def cmd_analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_graph(output_dir, Path(args.source) if args.source else None)
    print_json(GraphAnalyzer(output_dir / "medgraph.db").analyze(args.question))


def cmd_benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmarks(args.source, repeat=args.repeat)
    write_json(output_dir / "benchmark.json", result)
    print_json(result)


def cmd_nl2sql_eval(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_graph(output_dir, Path(args.source) if args.source else None)
    questions = read_json(args.eval_file)
    print_json(evaluate_nl2sql(output_dir / "medgraph.db", questions))


def cmd_quality(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    graph = load_graph_json(ensure_graph(output_dir, Path(args.source) if args.source else None))
    report = audit_graph(graph)
    write_json(output_dir / "quality_report.json", report)
    if args.format == "json":
        print_json(report)
    else:
        print(format_quality_summary(report))
    if not report["passed"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medgraph", description="MedGraph Insight Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the full preliminary-task demo")
    demo.add_argument("--source", default=str(default_source()))
    demo.add_argument("--output-dir", default=str(default_output_dir()))
    demo.add_argument("--task", default="完成医疗数据处理、知识图谱生成问答和图谱驱动分析展示")
    demo.add_argument("--repeat", type=int, default=20)
    demo.add_argument("--eval-file", default="knowledge/nl2sql_eval.json")
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="Run a planned data processing pipeline")
    run.add_argument("--task", required=True)
    run.add_argument("--source", default=str(default_source()))
    run.add_argument("--output-dir", default=str(default_output_dir()))
    run.set_defaults(func=cmd_run)

    qa = sub.add_parser("qa", help="Ask a graph-grounded medical question")
    qa.add_argument("--question", required=True)
    qa.add_argument("--source", default=None)
    qa.add_argument("--output-dir", default=str(default_output_dir()))
    qa.set_defaults(func=cmd_qa)

    analyze = sub.add_parser("analyze", help="Run graph-driven analysis/NL2SQL")
    analyze.add_argument("--question", required=True)
    analyze.add_argument("--source", default=None)
    analyze.add_argument("--output-dir", default=str(default_output_dir()))
    analyze.set_defaults(func=cmd_analyze)

    benchmark = sub.add_parser("benchmark", help="Run CPU/GPU/NPU backend availability and performance checks")
    benchmark.add_argument("--source", default=str(default_source()))
    benchmark.add_argument("--output-dir", default=str(default_output_dir()))
    benchmark.add_argument("--repeat", type=int, default=50)
    benchmark.set_defaults(func=cmd_benchmark)

    eval_parser = sub.add_parser("nl2sql-eval", help="Evaluate offline NL2SQL intent routing")
    eval_parser.add_argument("--eval-file", default="knowledge/nl2sql_eval.json")
    eval_parser.add_argument("--source", default=None)
    eval_parser.add_argument("--output-dir", default=str(default_output_dir()))
    eval_parser.set_defaults(func=cmd_nl2sql_eval)

    quality_parser = sub.add_parser("quality", help="Audit graph integrity and known precision guards")
    quality_parser.add_argument("--source", default=None)
    quality_parser.add_argument("--output-dir", default=str(default_output_dir()))
    quality_parser.add_argument("--format", choices=["json", "text"], default="json")
    quality_parser.set_defaults(func=cmd_quality)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
