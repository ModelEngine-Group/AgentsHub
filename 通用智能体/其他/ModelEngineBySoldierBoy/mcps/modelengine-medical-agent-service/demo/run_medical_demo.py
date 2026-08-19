"""CLI for the medical demo pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.analysis_agent import run_analysis_agent, run_configured_analysis, run_interactive_analysis, run_nl2sql_eval
from agents.data_processing_agent import run_data_processing_agent
from agents.integration_agent import run_platform_integration_agent
from agents.medical_kg_agent import ask_medical_question, build_medical_demo
from agents.orchestrator import run_orchestrator
from demo.app_server import export_dashboard_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the medical demo pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Run the task-1 data processing pipeline.")
    process_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/medical_demo.jsonl"),
        help="Input data file.",
    )
    process_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root where artifacts will be written.",
    )
    process_parser.add_argument(
        "--task",
        default=None,
        help="Natural-language processing task description used by the task-1 agent planner.",
    )

    orchestrate_parser = subparsers.add_parser(
        "orchestrate",
        help="Run the full task-1/task-2/task-3 pipeline through the engineering orchestrator.",
    )
    orchestrate_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/medical_demo.jsonl"),
        help="Input data file.",
    )
    orchestrate_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root where artifacts will be written.",
    )
    orchestrate_parser.add_argument(
        "--goal",
        default="形成数据-知识-洞察端到端医疗智能体链路",
        help="Human-readable orchestration goal.",
    )
    orchestrate_parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Optional QA question to include in the generated trace. Can be repeated.",
    )
    orchestrate_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Force GPT-5.4 rewriting for generated QA answers.",
    )
    orchestrate_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable GPT-5.4 rewriting even if credentials are configured.",
    )
    orchestrate_parser.add_argument(
        "--kg-backend",
        choices=["rule", "tensor", "llm"],
        default=None,
        help="KG extraction backend. 'tensor' enables CPU/Ascend NPU tensor candidate ranking.",
    )
    orchestrate_parser.add_argument(
        "--external-documents",
        type=Path,
        default=None,
        help="Optional JSONL documents converted from external datasets such as CBLUE.",
    )
    orchestrate_parser.add_argument(
        "--external-triples",
        type=Path,
        default=None,
        help="Optional JSONL triples converted from external datasets such as CBLUE CMeIE.",
    )
    orchestrate_parser.add_argument(
        "--gold-triples",
        type=Path,
        default=None,
        help="Optional gold JSONL triples for KG quality evaluation. Defaults to --external-triples when present.",
    )
    orchestrate_parser.add_argument(
        "--external-entities",
        type=Path,
        default=None,
        help="Optional JSONL entities converted from external datasets such as CBLUE CMeEE.",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Run the task-3 analysis pipeline.")
    analyze_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root containing processed data and graph artifacts.",
    )
    analyze_parser.add_argument(
        "--eval-path",
        type=Path,
        default=None,
        help="Optional NL2SQL eval JSONL path. Defaults to data/eval/nl2sql_questions.jsonl.",
    )

    insight_parser = subparsers.add_parser("insight", help="Run a single task-3 natural-language insight query.")
    insight_parser.add_argument(
        "question",
        help="Natural-language analysis question.",
    )
    insight_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root containing processed data and graph artifacts.",
    )

    analyze_config_parser = subparsers.add_parser(
        "analyze-config",
        help="Run a task-3 configured analysis plan with SQL and graph tools.",
    )
    analyze_config_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON config describing queries, chart overrides, focus entity, and graph tools.",
    )
    analyze_config_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root containing processed data and graph artifacts.",
    )

    nl2sql_eval_parser = subparsers.add_parser("nl2sql-eval", help="Run task-3 NL2SQL accuracy evaluation.")
    nl2sql_eval_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root containing processed data and graph artifacts.",
    )
    nl2sql_eval_parser.add_argument(
        "--eval-path",
        type=Path,
        default=None,
        help="Optional NL2SQL eval JSONL path. Defaults to data/eval/nl2sql_questions.jsonl.",
    )
    nl2sql_eval_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Accuracy threshold used for pass/fail reporting.",
    )

    dashboard_parser = subparsers.add_parser("dashboard", help="Export the task-3 dashboard as standalone HTML.")
    dashboard_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root containing generated graph, analysis, and integration artifacts.",
    )
    dashboard_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to demo/dashboard.html under --output-root.",
    )

    build_parser = subparsers.add_parser("build", help="Build demo artifacts from raw JSONL data.")
    build_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/medical_demo.jsonl"),
        help="Input JSONL file.",
    )
    build_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root where artifacts will be written.",
    )
    build_parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Optional QA question to include in the generated trace. Can be repeated.",
    )
    build_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Force GPT-5.4 rewriting for generated QA answers.",
    )
    build_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable GPT-5.4 rewriting even if credentials are configured.",
    )
    build_parser.add_argument(
        "--kg-backend",
        choices=["rule", "tensor", "llm"],
        default=None,
        help="KG extraction backend. 'tensor' enables CPU/Ascend NPU tensor candidate ranking.",
    )
    build_parser.add_argument(
        "--external-documents",
        type=Path,
        default=None,
        help="Optional JSONL documents converted from external datasets such as CBLUE.",
    )
    build_parser.add_argument(
        "--external-triples",
        type=Path,
        default=None,
        help="Optional JSONL triples converted from external datasets such as CBLUE CMeIE.",
    )
    build_parser.add_argument(
        "--gold-triples",
        type=Path,
        default=None,
        help="Optional gold JSONL triples for KG quality evaluation. Defaults to --external-triples when present.",
    )
    build_parser.add_argument(
        "--external-entities",
        type=Path,
        default=None,
        help="Optional JSONL entities converted from external datasets such as CBLUE CMeEE.",
    )

    ask_parser = subparsers.add_parser("ask", help="Answer a single question from an existing graph.")
    ask_parser.add_argument(
        "question",
        help="Question to answer.",
    )
    ask_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root containing the generated graph.",
    )
    ask_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Force GPT-5.4 rewriting for this answer.",
    )
    ask_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable GPT-5.4 rewriting for this answer.",
    )

    integrate_parser = subparsers.add_parser(
        "integrate",
        help="Collect DataMate/Nexent integration evidence for stage 4.",
    )
    integrate_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Repository root containing generated graph artifacts.",
    )
    integrate_parser.add_argument(
        "--query",
        default="高血压有哪些并发症？",
        help="Query used for the datamate_search evidence call.",
    )
    integrate_parser.add_argument(
        "--knowledge-base-id",
        default=None,
        help="Optional DataMate knowledge base id. If omitted, the collector tries to discover one.",
    )
    integrate_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=1.5,
        help="HTTP timeout for platform probes.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "process":
        result = run_data_processing_agent(args.input, args.output_root, task=args.task)
        print(
            f"Processed {len(result['documents'])} docs; "
            f"quality avg length {result['quality_report']['avg_length']}; "
            f"warnings {result['quality_report'].get('warning_count', 0)}."
        )
        print(
            f"Plan steps: {len(result['plan'])}; "
            f"state: {result['state']['status']}; "
            f"recovery actions: {result['state']['recovery_action_count']}."
        )
        print(f"Outputs written to {result['output_paths']['documents']}")
        return 0

    if args.command == "orchestrate":
        use_llm = None
        if args.use_llm and not args.no_llm:
            use_llm = True
        elif args.no_llm:
            use_llm = False
        result = run_orchestrator(
            input_path=args.input,
            output_root=args.output_root,
            goal=args.goal,
            questions=args.question or None,
            use_llm=use_llm,
            kg_backend=args.kg_backend,
            external_documents_path=args.external_documents,
            external_triples_path=args.external_triples,
            external_entities_path=args.external_entities,
            gold_triples_path=args.gold_triples,
        )
        metrics = result["metrics"]
        print(
            "Orchestration completed: "
            f"{metrics['document_count']} docs, "
            f"{metrics['entity_count']} entities, "
            f"{metrics['triple_count']} triples, "
            f"NL2SQL {metrics['nl2sql_accuracy']:.2%}."
        )
        print(f"Report written to {args.output_root / 'demo' / 'orchestration_report.md'}")
        return 0

    if args.command == "analyze":
        result = run_analysis_agent(args.output_root, nl2sql_eval_path=args.eval_path)
        print(
            f"Analysis completed: {result['analysis_summary']['query_count']} queries, "
            f"{result['analysis_summary']['subgraph_count']} subgraphs."
        )
        if "nl2sql_eval" in result:
            print(
                "NL2SQL eval: "
                f"{result['nl2sql_eval']['accuracy']:.2%} "
                f"({result['nl2sql_eval']['passed']}/{result['nl2sql_eval']['total']})."
            )
        print(f"Report written to {args.output_root / 'demo' / 'analysis_report.md'}")
        return 0

    if args.command == "insight":
        result = run_interactive_analysis(args.output_root, args.question)
        query = result.get("query") or {}
        chart = result.get("chart") or {}
        nl2sql_plan = result.get("nl2sql_plan") or {}
        sql_validation = nl2sql_plan.get("sql_validation") or {}
        print(
            "Insight completed: "
            f"query_id={query.get('query_id', '')}, "
            f"rows={result.get('row_count', 0)}, "
            f"chart={chart.get('chart_type', '')}, "
            f"sql_valid={sql_validation.get('valid', False)}."
        )
        print(result.get("explanation", ""))
        print(f"Latest insight written to {args.output_root / 'demo' / 'interactive_analysis_latest.json'}")
        return 0

    if args.command == "analyze-config":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_configured_analysis(args.output_root, config)
        summary = result["summary"]
        print(
            "Configured analysis completed: "
            f"analysis_id={result['analysis_id']}, "
            f"queries={summary['query_count']}, "
            f"charts={summary['chart_count']}, "
            f"graph_tools={summary['graph_tool_count']}."
        )
        print(f"Latest configured analysis written to {args.output_root / 'demo' / 'configured_analysis_latest.json'}")
        return 0

    if args.command == "nl2sql-eval":
        result = run_nl2sql_eval(args.output_root, eval_path=args.eval_path, threshold=args.threshold)
        print(
            "NL2SQL eval completed: "
            f"accuracy={result['accuracy']:.2%}, "
            f"passed={result['passed']}/{result['total']}, "
            f"meets_threshold={result['meets_threshold']}."
        )
        print(f"Report written to {args.output_root / 'demo' / 'nl2sql_eval_report.md'}")
        return 0

    if args.command == "dashboard":
        output_path = args.output
        if output_path is not None and not output_path.is_absolute():
            output_path = args.output_root / output_path
        target = export_dashboard_html(args.output_root, output_path)
        print(f"Dashboard written to {target}")
        return 0

    if args.command == "build":
        use_llm = None
        if args.use_llm and not args.no_llm:
            use_llm = True
        elif args.no_llm:
            use_llm = False
        result = build_medical_demo(
            input_path=args.input,
            output_root=args.output_root,
            questions=args.question or None,
            use_llm=use_llm,
            kg_backend=args.kg_backend,
            external_documents_path=args.external_documents,
            external_triples_path=args.external_triples,
            external_entities_path=args.external_entities,
            gold_triples_path=args.gold_triples,
        )
        print(
            f"Built demo artifacts: {len(result['documents'])} docs, "
            f"{len(result['entities'])} entities, {len(result['triples'])} triples."
        )
        print(
            "KG validation: "
            f"{result['kg_validation_report']['accepted_triples']} accepted, "
            f"{result['kg_validation_report']['rejected_triples']} rejected."
        )
        print(f"Subgraphs built: {len(result.get('subgraphs', []))}")
        external_summary = result.get("external_summary", {})
        if external_summary and any(
            external_summary.get(key, 0)
            for key in ("external_document_count", "external_triple_count", "external_entity_count")
        ):
            print(
                "External artifacts: "
                f"{external_summary.get('external_document_count', 0)} docs, "
                f"{external_summary.get('external_triple_count', 0)} triples, "
                f"{external_summary.get('external_entity_count', 0)} entities."
            )
        if "analysis_summary" in result:
            summary = result["analysis_summary"]
            print(
                "Analysis: "
                f"{summary['query_count']} queries, "
                f"{summary['subgraph_count']} subgraphs."
            )
            if "nl2sql_accuracy" in summary:
                print(f"NL2SQL eval: {summary['nl2sql_accuracy']:.2%}.")
        kg_quality_eval = result.get("kg_quality_eval", {})
        if kg_quality_eval.get("enabled"):
            overall = kg_quality_eval.get("overall", {})
            print(
                "KG gold eval: "
                f"P={overall.get('precision', 0):.2%}, "
                f"R={overall.get('recall', 0):.2%}, "
                f"F1={overall.get('f1', 0):.2%}."
            )
        kg_backend_probe = result.get("kg_backend_probe", {})
        if kg_backend_probe.get("enabled"):
            overall = kg_backend_probe.get("best_overall", {})
            print(
                "KG backend probe: "
                f"mode={kg_backend_probe.get('best_mode')}, "
                f"P={overall.get('precision', 0):.2%}, "
                f"R={overall.get('recall', 0):.2%}, "
                f"F1={overall.get('f1', 0):.2%}."
            )
        print(f"LLM rewriting: {'enabled' if result.get('llm_enabled') else 'disabled'}")
        print("Outputs written under data/, graph/, and demo/.")
        return 0

    if args.command == "ask":
        use_llm = None
        if args.use_llm and not args.no_llm:
            use_llm = True
        elif args.no_llm:
            use_llm = False
        trace = ask_medical_question(args.output_root, args.question, use_llm=use_llm)
        print(trace["answer"])
        if trace["supporting_triples"]:
            print("\n依据：")
            for idx, triple in enumerate(trace["supporting_triples"], start=1):
                print(
                    f"{idx}. {triple['subject']} -{triple['predicate']}-> {triple['object']}，来源 {triple['source_doc_id']}"
                )
        return 0

    if args.command == "integrate":
        result = run_platform_integration_agent(
            args.output_root,
            query=args.query,
            knowledge_base_id=args.knowledge_base_id,
            timeout_seconds=args.timeout_seconds,
        )
        summary = result["summary"]
        print(
            "Integration evidence collected: "
            f"DataMate reachable={summary['datamate_reachable']}, "
            f"Nexent reachable={summary['nexent_reachable']}, "
            f"datamate_search success={summary['datamate_search_success']}."
        )
        print(f"Report written to {args.output_root / 'demo' / 'integration' / 'integration_evidence_report.md'}")
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
