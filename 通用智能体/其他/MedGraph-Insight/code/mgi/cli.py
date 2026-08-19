"""MedGraph-Insight 命令行入口。

子命令：
  clean     任务一：数据处理智能体清洗语料
  build     任务二：知识图谱生成智能体 从语料构建图谱
  ask       任务二：知识图谱问答智能体
  analyze   任务三：数据分析智能体 生成 BI 可视化报告
  bench     算子性能评测（CPU/NPU 对比）
  eval-kg   NER/关系抽取 金标准 P/R/F1 评测
  eval-sql  NL2SQL 准确率评测
  dag       任务一：DAG 算子编排 + 数据质量指标 + 血缘追踪
  pipeline  端到端运行 任务一→二→三 全链路
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"


def _p(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_clean(args: argparse.Namespace) -> None:
    from .agents.data_agent import DataProcessingAgent

    agent = DataProcessingAgent()
    run = agent.process_file(args.task, args.input, args.output)
    _p(run.report())
    backend = "datamate-remote" if agent.client.online else "local-fallback"
    print(f"\n后端: {backend} | 输出 -> {args.output}")


def cmd_build(args: argparse.Namespace) -> None:
    from .agents.kg_agent import KGBuildAgent

    agent = KGBuildAgent(use_npu=args.npu)
    run = agent.build_from_corpus(args.corpus)
    _p(run.report())
    OUT.mkdir(exist_ok=True)
    agent.kg.save(args.graph)
    print(f"\n图谱统计: {json.dumps(agent.kg.stats(), ensure_ascii=False)}")
    print(f"图谱已保存 -> {args.graph}")


def cmd_ask(args: argparse.Namespace) -> None:
    from .agents.kg_agent import KGQAAgent
    from .kg import MedicalKG
    from .kg.graphrag import GraphRAG

    kg = MedicalKG.load(args.graph)
    ask = GraphRAG(kg).answer if args.rag else KGQAAgent(kg).answer
    if args.question:
        _p(ask(args.question))
    else:
        for q in ["T2DM有哪些症状？", "高血压吃什么药？", "coronary heart disease需要做哪些检查？",
                  "脑梗死怎么治疗？", "2型糖尿病的并发症有哪些症状？", "痛风好发于哪些人群？"]:
            r = ask(q)
            print(f"Q: {q}\nA: {r['answer']}\n")


def cmd_analyze(args: argparse.Namespace) -> None:
    from .agents.analysis_agent import AnalysisAgent

    OUT.mkdir(exist_ok=True)
    agent = AnalysisAgent(args.graph, args.db)
    run = agent.analyze(args.task, out_path=args.output)
    _p(run.report())
    if args.trace:
        Path(args.trace).write_text(
            json.dumps(run.report(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"执行轨迹 -> {args.trace}")
    print(f"\n报告 -> {run.result.get('report_path', args.output)}")


def cmd_bench(args: argparse.Namespace) -> None:
    from .benchmark import run_benchmark, to_markdown

    payload = run_benchmark(args.corpus, repeat=args.repeat, scale=args.scale)
    _p(payload)
    md = to_markdown(payload)
    print("\n" + md)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"Markdown -> {args.markdown}")


def cmd_dag(args: argparse.Namespace) -> None:
    """任务一：以 DAG 方式编排标准医疗文本 ETL，输出逐节点质量指标与血缘。"""
    from .datamate.dag import OperatorDAG

    records = []
    for ln in Path(args.input).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            records.append(json.loads(ln)["text"])

    dag = (OperatorDAG()
           .add("clean_html", "HtmlTagCleaner")
           .add("t2s", "TraditionalChineseCleaner", inputs=["clean_html"])
           .add("anon_phone", "PhoneAnonymizer", inputs=["t2s"])
           .add("anon_id", "IDCardAnonymizer", inputs=["anon_phone"])
           .add("ws", "WhitespaceNormalizer", inputs=["anon_id"])
           .add("dedup", "DedupFilter", inputs=["ws"])
           .add("minlen", "MinLengthFilter", inputs=["dedup"], min_len=10))
    run = dag.run(records)

    print(f"拓扑序: {' -> '.join(run['order'])}")
    print("\n逐节点执行与数据质量（DQ）:")
    for nr in run["results"]:
        print(f"  {nr.id}({nr.op}) {nr.rows_in}->{nr.rows_out}条 "
              f"{nr.elapsed_ms}ms | dup {nr.dq_in['dup_rate']}->{nr.dq_out['dup_rate']} "
              f"pii {nr.dq_in['pii_leak_rate']}->{nr.dq_out['pii_leak_rate']} "
              f"html {nr.dq_in['html_rate']}->{nr.dq_out['html_rate']}")
    print("\n血缘追踪:")
    for nid, chain in dag.lineage().items():
        print(f"  {nid} <- {' <- '.join(reversed(chain)) if chain else '(源)'}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in run["final"]:
                f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
        print(f"\n清洗结果 -> {args.output} ({len(run['final'])} 条)")


def cmd_eval_kg(args: argparse.Namespace) -> None:
    from .eval.runner import run_eval

    rep = run_eval(args.corpus, args.gold)
    ner, rel = rep["ner"], rep["re"]
    print(f"金标准文档数: {rep['docs']}")
    print(f"NER  实体级  P={ner['precision']:.3f} R={ner['recall']:.3f} "
          f"F1={ner['f1']:.3f}  (TP={ner['tp']} FP={ner['fp']} FN={ner['fn']})")
    print(f"RE   三元组级 P={rel['precision']:.3f} R={rel['recall']:.3f} "
          f"F1={rel['f1']:.3f}  (TP={rel['tp']} FP={rel['fp']} FN={rel['fn']})")


def cmd_eval_sql(args: argparse.Namespace) -> None:
    from .analysis.nl2sql import NL2SQL, eval_accuracy
    from .kg import MedicalKG

    OUT.mkdir(exist_ok=True)
    MedicalKG.load(args.graph).to_sqlite(args.db)
    report = eval_accuracy(NL2SQL(args.db), args.eval)
    _p({k: v for k, v in report.items() if k != "failures"})
    print(f"\nNL2SQL 准确率: {report['accuracy'] * 100:.1f}% "
          f"({report['correct']}/{report['total']})")
    if report["failures"]:
        print("失败样例:")
        _p(report["failures"])


def cmd_pipeline(args: argparse.Namespace) -> None:
    """端到端：清洗 → 建图 → 问答 → 分析报告 → 评测。"""
    from .agents.data_agent import DataProcessingAgent
    from .agents.kg_agent import KGBuildAgent, KGQAAgent
    from .agents.analysis_agent import AnalysisAgent
    from .analysis.nl2sql import NL2SQL, eval_accuracy

    OUT.mkdir(exist_ok=True)
    cleaned = str(OUT / "cleaned.jsonl")
    graph = str(OUT / "medical_kg.json")
    db = str(OUT / "medical_kg.sqlite")
    report = str(OUT / "analysis_report.html")

    print("=== 任务一：数据处理智能体 ===")
    d_agent = DataProcessingAgent()
    d_run = d_agent.process_file("清洗医疗文本：去HTML、繁简转换、脱敏、去重、过滤短文本",
                                 args.corpus, cleaned)
    for r in d_run.records:
        print(f"  [{r.status}] {r.step.tool}: {r.summary}")

    print("\n=== 任务二：知识图谱生成智能体 ===")
    kg_agent = KGBuildAgent(use_npu=args.npu)
    kg_run = kg_agent.build_from_corpus(args.corpus)
    for r in kg_run.records:
        print(f"  [{r.status}] {r.step.tool}: {r.summary}")
    kg_agent.kg.save(graph)

    print("\n=== 任务二：知识图谱问答 ===")
    qa = KGQAAgent(kg_agent.kg)
    for q in ["2型糖尿病有哪些症状？", "高血压吃什么药？", "痛风好发于哪些人群？"]:
        print(f"  Q: {q}\n  A: {qa.answer(q)['answer']}")

    print("\n=== 任务三：数据分析智能体 + BI 可视化 ===")
    a_agent = AnalysisAgent(graph, db)
    a_run = a_agent.analyze("分析2型糖尿病相关的疾病知识并生成报告", out_path=report)
    for r in a_run.records:
        print(f"  [{r.status}] {r.step.tool}: {r.summary}")

    print("\n=== NL2SQL 准确率评测 ===")
    ev = eval_accuracy(NL2SQL(db), str(DATA / "eval" / "nl2sql_eval.jsonl"))
    print(f"  准确率: {ev['accuracy'] * 100:.1f}% ({ev['correct']}/{ev['total']})")

    trace = str(OUT / "pipeline_trace.json")
    Path(trace).write_text(json.dumps(
        {"data_agent": d_run.report(), "kg_agent": kg_run.report(),
         "analysis_agent": a_run.report()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全链路智能体执行轨迹 -> {trace}")
    print(f"全链路完成。报告 -> {report}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mgi", description="MedGraph-Insight 数据→知识→洞察 智能体")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g_corpus = str(DATA / "corpus" / "medical_notes.jsonl")
    g_graph = str(OUT / "medical_kg.json")
    g_db = str(OUT / "medical_kg.sqlite")

    p = sub.add_parser("clean", help="任务一：数据处理智能体")
    p.add_argument("--task", default="清洗医疗文本：去HTML、繁简转换、脱敏、去重、过滤短文本")
    p.add_argument("--input", default=g_corpus)
    p.add_argument("--output", default=str(OUT / "cleaned.jsonl"))
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("build", help="任务二：知识图谱生成")
    p.add_argument("--corpus", default=g_corpus)
    p.add_argument("--graph", default=g_graph)
    p.add_argument("--npu", action="store_true", help="使用 NPU 张量化 NER 算子")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="任务二：知识图谱问答")
    p.add_argument("--graph", default=g_graph)
    p.add_argument("--question", default="")
    p.add_argument("--rag", action="store_true", help="启用 GraphRAG 语义检索问答（TF-IDF+子图证据+多跳）")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("analyze", help="任务三：数据分析 + BI 可视化")
    p.add_argument("--task", default="分析2型糖尿病相关的疾病知识并生成报告")
    p.add_argument("--graph", default=g_graph)
    p.add_argument("--db", default=g_db)
    p.add_argument("--output", default=str(OUT / "analysis_report.html"))
    p.add_argument("--trace", default=str(OUT / "analysis_trace.json"),
                   help="智能体规划/执行轨迹 JSON 输出路径（空字符串关闭）")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("bench", help="算子性能评测 CPU/NPU 对比")
    p.add_argument("--corpus", default=g_corpus)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--scale", type=int, default=20)
    p.add_argument("--markdown", default=str(OUT / "benchmark.md"))
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("dag", help="任务一：DAG 算子编排 + 数据质量 + 血缘")
    p.add_argument("--input", default=g_corpus)
    p.add_argument("--output", default=str(OUT / "cleaned_dag.jsonl"))
    p.set_defaults(func=cmd_dag)

    p = sub.add_parser("eval-kg", help="NER/关系抽取 金标准 P/R/F1 评测")
    p.add_argument("--corpus", default=g_corpus)
    p.add_argument("--gold", default=str(DATA / "eval" / "kg_gold.jsonl"))
    p.set_defaults(func=cmd_eval_kg)

    p = sub.add_parser("eval-sql", help="NL2SQL 准确率评测")
    p.add_argument("--graph", default=g_graph)
    p.add_argument("--db", default=g_db)
    p.add_argument("--eval", default=str(DATA / "eval" / "nl2sql_eval.jsonl"))
    p.set_defaults(func=cmd_eval_sql)

    p = sub.add_parser("pipeline", help="端到端全链路")
    p.add_argument("--corpus", default=g_corpus)
    p.add_argument("--npu", action="store_true")
    p.set_defaults(func=cmd_pipeline)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
