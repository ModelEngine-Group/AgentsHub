#!/usr/bin/env python3
"""MedGraph-Insight MCP 服务。

把「数据处理 → 医疗知识图谱 → 分析洞察」全链路能力暴露为 MCP 工具，
供 Nexent 平台（智能体开发 → 选择工具 → MCP 配置）或任意 MCP 客户端调用。

启动方式
--------
SSE（Nexent 平台接入用，MCP URL = http://<host>:<port>/sse）::

    python mgi_mcp_server.py --transport sse --host 0.0.0.0 --port 8765

stdio（Claude Desktop / MCP Inspector 等本地客户端）::

    python mgi_mcp_server.py

依赖：mcp>=1.2，以及 ../code 中 MedGraph-Insight 自身的依赖
（networkx / numpy / pandas / matplotlib / requests）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# 路径准备：让本服务可以直接 import 归档目录中的 mgi 包，无需 pip install
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

DATA = CODE / "data"
OUTPUT = CODE / "output"
CORPUS = DATA / "corpus" / "medical_notes.jsonl"
GRAPH_PATH = OUTPUT / "medical_kg.json"
DB_PATH = OUTPUT / "medical_kg.sqlite"

mcp = FastMCP(
    "medgraph-insight",
    instructions=(
        "医疗领域「数据→知识→洞察」智能体工具集：清洗医疗文本、构建医疗知识图谱、"
        "基于图谱做检索问答（GraphRAG，支持多跳）、用自然语言查询图谱关系数据（NL2SQL）、"
        "生成图文 BI 分析报告。所有工具返回 JSON 字符串。"
    ),
)

_STATE: dict[str, Any] = {"kg": None}


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _load_kg(rebuild: bool = False):
    """获取医疗知识图谱：优先复用内存缓存，其次磁盘，最后从语料重建。"""
    from mgi.kg import MedicalKG

    if not rebuild and _STATE["kg"] is not None:
        return _STATE["kg"]
    if not rebuild and GRAPH_PATH.exists():
        _STATE["kg"] = MedicalKG.load(str(GRAPH_PATH))
        return _STATE["kg"]

    from mgi.agents.kg_agent import KGBuildAgent

    agent = KGBuildAgent()
    agent.build_from_corpus(str(CORPUS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    agent.kg.save(str(GRAPH_PATH))
    _STATE["kg"] = agent.kg
    return _STATE["kg"]


def _ensure_db() -> str:
    """确保 NL2SQL 使用的 SQLite 视图与当前图谱一致。"""
    kg = _load_kg()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    kg.to_sqlite(str(DB_PATH))
    return str(DB_PATH)


# ---------------------------------------------------------------------------
# 任务一：数据处理
# ---------------------------------------------------------------------------
@mcp.tool()
def clean_medical_texts(texts: list[str]) -> str:
    """清洗医疗文本：去 HTML 标签、繁体转简体、手机号/身份证脱敏、空白归一、去重、过滤过短文本。

    以 DAG 方式编排 7 个算子，返回清洗结果、拓扑序、逐节点数据质量指标（重复率/PII 泄漏率/
    HTML 残留率）与血缘链路，可用于审计每一步对数据质量的改善。

    Args:
        texts: 待清洗的医疗文本列表。
    """
    from mgi.datamate.dag import OperatorDAG

    if not texts:
        return _dumps({"error": "texts 不能为空"})

    dag = (OperatorDAG()
           .add("clean_html", "HtmlTagCleaner")
           .add("t2s", "TraditionalChineseCleaner", inputs=["clean_html"])
           .add("anon_phone", "PhoneAnonymizer", inputs=["t2s"])
           .add("anon_id", "IDCardAnonymizer", inputs=["anon_phone"])
           .add("ws", "WhitespaceNormalizer", inputs=["anon_id"])
           .add("dedup", "DedupFilter", inputs=["ws"])
           .add("minlen", "MinLengthFilter", inputs=["dedup"], min_len=10))
    run = dag.run(list(texts))

    return _dumps({
        "topological_order": run["order"],
        "input_count": len(texts),
        "output_count": len(run["final"]),
        "cleaned": run["final"],
        "nodes": [
            {
                "id": nr.id, "operator": nr.op,
                "rows": f"{nr.rows_in} -> {nr.rows_out}", "elapsed_ms": nr.elapsed_ms,
                "dup_rate": f"{nr.dq_in['dup_rate']} -> {nr.dq_out['dup_rate']}",
                "pii_leak_rate": f"{nr.dq_in['pii_leak_rate']} -> {nr.dq_out['pii_leak_rate']}",
                "html_rate": f"{nr.dq_in['html_rate']} -> {nr.dq_out['html_rate']}",
            }
            for nr in run["results"]
        ],
        "lineage": dag.lineage(),
    })


# ---------------------------------------------------------------------------
# 任务二：知识图谱构建与问答
# ---------------------------------------------------------------------------
@mcp.tool()
def build_medical_kg(rebuild: bool = False) -> str:
    """从内置医疗语料构建医疗知识图谱（实体识别 → 关系抽取 → 三元组构建校验）。

    Args:
        rebuild: True 表示忽略已有图谱强制重建；默认复用缓存。
    """
    kg = _load_kg(rebuild=rebuild)
    return _dumps({
        "graph_path": str(GRAPH_PATH),
        "stats": kg.stats(),
        "rebuilt": bool(rebuild),
    })


@mcp.tool()
def ask_medical_kg(question: str, use_graphrag: bool = True) -> str:
    """基于医疗知识图谱回答自然语言问题，返回答案与图谱证据。

    支持疾病别名/英文/缩写链接（如 T2DM → 2型糖尿病），GraphRAG 模式下支持多跳推理
    （如「心力衰竭的并发症用什么药」沿并发症链二跳汇聚）。

    Args:
        question: 自然语言问题，例如「2型糖尿病有哪些症状？」。
        use_graphrag: True 使用 GraphRAG 语义检索（默认）；False 使用规则式图谱问答。
    """
    if not question.strip():
        return _dumps({"error": "question 不能为空"})

    kg = _load_kg()
    if use_graphrag:
        from mgi.kg.graphrag import GraphRAG

        result = GraphRAG(kg).answer(question)
    else:
        from mgi.agents.kg_agent import KGQAAgent

        result = KGQAAgent(kg).answer(question)
    return _dumps({"question": question,
                   "mode": "graphrag" if use_graphrag else "rule",
                   **result})


@mcp.tool()
def kg_overview() -> str:
    """查看医疗知识图谱概览：实体总数、关系总数、按类型分布。"""
    return _dumps({"graph_path": str(GRAPH_PATH), "stats": _load_kg().stats()})


# ---------------------------------------------------------------------------
# 任务三：分析与洞察
# ---------------------------------------------------------------------------
@mcp.tool()
def query_kg_with_nl2sql(question: str) -> str:
    """把自然语言问题翻译为 SQL 并在图谱关系库上执行，返回 SQL 与结果集。

    支持别名链接、多条件交集（INTERSECT）、聚合 + HAVING、JOIN、置信度过滤与反向查询。

    Args:
        question: 例如「既有胸痛又有心悸的疾病」「症状超过4个的疾病」。
    """
    if not question.strip():
        return _dumps({"error": "question 不能为空"})

    from mgi.analysis.nl2sql import NL2SQL

    engine = NL2SQL(_ensure_db())
    result = engine.query(question)
    return _dumps({"question": question, "db_path": str(DB_PATH), **result})


@mcp.tool()
def generate_bi_report(task: str = "分析2型糖尿病相关的疾病知识并生成报告") -> str:
    """生成图文 BI 分析报告（自包含 HTML）：分布图、覆盖热力图、TopN 排行、关系网络与子图。

    Args:
        task: 分析任务描述，例如「分析高血压相关的疾病知识并生成报告」。
    """
    from mgi.agents.analysis_agent import AnalysisAgent

    _load_kg()
    db = _ensure_db()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "analysis_report.html"

    agent = AnalysisAgent(str(GRAPH_PATH), db)
    run = agent.analyze(task, out_path=str(report_path))
    report = run.report()
    return _dumps({
        "task": task,
        "report_path": str(report_path),
        "planner": report.get("planner"),
        "steps": [
            {"tool": s.get("tool"), "status": s.get("status"),
             "elapsed_ms": s.get("elapsed_ms"), "summary": s.get("summary")}
            for s in report.get("steps", [])
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="MedGraph-Insight MCP 服务")
    parser.add_argument("--transport", default="stdio",
                        choices=["stdio", "sse", "streamable-http"],
                        help="传输方式；Nexent 平台接入请使用 sse")
    parser.add_argument("--host", default="127.0.0.1", help="SSE/HTTP 监听地址")
    parser.add_argument("--port", type=int, default=8765, help="SSE/HTTP 监听端口")
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    if args.transport != "stdio":
        print(f"MCP URL: http://{args.host}:{args.port}/sse", file=sys.stderr)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
