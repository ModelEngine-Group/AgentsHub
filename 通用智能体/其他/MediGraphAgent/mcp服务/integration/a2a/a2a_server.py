"""A2A server exposing MediGraph's QA + Analysis agents to Nexent.

Nexent can discover this agent via "A2A Agent 发现 -> URL 发现" using the Agent
Card URL, then collaborate with it as a managed sub-agent of a supervisor.

Implements the minimal A2A surface Nexent needs:
  - GET /.well-known/agent.json   -> Agent Card (A2AAgentCard schema)
  - POST /                        -> JSON-RPC 2.0 `message/send`

The agent routes a question to AnalysisAgent (statistics/trend/ranking) or
QAAgent (knowledge/association), reusing the local medigraph implementations.

Run (host):  python integration/a2a/a2a_server.py   # serves on :8100
Nexent(Docker) discovers: http://host.docker.internal:8100/.well-known/agent.json

NOTE: A2A JSON-RPC has many optional fields; this is a faithful minimal server.
If Nexent's "测试连接" reports a schema mismatch, adjust the card/result shape per
nexent/backend/consts/a2a_models.py and a2a_agent_adapter.py.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config.settings import OUTPUTS_DIR

HOST_FOR_NEXENT = os.getenv(
    "MEDIGRAPH_A2A_PUBLIC_URL",
    "http://host.docker.internal:8100",
).rstrip("/")
app = FastAPI(title="MediGraphAnalyst A2A")


def _agent_card() -> dict:
    return {
        "name": "MediGraphAnalyst",
        "description": "医疗知识图谱问答与数据分析智能体：基于已构建的医疗图谱做带溯源的问答，"
                       "以及统计/趋势/关联分析，复用 MediGraph 的 QA 与 Analysis 能力。",
        "version": "1.0.0",
        "url": f"{HOST_FOR_NEXENT}/",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "supportedTransportTypes": ["http-json-rpc"],
            "protocolVersion": "1.0",
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {"id": "medical_qa", "name": "医疗知识图谱问答",
             "description": "基于医疗知识图谱的混合检索问答，答案附三元组溯源。",
             "tags": ["medical", "graphrag", "qa"], "examples": ["嗜铬细胞瘤有哪些阳性标志物？"]},
            {"id": "medical_analysis", "name": "医疗数据分析",
             "description": "统计/趋势/关联分析（NL2SQL 或图谱遍历），返回数据与洞察。",
             "tags": ["medical", "analytics", "nl2sql"], "examples": ["各疾病的就诊人次是多少？"]},
        ],
        "supportedInterfaces": [
            {"protocolBinding": "http-json-rpc", "url": f"{HOST_FOR_NEXENT}/", "protocolVersion": "1.0"},
        ],
    }


_ANALYTIC_HINTS = ("多少", "数量", "平均", "比例", "占比", "趋势", "排名", "最多", "统计", "分布", "每月", "人次", "总")


def _route_and_answer(question: str) -> str:
    """Route to AnalysisAgent (analytic) or QAAgent (knowledge) and return text."""
    from medigraph.graph.local_store import LocalGraphStore
    graph_json = OUTPUTS_DIR / "graph.json"

    if any(h in question for h in _ANALYTIC_HINTS):
        from medigraph.analysis.analysis_agent import AnalysisAgent
        from medigraph.analysis.graph_profile import load_graph
        from medigraph.analysis.relational import build_db
        db = OUTPUTS_DIR / "analytics.db"
        store, _ = load_graph(graph_json if graph_json.exists() else None)
        build_db(db, store, n_visits=600, seed=42)
        agent = AnalysisAgent(str(db), graph_json=str(graph_json) if graph_json.exists() else None)
        res = agent.analyze(question, out_html=OUTPUTS_DIR / "a2a_analysis.html", verbose=False)
        return f"[分析/{res['route']}] {res['insight']}\n报告: {res['html']}"

    if not graph_json.exists():
        return "知识图谱尚未构建（缺 outputs/graph.json）。请先建图再问答。"
    from medigraph.agents.qa_agent import QAAgent
    store = LocalGraphStore.load_json(graph_json)
    res = QAAgent(store=store).answer(question)
    return res["answer"]


def _extract_question(params: dict) -> str:
    msg = (params or {}).get("message", {}) or {}
    parts = msg.get("parts", []) or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    return "\n".join(texts).strip()


@app.get("/.well-known/agent.json")
def agent_card():
    return JSONResponse(_agent_card())


@app.get("/.well-known/agent-card.json")
def agent_card_alt():
    return JSONResponse(_agent_card())


@app.post("/")
@app.post("/v1")
@app.post("/message:send")
async def jsonrpc(request: Request):
    body = await request.json()
    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {}) or {}

    # Nexent's A2A client appends /v1 for JSONRPC and uses method "SendMessage",
    # while some A2A examples use "message/send" on the base URL. Accept both so
    # discovery, direct chat and managed-agent delegation all work.
    if method not in ("message/send", "message/stream", "tasks/send", "SendMessage"):
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id,
                             "error": {"code": -32601, "message": f"method not found: {method}"}})
    question = _extract_question(params)
    try:
        answer = _route_and_answer(question) if question else "未收到问题文本。"
    except Exception as exc:  # noqa: BLE001
        answer = f"处理失败: {exc}"

    now = datetime.now(timezone.utc).isoformat()
    task = {
        "id": str(uuid.uuid4()),
        "status": {
            "state": "TASK_STATE_COMPLETED",
            "timestamp": now,
            "message": {"role": "ROLE_AGENT", "parts": [{"text": answer, "mediaType": "text/plain"}]},
        },
        "artifacts": [],
    }
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": {"task": task}})


if __name__ == "__main__":
    import uvicorn
    print("MediGraphAnalyst A2A on http://0.0.0.0:8100  (card: /.well-known/agent.json)")
    uvicorn.run(app, host="0.0.0.0", port=8100)
