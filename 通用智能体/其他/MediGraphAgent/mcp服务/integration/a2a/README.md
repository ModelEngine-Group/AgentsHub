# A2A 协作智能体（MediGraphAnalyst）

把 MediGraph 的「医疗图谱问答 + 数据分析」能力暴露成一个 **A2A 智能体**，让 Nexent 的总控 Agent 通过 A2A 协议发现并协作调用（多智能体协作）。

## 起服务（宿主机）
```bash
cd CCF
pip install fastapi uvicorn        # 若未装
python integration/a2a/a2a_server.py     # 监听 0.0.0.0:8100
```
- Agent Card: `http://localhost:8100/.well-known/agent.json`
- JSON-RPC: `POST http://localhost:8100/`（`message/send`）

## 接入 Nexent
1. Nexent →「A2A Agent 发现」→「URL 发现」。
2. Agent Card URL 填：`http://host.docker.internal:8100/.well-known/agent.json`
   （Nexent 在 Docker 内，用 `host.docker.internal` 访问宿主机；纯本机则用 `127.0.0.1`）。
3. 点「发现」→「测试连接」→ 通过后保存。
4. 在总控/Supervisor 智能体里把 `MediGraphAnalyst` 加为协作子 Agent（managed_agents）。

## 行为
- 问题含"多少/平均/趋势/比例/排名/统计/人次"等 → 走 **AnalysisAgent**（NL2SQL/图谱分析，返回洞察 + 报告路径）。
- 其它医疗知识/关联问题 → 走 **QAAgent**（混合 GraphRAG，答案带三元组溯源）。
- 依赖 `outputs/graph.json`（先完成医疗知识图谱构建）；分析会用图谱派生的 SQLite。

## 说明
A2A JSON-RPC 字段较多，本服务是**最小可用实现**（card + message/send，返回 Task）。
若 Nexent「测试连接」报 schema 不符，按 `nexent/backend/consts/a2a_models.py` 调整 card/result 形状即可。
