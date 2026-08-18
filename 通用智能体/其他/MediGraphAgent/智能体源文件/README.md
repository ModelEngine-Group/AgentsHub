# Nexent 智能体模板与复现说明

本目录提供 4 套可导入 Nexent 的 AgentConfig，覆盖医疗数据处理、知识图谱问答、
图谱驱动分析和端到端协作总控。所有业务能力均通过宿主机
`mcp_server/server.py` 暴露的 `MediGraphOperators` 调用。

> **路径约定**：本文档中所有相对路径（`mcp_server/`、`scripts/`、`data/` 等）
> 均以本包的 **`mcp服务/` 目录**为根，执行命令前请先 `cd mcp服务`。
>
> 其中涉及 **0.8B 微调编排模型**（`scripts/start_finetuned_model.ps1`）与
> `finetune/README.md` 的部分属于**可选增强**：模型权重约 320 MB，未随本包提供，
> 不影响四个智能体的任何核心功能——DAG 规划会自动改用所配置的大语言模型。
> 需要该模型时见完整参赛仓库。
>
> 快速上手请先读上一级目录的 [README.md](../README.md)。

## 1. 模板与工具

| 模板 | 智能体 | 最大步骤 | 核心工具 |
| --- | --- | ---: | --- |
| `dataproc_agent.json` | 医疗数据处理智能体 | 15 | `text_clean`、`chunker`、`medical_ner`、`medical_re`、`triple_validator`、`plan_datamate_pipeline`、`run_datamate_pipeline`、`inspect_extraction_models` |
| `kg_qa_agent.json` | 医疗知识图谱生成与问答智能体 | 4 | `build_medical_kg`、`medical_kg_qa`、`inspect_medical_kg`、`inspect_extraction_models`、NER/RE/校验工具 |
| `analysis_supervisor.json` | 图谱驱动医疗数据分析与 BI 智能体 | 4 | `analyze_medical_data`、`inspect_analysis_assets` |
| `collaboration_supervisor.json` | MediGraph 医疗数据闭环协作总控智能体 | 8 | DataMate 规划/执行、模型审计、建图、GraphRAG、图谱审计、SQL/GRAPH 分析 |

总控模板还包含 A2A Agent Card：

```text
http://host.docker.internal:8100/.well-known/agent.json
```

用于发现可选协作子智能体 `MediGraphAnalyst`。

## 2. 配置边界

- MCP 名称固定为 `MediGraphOperators`；
- MCP URL 固定为 `http://host.docker.internal:8011/sse`；
- 模板默认模型名为 `deepseek-ai/DeepSeek-V4-Pro`；
- `agent_id`、`model_id`、租户 ID 和导入后的版本号由目标 Nexent 环境重新分配，
  不应把本机数字当成跨环境固定值；
- `managed_agents` 默认留空，避免导入时引用其他环境不存在的本地智能体 ID；
- A2A 是可选增强；不启动 A2A 时，所有核心 MCP 工具仍可独立运行。

## 3. 前置服务

在项目根目录检查当前状态：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_demo_services.ps1
```

### 3.1 MCP（必需）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_mcp.ps1
```

Nexent Docker 内地址：

```text
http://host.docker.internal:8011/sse
```

### 3.2 DataMate（执行数据流水线时必需）

`run_datamate_pipeline` 依赖本机 DataMate 容器和已上传的五个自定义算子：

```powershell
python datamate_ops/build_zip.py
python integration/datamate/upload_operators.py
```

地址：

- DataMate 前端：`http://localhost:30000`
- DataMate API：`http://localhost:8080/api`

只运行图谱问答或 BI 分析时，不要求重新执行 DataMate 任务。

### 3.3 0.8B 编排模型（可选）

只有展示“小模型自然语言 → DataMate DAG”能力时才需要：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_finetuned_model.ps1
```

服务地址：

```text
http://host.docker.internal:18088/v1/
```

模型训练、评测和 Nexent 配置见 `finetune/README.md`。当前 held-out 112 条实测：

- DAG 准确率：97.3%
- 可执行率：100%

### 3.4 A2A（可选）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_a2a.ps1
```

用于总控智能体发现和调用 `MediGraphAnalyst`。

## 4. Nexent 导入流程

1. 在 Nexent「MCP 服务器」中新建 `MediGraphOperators`，填写 MCP URL；
2. 在「模型管理」确认 `deepseek-ai/DeepSeek-V4-Pro` 或其他兼容模型可用；
3. 按需导入本目录的 JSON 模板；
4. 导入后重新选择当前租户可用模型；
5. 检查工具是否完整显示；
6. 如需 A2A，在「A2A Agent 发现」中使用 Agent Card URL 发现
   `MediGraphAnalyst`；
7. 保存并发布智能体版本。

重点验收：

- `inspect_extraction_models` 能返回 neural GPLinker 路由与 CMeIE 指标；
- `build_medical_kg` 能生成独立 JSON/HTML 图谱；
- `medical_kg_qa` 返回三元组、置信度和来源；
- `inspect_analysis_assets` 返回表规模和 NL2SQL 双库指标；
- `analyze_medical_data` 返回 SQL/GRAPH 路由、数据、血缘和 HTML 报告。

## 5. 推荐验证提示词

2026-08-02 起，三个专才 agent 的 `duty_prompt` 已改为按用户意图（关键词/语义）路由，
不再要求提示词里带精确的文件名或工具名——下面两组写法都实测可用，自然口语化的写法
优先推荐（更接近真实用户会怎么问），括号内是这条提示词主要考验的路由分支。

### 数据处理（agent 2）

自然口语化（推荐）：

```text
我们最近收治的一批病历还没处理，能不能先帮我规划一下清洗、抽取这些病历需要走哪些步骤？先不用真的执行。
```

原始写法（同样可用，更明确指定范围）：

```text
先规划“清洗、切块、实体抽取、关系抽取、三元组校验”的 DataMate DAG，
说明每个算子的依赖关系，不要立即执行。
```

> 注意：DAG 规划依赖本机 0.8B 微调编排模型服务（`scripts/start_finetuned_model.ps1`），
> 服务未启动时 `planner_dag`/`selected_ops` 会返回空、并提示网络不可达——录制/验收前请
> 确认该服务已启动。

### 图谱问答（agent 3）

自然口语化（推荐，`graph_name` 默认落到主图 `graph.json`，不需要在提问里提到任何文件名）：

```text
高血压有哪些常见并发症？请说明依据。
```

针对指定案例图谱（如需要）：

```text
基于 new_case_cardiometabolic_20260701.json 回答：
高血压可能并发哪些疾病？列出证据三元组、confidence 和 source。
```

### 图谱驱动分析（agent 4）

自然口语化（推荐）：

```text
帮我看看目前接诊病人数最多的是哪个科室？给出具体数据和查询依据。
```

原始写法：

```text
先调用 inspect_analysis_assets，再回答“哪个科室接诊的不同病人数量最多？”。
给出实际 SQL、查询结果、数据血缘和 HTML 报告路径。
```

## 6. 最新端到端验收案例

输入位于：

```text
data/demo_cases/cardiometabolic_20260701
```

重新创建真实 Nexent 工具会话：

```powershell
.\.venv-neural\Scripts\python.exe -X utf8 integration/nexent/run_new_case.py
```

> 注意：仓库里**不再提供**任何"把预先写好的答案直接注入会话记录、伪装成实时问答"的脚本
> （曾经的 `create_recording_conversation.py` 已删除）——录屏演示只应使用 `run_new_case.py`
> 触发的真实工具执行会话，或直接在 Nexent 网页端手动提问、实时录屏，避免任何形式的
> "虚假演示作品效果"。

当前已经完成：

- 真实工具执行会话：28
- Nexent 独立图谱：29 个实体、37 条有效三元组、校验通过率 100%
- 路由：`ner:L1_neural`、`re:L1_neural`
- NL2SQL 人工集与 128 题压力集：双数据库执行准确率均为 100%

核心证据：

```text
outputs/nexent_new_case_answer_qa.md
outputs/new_case_cardiometabolic_20260701.json
outputs/new_case_cardiometabolic_20260701.html
docs/EVIDENCE_MAP.md
```

`run_new_case.py` 会生成 `outputs/nexent_new_case_stream_*.txt` 原始 SSE 调试流；
它们不是核心结果，完成排错后可以删除。

## 7. 停止服务

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_demo_services.ps1
```

该脚本只停止本项目的 MCP、A2A 和 0.8B 模型服务，不停止 Nexent/DataMate Docker。

## 8. 常见问题

### 导入后工具为空

确认 MCP 服务已启动，并从 Nexent 容器访问
`http://host.docker.internal:8011/sse`，不要填写 `localhost:8011`。

### 导入后模型不可用

模板中的模型 ID 仅是导出环境记录。重新选择目标租户已配置的模型并发布新版本。

### DataMate 调用失败

确认 DataMate 容器、API 网关和五个自定义算子均可用，并检查
`outputs/datamate_operator_ids.json`。

### 页面看不到 `run_new_case.py` 创建的会话

先登录创建该会话的 Nexent 用户，再打开脚本打印出的 `conversationId=...` 对应 URL。不同数据库或
租户不会共享本机历史会话。
