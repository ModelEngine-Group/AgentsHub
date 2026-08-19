# ModelEngineBySoldierBoy

ModelEngineBySoldierBoy 是基于 DataMate 与 Nexent 构建的医疗“数据 → 知识 → 洞察”多智能体链路。

| 顺序 | 智能体 | 核心能力 |
| ---: | --- | --- |
| 1 | DataProcessingAgent | 结构化/非结构化医疗数据接入、清洗规划、质量检查与异常恢复 |
| 2 | MedicalKGAgent | 实体关系抽取、三元组校验、知识图谱构建与证据化问答 |
| 3 | InsightAnalysisAgent | 图谱理解、NL2SQL、统计/关联分析、BI 图表与图谱洞察 |

本项目用于技术演示和研究，不提供医疗诊断、治疗或临床决策建议。

## 归档内容

```text
.
├── agent.json                         # Nexent 官方导出的根 Agent（含三个子 Agent）
├── config/                            # 环境模板与三个 OpenAPI 配置
├── knowledge/                         # DataMate 知识库导入材料
├── mcps/modelengine-medical-agent-service/
│   ├── agents/                        # 三个智能体及平台编排代码
│   ├── operators/                     # 数据、KG、分析和可选 tensor 算子
│   ├── demo/                          # CLI、HTTP API、OpenAPI 与 Dashboard
│   ├── data/                          # 合成样例和必要的转换产物
│   └── graph/                         # 图谱 schema 与运行产物目录
├── skills/                            # 项目开发维护 Skill 及可上传 zip
├── evidence/                          # 精简项目书和验收材料
└── LICENSE
```

`agent.json` 由 Nexent 官方 `/agent/export` 接口导出，包含
`ModelEngineBySoldierBoy` 根 Agent 和任务一、二、三三个子 Agent。导出文件中的
源环境 Agent/模型/工具 ID 仅用于描述关系；导入新环境后应重新选择模型并检查工具绑定。

当前运行链路不依赖自定义 Nexent Skill。`skills/` 中的 Skill 用于开发维护和复现指导，
不是基础运行的前置依赖。

## 1. 快速离线复现

离线规则链路只需要 Python 3.10 或更高版本，不需要 LLM、DataMate、Nexent、NPU 或
第三方 Python 包。

```bash
cd mcps/modelengine-medical-agent-service
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

TASK3_ANALYSIS_ENABLE_TENSOR=false MEDICAL_DEMO_USE_LLM=false \
python -m demo.run_medical_demo orchestrate \
  --output-root . \
  --input data/raw/medical_demo.jsonl \
  --no-llm \
  --kg-backend rule
```

该命令依次运行：

```text
DataProcessingAgent -> MedicalKGAgent -> InsightAnalysisAgent
```

当前归档已用上述规则链路验证通过。它会生成处理数据、实体、三元组、分析报告、图表和
编排 trace；这验证的是本地确定性链路，不代表 DataMate/Nexent 平台服务已经启动。

## 2. 启动 OpenAPI/MCP 服务

仍在 `mcps/modelengine-medical-agent-service` 目录执行：

```bash
python -m demo.app_server --host 0.0.0.0 --port 8088
```

可访问：

- 健康检查：`http://127.0.0.1:8088/health`
- Dashboard：`http://127.0.0.1:8088/dashboard`
- 任务一 OpenAPI：`http://127.0.0.1:8088/api/task1/openapi.json`
- 任务二 OpenAPI：`http://127.0.0.1:8088/api/task2/openapi.json`
- 任务三 OpenAPI：`http://127.0.0.1:8088/api/task3/openapi.json`

`/api/datamate-process` 需要可访问的 DataMate；KG、分析和 OpenAPI 查看可先使用离线规则链路。

## 3. 部署 DataMate 与 Nexent

完整平台链路需要 Linux/WSL、Docker Engine、Docker Compose v2、DataMate、Nexent，
以及一个已在 Nexent 中验证可用的对话模型。请优先参考两个上游项目的安装文档，也可将
源码克隆到任意目录：

```bash
git clone https://gitlink.org.cn/modelengine-group/DataMate.git /absolute/path/to/DataMate
git clone https://gitlink.org.cn/modelengine-group/nexent.git /absolute/path/to/nexent
```

本归档最近验证的源码版本为 DataMate `25bf497`、Nexent `7d811c6`。复制配置模板并替换
所有占位值，切勿提交真实密钥：

在归档根目录执行：

```bash
cp config/.env.example mcps/modelengine-medical-agent-service/.env
```

至少配置：

```text
DATAMATE_HOME=/absolute/path/to/DataMate
NEXENT_HOME=/absolute/path/to/nexent
DATAMATE_KNOWLEDGE_BASE_ID=导入后获得的新ID
OPENAI_API_BASE=你的模型服务地址
OPENAI_API_KEY=你的模型密钥
LLM_MODEL=目标模型名称
```

默认宿主机端口为 DataMate UI `30000`、DataMate API `8080`、Nexent UI `3000`、
Nexent Config API `5010`、Nexent Runtime API `5014`、本项目 API `8088`。

## 4. 导入知识库

1. 登录 DataMate，创建一个 DOCUMENT 类型知识库。
2. 上传 `knowledge/medical_kg_knowledge.md`，等待解析和索引完成。
3. 记录目标环境生成的新知识库 ID，不要沿用历史证据里的 UUID。
4. 在 Nexent 中配置 DataMate 地址并同步该知识库。
5. 为需要检索的 Agent 启用 `datamate_search`。

`knowledge/medical_kg_documents.jsonl` 是结构化审阅/备用导入文件；若目标 DataMate 不支持
JSONL，上传 Markdown 文件即可。

## 5. 导入 OpenAPI/MCP 与 Agent

先确保 Nexent 容器能够访问本项目 API。三个配置默认使用：

```text
http://host.docker.internal:8088
```

如果该地址在目标 Docker 环境不可达，请将 `config/task*_nexent_openapi.json` 中的
`servers[0].url` 改为容器实际可访问的宿主机地址。

按以下顺序操作：

1. 在 Nexent 中添加并验证对话模型。
2. 依次导入 `config/task1_nexent_openapi.json`、`task2_nexent_openapi.json`、
   `task3_nexent_openapi.json`。
3. 配置并同步上一步创建的 DataMate 知识库。
4. 在“智能体开发 → 导入”中导入根目录 `agent.json`。
5. 重新选择目标环境模型，并确认三个子 Agent 的六个 OpenAPI/MCP 工具均已绑定。

应看到的工具为：

```text
modelengine_task1_data_processing_agent_run_datamate_processing_agent
modelengine_task2_medical_kg_agent_build_medical_kg_agent
modelengine_task2_medical_kg_agent_ask_medical_kg_agent
modelengine_task3_analysis_agent_run_analysis_agent
modelengine_task3_analysis_agent_run_interactive_analysis
modelengine_task3_analysis_agent_run_configured_analysis
```

## 6. 使用示例

- 任务一：`使用 DataMate 处理 data/raw/medical_demo.jsonl，完成清洗、质量检查和异常恢复。`
- 任务二：`基于 data/raw/medical_demo.jsonl 构建医疗知识图谱并说明校验结果。`
- 任务三：`统计每种疾病的并发症数量，生成图表并说明主要洞察。`
- 根 Agent：`对医疗样例完成清洗、构建知识图谱并分析疾病并发症。`

推荐顺序仍为任务一 → 任务二 → 任务三，因为后续 Agent 会复用前序产物。

## 7. 可选增强

- `requirements-optional-analytics.txt`：tensor/分析和策略模型依赖。
- `requirements-optional-model.txt`：LoRA/Transformer 训练与推理依赖。
- Ascend NPU 需要与环境匹配的 `torch_npu`，不属于可移植基础依赖。

基础规则链路不依赖大型模型权重。本归档未包含训练大数据集、模型权重或原始 CBLUE 大包。

## 8. 数据、许可与安全

- 项目源码采用 MIT License，详见 `LICENSE`。
- `data/raw/medical_demo*` 是合成演示数据，不包含真实患者身份信息。
- 知识库及转换产物含 CBLUE 衍生内容；第三方数据仍受其原始许可和使用条款约束。
- 使用真实医疗数据前，使用者须自行完成授权、脱敏、隐私保护和合规审查。
- 不要提交 `.env`、API Key、Token、患者数据或源环境私有地址。

原始开发仓库：<https://gitlink.org.cn/ohanabi/ModelEngineBySoldierBoy>
