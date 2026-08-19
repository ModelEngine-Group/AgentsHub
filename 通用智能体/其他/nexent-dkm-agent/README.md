# nexent-dkm-agent

## 简介

**nexent-dkm-agent** 是 ModelEngine 开源项目贡献赛作品（队伍：**龙卷风摧毁停车场**），基于 [Nexent](https://gitcode.com/ModelEngine/nexent) 与 DataMate，实现「数据清洗 → 医疗知识图谱 → 图谱驱动分析与可视化」端到端闭环。

源仓库（`final-version`）：[hybdaxia/nexent-dkm-agent](https://gitlink.org.cn/hybdaxia/nexent-dkm-agent.git)

## 核心功能

| 模块 | 能力 |
| --- | --- |
| 任务一 数据处理 | 自然语言理解、数据画像、自动规划、本地清洗、DataMate dry_run/submit |
| 任务二 医疗知识图谱 | 实体/关系抽取、三元组校验、图谱 JSON、KG 问答、可选 Neo4j 持久化 |
| 任务三 图谱分析 | NL2SQL、统计/关联/趋势、中心性与社区、ECharts 交互仪表盘 |
| Nexent 集成 | 三套 OpenAPI 注册为 MCP 工具，创建 `dkm_end_to_end_agent` 编排 Agent |

## 目录结构

```text
nexent-dkm-agent/
├── agent.json                 # Nexent 智能体描述（可导入）
├── README.md                  # 本文件
├── mcps/                      # 三套任务 OpenAPI / MCP 契约
├── knowledge/                 # 知识库与样例数据入口
├── configs/                   # 任务配置模板
├── src/                       # Agent / Operator / Pipeline 源码
├── demos/                     # 可运行演示与在线集成入口
├── data/samples/              # 样例语料与术语表
├── docs/                      # 架构、依赖、在线集成说明
├── requirements.txt           # 主依赖
├── docker-compose.neo4j.yml   # 可选 Neo4j
└── ...
```

## 运行环境依赖

- Python 3.12+
- （可选）Docker：Nexent、DataMate、Neo4j
- （可选）LLM API（DeepSeek 等）用于规划 / NL2SQL 增强
- （可选）Ascend NPU：见 `requirements-npu.txt`

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # 测试可选
```

复制 `.env.example` 为本地 `.env` / `.local/` 配置（勿提交真实密钥）。

## 启动命令

### 离线最小复现（不依赖外部服务）

```powershell
python demos/task1_demo.py --input data/samples/task1_patients.csv --output-dir outputs/repro/task1
python demos/task2_demo.py --input data/samples/task2_medical_notes.txt --output-dir outputs/repro/task2 --question "高血压有哪些症状和用药？"
python demos/task3_demo.py --graph-file outputs/repro/task2/medical_kg.json --output-dir outputs/repro/task3 --question "哪些疾病关联最多症状？"
python demos/end_to_end_demo.py --output-root outputs/repro/end_to_end
```

### 启动三套任务 API（供 Nexent MCP 调用）

```bash
python demos/task1_demo.py --serve --host 0.0.0.0 --port 8000
python demos/task2_demo.py --serve --host 0.0.0.0 --port 8002
python demos/task3_demo.py --serve --host 0.0.0.0 --port 8003
```

### 导入 OpenAPI MCP 并创建 Agent

1. 在 Nexent「选择工具 → MCP 配置」中导入 `mcps/*.openapi.json`（或运行在线集成脚本）。
2. 导入本目录 `agent.json`，或执行：

```bash
python demos/dkm_online_integration.py --mode submit --create-agent --allow-write
```

详细步骤见 `docs/online_integration.md`。

## 使用示例

```text
用户：清洗患者样例 CSV，再构建医疗知识图谱，并回答「高血压有哪些症状和用药？」
Agent：调用 task1 → task2，返回清洗质量报告与图谱问答结果。

用户：对图谱做统计分析，并生成交互仪表盘
Agent：调用 task3_graph_analysis，输出报告与 ECharts 仪表盘。
```

导出统一 tool/agent spec：

```bash
python demos/dkm_nexent_spec.py
```

## MCP 服务

| 服务名 | 说明 | 契约文件 |
| --- | --- | --- |
| `nexent_dkm_task1` | 数据处理 API | `mcps/nexent_dkm_task1.openapi.json` |
| `nexent_dkm_task2` | 医疗知识图谱 API | `mcps/nexent_dkm_task2.openapi.json` |
| `nexent_dkm_task3` | 图谱分析 API | `mcps/nexent_dkm_task3.openapi.json` |

## 知识库文件

| 路径 | 说明 |
| --- | --- |
| `knowledge/` | 归档入口（指向样例与术语） |
| `data/samples/medical_terminology.json` | 医疗术语表 |
| `data/samples/` | 任务一/二样例数据 |

## 配置参数

| 参数 | 值 |
| --- | --- |
| Agent 名称 | `nexent-dkm-agent` |
| Nexent name | `dkm_end_to_end_agent` |
| 最大执行步骤 | 12 |
| 队伍 | 龙卷风摧毁停车场 |
| 作者字段 | 龙卷风摧毁停车场 |

## 更多文档

- 架构：`docs/architecture.md`
- 依赖：`docs/dependencies.md`
- 任务一/二/三：`docs/task1_data_agent.md` 等
- 在线集成：`docs/online_integration.md`
