# Demo 与 DataMate / Nexent 结构对照说明

## 1. 本文目标

本文说明当前 `data_knowledge_insight_demo` 与 DataMate、Nexent 的对应关系，用来回答三个问题：

1. Demo 里的算子和 DataMate 的 `runtime/ops` 算子生态怎么对应？
2. Demo 里的智能体和 Nexent 的 Agent / MCP Tool 体系怎么对应？
3. 下一步哪些能力适合迁移，哪些暂时不应该做？

本文只做结构对照和迁移设计，不直接修改 DataMate 或 Nexent 主项目源码。

## 2. 三个项目的角色分工

| 项目 | 当前角色 | 适合承担的职责 |
| --- | --- | --- |
| `data_knowledge_insight_demo` | 独立可运行 Demo | 验证“数据 → 知识 → 洞察”业务闭环，沉淀可测试的算子和智能体能力 |
| `DataMate/runtime/ops` | 算子生态 | 承载数据清洗、格式转换、文本切分、质量过滤等可复用数据处理算子 |
| `Nexent` | Agent 与工具编排平台 | 通过 Agent 调用工具，适合把图谱问答、数据分析等能力注册为 MCP Tool |

推荐理解方式：

```text
DataMate 更像“可复用工具箱”：负责单个数据处理动作。
Nexent 更像“智能调度员”：负责理解任务并调用工具。
Demo 是两者之间的验证场：先把能力跑通，再选择哪些能力迁移。
```

## 3. Demo 算子与 DataMate `runtime/ops` 对照

### 3.1 DataMate 算子规范摘要

DataMate `runtime/ops/README.md` 中定义的标准算子包结构如下：

```text
operator_package/
├── __init__.py        # 算子注册入口
├── metadata.yml       # 算子元数据、UI 参数定义、资源配置
├── process.py         # 算子核心逻辑
├── requirements.txt   # 可选依赖
└── README.md          # 可选说明文档
```

核心约束：

- `process.py` 中的类通常继承 `datamate.core.base_op.Mapper` 或 `Filter`。
- 类名应与 `metadata.yml` 中的 `raw_id` 保持一致。
- 必须实现 `execute(sample: dict) -> dict`。
- `sample` 通常包含 `text`、`data`、`fileName`、`fileType`、`filePath` 等字段。
- `__init__.py` 通过 `OPERATORS.register_module(...)` 注册算子。

### 3.2 Demo 算子规范摘要

Demo 当前算子集中在：

- `operators/base.py`
- `operators/csv_operators.py`
- `operators/text_operators.py`
- `operators/kg_operators.py`
- `operators/analysis_operators.py`
- `operators/nl2sql_operators.py`

Demo 的基础约束：

- 所有算子继承 `BaseOperator`。
- 子类实现 `run(...)`。
- 外部推荐调用 `execute(...)`，由基类负责状态追踪和异常包装。

### 3.3 核心差异

| 维度 | Demo Operator | DataMate Operator | 迁移含义 |
| --- | --- | --- | --- |
| 基类 | `BaseOperator` | `Mapper` / `Filter` | 迁移时需要替换基类或加 adapter |
| 执行入口 | `execute(*args, **kwargs)` 调 `run(...)` | `execute(sample: dict) -> dict` | 需要把 Demo 参数改造成 `sample` 输入输出 |
| 元数据 | Python 类属性为主 | `metadata.yml` | 迁移时要补 UI 参数和资源配置 |
| 注册方式 | `operators/__init__.py` 导出 | `OPERATORS.register_module(...)` | 迁移时要写 DataMate 风格注册入口 |
| 数据单位 | 文件路径、DataFrame、Graph dict 等 | 单条 `sample` 字典 | 文件型算子要拆成样本级或批处理封装 |

### 3.4 Demo 算子迁移优先级

| Demo 算子 | 所在文件 | 对应 DataMate 类型 | 迁移优先级 | 原因 |
| --- | --- | --- | --- | --- |
| `CleanTextOperator` | `operators/text_operators.py` | `mapper` | 高 | 输入输出都是文本，最接近 `sample['text'] -> sample['text']` |
| `SplitParagraphOperator` | `operators/text_operators.py` | `slicer` | 高 | 对应 DataMate 的文本切分能力，容易解释 |
| `NormalizeTextOperator` | `operators/csv_operators.py` | `mapper` | 中 | 适合字段标准化，但当前依赖表格字段 |
| `CleanMissingOperator` | `operators/csv_operators.py` | `mapper` / `filter` | 中 | 可用于结构化数据清洗，但 DataMate sample 更偏单样本 |
| `EntityRecognitionOperator` | `operators/kg_operators.py` | `mapper` / `llms` | 中 | 可迁移，但更像领域模型或 LLM 算子 |
| `RelationExtractionOperator` | `operators/kg_operators.py` | `mapper` / `llms` | 中 | 适合作为智能抽取算子，但要补模型配置 |
| `GraphBuildOperator` | `operators/kg_operators.py` | 不建议直接作为 DataMate 单算子 | 低 | 它是聚合型图构建，超出单条 sample 处理范围 |
| `GraphStatisticsOperator` | `operators/analysis_operators.py` | 不建议迁移到 DataMate ops | 低 | 更适合分析服务或 Nexent 工具，而不是清洗算子 |
| `NL2SQLOperator` | `operators/nl2sql_operators.py` | 不建议第一批迁移 | 低 | 更像智能查询工具，适合 Nexent Tool |

### 3.5 DataMate 侧建议下一步

先实现一个最小适配样例：

```text
Demo CleanTextOperator / SplitParagraphOperator
    ↓ adapter 包装
DataMate 风格 process.py
    ↓ execute(sample)
返回更新后的 sample
```

建议先选 `CleanTextOperator`，因为它最容易对应 DataMate 的 `Mapper`：

```python
sample = {"text": "  原始 文本  "}
result = CleanTextMapper().execute(sample)
assert result["text"] == "原始 文本"
```

## 4. Demo Agent 与 Nexent Agent / MCP Tool 对照

### 4.1 Nexent 结构摘要

Nexent 文档中说明：

- `CoreAgent` 是继承并增强 SmolAgents `CodeAgent` 的执行引擎。
- Agent 按 ReAct 模式运行：思考、行动、观察、迭代。
- 工具系统基于 MCP。
- 本地工具注册示例位于：`backend/tool_collection/mcp/local_mcp_service.py`。

当前本地 MCP 服务示例：

```python
from fastmcp import FastMCP

local_mcp_service = FastMCP("local")

@local_mcp_service.tool(name="test_tool_name", description="test_tool_description")
async def demo_tool(para_1: str, para_2: int) -> str:
    return "success"
```

### 4.2 Demo Agent 摘要

Demo 当前 Agent：

| Agent | 文件 | 职责 |
| --- | --- | --- |
| `DataProcessingAgent` | `agents/data_processing_agent.py` | 编排任务一数据处理算子 |
| `KnowledgeGraphAgent` | `agents/knowledge_graph_agent.py` | 编排文本读取、实体识别、关系抽取、图谱构建 |
| `KGQAAgent` | `agents/kgqa_agent.py` | 基于 `medical_kg.json` 回答图谱问题 |
| `DataAnalysisAgent` | `agents/data_analysis_agent.py` | 统计图谱、生成图表、构建 SQLite、评估 NL2SQL |

### 4.3 Agent / Tool 对照关系

| Demo 能力 | Nexent 中更适合的形态 | 迁移优先级 | 原因 |
| --- | --- | --- | --- |
| `DataProcessingAgent.run(...)` | MCP Tool：`process_medical_data(task_description: str)` | 高 | 清洗医疗 CSV 和文本，是后续图谱/分析的前置能力 |
| `KnowledgeGraphAgent.run(...)` | MCP Tool：`build_medical_knowledge_graph(task_description: str)` | 高 | 生成 `medical_kg.json`，是 KGQA 和图谱分析的前置能力 |
| `KGQAAgent.run(question)` | MCP Tool：`ask_medical_kg(question: str)` | 高 | 输入输出简单，适合 Agent 调用；缺少图谱时返回 `missing_prerequisite` |
| `DataAnalysisAgent.run(task_description)` | MCP Tool：`analyze_medical_data(task_description: str)` | 高 | 可返回统计结果、图表和报告路径；缺少清洗数据或图谱时返回 `missing_prerequisite` |
| 单个底层算子 | 不建议直接暴露给 Nexent Agent | 低 | Agent 更适合调用业务级工具，而不是微小算子 |

### 4.4 Nexent 侧建议下一步

当前已实现 4 个业务级 MCP 工具适配样例，不直接改 Nexent 主项目：

```text
Demo Agents
    ↓ adapters/nexent_mcp_tools.py
process_medical_data_tool(...) / build_medical_knowledge_graph_tool(...)
ask_medical_kg_tool(...) / analyze_medical_data_tool(...)
    ↓ FastMCP tool registration when fastmcp is installed
Nexent Agent 可通过远程 MCP 调用数据处理、知识图谱、问答和分析能力
```

当前工具函数：

```python
process_medical_data_tool(task_description: str, data_dir: str | None = None, outputs_dir: str | None = None) -> dict
build_medical_knowledge_graph_tool(task_description: str, data_dir: str | None = None, outputs_dir: str | None = None) -> dict
ask_medical_kg_tool(question: str, outputs_dir: str | None = None) -> dict
analyze_medical_data_tool(task_description: str, outputs_dir: str | None = None) -> dict
```

`ask_medical_kg_tool(...)` 需要先生成 `outputs/medical_kg.json`；`analyze_medical_data_tool(...)` 需要先生成 `outputs/medical_kg.json` 和 `outputs/cleaned_records.csv`。缺少前置产物时，工具返回结构化 `missing_prerequisite`，方便 Nexent Agent 引导用户先运行前置工具。

迁移到 Nexent 时，再包成：

```python
from fastmcp import FastMCP

mcp = FastMCP("data-knowledge-insight")
mcp.tool(name="process_medical_data")(process_medical_data_tool)
mcp.tool(name="build_medical_knowledge_graph")(build_medical_knowledge_graph_tool)
mcp.tool(name="ask_medical_kg")(ask_medical_kg_tool)
mcp.tool(name="analyze_medical_data")(analyze_medical_data_tool)
```

启动 Demo MCP server 后，Nexent 远程 MCP 地址通常为：

- 本机 Nexent：`http://127.0.0.1:8001/sse`
- Docker Nexent：`http://host.docker.internal:8001/sse`

## 5. 可迁移能力清单

### 5.1 适合迁移到 DataMate 的能力

| 能力 | 推荐形式 | 备注 |
| --- | --- | --- |
| 文本清洗 | `mapper` 算子 | 第一优先级 |
| 文本切分 | `slicer` 算子 | 第二优先级 |
| 字段标准化 | `mapper` 算子 | 需要明确 sample 中字段结构 |
| 实体识别 | `mapper` / `llms` 算子 | 后续可接模型或 LLM |
| 关系抽取 | `mapper` / `llms` 算子 | 后续可接模型或 LLM |

### 5.2 适合迁移到 Nexent 的能力

| 能力 | 推荐形式 | 备注 |
| --- | --- | --- |
| 医疗数据清洗 | MCP Tool / 调用 DataMate pipeline | 已提供 `process_medical_data_tool(...)` |
| 一键构建知识图谱 | MCP Tool | 已提供 `build_medical_knowledge_graph_tool(...)` |
| 医疗图谱问答 | MCP Tool | 已提供 `ask_medical_kg_tool(...)` |
| 医疗图谱分析 | MCP Tool | 已提供 `analyze_medical_data_tool(...)`，可返回统计结果和报告路径 |

## 6. 暂时不做的内容

为了保持当前项目稳定，以下内容本阶段暂时不做：

- 不直接修改 DataMate `runtime/ops` 主目录下已有算子。
- 不启动 DataMate PostgreSQL、Milvus、Ray 等完整服务。
- 不上传真实 DataMate 算子包。
- 不修改 Nexent 数据库、租户、权限、前端配置。
- 默认不要求启动真实 Nexent 服务；Demo 已提供可选 MCP server 入口。
- 不把所有底层算子都暴露为 Nexent 工具，避免工具粒度过细。

## 7. 下一步执行建议

建议按下面顺序继续推进：

1. **Planner 增强**：后续可把 `RuleBasedTaskPlanner` 扩展为 LLM planner。
2. **异常策略增强**：后续可增加可配置 retry / skipped 策略。
3. **真实平台验证**：在 DataMate 上传 `datamate_task1_package`，在 Nexent 注册 MCP server。
4. **最终验证**：每次增强后运行 `uv run pytest` 和 `uv run python run_demo.py task1`。

## 8. 当前结论

当前 Demo 与 DataMate / Nexent 的关系可以概括为：

```text
Demo operators 是可迁移到 DataMate ops 的候选算子。
Demo agents 是可迁移到 Nexent MCP Tools 的候选业务能力。
DataMate 负责沉淀可复用数据处理算子。
Nexent 负责通过 Agent 编排和调用这些工具。
```

因此，第 7 阶段不应该追求“一次性深度接入”，而应该先完成“结构对照 → 适配样例 → 集成说明”的渐进式增强。
