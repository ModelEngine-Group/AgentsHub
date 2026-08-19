# 任务一：数据处理智能体实现说明

## 目标

基于 Nexent 智能体思想，利用 DataMate 风格的数据处理算子，构建一个可运行的数据处理智能体 `DataProcessingAgent`。

该智能体支持：

- 接收自然语言任务描述
- 通过 `RuleBasedTaskPlanner` 生成可解释的执行计划
- 调用多个数据处理算子
- 跟踪任务执行状态
- 捕获算子异常并返回结构化失败摘要
- 输出清洗后的结构化数据和文本结构化结果

## 输入数据

| 文件 | 说明 |
|---|---|
| `data/medical_records.csv` | 结构化医疗记录，共 15 条 |
| `data/medical_text.txt` | 非结构化医疗文本，包含 15 个疾病章节 |

## 智能体入口

```text
agents/data_processing_agent.py
```

核心类：

```python
DataProcessingAgent
```

主要方法：

```python
plan(task_description: str) -> list[dict]
run(task_description: str) -> dict
```

## 执行计划

当前实现把规划逻辑放在：

```text
planners/task_planner.py
```

`DataProcessingAgent.plan(...)` 会委托给可插拔 planner。当前提供两种 planner：

- `RuleBasedTaskPlanner`：默认规则规划器，离线稳定可复现。
- `LLMTaskPlanner`：LLM 驱动规划器，接收一个可注入的 LLM callable；LLM 输出异常、JSON 无效或 step 不合法时，会回退到规则规划器。

规划器可以根据任务描述生成三类计划：

- CSV-only：只处理结构化 CSV。
- text-only：只处理医疗文本。
- CSV + text：同时处理结构化 CSV 和非结构化文本。

默认完整流程包含以下步骤：

1. `read_csv`：读取结构化医疗记录
2. `clean_missing`：删除空值记录
3. `normalize_text`：规范化文本字段
4. `extract_fields`：抽取核心字段
5. `save_csv`：保存清洗后的结构化数据
6. `read_text`：读取非结构化医疗文本
7. `split_paragraph`：按疾病章节切分文本
8. `save_text_json`：保存清洗后的文本结构

## 使用的算子

| 算子 | 文件 | 作用 |
|---|---|---|
| `ReadCsvOperator` | `operators/csv_operators.py` | 读取 CSV |
| `CleanMissingOperator` | `operators/csv_operators.py` | 清洗空值 |
| `NormalizeTextOperator` | `operators/csv_operators.py` | 文本字段规范化 |
| `ExtractFieldOperator` | `operators/csv_operators.py` | 抽取核心字段 |
| `SaveCsvOperator` | `operators/csv_operators.py` | 保存 CSV |
| `ReadTextOperator` | `operators/text_operators.py` | 读取文本 |
| `SplitParagraphOperator` | `operators/text_operators.py` | 分割段落 |
| `SaveJsonOperator` | `operators/json_operators.py` | 保存 JSON |

## 异常处理与状态记录

`DataProcessingAgent.run(...)` 会在执行失败时返回结构化结果，而不是只让异常向外冒泡。

失败摘要示例：

```json
{
  "status": "failed",
  "execution_state": {
    "steps": [
      {"step": "read_csv", "status": "running", "detail": "..."},
      {"step": "read_csv", "status": "failed", "detail": "..."}
    ],
    "errors": [
      {"step": "read_csv", "error": "[ReadCsvOperator] 执行失败: ..."}
    ]
  }
}
```

成功摘要中 `execution_state.errors` 为空列表。

异常策略现在可以通过 `StepExecutionPolicy` 显式配置：

```python
from agents.data_processing_agent import DataProcessingAgent, StepExecutionPolicy

agent = DataProcessingAgent(
    data_dir="data",
    outputs_dir="outputs",
    step_policies={"read_text": StepExecutionPolicy(retries=2, required=False)},
)
```

这表示 `read_text` 失败时最多重试 2 次，最终仍失败则标记为 `skipped`，整体任务可返回 `partial_success`。

## DataMate 适配

任务一现在提供两类 DataMate 风格适配：

- `adapters/datamate_adapter.py`：`CleanTextDataMateMapper`，证明文本清洗算子可迁移为 `sample -> sample`。
- `adapters/datamate_task1_mappers.py`：`NormalizeMedicalRecordMapper`、`ExtractMedicalFieldsMapper`，用于结构化医疗记录样本级处理。

同时提供一个可迁移 package 示例：

```text
adapters/datamate_task1_package/
├── __init__.py
├── metadata.yml
└── process.py
```

该目录展示了 DataMate 要求的 `metadata.yml`、`process.py` 和 `OPERATORS.register_module(...)` 注册形态。

## Nexent / MCP Tool 适配

Demo 现在提供 Nexent/FastMCP 风格工具包装：

```text
adapters/nexent_mcp_tools.py
```

核心函数：

```python
process_medical_data_tool(task_description: str, data_dir: str | None = None, outputs_dir: str | None = None) -> dict
build_medical_knowledge_graph_tool(task_description: str, data_dir: str | None = None, outputs_dir: str | None = None) -> dict
ask_medical_kg_tool(question: str, outputs_dir: str | None = None) -> dict
analyze_medical_data_tool(task_description: str, outputs_dir: str | None = None) -> dict
```

这些函数分别调用 `DataProcessingAgent`、`KnowledgeGraphAgent`、`KGQAAgent` 和 `DataAnalysisAgent`，返回 JSON 可序列化结果。

前置依赖处理：

- `ask_medical_kg_tool(...)` 需要 `outputs/medical_kg.json`，缺失时返回 `status = "missing_prerequisite"`。
- `analyze_medical_data_tool(...)` 需要 `outputs/medical_kg.json` 和 `outputs/cleaned_records.csv`，缺失时返回 `status = "missing_prerequisite"`。

安装 FastMCP 后，也可以作为 MCP server 启动：

```bash
uv run python -m adapters.nexent_mcp_tools
```

启动后可在 Nexent 的 MCP 配置中注册 `http://127.0.0.1:8001/sse`。如果 Nexent 运行在 Docker 容器中，地址通常需要改成 `http://host.docker.internal:8001/sse`。

## 真实 LLM 测试配置

工程提供 `.env.example` 作为模板，真实密钥写入本地 `.env`，该文件已被 `.gitignore` 忽略。

```env
RUN_REAL_LLM_TESTS=1
DKI_REAL_LLM_API_KEY=replace-with-your-key
DKI_REAL_LLM_BASE_URL=http://82.157.105.8:37853/v1
DKI_REAL_LLM_MODEL=MiniMax-M2.7-highspeed
```

运行真实 LLM planner 测试：

```bash
uv run pytest tests/test_real_llm_planner.py -v
```

## 运行方式

```bash
uv run python run_demo.py task1
```

## 输出文件

| 文件 | 说明 |
|---|---|
| `outputs/cleaned_records.csv` | 清洗后的结构化医疗记录 |
| `outputs/cleaned_text.json` | 清洗并切分后的医疗文本 |
| `outputs/task1_summary.json` | 智能体执行摘要和状态记录 |

## 验证命令

```bash
uv run pytest tests/test_data_processing_agent.py
uv run pytest tests/test_task_planner.py tests/test_data_processing_agent_resilience.py tests/test_datamate_task1_mappers.py tests/test_nexent_mcp_tools.py tests/test_real_llm_planner.py
uv run python run_demo.py task1
```

## 当前验证结果

- `uv run pytest tests/test_data_processing_agent.py`：2 passed
- `uv run pytest tests/test_task_planner.py tests/test_data_processing_agent_resilience.py tests/test_datamate_task1_mappers.py tests/test_nexent_mcp_tools.py`：专项测试通过
- `uv run pytest tests/test_real_llm_planner.py -v`：真实 MiniMax planner 测试通过（需要本地 `.env`）
- `uv run python run_demo.py task1`：运行成功
- 输出 CSV 行数：15
- 输出文本段落数：15

## 后续增强方向

- 将 `LLMTaskPlanner` 接入 CLI/API 的可选运行参数
- 为异常处理增加更细粒度的 backoff / step timeout 策略
- 在真实 DataMate 环境中上传并运行 task1 package
- 在真实 Nexent 环境中注册并调用 MCP Tool

