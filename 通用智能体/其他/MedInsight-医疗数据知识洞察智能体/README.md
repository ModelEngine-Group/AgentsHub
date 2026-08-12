# MedInsight 医疗数据知识洞察智能体

MedInsight 是一个基于 Nexent 的医疗“数据—知识—洞察”智能体。它通过一组可复用 MCP 工具，将原始医疗记录依次转换为高质量文档、可追溯知识图谱与可解释分析结果。

> 本智能体用于数据工程、知识组织与技术演示，不构成医疗诊断、治疗或用药建议。

## 核心能力

| 阶段 | 能力 | 主要产物 |
|---|---|---|
| 数据 | 数据画像、规范化、清洗、切分、异常处理、DataMate 能力发现与可用性门控 | 处理文档、切分结果、质量报告 |
| 知识 | 实体/关系抽取、Schema 校验、三元组修复、冲突检测、证据绑定 | 医疗知识图谱、三元组、质量门控报告 |
| 洞察 | 图谱 Schema 感知、Graph-aware NL2SQL、指标计算、知识缺口检测、BI 可视化 | SQLite 分析库、SQL、BI 看板、洞察报告 |

## 架构

```text
用户请求
   |
   v
Nexent Agent (agent.json)
   |
   v
MedInsight MCP
   |-- process_medical_data              -> Task1 / DataMate 适配层
   |-- build_medical_knowledge_graph     -> Task2 知识构建算子链
   |-- ask_medical_knowledge_graph       -> 证据路径问答
   |-- query_medical_graph_analysis      -> Task3 Graph-aware NL2SQL
   |-- analyze_medical_graph             -> 指标 + BI + 洞察报告
   `-- run_data_knowledge_insight_pipeline -> 端到端闭环
```

## 文件结构

```text
MedInsight-医疗数据知识洞察智能体/
|-- agent.json                  # Nexent 智能体导入文件
|-- README.md
|-- CHANGELOG.md
|-- VERIFICATION.md            # 本地结构、闭环与 MCP 验证记录
|-- LICENSE
|-- mcps/                       # 自定义 MCP 服务及完整执行源码
|-- skills/                     # 智能体规划与工具路由技能
|-- knowledge/                  # 医疗 Schema 与空图谱模板
|-- config/                     # MCP、DataMate 与分析配置
|-- data/                       # 可替换的输入契约示例
|-- tests/                      # 离线冒烟测试
`-- runtime/                    # 本地运行产物，不提交
```

## 运行环境

- Python 3.10 或更高版本
- 可访问的 Nexent 平台实例
- Docker Desktop 或 Docker Engine（用于运行 Nexent、DataMate）
- DataMate 可选；启用后执行健康检查、清洗模板与算子目录发现，并输出可审计连接状态

## 快速启动

### 1. 安装 MCP 依赖

```bash
cd "通用智能体/其他/MedInsight-医疗数据知识洞察智能体"
python3 -m venv .venv
source .venv/bin/activate
pip install -r mcps/requirements.txt
```

### 2. 启动 MCP 服务

```bash
cp config/env.example .env
set -a
source .env
set +a
bash mcps/start.sh
```

默认 SSE 地址：

```text
http://127.0.0.1:18100/sse
```

启动后通过健康接口检查服务状态：

```text
http://127.0.0.1:18100/health
```

健康接口返回 `"status":"ok"` 且 `"tool_count":6` 表示 MCP 已就绪。`/sse` 是提供给 MCP 客户端的持续事件流，浏览器访问后保持空白或持续加载属于正常现象，不应将其作为普通网页使用；服务根路径不提供网页界面。

Nexent 使用 Docker 运行时，MCP 地址填写：

```text
http://host.docker.internal:18100/sse
```

### 3. 在 Nexent 中注册 MCP 和 Agent

确认 Nexent 配置 API 可访问：

```text
http://127.0.0.1:5010/openapi.json
```

运行自动注册脚本：

```bash
python3 scripts/register_in_nexent.py
```

脚本会依次完成 MCP 握手、服务注册、6 个工具扫描和 `agent.json` 导入。也可在页面中手动配置：

1. 进入 **智能体开发** -> **选择工具** -> **MCP 配置**。
2. 根据 `config/mcp.json` 添加 `medinsight` MCP 服务。
3. 刷新工具列表并确认 Nexent 能识别 6 个 MedInsight 工具。
4. 上传本目录下的 `agent.json`。

### 4. 配置模型并使用

`agent.json` 不绑定特定厂商或模型编号。导入后，在目标 Nexent 实例中选择一个支持工具调用的模型并保存，即可通过聊天界面进行自然语言规划和工具路由。未配置模型时，仍可在工具调试页直接验证 6 个 MCP 工具。

聊天模式：

1. 进入 **智能体开发** -> **模型配置**，添加并验证一个支持工具调用的模型。
2. 进入 **智能体配置**，打开 `MedInsight 医疗数据知识洞察智能体`。
3. 选择该模型并保存，然后在聊天页选择 MedInsight Agent。

无模型工具验证：

1. 进入 **智能体开发** -> **智能体配置**，打开 MedInsight Agent。
2. 在工具列表找到 `run_data_knowledge_insight_pipeline`，点击右侧配置按钮。
3. 点击 **测试工具**，传入有权使用的医疗记录 JSON，并将 `require_datamate` 设为 `true` 以验证 DataMate 强制连接链路。

模型只负责自然语言理解和工具路由；六个 MCP 工具及其确定性处理流程不依赖特定模型厂商。

`skills/medinsight-workflow.md` 是可选的工作流说明；需要时可在 **选择技能** -> **构建技能** 中上传并绑定。

## DataMate 配置

默认配置位于 `config/env.example`：

```bash
DATAMATE_ENABLED=auto
DATAMATE_BASE_URL=http://127.0.0.1:18000
DATAMATE_MANAGEMENT_BASE_URL=http://127.0.0.1:8080
MEDINSIGHT_RUNTIME_ROOT=runtime
```

DataMate 由官方项目独立部署。本项目的适配层会读取其健康状态、清洗模板、算子目录和网关状态；确定性 MedInsight 算子在 MCP 进程内执行。完整 DataMate 验证时，调用 `process_medical_data` 或端到端工具并设置 `require_datamate=true`。若服务未连接，运行会明确失败；未强制时会返回透明的降级记录。

参考部署仓库：

- Nexent：<https://gitcode.com/ModelEngine/nexent>
- DataMate：<https://www.gitlink.org.cn/modelengine-group/DataMate>

## 使用示例

### 端到端闭环

```text
请处理我提供的 JSON 医疗记录，生成知识图谱，并统计图谱的实体类型分布。
```

Agent 将调用 `run_data_knowledge_insight_pipeline`，返回任务一质量报告、任务二图谱质量门控和任务三 SQL 分析结果。

### 图谱问答

```text
根据刚刚生成的知识图谱，查询目标疾病的临床表现和检查项目。
```

Agent 仅根据当前图谱中的关系和证据路径组织答案。

### 图谱分析

```text
分析当前图谱中哪些疾病关系最多，并给出执行的 SQL。
```

Agent 将自动选择 `query_medical_graph_analysis`，返回意图、只读 SQL、查询结果和证据摘要。

## 测试

```bash
PYTHONPATH="mcps/src:mcps" python -m unittest discover -s tests -v
python scripts/verify_package.py
```

已验证平台版本和真实调用结果见 [`VERIFICATION.md`](VERIFICATION.md)。

## 数据与隐私

- 归档包不附带授权状态不清晰的医疗事实数据。
- `data/input_records.example.json` 只定义输入结构，运行前应替换为有权处理的数据。
- 本地生成的文档、图谱、SQL 分析库和报告均写入 `runtime/`，默认不进入版本控制。

## 许可证

代码与自定义配置采用 MIT License。用户自行导入的医疗数据仍受其原始授权协议约束。
