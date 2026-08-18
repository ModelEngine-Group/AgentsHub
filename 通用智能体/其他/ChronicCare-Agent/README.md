# ChronicCare-Agent：基于Nexent的慢病“数据—知识—洞察”智能体

**正式交付日期：2026年8月10日**

ChronicCare-Agent面向慢病随访数据处理、知识组织和辅助分析场景，以Nexent单智能体作为自然语言入口，通过MCP连接DataMate算子链路、医疗知识图谱、受控Open SQL、BI可视化和Ascend NPU增强能力。

项目已经形成从原始数据到前端回答的完整闭环：

```text
慢病随访数据
  → DataMate清洗、标准化、实体/关系抽取
  → 三元组校验、知识图谱、SQLite分析库
  → Nexent Agent规划与MCP工具调用
  → 统计表格、趋势图、关系子图、Open SQL结果
  → 可追踪的报告和评测证据
```

> 本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。

## 开源仓库

| 工程 | GitLink仓库 | 发布分支 | 版本标识 |
| --- | --- | --- | --- |
| ChronicCare-Agent | [guotingxuan/ChronicCare-Agent](https://www.gitlink.org.cn/guotingxuan/ChronicCare-Agent) | `competition` | 以`competition`分支HEAD为准 |
| DataMate算子贡献 | [guotingxuan/DataMate](https://www.gitlink.org.cn/guotingxuan/DataMate) | `competition` | `5ef1f9619850fe3b640231784e4c9d2e6ce4ae56` |

ChronicCare-Agent仓库提供完整智能体工程；DataMate仓库的`competition`分支提供本项目贡献的慢病数据处理与NPU增强算子。复现时应使用表中对应分支。

## 核心能力

- **数据处理智能体**：11个DataMate CPU/通用主线算子，支持真实执行、状态追踪、异常处理和结果同步。
- **动态DAG**：根据“只清洗、只重建图谱、只刷新分析库、完整链路”等目标裁剪节点，支持dry-run、重试和恢复。
- **知识图谱问答**：支持实体、关系、患者路径、图谱概览和实时交互式子图。
- **数据分析智能体**：支持疾病、共病、风险、随访、指标、趋势和关联分析。
- **受控Open SQL**：Schema Linking、模板/LLM候选SQL、SQL AST Guard和只读SQLite执行。
- **图文可视化**：生成表格、饼图、柱状图、折线图、图谱预览和交互式HTML。
- **NPU增强**：两个BGE模型算子支持Ascend NPU，并提供CPU/NPU同样本和NPU全量独立实测。
- **Nexent集成**：通过Streamable HTTP MCP将工具注册给Nexent单智能体。

## 当前数据与图谱规模

| 指标 | 当前值 |
| --- | ---: |
| 数据版本 | `synthetic_chroniccare` |
| 患者 | 2,000人 |
| 随访记录 | 8,231条 |
| 检验记录 | 131,323条 |
| 用药记录 | 18,248条 |
| 慢病类型 | 20种 |
| 图谱节点 | 197,404个 |
| 图谱边 | 396,928条 |
| 实体类型 | 14类 |
| 关系类型 | 15类 |

事实来源为`configs/current_metrics.json`、当前SQLite数据库和图谱产物，前端不展示内部图谱评分。

当前数据为参考真实慢病随访业务的字段结构、业务关系和合理取值范围，采用程序化规则与固定随机种子合成的数据，不包含可识别的真实患者身份信息。

## 比赛任务覆盖

| 任务 | 项目实现 |
| --- | --- |
| 任务一：数据处理智能体 | Nexent任务理解、DataMate算子、动态DAG、状态和异常处理 |
| 任务二：知识图谱问答智能体 | 医疗实体/关系抽取、三元组校验、图谱构建、问答和子图 |
| 任务三：数据分析与图谱可视化 | 图谱理解、统计/趋势/关联分析、Open SQL、BI图表 |
| NPU加分项 | 实体标准化和关系重排的BGE NPU增强及真实性能对比 |

详细映射见[比赛任务要求映射](docs/task_requirements_mapping.md)。

## 快速启动

### 1. 准备配置

```bash
cp .env.example .env
```

默认模板Open SQL无需远程模型。启用LLM候选SQL时，API Key仅写入本地`.env`，该文件不进入提交内容。

### 2. 一行启动

```bash
docker compose up -d --build chroniccare-runtime
```

根目录`docker-compose.yml`是正式交付的唯一推荐入口。`docker-compose.server.yml`仅用于需要宿主机Docker编排能力的受控服务器环境。

### 3. 健康检查

```bash
curl --noproxy "*" -sS http://127.0.0.1:18088/health
curl --noproxy "*" -sS http://127.0.0.1:18188
```

### 4. 访问入口

- Tool Server：`http://127.0.0.1:18088`
- MCP Endpoint：`http://127.0.0.1:18188/mcp`
- 可选辅助Dashboard：`http://127.0.0.1:18501`（正式交互与功能测试以Nexent前端为准）

正式运行采用单容器`chroniccare-runtime`，不需要分别启动旧的Tool Server、MCP Adapter或Streamlit容器。

完整步骤见[部署与复现指南](docs/deployment_guide.md)。

## Nexent接入

1. 启动`chroniccare-runtime`。
2. 在Nexent中添加Streamable HTTP MCP服务器。
3. MCP地址配置为Nexent可访问的`http://宿主机地址:18188/mcp`。
4. 参考`integrations/nexent/chroniccare_agent_prompt.md`配置Agent提示词。
5. 后端MCP服务公开38个工具；按`integrations/nexent/chroniccare_tool_manifest.json`默认绑定33个工具，其中保留2个调试工具，不默认暴露5个历史兼容入口。
6. 从数据规模、疾病分布和图谱摘要开始验证，再运行DataMate/NPU长任务。

示例配置位于`integrations/nexent/chroniccare_mcp_config.example.json`。

## DataMate算子

主线包含11个CPU/通用算子：

```text
chronic_file_ingest
chronic_table_clean
chronic_field_normalize
chronic_text_split
chronic_entity_extract
chronic_relation_extract
chronic_triple_validate
chronic_kg_build
chronic_sqlite_loader
chronic_nl2sql_analyze
chronic_report_pack
```

NPU增强算子：

```text
chronic_entity_extract_model_npu
chronic_relation_extract_model_npu
```

完整算子源码和契约位于`integrations/datamate/`。

## NPU评测口径

当前正式口径为：

- CPU 64线程，batch size 64；
- NPU单卡，batch size 1024；
- 每轮CPU与NPU各预热一次；
- CPU 2,048条、NPU 2,048条、NPU全量分别真实计时并重复5轮；
- NPU 2,048条和全量分别启停资源采样器；
- 正式结果采用5轮耗时算术平均值，并保留逐轮原始值；
- NPU不可用时明确fallback，不声明加速效果。

当前机器可读报告位于`outputs/evaluation/npu_operator_benchmark_report.json`，可读说明见[测试与性能报告](docs/testing_and_benchmarks.md)。

## 测试摘要

- 默认单元、契约与集成测试：431项；另有6项显式开启的真实集成测试，覆盖已部署Tool Server、MCP JSON-RPC、DataMate容器接入和Ascend NPU硬件冒烟。
- 启用真实集成测试后的全源码语句与分支综合覆盖率：62.32%，其中语句覆盖率65.61%、分支覆盖率53.64%。重点模块中，MCP HTTP客户端、MCP Adapter客户端和问题流水线均为100%，问题分类器98.39%、动态DAG引擎98.13%、图谱总览渲染95.48%、图表渲染93.20%、报告入口91.49%、Supervisor编排90.00%、查询规划（`orchestration/planner.py`）86.49%、NPU runtime（`runtime_common/npu_runtime.py`）86.36%、Open SQL工具服务（`tool_server/open_sql_tools.py`）84.62%、DataMate流水线工具81.82%、工具通用层75.45%、NPU工具62.88%、DataMate runner 55.32%、MCP聚合入口（`mcp_adapter/server.py`）51.51%。
- 静态检查：使用Ruff检查语法、未定义或未使用符号、重复字典键和导入规范。
- NL2SQL盲测集：240题；通过`scripts/eval_nl2sql_blind.py`在当前数据库和外部模型环境生成结果。
- SQL Guard专项安全集：120例，总体正确率100%。
- Agent路由回归：80题，通过率100%。
- KGQA工程回归：150例通过；该结果不代表临床专家评审。
- 图谱精确重复边为0，边来源信息完整率100%。
- 42张Nexent前端截图已与当前SQLite和图谱数据交叉核验。

评测证据位于`outputs/evaluation/`。
本地执行质量检查：

```bash
python3 -m pip install -r requirements.txt
make quality
make coverage
```


## 项目目录

```text
analysis/                 Open SQL与NL2SQL分析
app/                      可选Streamlit辅助Dashboard
configs/                  当前指标、字典、契约和评测配置
data/                     原始/处理数据、SQLite和知识图谱
deploy/runtime/           单容器运行时
docs/                     中文技术与交付文档、前端截图
integrations/datamate/    DataMate算子交付源码和清单
integrations/nexent/      Nexent提示词、MCP配置和接入材料
kg/                       知识图谱构建与查询逻辑
mcp_adapter/              MCP工具定义与适配器
orchestration/dag/        动态DAG规划、执行和恢复
outputs/                  正式评测、发布证据及轻量展示产物
scripts/                  运行、同步、检查和打包脚本
tests/                    单元、契约、集成测试和小型测试夹具
tool_server/              HTTP工具服务
visualization/            图表与图谱可视化
```

## 文档

- [文档中心](docs/README.md)
- [技术报告](docs/技术报告.md)（[PDF版](docs/技术报告.pdf)）
- [系统架构](docs/architecture.md)
- [任务要求映射](docs/task_requirements_mapping.md)
- [部署指南](docs/deployment_guide.md)
- [测试与性能](docs/testing_and_benchmarks.md)
- [数据与模型来源](docs/data_and_model_provenance.md)
- [开源合规](docs/open_source_compliance.md)
- [MCP与API参考](docs/mcp_and_api_reference.md)
- [前端证据索引](docs/frontend_evidence_index.md)

## 许可证与第三方组件

项目自研代码和文档采用MIT License，见`LICENSE`。第三方组件、模型和外部运行时说明见`NOTICE`和`THIRD_PARTY_NOTICES.md`。
