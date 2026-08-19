# 架构说明

## 总览

本项目实现“数据 → 知识 → 洞察”闭环，由三个任务智能体和一组可复用算子共同组成。设计目标是：离线可运行、结果可复现、模块可扩展，并且不强依赖外部服务。

## 分层结构

```text
Agent      任务理解、规划、算子编排、状态与异常处理
Operator   可复用算子：数据清洗、图谱抽取、分析、NL2SQL、NPU 基准测试
Pipeline   单任务与跨任务流程编排
演示程序   可执行命令行演示入口
Benchmark  CPU 基线对照、NL2SQL 准确率、NPU 探测与性能评测（业务指标，见 README「任务效果 benchmark」）
Test       单元与集成验证（pytest，见 README「工程校验」）
```

除跨任务 `end_to_end_pipeline` 外，每个 Agent 只依赖本任务所需的 Operator，不直接依赖其他 Agent。三任务串联发生在 Pipeline 层，从而保持智能体解耦，也让算子可以独立测试。

## 三个任务智能体

| 智能体 | 模块 | 输入 | 输出 |
| --- | --- | --- | --- |
| 任务一：数据处理 | `src/agents/data_processing_agent` | CSV / JSON / text | 清洗数据集与质量报告 |
| 任务二：医疗知识图谱 | `src/agents/kg_agent` | 医疗文本 | 图谱 JSON、问答结果与质量报告 |
| 任务三：图谱分析 | `src/agents/analysis_agent` | 任务二图谱 JSON | 统计、关联、趋势、NL2SQL 与 BI 仪表盘 |

### 规划与自主执行

三个任务都采用混合规划链路：本地小模型 → LLM → 规则回退。规划结果不是展示信息，而会驱动真实执行：

- **任务一**：规划结果决定清洗算子是否执行去重、填补缺失、类型规范化和字段转换。
- **任务二**：计划选择的算子会控制 QA 是否执行。只有构建或查询意图且没有问题时，会跳过 `answer_graph_question`；带问题的计划会执行问答。
- **任务三**：计划识别的意图会控制可选图分析算子。`graph_analytics` 意图触发社区发现和枢纽最短路径分析；否则只运行核心分析链路。Agent 会在 `plan_execution` 中记录所选意图和实际执行算子。

## 数据流闭环

```text
原始医疗文本
   │  任务一：清洗（去 HTML、Unicode/空白标准化、PII 脱敏）
   ▼
清洗文本  ─────────────────────────────────────────────┐
   │  任务二：实体抽取 → 关系抽取 → 三元组校验 → 建图 → QA │
   ▼                                                         │
medical_kg.json（nodes/edges）                               │ 复用
   │  任务三：加载图谱 → 统计/关联/趋势 → 中心性             │
   │          （可选社区/路径）→ SQLite 图投影 NL2SQL → BI   │
   ▼                                                         │
洞察报告（Markdown / HTML / ECharts）────────────────────────┘
```

跨任务编排入口为 `src/pipelines/end_to_end_pipeline.py` 的 `run_end_to_end_pipeline`，由 `demos/end_to_end_demo.py` 和 `tests/test_end_to_end.py` 验证。任务一在结构化路径中也支持 JSON records：先转为扁平 CSV，再复用 CSV 清洗链路。

## NL2SQL 设计（任务三）

问题会先分类为标准意图（见 `src/operators/analysis_ops/nl2sql.py` 中的 `INTENT_SQL`），每个意图对应一条安全、只读的 SQL 模板，查询对象是图谱的 SQLite 投影表 `nodes` 和 `edges`。由于每个意图唯一对应 SQL 模板，意图分类正确即可代表模板 SQL 正确。

准确率由 `benchmarks/data/nl2sql_benchmark.json`（意图分类）、`benchmarks/data/nl2sql_execution_benchmark.json`（执行级）和 `benchmarks/data/nl2sql_holdout_benchmark.json`（扩展改写回归，文件名保留 holdout 仅为兼容）共同验证：`task3_nl2sql_benchmark.py` 产出报告 JSON，pytest 回归测试（如 `test_nl2sql_*_above_threshold`）防止准确率回退。配置 LLM 时，会优先尝试 LLM 生成 SQL，并在失败时回退模板翻译器。

## 外部集成

- **DataMate**：任务一通过 `DataMateClient` 探测健康状态、算子目录、模板和任务，
  生成试运行或正式提交数据；提交后提取资源 ID 并调用详情接口回查，默认不提交。
- **Neo4j**：任务二可选持久化和查询图谱；驱动或服务不可用时会降级为本地图谱 JSON。
- **Nexent**：三个任务分别通过 `nexent_adapter` 导出 tool/agent spec，并可由
  `demos/dkm_online_integration.py` 调用官方接口导入三套 OpenAPI 服务、刷新工具目录、
  创建和回查 DKM Agent；所有写入均需显式授权，不修改官方 Nexent 源码。
- **LLM**：可选用于规划、抽取和 NL2SQL 增强；配置只从本地 `.env` 或 `.json` 加载，真实 key 不提交。
- **NPU**：`src/operators/npu_ops` 提供 CPU 基线对照、Ascend 运行时探测和算子级加速路径；没有硬件时只报告 `unavailable`（不可用），不伪造加速比或能效。自 2026-07-03 起，任务二张量关系打分在 NPU 模式下默认走 `cached_argmax_labels` 路径（跳过全量 logits 回拷），失败再回退 legacy / CPU。
- **OOV 模式抽取**：`pattern_extractor.py` 用中文医疗后缀（…病、…综合征、…替丁 等）补充词典匹配；与别名表合并后最长优先。封闭 gold 与 OOV 8 条语料评测 F1 均为 1.0，词典外实体 **22/22**。

Nexent 与 DataMate 官方仓库只作为部署、接口参考和集成目标；比赛工程实现集中在本仓库内。默认服务端口与 WSL/Linux 部署边界见 [preparation.md](preparation.md)。2026-07-02 在线集成按 L1 任务 API → L2 DataMate → L3 Nexent → L4 Neo4j 验证，详见 [online_integration.md](online_integration.md) 与答辩 [§6.3](competition_defense_document.md)。Windows 主链路与 NPU 服务器 pytest 见根目录 [README.md](../README.md#工程校验)（**437/437**、**770/770** 等）。

## 输出边界

- 临时演示输出统一放入 `outputs/`，该目录被 git 忽略。
- 基准测试报告放入 `benchmarks/reports/`。
- 大数据、隐私数据、模型权重、日志和本地 `.env` 文件不提交。
