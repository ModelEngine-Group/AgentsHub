# 比赛任务要求映射

## 1. 总体映射

| 比赛任务 | 项目实现 | 主要代码/产物 | 前端证据 |
| --- | --- | --- | --- |
| 任务一：数据处理智能体 | Nexent 规划、DataMate 11 算子、动态 DAG、状态追踪与恢复 | `integrations/datamate/`、`orchestration/dag/`、`outputs/release/` | 01、02、18 组截图 |
| 任务二：知识图谱问答智能体 | 医疗实体/关系抽取、三元组校验、图谱构建、图谱问答与子图 | `kg/`、`data/graph/`、`docs/assets/` | 12—17 组截图 |
| 任务三：数据分析与图谱驱动可视化 | 专用统计工具、受控 Open SQL、趋势/关联分析、图表和图谱可视化 | `analysis/`、`visualization/`、`docs/assets/` | 04—11、19—20 组截图 |
| NPU 加分项 | 实体与关系 BGE 推理 NPU 增强，CPU/NPU 同样本实测和 NPU 全量实测 | `integrations/datamate/operators/`、`outputs/evaluation/npu_operator_benchmark_report.json` | 03 组截图 |

## 2. 任务一：基于 Nexent 的数据处理智能体

| 要求 | 实现情况 | 验证依据 |
| --- | --- | --- |
| DataMate 数据接入与处理 | 提供文件接入、表清洗、字段标准化、文本切分等算子 | `integrations/datamate/operator_catalog.yml` |
| 理解清洗、抽取、转换任务 | Nexent Agent 按问题路由到 DataMate 或动态 DAG | `integrations/nexent/chroniccare_agent_prompt.md` |
| 多算子组合与调度 | 11 个主线算子形成完整链路；动态 DAG 可按目标裁剪 | `orchestration/dag/` |
| 状态跟踪与异常处理 | 记录节点状态、耗时和错误，支持重试及按 `run_id` 恢复 | `chroniccare_datamate_dag_status`、`chroniccare_datamate_dag_resume` |
| 典型流程自动执行 | CPU 全流程可由 Nexent 前端真实触发 | `chroniccare_datamate_pipeline_run` |
| 可运行 Demo | Docker 单容器运行，通过 MCP 接入 Nexent | `docker-compose.yml` |
| 稳定性与可复现性 | 固定依赖、Compose、当前指标快照和机器可读评测证据 | `requirements.txt`、`configs/current_metrics.json` |

## 3. 任务二：基于知识图谱的问答智能体

| 要求 | 实现情况 | 验证依据 |
| --- | --- | --- |
| 医疗相关数据 | 使用不含真实患者身份信息的合成慢病随访数据 | `data/`、`docs/data_and_model_provenance.md` |
| 实体抽取 | 规则召回 + 可选 BGE 标准化 | `chronic_entity_extract`、`chronic_entity_extract_model_npu` |
| 关系抽取 | 规则召回 + 可选 BGE 重排 | `chronic_relation_extract`、`chronic_relation_extract_model_npu` |
| 三元组校验 | 校验实体引用、关系类型和数据契约 | `chronic_triple_validate` |
| 自动编排 | Nexent 可触发完整链路或只重建图谱 DAG | `chroniccare_datamate_dag_plan`、`chroniccare_datamate_dag_run` |
| 图谱结构合理 | 当前图谱 197,404 节点、396,928 边、14 类实体、15 类关系 | `configs/current_metrics.json` |
| 图谱问答 | 支持实体、关系、患者路径和子图查询 | `chroniccare_kg_*` 工具 |
| 结果可解释 | 返回表格、关系证据、节点边统计和交互式子图 | `docs/assets/`中的12—17组前端实测截图；交互式子图由运行时按需生成 |
| NPU 优化 | 两个 BGE 算子提供真实 NPU 分支和 fallback 标记 | `npu_operator_benchmark_report.json` |

## 4. 任务三：数据分析智能体与图谱驱动可视化

| 要求 | 实现情况 | 验证依据 |
| --- | --- | --- |
| 自动感知图谱结构 | 图谱摘要、实体/关系类型和子图接口可被 Agent 调用 | `chroniccare_kg_summary` |
| 基于图谱分析 | 支持疾病相关指标、药物、风险及跨实体关系分析 | `chroniccare_kg_entity_query`、`chroniccare_kg_relation_query` |
| 自然语言触发 | Nexent 前端自然语言调用专用工具或 Open SQL | `chroniccare_agent_prompt.md` |
| 统计分析 | 疾病、共病、风险、随访、指标均值等确定性查询 | 专用分析工具 |
| 关联分析 | 疾病—指标—药物—风险关系和人群证据 | 图谱关系工具 |
| 趋势分析 | 月度指标趋势和未来任意 1—200 天随访趋势 | Open SQL、随访工具 |
| 基本规划与工具调用 | 单 Agent 通过明确路由规则选择工具 | Nexent Prompt + MCP Adapter |
| NL2SQL 准确率 | 240题本地固定工程回归盲测正确239题，总准确率99.58%；其中80题真实调用外部LLM，准确率98.75% | `outputs/evaluation/nl2sql_blind_eval_report.json` |
| BI 图表 | 支持表格、饼图、柱状图和折线图 | `docs/assets/`中的04—11、19—20组前端实测截图；图表由运行时按需生成 |
| 图谱可视化 | 支持概览页和实时交互式子图 | `docs/assets/`中的12—17组前端实测截图；子图由运行时按需生成 |
| 图文说明 | 回答同时返回统计口径、表格、图表/图谱和安全说明 | Nexent 前端截图 |
| 系统集成 | 复用任务一的数据和任务二的图谱 | 统一 SQLite、图谱和 MCP 工具层 |

## 5. 通用评审维度

### 5.1 技术完整性

- 三项任务和 NPU 加分项形成一条可运行链路。
- 关键结果读取真实数据库、图谱或本轮执行 Observation，不依赖固定答案文本。
- 提供结构化状态、错误信息、fallback 标记和可复核的 JSON 证据。
- 当前指标、前端回答和数据库基准值保持一致。

### 5.2 开源合规性

- 项目自研代码采用 MIT License。
- 提供 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md`。
- 模型权重、Nexent、DataMate、CANN、PyTorch 和 `torch_npu` 作为外部依赖，不随包冒充自研成果。
- `.env`、密钥、模型权重、日志、缓存和历史运行中间产物不进入正式发布包。

### 5.3 场景适配性

- 面向慢病随访中的多病共存、长期指标、风险分层和复诊计划。
- 普通查询为确定性快速路径，长任务提供状态和超时边界。
- CPU 环境可完成主线功能；Ascend 环境可启用 NPU 增强。
- 图谱全局入口采用概览，避免浏览器加载约 20 万节点。

### 5.4 创新性

- 将 Nexent 的自然语言规划、DataMate 算子、动态图谱和受控 SQL 统一到单 Agent。
- 通过动态 DAG 按目标裁剪数据流程，并提供失败恢复。
- 用专用工具和 SQL Guard 降低医疗数据分析中的自由生成风险。
- 只把 BGE 推理热点迁移到 NPU，保留 CPU 规则链路，形成可解释的软硬件协同方案。

