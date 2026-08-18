# MCP与API参考

## 1. 服务入口

| 服务 | 默认地址 | 作用 |
| --- | --- | --- |
| Tool Server | `http://127.0.0.1:18088` | 业务HTTP API和artifacts |
| MCP Adapter | `http://127.0.0.1:18188/mcp` | Nexent Streamable HTTP MCP入口 |
| 可选辅助Dashboard | `http://127.0.0.1:18501` | 本地产物查看；正式交互与测试以Nexent为准 |

当Nexent与ChronicCare不在同一主机或容器时，需要把`127.0.0.1`改为Nexent可访问的地址。

## 2. 工具设计原则

- 一个工具只负责一类明确问题。
- 统计值直接复用工具Observation，不由大模型二次计算。
- 普通查询读取当前SQLite或图谱；DataMate、DAG执行和NPU属于长任务。
- “重新执行”必须等待本轮执行结果，不能读取历史报告秒回。
- 图表和图谱返回浏览器可访问URL。
- Open SQL只执行通过SQL Guard的SELECT。
- 所有医疗输出带非诊断安全说明。

## 3. 核心工具分组

当前后端MCP服务公开38个工具；`integrations/nexent/chroniccare_tool_manifest.json`从中选择33个作为Nexent Agent默认绑定范围。后端保留的5个兼容入口不进入默认清单，2个调试工具继续保留绑定。

### 3.1 系统和数据

| 工具 | 用途 | 典型问题 |
| --- | --- | --- |
| `chroniccare_health_check` | 服务健康检查 | 系统现在是否正常？ |
| `chroniccare_data_summary` | 患者、随访、检验、用药和图谱规模 | 现在有多少患者、随访和检验记录？ |
| `chroniccare_report_summary` | 当前报告和产物入口 | 当前有哪些报告？ |

### 3.2 DataMate和动态DAG

| 工具 | 用途 | 重要说明 |
| --- | --- | --- |
| `chroniccare_datamate_pipelines` | 三条逻辑pipeline和11个主线算子 | 不用于回答NPU算子 |
| `chroniccare_datamate_pipeline_run` | 真实执行CPU/通用全流程 | 长任务，等待本轮结果 |
| `chroniccare_datamate_pipeline_run_npu` | 真实执行NPU增强全流程 | 必须返回fallback状态 |
| `chroniccare_datamate_pipeline_status` | 查询运行状态 | 只读 |
| `chroniccare_datamate_pipeline_latest` | 最近一次运行摘要 | 不等于重新运行 |
| `chroniccare_datamate_pipeline_report` | 完整运行报告 | 返回报告路径 |
| `chroniccare_datamate_dag_plan` | 生成动态DAG计划 | 不写业务产物 |
| `chroniccare_datamate_dag_run` | 执行或dry-run动态DAG | 返回run_id |
| `chroniccare_datamate_dag_status` | 查询DAG节点状态 | 需要run_id |
| `chroniccare_datamate_dag_resume` | 从失败节点恢复 | 输入变化时不能复用旧缓存 |

### 3.3 NPU

| 工具 | 用途 |
| --- | --- |
| `chroniccare_npu_readiness` | 检测NPU、CANN、模型和fallback条件 |
| `chroniccare_npu_supported_operators` | 返回2个NPU增强算子 |
| `chroniccare_npu_operator_benchmark` | 执行实体/关系CPU-NPU基准 |
| `chroniccare_datamate_pipeline_run_npu` | 运行包含NPU增强分支的完整链路 |

NPU支持工具只能宣称两个增强算子，不能把11个CPU/通用主线算子写成NPU算子。

### 3.4 知识图谱

| 工具 | 用途 |
| --- | --- |
| `chroniccare_kg_summary` | 节点、边、实体类型、关系类型和概览页 |
| `chroniccare_kg_entity_query` | 疾病关联指标、药物、风险等实体查询 |
| `chroniccare_kg_relation_query` | 两个实体间的关系和统计证据 |
| `chroniccare_kg_patient_path_query` | 患者相关路径查询 |
| `chroniccare_kg_subgraph_query` | 返回结构化局部图JSON |
| `chroniccare_kg_subgraph_render` | 实时生成预览和交互式HTML子图 |

询问图谱规模时，概览链接只展示规模与类型摘要，不直接渲染全部197,404个节点。询问具体疾病或关系图时使用实时子图工具。

### 3.5 专用分析

| 工具 | 用途 |
| --- | --- |
| `chroniccare_disease_distribution` | 20种疾病或单病人数/占比 |
| `chroniccare_disease_combination_distribution` | 精确共病组合 |
| `chroniccare_risk_level_distribution` | 最新风险等级人数 |
| `chroniccare_followup_high_risk` | 未来1—200天高风险随访人数和趋势 |
| `chroniccare_cohort_disease_distribution` | 指定人群疾病分布 |
| `chroniccare_graph_driven_analysis` | 图谱约束下的统计、关联和图表分析 |

询问系统支持的分析能力时，路由到 `chroniccare_open_sql_examples`，动态返回当前能力边界和示例问题，不使用固定题目总数描述系统能力。

### 3.6 Open SQL

| 工具 | 用途 |
| --- | --- |
| `chroniccare_open_sql_query` | 自然语言生成并执行受控查询 |
| `chroniccare_open_sql_schema` | 9张白名单表、字段和16个JOIN |
| `chroniccare_open_sql_eval` | Open SQL评测摘要 |
| `chroniccare_open_sql_examples` | 示例问题 |

`chroniccare_open_sql_query`可返回答案Markdown、结果表、生成SQL、来源表、SQL Guard结论、图表URL和`trace_id`。禁止DDL/DML、危险函数、多语句和非白名单关联。

## 4. 后端保留但不默认绑定给Nexent的兼容工具

以下5个历史兼容或重复入口保留后端实现，但不默认绑定到当前Nexent Agent：

| 工具 | 原因 |
| --- | --- |
| `chroniccare_analysis_query` | 旧固定分析兼容入口，已被专用工具替代 |
| `chroniccare_metric_query` | 历史指标入口，最终仍转发到Open SQL |
| `chroniccare_trend_query` | 历史趋势入口，当前优先Open SQL或随访工具 |
| `chroniccare_open_analysis_query` | 通用兜底，容易与专用工具重复选路 |
| `chroniccare_agent_run` | 再嵌套一层Agent会造成重复规划和超时 |

隐藏这些工具只改变Nexent默认工具面，不删除后端代码，便于兼容和回归。

## 5. 保留绑定的调试工具

| 工具 | 用途 | 使用边界 |
| --- | --- | --- |
| `chroniccare_open_sql_eval` | 查询开放式 NL2SQL 评测结果 | 不用于普通业务查询 |
| `chroniccare_trace_summary` | 查看最近MCP工具调用摘要 | 不返回患者敏感明细 |

调试工具用于工程证据查询和问题排查，Agent提示词限制其被普通医疗分析问题选中。

## 6. Open SQL白名单

当前9张表：

1. `patient_profile`
2. `visit_record`
3. `lab_result`
4. `medication_record`
5. `followup_plan`
6. `patient_risk_score`
7. `risk_event`
8. `lifestyle_record`
9. `doctor_advice`

允许的16个JOIN和字段级权限详见：

```text
outputs/evaluation/open_sql_schema_catalog.md
outputs/evaluation/open_sql_schema_catalog.json
```

## 7. HTTP接口示例

```bash
curl --noproxy "*" -sS http://127.0.0.1:18088/health
curl --noproxy "*" -sS http://127.0.0.1:18088/analysis/open-sql/schema
curl --noproxy "*" -sS http://127.0.0.1:18088/analysis/open-sql/eval
```

具体HTTP路由以`tool_server/`当前实现为准；Nexent集成优先使用MCP，不应依赖内部文件路径。

## 8. 长任务和错误处理

- DataMate和NPU任务需要更长超时。
- 工具返回`run_id`时，优先查询状态而不是重复启动。
- 任务失败必须显示错误和fallback，不返回旧成功结果。
- 图表/图谱URL不可访问时，检查public host、端口和SSH转发。
- NPU不可用时，功能可回退CPU，但性能结论无效。
