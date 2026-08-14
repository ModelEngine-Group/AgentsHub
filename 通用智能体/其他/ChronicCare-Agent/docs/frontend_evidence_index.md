# Nexent前端证据索引

## 1. 说明

本目录的42张图片来自Nexent前端真实提问结果，按当前20个代表性问题重新命名。问题的正确答案和解释统一写在[技术报告](技术报告.md)第7节；本文件只负责问题、能力、工具和图片证据之间的映射。

素材目录：`docs/assets/`

## 2. 证据映射

| 编号 | 主题 | 主要工具 | 图片 |
| --- | --- | --- | --- |
| 1 | DataMate支持算子 | `chroniccare_datamate_pipelines`、`chroniccare_npu_supported_operators` | `01-datamate-operator-overview-01.png`、`01-datamate-operator-overview-02.png` |
| 2 | DataMate CPU全流程重跑 | `chroniccare_datamate_pipeline_run` | `02-datamate-cpu-pipeline-run-summary.png`、`02-datamate-cpu-pipeline-run-operator-timings.png` |
| 3 | NPU增强全流程与CPU/NPU对比 | `chroniccare_datamate_pipeline_run_npu` | `03-npu-full-pipeline-run-summary.png`、`03-npu-cpu-same-sample-comparison.png`、`03-npu-full-run-conclusions.png` |
| 4 | 当前数据规模 | `chroniccare_data_summary` | `04-data-summary.png` |
| 5 | 20种疾病分布 | `chroniccare_disease_distribution` | `05-disease-distribution-table.png`、`05-disease-distribution-chart.png` |
| 6 | 高血压患者人数 | `chroniccare_disease_distribution` | `06-hypertension-patient-count.png` |
| 7 | 共病组合分布 | `chroniccare_disease_combination_distribution` | `07-comorbidity-distribution-table.png`、`07-comorbidity-distribution-chart.png` |
| 8 | 风险等级分布 | `chroniccare_risk_level_distribution` | `08-risk-level-distribution.png` |
| 9 | 未来30天高风险随访 | `chroniccare_followup_high_risk` | `09-high-risk-followup-30d-table.png`、`09-high-risk-followup-30d-chart.png` |
| 10 | 未来67天高风险随访 | `chroniccare_followup_high_risk` | `10-high-risk-followup-67d-summary.png`、`10-high-risk-followup-67d-daily-table.png` |
| 11 | 最近6个月血压异常趋势 | `chroniccare_open_sql_query` | `11-blood-pressure-abnormal-trend-6m.png` |
| 12 | 知识图谱规模和概览 | `chroniccare_kg_summary` | `12-knowledge-graph-summary.png`、`12-knowledge-graph-overview-page.png` |
| 13 | 高血压关联指标、药物和风险 | `chroniccare_kg_entity_query` | `13-hypertension-knowledge-query-01.png`、`13-hypertension-knowledge-query-02.png` |
| 14 | 高血压知识图谱子图 | `chroniccare_kg_subgraph_render` | `14-hypertension-subgraph-answer.png`、`14-hypertension-subgraph-overview.png`、`14-hypertension-subgraph-interactive-graph.png`、`14-hypertension-subgraph-node-table.png` |
| 15 | 高血压合并糖尿病子图 | `chroniccare_kg_subgraph_render` | `15-hypertension-diabetes-subgraph-answer.png`、`15-hypertension-diabetes-subgraph-overview.png`、`15-hypertension-diabetes-subgraph-interactive-graph.png` |
| 16 | 高盐饮食与血压异常证据 | `chroniccare_kg_relation_query`、`chroniccare_kg_subgraph_render` | `16-high-salt-blood-pressure-relation-answer.png`、`16-high-salt-blood-pressure-evidence.png`、`16-high-salt-blood-pressure-subgraph-overview.png`、`16-high-salt-blood-pressure-interactive-graph.png` |
| 17 | 糖尿病、HbA1c、用药和风险子图 | `chroniccare_kg_subgraph_render` | `17-diabetes-hba1c-medication-risk-subgraph-answer.png`、`17-diabetes-hba1c-medication-risk-subgraph-overview.png`、`17-diabetes-hba1c-medication-risk-interactive-graph.png` |
| 18 | 动态DAG规划 | `chroniccare_datamate_dag_plan` | `18-dynamic-dag-plan-summary.png`、`18-dynamic-dag-plan-dependencies.png` |
| 19 | Open SQL Schema与安全规则 | `chroniccare_open_sql_schema` | `19-open-sql-schema-tables.png`、`19-open-sql-join-whitelist-safety.png` |
| 20 | 高血压合并糖尿病平均HbA1c | `chroniccare_open_sql_query` | `20-open-sql-hba1c-result-and-sql.png` |

## 3. 证据说明

- 42张截图构成完整的Nexent前端验证证据集，技术报告收录其中能够对应关键评分点的代表性截图。
- NPU证据由CPU 2,048条、NPU 2,048条和NPU全量三列同批次结果共同构成。截图记录一轮独立前端运行；正式基准采用`outputs/evaluation/npu_operator_benchmark_report.json`中的另一轮独立实测，受共享服务器负载、预热和测量时点影响，两轮数值允许波动且不混算。
- 图谱证据同时包含Nexent回答和成功打开的交互式页面。
- 数据规模、疾病、风险和随访证据保留表头与统计口径。
- 截图不包含API Key、服务器账号、私有域名或真实个人信息。

## 4. 技术报告代表图

1. DataMate算子总览：`01-datamate-operator-overview-01.png`
2. DataMate本轮执行：`02-datamate-cpu-pipeline-run-summary.png`
3. CPU/NPU三列对比：`03-npu-cpu-same-sample-comparison.png`
4. 数据规模：`04-data-summary.png`
5. 疾病图表：`05-disease-distribution-chart.png`
6. 未来30天随访趋势：`09-high-risk-followup-30d-chart.png`
7. 图谱规模和概览：`12-knowledge-graph-overview-page.png`
8. 高血压子图：`14-hypertension-subgraph-interactive-graph.png`
9. 高盐饮食关系证据：`16-high-salt-blood-pressure-evidence.png`
10. 动态DAG：`18-dynamic-dag-plan-dependencies.png`
11. Open SQL安全：`19-open-sql-join-whitelist-safety.png`
12. SQL和真实结果：`20-open-sql-hba1c-result-and-sql.png`
