# 最终验收报告

- 生成时间：2026-07-02T23:05:53+08:00
- 总体结论：通过
- CBLUE limit：300
- CBLUE 数据模式：official/local source
- 命令日志目录：`data/processed/final_acceptance`

## 命令执行

| 步骤 | 状态 | 日志 |
| --- | --- | --- |
| Python unit tests | 通过 | `data/processed/final_acceptance/unit_tests.log` |
| Extract and convert CBLUE CMeIE/CMeEE | 通过 | `data/processed/final_acceptance/prepare_cblue.log` |
| Build demo with converted CBLUE artifacts | 通过 | `data/processed/final_acceptance/build_cblue_demo.log` |
| Run end-to-end three-agent orchestration | 通过 | `data/processed/final_acceptance/orchestrate_cblue_demo.log` |
| Task-3 NL2SQL evaluation | 通过 | `data/processed/final_acceptance/nl2sql_eval.log` |
| CPU KG operator benchmark | 通过 | `data/processed/final_acceptance/cpu_benchmark.log` |
| Refresh DataMate/Nexent integration evidence | 通过 | `data/processed/final_acceptance/integration_evidence.log` |

## 验收检查

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| Command steps | 通过 | 0 failed |
| Required artifacts | 通过 | 0 missing |
| CBLUE CMeIE converted | 通过 | 300 docs |
| CBLUE CMeEE converted | 通过 | 300 docs |
| CBLUE CMedCausal converted | 通过 | 2009 triples |
| KG rejected triples | 通过 | 0 rejected |
| KG relation coverage | 通过 | 15 relation types |
| KG size | 通过 | 3403 accepted triples |
| KG gold quality evaluation | 通过 | P=94.62%, R=100.00%, F1=97.24%, matched=3220/3220 |
| KG backend extraction probe | 通过 | mode=lexicon_assisted_backend, P=79.22%, R=75.31%, F1=77.22%, docs=40 |
| Artifact count consistency | 通过 | jsonl=320/4468/3403; analysis=320/4468/3403; orchestration=320/4468/3403; kg accepted=3403 |
| Three-agent orchestration | 通过 | DataProcessingAgent -> MedicalKGAgent -> InsightAnalysisAgent |
| Orchestration agent count | 通过 | 3 agents |
| NL2SQL threshold | 通过 | 100.00% accuracy |
| CPU benchmark | 通过 | 908.1434 docs/s |
| Integration refresh | 通过 | fresh collection requested; datamate_search=success, results=5 |

## CBLUE 转换

| 数据集 | 文档数 | 记录数 |
| --- | ---: | ---: |
| CMeIE | 300 | 1224 triples |
| CMeEE | 300 | 793 entities |
| CMedCausal | 300 | 2009 triples |

## 知识图谱

- 文档数：320
- 实体数：4468
- 三元组数：3403
- 候选三元组数：3418
- 接受三元组数：3403
- 拒绝三元组数：0
- Gold 标注级 F1：97.24%
- Gold 标注级 Recall：100.00%
- 后端抽取 Probe：lexicon_assisted_backend，F1 77.22%，Recall 75.31%

### 计数一致性

| 来源 | 文档 | 实体 | 三元组 |
| --- | ---: | ---: | ---: |
| JSONL 文件 | 320 | 4468 | 3403 |
| analysis_summary | 320 | 4468 | 3403 |
| orchestration_summary | 320 | 4468 | 3403 |
| kg_validation_report | - | - | 3403 |

| 关系 | 数量 |
| --- | ---: |
| `belongs_to_department` | 3 |
| `causes` | 2153 |
| `contraindicated_for` | 2 |
| `diagnosed_by` | 158 |
| `has_complication` | 47 |
| `has_epidemiology` | 10 |
| `has_history` | 5 |
| `has_patient_group` | 26 |
| `has_risk_factor` | 69 |
| `has_stage` | 138 |
| `has_symptom` | 414 |
| `located_in` | 43 |
| `prevented_by` | 40 |
| `spreads_via` | 1 |
| `treated_by` | 294 |

## 实体覆盖

| 类型 | 数量 |
| --- | ---: |
| Anatomy | 42 |
| ClinicalStage | 135 |
| Department | 6 |
| Disease | 880 |
| Drug | 190 |
| Epidemiology | 10 |
| History | 5 |
| PatientGroup | 24 |
| RiskFactor | 954 |
| Symptom | 1704 |
| Test | 216 |
| TransmissionRoute | 1 |
| Treatment | 301 |

## 任务三与 Benchmark

- NL2SQL：70/70，准确率 100.00%，阈值 85.00%
- 端到端编排：DataProcessingAgent -> MedicalKGAgent -> InsightAnalysisAgent，状态 completed
- CPU benchmark 吞吐：908.1434 docs/s
- CPU benchmark pipeline p50：22.023 ms

## 集成证据

- 本次已请求刷新 DataMate/Nexent 集成证据。
- 现有 platform evidence 时间：2026-07-02T23:05:53+08:00
- 服务探测通过数：6/6
- datamate_search 状态：success

## 后续

- 本地代码、转换、三智能体编排、NL2SQL 和 CPU benchmark 已形成可复现验收闭环。
- 如需刷新平台链路证据，在本地 WSL 终端确认 Docker/DataMate/Nexent 可达后运行：

```bash
bash scripts/run_final_acceptance.sh --run-integration --knowledge-base-id <DataMate知识库ID>
```
