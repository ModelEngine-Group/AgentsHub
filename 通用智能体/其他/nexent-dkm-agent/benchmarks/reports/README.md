# 基准测试报告目录

本目录存放已验证的 CPU/NPU 基准测试 JSON 与总结文档。脚本可复跑覆盖同名 JSON，重要历史快照复制到 `history/`。

## 当前最新报告（2026-06-24 Ascend 910B3 快照）

| 文件 | 内容 |
| --- | --- |
| `task1_data_quality.json` | 任务一确定性数据质量基准测试（去重、缺失值、质量分、时延） |
| `task1_datamate_submit.json` | 任务一 DataMate submit 基准（DataMate 不可用时跳过并指向在线证据） |
| `task2_topk_4k.json` | 任务二关系张量 4k 候选，全模式 CPU/NPU 对比 |
| `task2_topk_65k.json` | 任务二关系张量 65k 候选，含 profile 分解（主加速比证据） |
| `task2_relation_tensor_ascend_910b2c_xlarge.json` | 任务二 131072/768/16 超大负载 + NPU 利用率/功率采样 |
| `task2_relation_quality_ascend_910b2c_npu.json` | 任务二关系级 P/R/F1（NPU backend） |
| `task2_kg_extraction_quality.json` | 任务二实体/关系抽取质量（30 条人工标注病历） |
| `task2_oov_extraction_quality.json` | 任务二 OOV 评测（8 条语料，词典外实体 **22/22**） |
| `task2_pipeline_latency.json` | 任务二端到端流水线各阶段时延 |
| `task2_relation_quality_rule.json` | 任务二关系级质量（规则基线） |
| `task2_neo4j_live_smoke.json` | 任务二 Neo4j 冒烟测试（**26/29**，与 §3.3 同输入）；在线读回见 `../../competition_submission/defense-package-final/evidence/online_integration/task2-neo4j-live-smoke-20260702-final.json` |
| `task2_relation_quality_cpu.json` | 任务二关系级质量（CPU 张量路径，30 条人工标注病历） |
| `task2_relation_tensor_real_corpus.json` | 任务二真实语料张量正确性 |
| `task3_graph_tensor_ascend_910b2c_large.json` | 任务三图张量 5k/50k CPU/NPU 对比 |
| `task3_centrality_5k.json` | 任务三中心性 5k/50k，cached NPU 路径 |
| `task3_nl2sql_report.json` | 任务三 NL2SQL 准确率 |
| `service_reachability_ascend_910b2c.json` | NPU 节点服务可达性探测 |

## 可读总结

- `ascend_910b2c_experiment_summary.md`：NPU 实验时间线、关键结论与复现命令（推荐先读）。
- `task3_nl2sql_accuracy_report.md`：任务三模板 NL2SQL 准确率说明。

## 历史快照

`history/` 保存不可变的历史运行快照，命名规则 `<task>_ascend_910b2c_<YYYYMMDD>_<stage>.json`，供追溯优化过程，不作为当前结论依据。

## 复跑

一键全量验证（仅 Ascend Linux）：

```bash
bash benchmarks/scripts/run_npu_full_verify.sh
```

分步命令与可选参数见 [server_environment.md](../../docs/server_environment.md) 与 [npu_optimization.md](../../docs/npu_optimization.md)。
