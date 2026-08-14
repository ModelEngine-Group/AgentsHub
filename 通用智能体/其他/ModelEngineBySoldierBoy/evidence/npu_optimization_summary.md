# NPU 算子优化总报告

更新时间：2026-07-06

本文汇总当前项目中已经完成的 Ascend NPU 算子优化与 CPU/NPU 对比证据。口径以赛题 `docs/比赛要求.md` 的加分项为准：完成算子的 NPU 优化实现，并验证性能与能效或资源利用率提升效果。

## 总体结论

当前已覆盖三项任务的真实算子层：

| 任务 | NPU 优化对象 | 证据结论 | 可声明范围 |
| --- | --- | --- | --- |
| 任务一 | DataMate operator market 候选排序 / DAG 规划输入 tensor ranker | wall-clock 基本持平，host CPU seconds/operator-score 改善 1.5921x | 智能体侧多算子组合与调度 ranker 支持 Ascend NPU |
| 任务二 | KG 实体候选召回、关系意图排序、三元组生成支撑 tensor 候选排序 | 核心 candidate scoring + top-k kernel 达到 15.3177x；4x 更大 score count 复测仍达到 14.0649x；包含 Python 后处理的大 batch 链路为 1.0165x | 医疗 KG 抽取前的 tensor candidate scoring 算子支持 Ascend NPU |
| 任务三 | 图谱多关系疾病相似度 tensor 算子 | pair-score 吞吐 10.1442x，p50 延迟 10.0889x | 图谱驱动分析中的 multi-relation similarity 算子支持 Ascend NPU |

不能宣称的内容：

- 不能宣称 DataMate 官方 Docker/API 内部清洗处理由本项目 NPU 加速。
- 不能宣称三任务全流程端到端都被 NPU 加速。
- 小 demo 端到端会包含 Python 解析、规则抽取、NPU 初始化和数据搬运开销，不作为强加速声明来源。

## 性能与资源对比

| 任务 | 指标 | CPU | NPU | 对比 |
| --- | --- | ---: | ---: | ---: |
| 任务一 ranker | throughput operator-scores/s | 16423.4631 | 16567.4057 | 1.0088x |
| 任务一 ranker | p50 latency ms | 795.2724 | 797.3402 | 0.9974x |
| 任务一 ranker | CPU seconds/operator-score | 0.0000995988 | 0.0000625586 | 1.5921x lower host CPU cost |
| 任务二 KG tensor | throughput docs/s | 2986.4442 | 3035.6406 | 1.0165x |
| 任务二 KG tensor | pipeline p50 ms | 260.8336 | 256.2488 | 1.0179x |
| 任务二 KG tensor | CPU seconds/document | 0.00085188 | 0.00033892 | 2.5135x lower host CPU cost |
| 任务二 KG kernel float32 | throughput candidate-scores/s | 664986983.827 | 10186039748.2617 | 15.3177x |
| 任务二 KG kernel float32 | p50 latency ms | 19.3511 | 1.309 | 14.7831x |
| 任务二 KG kernel bfloat16 | throughput candidate-scores/s | 3058408102.0865 | 16182824831.391 | 5.2913x |
| 任务二 KG kernel large-batch float32 | throughput candidate-scores/s | 822059726.2344 | 11562222602.0902 | 14.0649x |
| 任务二 KG kernel large-batch float32 | p50 latency ms | 62.89 | 4.5854 | 13.7153x |
| 任务二 KG kernel large-batch bfloat16 | throughput candidate-scores/s | 2111728385.7386 | 19751566921.909 | 9.3533x |
| 任务三 graph tensor | throughput pair-scores/s | 548830190.5659 | 5567461367.381 | 10.1442x |
| 任务三 graph tensor | p50 latency ms | 21.2987 | 2.1111 | 10.0889x |
| 任务三 graph tensor | CPU seconds/pair-score | 0.0000001349 | 0.0000000001 | 1349.0x lower host CPU cost |

## NPU 资源利用率口径

资源采样来自 benchmark 运行前后的 `resource.getrusage + npu-smi info`。其中 CPU seconds 是主机 CPU 占用代理指标，`npu-smi` 的 power/HBM 是设备快照；这些不是电表级能耗，但属于可获取的资源利用率证据。

| 任务 | NPU power before/after | HBM before/after | 证据文件 |
| --- | ---: | ---: | --- |
| 任务一 ranker | 89.2W / 89.3W | 3538MB / 3538MB | `docs/task1_tensor_ranker_cpu_npu_comparison.json` |
| 任务二 KG tensor | 89.1W / 89.3W | 3498MB / 3517MB | `benchmarks/results/tensor_kg_npu_comparison.json` |
| 任务二 KG kernel | 原始快照见 JSON | 原始快照见 JSON | `docs/task2_tensor_kernel_1_5x_npu_comparison.json` |
| 任务三 graph tensor | 90.3W / 94.3W | 3625MB / 3627MB | `docs/task3_tensor_analysis_cpu_npu_comparison.json` |

## 主链路接入

- 任务一：`agents/datamate_processing_agent.py` 在 DataMate operator market 候选排序后调用 `operators/task1_tensor_ranker.py`，输出 `operator_planning.tensor_ranker`，再进入原有 coverage-first DAG 搜索。
- 任务二：`agents/medical_kg_agent.py` 支持 `tensor` KG 后端，批量调用 `operators/npu_tensor_kg_ops.py`，trace 中记录 `tensor_device`、候选数量和算子描述；核心 kernel 使用预加载 tensor 的 matmul + top-k，CPU/NPU 对比达到 15.3177x。
- 任务三：`agents/analysis_agent.py` 在图谱分析中使用 `task3_multi_relation_similarity_tensor`，输出 `selected_device`、`pair_count` 和 `elapsed_ms`。

## 证据文件

- 任务一：`docs/task1_tensor_ranker_npu_optimization_report.md`、`docs/task1_tensor_ranker_cpu_npu_comparison.json`
- 任务二：`docs/task2_tensor_kernel_1_5x_npu_report.md`、`docs/task2_tensor_kernel_1_5x_npu_comparison.json`、`docs/task2_tensor_kernel_large_batch_npu_comparison.json`、`docs/tensor_kg_npu_optimization_report.md`、`benchmarks/results/tensor_kg_npu_comparison.json`、`docs/task2_tensor_pipeline_cpu_npu_evidence.json`
- 任务三：`docs/task3_tensor_npu_optimization_report.md`、`docs/task3_tensor_analysis_cpu_npu_comparison.json`、`docs/task2_task3_tensor_npu_chain_evidence.json`
