# 基准测试目录

本目录存放基准测试脚本及 CPU/NPU 对比辅助工具。

## 任务一数据质量基准测试

```bash
python benchmarks/task1_data_quality_benchmark.py
python benchmarks/task1_data_quality_benchmark.py --report benchmarks/reports/task1_data_quality.json
```

任务一基准测试衡量确定性数据清洗质量，而不是 NPU 加速。默认样例为
`data/samples/task1_patients.csv`，指标包括输入/输出行数、重复行删除数、缺失值填补数、
清洗前后质量分和平均运行耗时。当前报告中样例 CSV 为 5 行 -> 4 行，重复行 1 -> 0，
缺失值 3 -> 0，质量分 0.8 -> 1.0。

## 任务二抽取质量基准测试

```bash
python benchmarks/task2_extraction_quality_benchmark.py \
  --gold benchmarks/data/kg_extraction_gold.json \
  --report benchmarks/reports/task2_kg_extraction_quality.json

python benchmarks/task2_relation_quality_benchmark.py \
  --gold benchmarks/data/kg_relation_gold.json \
  --report benchmarks/reports/task2_relation_quality_rule.json
```

在 30 条人工标注病历（`benchmarks/data/kg_extraction_gold.json`、`kg_relation_gold.json`）上逐条对比预测与标注，产出实体 161/161、关系 145/145 等指标。与 `task2_kg_benchmark.py`（算子性能）和默认 demo（4 条样例 26/29）不是同一评测。

## 任务二知识图谱基准测试

```bash
python benchmarks/task2_kg_benchmark.py --iterations 20
python benchmarks/task2_kg_benchmark.py --iterations 20 --report outputs/task2/task2_kg_benchmark.json
```

任务二基准测试衡量确定性 CPU 算子链，并探测支持的 NPU 运行时。若不存在可用 NPU 运行时，报告将 NPU 指标标记为 `unavailable`，而不是虚构加速比或能耗数据。

## 任务二关系张量基准测试

```bash
python benchmarks/task2_relation_tensor_benchmark.py --candidate-count 4096 --feature-dim 256 --relation-count 5 --iterations 20
python benchmarks/task2_relation_tensor_benchmark.py --candidate-count 65536 --feature-dim 256 --relation-count 5 --iterations 20 --benchmark-modes all --profile-breakdown --report benchmarks/reports/task2_topk_65k.json
```

本基准测试是任务二算子级 NPU 路径，将关系抽取候选打分建模为张量矩阵乘法：

- 输入：确定性合成实体对特征
- CPU 基线对照：`torch` CPU 关系打分
- NPU 实现：`npu:0` 上的 `torch_npu` 关系打分
- 正确性：对比 CPU/NPU logits 与预测关系标签
- 报告字段：候选数、特征维度、关系数、CPU/NPU 时延、吞吐、加速比、正确性及运行时/设备信息
- 测试模式：`baseline_full_logits`、`cached_full_logits`、`cached_argmax_labels`、`cached_topk_labels`、`cpu_topk_labels`
- `--profile-breakdown`：记录 NPU 各步骤耗时，如 h2d 传输、matmul、d2h 回传、argmax 及 Python 结果格式化

Ascend 服务器上无需 DataMate 或 Nexent Web 服务，可直接针对 NPU 运行时验证可复用关系打分算子。

Ascend 910B3 最新要点（2026-06-24 复跑，详见 `reports/ascend_910b2c_experiment_summary.md`）：

- 65k `cached_topk_labels`：61.64×（对 full-format CPU），1.078 ms
- xlarge `cached_topk_labels`：99.95×（对 full-format CPU），1.389 ms
- 5k/50k `cached_bincount_topk`：27.77×；中心性 cached 路径：1.16×

## 任务三图分析基准测试

```bash
python benchmarks/task3_analysis_benchmark.py --iterations 20
python benchmarks/task3_analysis_benchmark.py --iterations 20 --report outputs/task3/task3_analysis_benchmark.json
```

任务三基准测试衡量确定性图分析链：统计、关联分析、趋势、NL2SQL 与可视化 spec 生成。仅在检测到支持的运行时时才报告 NPU 指标。

## 任务三图张量基准测试

```bash
python benchmarks/task3_graph_tensor_benchmark.py --nodes 1000 --edges 10000 --iterations 20
python benchmarks/task3_graph_tensor_benchmark.py --nodes 1000 --edges 10000 --iterations 20 --report benchmarks/reports/task3_graph_tensor_ascend_910b2c.json
python benchmarks/task3_graph_tensor_benchmark.py --nodes 5000 --edges 50000 --iterations 20 --amortized-runs 1,2,5,10,20
```

本基准测试是首个算子级 CPU/NPU 对比路径，在 Ascend NPU 可用时将 Python 度中心性与 `npu:0` 上 `torch` 张量 `index_add_` 实现对比。合成图生成是确定性的，且不计入 CPU/NPU 计时区间。

报告区分两类 NPU 测量：

- `npu`：端到端图到张量执行，含边索引解析、设备张量创建、NPU 同步及度向量回传 CPU。
- `npu_prepared`：已预热的 kernel 执行，`source/target/ones/output` 张量已驻留 `npu:0`；计时区间仅含重复 `zero_` + `index_add_`，不含结果回传开销。
- `amortized`：重复图分析运行的 prepare-once 总耗时。使用 `cpu_total_ms = cpu.latency_ms_avg * runs` 与 `npu_prepared_total_ms = npu_prepared.prepare_latency_ms + npu_prepared.latency_ms_avg * runs`，并报告对应加速比与首次盈亏运行次数。

## 任务三中心性集成基准测试

```bash
python benchmarks/task3_centrality_benchmark.py --nodes 1000 --edges 10000 --iterations 20
python benchmarks/task3_centrality_benchmark.py --nodes 1000 --edges 10000 --iterations 20 --report benchmarks/reports/task3_centrality_ascend_910b2c.json
python benchmarks/task3_centrality_benchmark.py --nodes 5000 --edges 50000 --iterations 20 --benchmark-modes all --amortized-runs 1,2,5,10,20
python benchmarks/task3_centrality_benchmark.py --nodes 5000 --edges 50000 --iterations 20 --benchmark-modes all --multi-type --report benchmarks/reports/task3_centrality_5k_multi.json
```

本基准测试对比完整 CPU `compute_centrality` 路径与优化后的任务三路径：在可用时使用缓存 NPU `bincount + topk` 算子计算 `top_hubs`，并用 NPU `scatter_add_` 做类型聚合。报告图规模、CPU 时延、未缓存优化时延、缓存优化时延、`top_hubs_backend`、类型中心性后端、正确性、加速比及摊销 prepare-once 总耗时。使用 `--multi-type` 可生成含多种节点类型的合成医疗 KG。

经整理的报告可提交到 `benchmarks/reports/`。本地运行生成的 JSON 应保留在已忽略的 `outputs/` 目录。

## 任务三 NL2SQL 准确率基准测试

```bash
python benchmarks/task3_nl2sql_benchmark.py --report benchmarks/reports/task3_nl2sql_report.json
```

NL2SQL 基准测试在标注数据集上衡量模板意图、执行级与改写回归准确率。`--holdout-benchmark` 参数名保留历史兼容，对应扩展改写集。经整理的 Markdown 摘要位于 `benchmarks/reports/`；JSON 报告可按需重新生成。

## Ascend NPU 一键复验（Linux 服务器）

NPU 命令**不在 Windows/WSL 上执行**。在 Ascend 910B3 Linux 服务器上：

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
source /data/npu_env.sh
python -m pip install -r requirements-npu.txt
bash benchmarks/scripts/run_npu_full_verify.sh
```

脚本步骤（16 步）：NPU 自检 → 环境快照 → 任务一/二/三演示（含 `task2 --relation-backend npu`）→ 全量 pytest + NPU 子集 → 任务一质量基准测试 → 任务二 4k/65k/xlarge 关系张量 → 关系级 NPU 质量 → 任务三图张量与中心性 → 服务可达性探测。`run_full_verify.sh` 为兼容入口。

可选：`SKIP_XLARGE=1` 跳过 131072 超大负载；`SKIP_REACHABILITY=1` 跳过可达性探测。详见 [docs/npu_optimization.md](../docs/npu_optimization.md) 与 [docs/server_environment.md](../docs/server_environment.md)。
