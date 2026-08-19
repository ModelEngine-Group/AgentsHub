# Ascend NPU 实验总结

最后更新：2026-06-24（NPU 服务器在线复跑验证）

本文汇总当前 NPU 优化工作的最新结论与复现入口。文件名中的 `910b2c` 为历史命名，正文硬件以 **910B3** 为准。`benchmarks/reports/` 下最新 JSON 可能被服务器复跑覆盖，重要历史快照保存在 `benchmarks/reports/history/`。

## 2026-06-24 在线 NPU 复跑结论

本轮在 Ascend **910B3**（aarch64，CANN 8.5.0，npu-smi 25.5.0）服务器上拉取 `final-version` 分支后复跑。`/data/npu_env.sh` 已配置（CANN + ModelScope 缓存 + PYTHONPATH）。

### 环境与全量验证

| 检查项 | 结果 |
| --- | --- |
| NPU 张量检查 | `torch.randn(..., device='npu:0')` 通过 |
| `npu-smi info` | 910B3，Health=OK，HBM 65536 MB（已用 3446 MB），npu-smi 25.5.0，功率 99.6W / 温度 33°C |
| 一键验证 | `bash benchmarks/scripts/run_npu_full_verify.sh`（16 步全部完成，exit 0） |
| pytest | **770/770 passed** in 369.30s（插卡后；插卡前无卡预配置 **737/737**，2026-06-24） |
| NPU 专项 pytest | **43/43 passed** in 21.60s（kg/graph tensor ops + task2/task3 benchmark） |
| Task 2 真实链路 | `demos/task2_demo.py --relation-backend npu` 完成，`relation=npu device=npu:0`，29 triples / 26 nodes / 29 edges |
| Task 1 数据质量 | 通过（3 iterations，5→4 行，quality 0.8→1.0） |
| 服务可达性 | Neo4j/DataMate/Nexent 均 unreachable（NPU 节点外部服务不可达） |

完整服务器环境见 [docs/server_environment.md](../../docs/server_environment.md)。

### 关键 NPU 证据

| 报告 | 核心结果 |
| --- | --- |
| `task2_topk_65k.json` | `correctness.status=passed`，NPU `backend=torch_npu device=npu:0`；65k/256/5 下 CPU full 66.43 ms，NPU end-to-end 115.42 ms，`cached_topk_labels` 1.078 ms，对 full-format CPU 为 **61.64x** |
| `task2_relation_tensor_ascend_910b2c_xlarge.json` | `correctness.status=passed`；131072/768/16 下 CPU full 138.85 ms，`cached_topk_labels` 1.389 ms，对 full-format CPU 为 **99.95x** |
| `task2_relation_quality_ascend_910b2c_npu.json` | `backend=npu`，`passed=true`，关系级 **P/R/F1 均为 1.0**（46 TP / 0 FP / 0 FN） |
| `task3_graph_tensor_ascend_910b2c_large.json` | `correctness.status=passed`，`prepared_correctness.status=passed`；5k/50k 下 CPU 45.00 ms，NPU prepared 5.64 ms，`cached_bincount_topk` 1.62 ms / **27.77x**（prepared kernel 7.98x） |
| `task3_centrality_5k.json` | `correctness.status=passed`，`cached_correctness.status=passed`，`top_hubs_backend=torch_npu`；CPU 59.52 ms，cached NPU 51.47 ms / **1.16x** |

### NPU 利用率/功率采样

`task2_relation_tensor_ascend_910b2c_xlarge.json` 包含 `npu_utilization`：采样 44 次（0.2s 间隔），NPU 利用率 min/avg/max 为 0.0% / 0.068% / 3.0%，AICore 均为 0.0%，HBM 占用 5.0%，功率 min/avg/max 为 101.4W / 106.95W / 111.5W。该任务输出列数较小、kernel 很短，采样值偏低；设备执行证据以 `torch_npu` backend、`device=npu:0` 和 correctness passed 为主。

> 历史口径（2026-06-16：910B2C / 380/380 / 72.84x / 79.60x / 12.70x / 1.11x；2026-06-14：289/289 / 114.61x / 83.05x）数值保留于本文记录；同名 JSON 已被 2026-06-24 复跑覆盖，`benchmarks/reports/history/` 仅存 2026-05-30/31 早期快照。历史口径仅作对照，不作为当前结论依据。

## 实验最终目标

NPU 验证需分层覆盖以下能力，**上述四层均已达成**：

1. **运行时就绪**：CANN、`acl`、`torch_npu`、`torch.npu` 可用，张量 kernel 可在 `npu:0` 执行。
2. **设备适配就绪**：项目代码可通过统一适配器选择 `npu/cuda/cpu`，本地小模型推理路径不依赖仅 CUDA 的加载逻辑。
3. **算子加速**：任务二/任务三核心算子具备 CPU 基线对照、真实 NPU 实现、正确性检查、时延/吞吐测量及可重复服务器报告。
4. **业务路径集成**：任务三分析路径可明确使用 NPU 算子，演示输出可证明 `top_hubs` 来自 `torch_npu` 而非 Python 回退。

## 优化方法论

任务二与任务三采用相同三步方法：

1. **拆分 profiling（含 warmup）**：定位真实瓶颈。任务二瓶颈在 `format_result`（67%）与 `h2d_features`（11%）；任务三在 `edge_index_build`+`format_result`（81%）。
2. **针对性优化模式**：张量缓存到设备侧（`cached_*`）、仅返回 top-k/argmax（`cached_topk_labels`、`cached_bincount_topk`）、更高效 kernel（`bincount` 替代 `index_add_`）。
3. **正确性验证**：每种模式报告 `correctness.status`，使用容差检查处理 CPU/NPU 浮点差异，当前全部 passed。

详细方法论与 profiling 拆分表见 [docs/npu_optimization.md](../../docs/npu_optimization.md)。

## 复现命令

一键全量验证（仅 Ascend Linux，16 步）：

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
source /data/npu_env.sh
python -m pip install -r requirements-npu.txt
bash benchmarks/scripts/run_npu_full_verify.sh
# 或兼容入口：bash benchmarks/scripts/run_full_verify.sh
```

可选：`SKIP_XLARGE=1` · `SKIP_REACHABILITY=1` · `SKIP_ENV_SNAPSHOT=1`

分步命令与参数说明见 [docs/server_environment.md](../../docs/server_environment.md)。

## 关键文件

| 文件 | 用途 |
| --- | --- |
| `src/common/device.py` | 共享设备适配器（npu/cuda/cpu） |
| `src/operators/npu_ops/kg_tensor_ops.py` | 任务二关系打分算子 |
| `src/operators/npu_ops/graph_tensor_ops.py` | 任务三图度数算子 |
| `src/operators/analysis_ops/graph_analytics.py` | 任务三中心性业务路径集成 |
| `benchmarks/task2_relation_tensor_benchmark.py` | 任务二关系张量基准测试命令行（支持 `--real-corpus`/`--monitor-npu`） |
| `benchmarks/task3_graph_tensor_benchmark.py` | 任务三图张量基准测试命令行 |
| `benchmarks/task3_centrality_benchmark.py` | 任务三中心性基准测试命令行 |
| `benchmarks/npu_monitor.py` | `npu-smi` 后台利用率/功率采样器 |
| `benchmarks/task2_relation_quality_benchmark.py` | 关系级 P/R/F1 质量评测（rule/cpu/npu） |
| `benchmarks/service_reachability_probe.py` | Neo4j/DataMate/Nexent 协议级可达性校验 |
| `benchmarks/scripts/run_npu_full_verify.sh` | NPU 一键全量验证（16 步，推荐） |
| `benchmarks/scripts/collect_env.sh` | 服务器环境采集脚本 |
