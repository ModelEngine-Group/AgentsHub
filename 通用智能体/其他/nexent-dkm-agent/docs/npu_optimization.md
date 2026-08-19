# NPU 优化说明

本文整理 Nexent DKM Agent 项目的 NPU 优化工作，覆盖优化方法、已验证结果和结果解读。

> **文件名说明**：`benchmarks/reports/` 下若干 JSON 与 `ascend_910b2c_experiment_summary.md` 保留 `910b2c` 历史前缀；正文硬件与 2026-06-24 复跑结论以 Ascend 910B3 为准，内容已对应当前服务器快照。

## 目标

竞赛 NPU 算子优化要求在真实业务算子上展示 Ascend NPU 加速效果。本项目选择任务二（知识图谱）和任务三（图分析）中的核心算子进行验证，并交付以下能力：

1. **运行时可用性**：CANN、`acl`、`torch_npu` 和 `torch.npu` 均可用。
2. **设备适配能力**：本地代码可通过 `src.common.device.get_device()` 在 `npu/cuda/cpu` 之间选择执行设备。
3. **算子加速能力**：任务二关系打分算子和任务三图度数算子均包含 CPU 基线对照、NPU 实现和正确性检查。
4. **业务路径集成**：任务三演示能明确展示 NPU 图算子已接入业务路径（`top_hubs_backend=torch_npu`）。

所有结果均在 Ascend NPU（2026-06-24 复验为 910B3 / aarch64；历史口径 910B2C / x86_64）上完成验证，并包含完整正确性检查。只有在 `correctness.status=passed` 的前提下，本文才记录加速比。

> **版本边界**：NPU 数据已于 2026-06-24 在 Ascend 910B3（aarch64，CANN 8.5.0，npu-smi 25.5.0）上基于 `final-version` 分支复跑。
> 当前 NPU 服务器回归为 **770/770 passed**（369.30s），NPU 专项 **43/43**；任务二关系级 NPU 质量历史快照为 P/R/F1=1.0（46/46，10 条人工标注病历）；扩充至 30 条标注病历后，CPU 张量路径本地复跑为 P/R/F1=1.0（145/145），见 `task2_relation_quality_cpu.json`；NPU 30 条标注病历复跑需在 Ascend 上执行。
> 下文保留 2026-06-16 / 2026-06-14 历史快照段落用于解释优化过程；当前结论以 2026-06-24 JSON 与本节最新摘要为准。

## 近期补充能力（阶段 1A–6）

在原有「独立算子基准测试」之外，近期进一步把 NPU 接入真实业务链路，并补齐质量评测、能效与证据。最新可复现数值以 [实验总结](../benchmarks/reports/ascend_910b2c_experiment_summary.md) 为准。

- **1A 真实链路集成**：NPU 张量关系打分接入真实任务二抽取链路（`relation_features.py` + `extract_relations_tensorized`，`--relation-backend rule|cpu|npu`）；2026-06-24 复跑中 `demos/task2_demo.py --relation-backend npu` 完成，输出 `relation=npu device=npu:0`，当前图谱为 29 triples / 26 nodes / 29 edges。
- **1B/1C 能效与超大负载**：`benchmarks/npu_monitor.py` 后台采样 `npu-smi`（利用率/功率）；2026-06-24 复跑任务二 131072 候选/768 维/16 关系 `cached_topk_labels` ≈ **99.95x**（对 full-format CPU），任务三 5k/50k `cached_bincount_topk` ≈ **27.77x**，xlarge 采样 44 次，NPU 利用率 0.0%–3.0%、功率 101.4–111.5W。
- **3 关系级质量**：`kg_relation_gold.json` 已扩充至 30 条人工标注病历；实体抽取 P/R/F1=1.0（161/161），规则/CPU 张量关系 P/R/F1=1.0（145/145）。Ascend 910B3 历史 NPU 快照仍为 10 条标注病历 46/46（`task2_relation_quality_ascend_910b2c_npu.json`）；30 条标注病历 NPU 复跑待 Ascend 环境执行。
- **4 NL2SQL 分路径 + 图谱洞察**：`evaluate_nl2sql_execution_accuracy` 支持分路径准确率；`insight_report` 新增图谱驱动自然语言洞察。
- **5 服务可达性**：`benchmarks/service_reachability_probe.py` 协议级校验记录了 NPU 节点的服务状态：Neo4j/DataMate/Nexent 均不可达，端口 3000 经 Server 头确认为 Jupyter，而非 Nexent。
- **6 证据**：`figure_export` 输出 NPU 模式加速图与利用率/功率图，证据收集器补 `npu-smi` 快照与新报告。

## 服务器环境

| 组件 | 值（2026-06-24 实测） |
| --- | --- |
| 硬件 | Ascend 910B3（NPU 0），65536 MB HBM，OK |
| 架构 | aarch64 |
| OS | openEuler 24.03 (LTS-SP2) |
| CANN | 8.5.0 |
| npu-smi | 25.5.0 |
| Python | 3.11.14 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0 |
| pytest | **770/770 passed**，0 个失败，369.30s（2026-06-24 最新 NPU 复跑；2026-06-16 历史为 380/380；2026-06-14 历史快照为 289/289） |

完整环境信息见：[server_environment.md](server_environment.md)。

### 无卡预配置（2026-06-24）

软件栈与依赖可在无 NPU 设备时预先安装。当前容器已配置 `/data/npu_env.sh`、清华 pip 镜像、ModelScope 缓存目录与 `scripts/setup_npu_env.sh` 一键脚本。插卡前无卡预配置环境已完成 **737/737 passed**（2026-06-24）；NPU 基准测试与 `torch.randn(..., device='npu:0')` 需插卡后执行 `bash benchmarks/scripts/run_npu_full_verify.sh`；2026-06-24 插卡复验已完成（**770/770** + 全部基准测试）。

## 优化方法论

任务二和任务三都采用同一套三步优化方法：

### 步骤一：拆分 profiling

将 NPU 执行链路拆成多个阶段分别计时。关键经验是：**profiling 必须包含 warmup 迭代**，否则容易把 kernel JIT 或编译开销误计入稳定运行时延。

| 任务 | 主要瓶颈 | 占比 |
| --- | --- | --- |
| 任务二关系打分 | `format_result`（Python 列表格式化） | 79-88% |
| 任务二关系打分 | `h2d_features`（主机到设备传输） | 9-12% |
| 任务三图度数 | `edge_index_build` + `format_result` | 53% |

**关键结论**：瓶颈通常不在 NPU kernel 本身，而在**结果格式化**和**数据传输**。因此优化重点应放在缩小返回结果面、减少 CPU 侧格式化，以及将张量缓存到设备侧。

### 步骤二：定向优化

根据 profiling 结果，为每类瓶颈实现有针对性的优化模式：

| 策略 | 任务二模式 | 任务三模式 |
| --- | --- | --- |
| 将张量缓存到设备侧 | `cached_full_logits` | `cached_*` |
| 仅返回 argmax，避免完整格式化 | `cached_argmax_labels` | - |
| 仅返回 top-k，最小化传输 | **`cached_topk_labels`** | `cached_bincount_topk` |
| 使用更高效 kernel | - | 用 `bincount` 替代 `index_add_` |
| NPU 向量化类型聚合 | - | `compute_type_centrality_npu()` |

### 步骤三：正确性验证

每个模式都会输出 `correctness.status`，并使用容差检查处理 CPU 与 NPU 之间可能出现的浮点差异。目前所有模式均通过正确性验证。

## 任务二：关系张量打分

**算子含义**：模拟知识图谱抽取中的候选实体关系打分阶段，即：
`entity-pair feature tensor -> relation weight matrix -> relation logits/label`。

**实现文件**：`src/operators/npu_ops/kg_tensor_ops.py`

### 历史验证结果（2026-06-14 快照，每项 20 次迭代）

> 本表为 2026-06-14 历史快照，用于解释优化过程。当前结论见上文“版本边界”与下文“已验证结论”的 2026-06-24 数值。

| 模式 | 4k 候选 | 加速比 | 65k 候选 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| CPU 基线对照（full format） | 44.13 ms | 1.00x | 113.78 ms | 1.00x |
| NPU end-to-end | 22.23 ms | 1.98x | 91.63 ms | 1.24x |
| `cached_full_logits` | 20.75 ms | 2.13x | 73.98 ms | 1.54x |
| `cached_argmax_labels` | 1.96 ms | 22.55x | 67.99 ms | 1.67x |
| **`cached_topk_labels`** (NPU) | **0.39 ms** | **114.61x** | **1.37 ms** | **83.05x** |
| `cpu_topk_labels` (CPU topk) | 40.92 ms | 1.08x | 61.81 ms | 1.84x |

### 稳态对比（NPU topk vs CPU topk，同等逻辑工作）

| 工作负载 | NPU | CPU | NPU/CPU |
| --- | ---: | ---: | ---: |
| 4k candidates | **0.39 ms** | 40.92 ms | **106.27x（NPU 更快）** |
| 65k candidates | **1.37 ms** | 61.81 ms | **45.11x（NPU 更快）** |

### 结果解读

- **Full-format 路径受 CPU 格式化主导**：65k 报告中 `format_result_ms` 为 59.52 ms，占性能剖析的约 67%，远高于 NPU matmul 本身。
- **`cached_topk_labels` 缩小返回结果面**：只返回 top-10，相对 full-format CPU 基线对照为 83.05x；该数字同时包含返回语义从全量结果变为 top-k 的收益。
- **稳态对比口径**：NPU 和 CPU 都执行 matmul、逐候选 max、全局 top-k 和 top-k 格式化时，65k 为 45.11x。NPU 张量已预先缓存，计时不含一次性准备成本，因此只代表重复推理场景。
- **冷启动边界**：以上稳态结果不能替代包含张量准备和 H2D 的端到端数字；65k NPU 端到端相对 CPU full-format 为 1.24x。

### 经验总结

1. **先做拆分 profiling**：最初观察到 `argmax_npu` 为 4.6 ms，加入 warmup 后稳定值只有 0.038 ms。前者是 JIT 编译伪影，因此 NPU profiling 必须包含 warmup。
2. **结果格式化才是主导瓶颈**：任务二 79-88% 的时间耗在 Python 列表格式化，而不是 NPU 计算。缩小返回结果面是核心优化点。
3. **基线对照口径必须清晰**：83.05x 包含“top-10 替代全量 65k 返回”的结果面变化；45.11x 是相同 top-k 逻辑、排除 NPU 缓存准备后的稳态测量。

## 任务三：图张量算子

**算子含义**：根据边列表计算无向图节点度数，并返回 top hubs。

**实现文件**：`src/operators/npu_ops/graph_tensor_ops.py`

### 历史验证结果（2026-06-14 快照）

> 本表为 2026-06-14 历史快照。当前结论：`cached_bincount_topk` 为 27.77x（见下文“已验证结论”）。

| 模式 | 5k/50k 时延 | 加速比 |
| --- | ---: | ---: |
| CPU 基线对照 | 60.91 ms | 1.00x |
| NPU end-to-end | 118.70 ms | 0.51x |
| NPU prepared kernel | 5.70 ms | 10.68x |
| `cached_index_add_topk` | 6.11 ms | 9.97x |
| `cached_bincount_topk` | **1.93 ms** | **31.49x** |

### 结果解读

- 对度数统计而言，`bincount` kernel 比 `index_add_` 更高效。
- 将图张量缓存到设备侧，可以避免重复 H2D 传输。
- 只返回 top-k hubs，可以避免完整度数向量的格式化成本。
- 5k/50k 规模下 `cached_bincount_topk` 为 31.49x；端到端 NPU 路径仍为 0.51x，说明缓存准备和 Python 图解析不能从口径中隐去。

## 任务三：中心性业务路径集成

**目标**：将 NPU 图算子接入真实任务三分析流水线，而不是只停留在独立基准测试。

**实现文件**：`src/operators/analysis_ops/graph_analytics.py`；其中 `compute_type_centrality_npu()` 使用 NPU `scatter_add_` 完成类型中心性聚合。

### 历史验证结果（2026-06-14 快照，5k/50k，6 类型节点）

> 本表为 2026-06-14 历史快照。当前结论：中心性业务路径为 1.16x（5k/50k cached，见下文“已验证结论”）。

| 路径 | 时延 | 加速比 |
| --- | ---: | ---: |
| CPU 基线对照 | 118.14 ms | 1.00x |
| NPU path（未复用缓存） | 128.97 ms | 0.92x |
| **Cached NPU + type aggregation** | **77.60 ms** | **1.52x** |
| Cache prepare (one-time) | 53.46 ms | — |
| Breakeven runs | 2 | — |

### 为什么业务路径加速比较克制

Amdahl 定律分析显示：即使核心度数和类型聚合已放到 NPU，业务路径加速仍明显低于独立算子，主要原因包括：

- NPU kernel launch 和同步每次调用会增加约 1-2 ms 固定开销。
- 图节点整理、结果对象构建等 Python 工作仍留在 CPU。
- NPU 张量分配和内存管理会引入 CPU 路径中不存在的额外开销。

该集成证明 NPU 算子可以正确接入真实分析路径。缓存复用后为 1.52x，未复用缓存时为 0.92x，因此必须同时披露一次性准备成本与复用条件。

## 验证范围与结果边界

### 已验证结论

- Ascend 910B3 runtime 已验证（2026-06-24）：CANN 8.5.0、`torch_npu==2.9.0`、`acl` OK、`torch.randn(..., device='npu:0')` 通过。历史 910B2C 口径：CANN 8.5.1、`torch_npu==2.9.0rc1`。
- 共享设备适配器（`src/common/device.py`）可自动选择 `npu:0`。
- 任务二关系打分 NPU 算子：2026-06-24 最新 65k 报告中 `cached_topk_labels` 相对 full-format CPU 为 **61.64x**（1.078 ms）；xlarge 报告中相对 full-format CPU 为 **99.95x**（1.389 ms）。（历史 2026-06-16 口径：65k 72.84x / xlarge 79.60x。）
- 任务三图度数算子：5k/50k 规模下 `cached_bincount_topk` 达到 **27.77x**；prepared kernel 为 **7.98x**。（历史 2026-06-16 口径：12.70x / 2.53x。）
- 任务三中心性集成：cached NPU-backed top hubs 在 5k/50k 工作负载下为 **1.16x**（`top_hubs_backend=torch_npu`，cache device `npu:0`）。（历史 2026-06-16 口径：1.11x。）
- 项目全量验证：NPU 服务器上三个任务演示全部通过；pytest **770/770 passed**（2026-06-24 服务器复跑，369.30s）；NPU 专项 **43/43 passed**，21.60s。
- 能效为实测：2026-06-24 task2 xlarge 报告包含 `npu_utilization` 字段，采样 44 次（0.2s 间隔），NPU 利用率 0.0%–3.0%（avg 0.068%），功率 min/avg/max 为 101.4W / 106.95W / 111.5W；由于关系打分 kernel 很短，利用率采样值偏低，设备执行证据以 `torch_npu`、`device=npu:0` 与 correctness passed 为主。

### 适用边界

- 当前结果主要覆盖算子级与部分业务子路径，任务二/任务三完整流水线没有作为整体加速对象评测。
- 能效数据来自指定基准测试期间的利用率与功率采样，不外推为完整系统能效提升。
- 加速比必须与基线对照、工作负载规模、缓存复用条件和结果返回口径同时阅读。

## 复现命令

> **环境边界**：NPU 命令仅在 **Ascend Linux 服务器**上执行（如 openEuler + CANN 8.5.0）。Windows 开发机与 WSL **不能**替代 NPU 复验；Windows 侧 Docker/WSL 只用于 Nexent/DataMate/Neo4j 在线集成，见 [初步准备与部署记录](preparation.md)。

**一键全量验证（推荐）** 与 **分步命令** 的完整说明见 [服务器环境](server_environment.md) 一节，此处不再重复。简要入口：

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
source /data/npu_env.sh
python -m pip install -r requirements-npu.txt
bash benchmarks/scripts/run_npu_full_verify.sh   # 16 步全量；可选 SKIP_XLARGE=1
```

报告默认写入 `benchmarks/reports/`；环境快照为 `benchmarks/reports/npu_env_snapshot.log`。脚本失败时可按 `server_environment.md` 中的分步命令逐条排查。

## 关键文件

| 文件 | 用途 |
| --- | --- |
| `src/common/device.py` | 共享设备适配器（npu/cuda/cpu） |
| `src/operators/npu_ops/kg_tensor_ops.py` | 任务二关系打分算子 |
| `src/operators/npu_ops/graph_tensor_ops.py` | 任务三图度数算子 |
| `src/operators/analysis_ops/graph_analytics.py` | 任务三中心性业务路径集成 |
| `benchmarks/task2_relation_tensor_benchmark.py` | 任务二关系张量基准测试命令行 |
| `benchmarks/task3_graph_tensor_benchmark.py` | 任务三图张量基准测试命令行 |
| `benchmarks/task3_centrality_benchmark.py` | 任务三中心性基准测试命令行 |
| `benchmarks/npu_monitor.py` | `npu-smi` 后台利用率/功率采样器 |
| `benchmarks/task2_relation_quality_benchmark.py` | 关系级 P/R/F1 质量评测（rule/cpu/npu） |
| `benchmarks/service_reachability_probe.py` | Neo4j/DataMate/Nexent 协议级可达性校验 |
| `src/operators/kg_ops/relation_features.py` | 真实候选对张量编码与关系投影权重 |
| `benchmarks/scripts/collect_env.sh` | 服务器环境采集脚本 |
| `benchmarks/scripts/run_npu_full_verify.sh` | NPU 一键全量验证（16 步，推荐） |
| `benchmarks/scripts/run_full_verify.sh` | 兼容入口，内部调用 `run_npu_full_verify.sh` |
| `benchmarks/reports/ascend_910b2c_experiment_summary.md` | 详细实验记录 |
| `server_environment.md` | 服务器环境说明 |
