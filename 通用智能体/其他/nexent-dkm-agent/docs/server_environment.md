# Ascend 910B3 服务器环境

硬件与软件栈采集时间：2026-06-24；项目验证结果更新：2026-06-24（NPU 已验证可用）

## 无卡预配置（插卡前快照，2026-06-24）

插卡前容器**尚未映射 NPU 设备**（仅 `/dev/davinci_manager`，无 `/dev/davinci0`），`npu-smi` 与 `torch.randn(..., device='npu:0')` 暂不可用。软件栈与项目依赖已预先配置完毕，插卡或映射设备后可直接复验。

| 检查项 | 当前状态 |
| --- | --- |
| 代码分支 | `final-version`（`/data/nexent-dkm-agent/nexent-dkm-agent`） |
| `/data/npu_env.sh` | 已创建（CANN + ModelScope 缓存 + `PYTHONPATH`） |
| pip 镜像 | 清华源（`~/.pip/pip.conf`） |
| Python 依赖 | `requirements-npu.txt` + `requirements.txt` + `requirements-dev.txt` + `modelscope` |
| 模型缓存目录 | `/data/modelscope_cache`、`/data/huggingface_home` |
| DeepSeek 配置 | `.local/llm_deepseek_v4.env`（已写入，权限 600） |
| CPU pytest（无卡） | **737/737 passed** in 57.87s（2026-06-24，`final-version`，无 NPU 设备；未跑 NPU 专项用例，插卡后全量为 **770/770**） |
| 任务一/二演示 | 无卡通过（规则后端） |
| NPU 全量验证 | 已于 **2026-06-24** 插卡后执行完成（**770/770 passed**）；见下文「项目验证」 |

一键初始化（新机器或重装后）：

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
bash scripts/setup_npu_env.sh
```

插卡后复验：

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
git pull origin final-version
source /data/npu_env.sh
python -m pip install -r requirements-npu.txt
bash benchmarks/scripts/run_npu_full_verify.sh
```

> 下文「硬件」「项目验证」等表格已按 **2026-06-24 有卡复跑** 结果更新。

## 硬件

> 2026-06-24 复验实测：本容器 NPU 型号为 **Ascend 910B3**（非 910B2C），架构 **aarch64**。历史 910B2C/x86_64 描述保留于本段下方供对照，当前 NPU 结论以 910B3 / aarch64 实测为准。

| 组件 | 详情（2026-06-24 实测） |
| --- | --- |
| **NPU** | Ascend 910B3（NPU 0），65536 MB HBM，健康状态 OK，功率 99.6W / 温度 33°C |
| **npu-smi** | 25.5.0 |
| **CPU 架构** | aarch64 |
| **内存** | 见 `free -h` |
| **磁盘（/data）** | 见 `df -h /data` |

历史口径（2026-06-16，910B2C / x86_64 服务器，仅作对照）：

| 组件 | 详情 |
| --- | --- |
| **NPU** | Ascend 910B2C（NPU 11），65536 MB HBM，健康状态为 OK |
| **CPU 架构** | x86_64 |
| **CPU 核心** | 2 路 CPU，每路 32 核；开启超线程后共 128 线程（精确型号可用 `lscpu` 确认） |
| **NUMA** | 2 个节点（node0: 0-31,64-95；node1: 32-63,96-127） |
| **内存** | 总计 64 GiB，约 57 GiB 可用 |
| **磁盘（/data）** | 59 GB 持久化空间，56 GB 可用 |

## 操作系统

| 组件 | 版本 |
| --- | --- |
| **操作系统** | openEuler 24.03（LTS-SP2） |
| **内核** | 5.15.0-25-generic（2026-06-24 容器）；历史 5.15.0-113-generic |

## Ascend / CANN 栈

| 组件 | 版本或路径（2026-06-24 实测） |
| --- | --- |
| **CANN** | 8.5.0 (`/usr/local/Ascend/cann-8.5.0`，`cann` 软链指向此目录) |
| **Ascend Toolkit** | `/usr/local/Ascend/ascend-toolkit/latest` |
| **NNAL / ATB** | `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1` |
| **驱动** | `/usr/local/Ascend/driver`（driver version 25.5.0） |
| **npu-smi** | 25.5.0 |
| **NPU 状态** | **OK**（2026-06-24 已验证，NPU 0，910B3，已用 3446/65536 MB） |

## Python 与机器学习框架

> 2026-06-24 实测版本（`pip list`）。

| 包 | 版本 |
| --- | --- |
| **Python** | 3.11.14 (`/usr/local/python3.11.14/bin/python3`) |
| **torch** | 2.9.0+cpu |
| **torch_npu** | 2.9.0 |
| **numpy** | 2.4.5 |
| **transformers** | 5.12.1 |
| **pydantic** | 2.13.4 |
| **fastapi** | 0.138.0 |
| **pandas** | 3.0.3 |
| **peft** | 0.19.1 |
| **datasets** | 5.0.0 |
| **accelerate** | 1.14.0 |
| **neo4j** | 6.2.0 |
| **pytest** | 9.1.1 |
| **modelscope** | 1.37.1 |

## NPU 计算验证

```
$PY -c "import torch; import torch_npu; t=torch.randn(100,100,device='npu:0'); print('NPU OK:', t.device)"
NPU OK: npu:0
```

## 项目验证（最新 2026-06-24）

| 验证项 | 结果 |
| --- | --- |
| 任务一演示 | 通过（CSV 清洗，5 行 -> 4 行，quality 0.8 -> 1.0） |
| 任务二演示（规则） | 通过（29 条三元组 / 26 节点 / 29 边） |
| 任务二演示（NPU 后端） | 通过，`relation=npu device=npu:0` |
| 任务三演示 | 通过（`top_hubs_backend=torch_npu`） |
| pytest | **770/770 passed**，0 个失败，369.30s（2026-06-24；2026-06-16 历史为 380/380） |
| NPU 专项 pytest | **43/43 passed**，21.60s（`test_npu_kg_tensor_ops` + `test_npu_graph_tensor_ops` + `test_task2_benchmark` + `test_task3_centrality_benchmark`） |
| 任务一数据质量 | 通过（3 iterations，quality 0.8 -> 1.0） |
| 任务二基准测试 4k/65k | 通过，覆盖全部模式 |
| 任务二 relation top-k 65k | `cached_topk_labels` **61.64x**（对 full-format CPU），1.078 ms |
| 任务二 relation xlarge | `cached_topk_labels` **99.95x**（对 full-format CPU），1.389 ms |
| 任务二关系级质量 | NPU P/R/F1=1.0（46 TP / 0 FP / 0 FN） |
| 任务三 Graph Tensor 5k/50k | `cached_bincount_topk` **27.77x**；prepared kernel 7.98x |
| 任务三 Centrality 5k/50k | cached NPU 路径 **1.16x**，`top_hubs_backend=torch_npu` |
| 服务可达性 | Neo4j/DataMate/Nexent 均 unreachable（NPU 节点外部服务不可达，仅记录状态） |

## 环境初始化脚本

```bash
source /data/npu_env.sh
```

`/data/npu_env.sh` 内容如下：

```bash
#!/usr/bin/env bash

for f in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/cann-*/set_env.sh \
  /usr/local/Ascend/nnal/atb/set_env.sh
do
  if [ -f "$f" ]; then
    source "$f"
  fi
done

if [ -x /usr/local/python3.11.14/bin/python3 ]; then
  export PY=/usr/local/python3.11.14/bin/python3
else
  export PY=$(command -v python3 || command -v python)
fi

export PYTHONUNBUFFERED=1

# ModelScope / HuggingFace cache on persistent /data volume
export MODELSCOPE_CACHE=/data/modelscope_cache
export HF_HOME=/data/huggingface_home
export HF_ENDPOINT=https://hf-mirror.com

# Project source root
export NEXENT_DKM_ROOT=/data/nexent-dkm-agent/nexent-dkm-agent
export PYTHONPATH="${NEXENT_DKM_ROOT}/src:${PYTHONPATH:-}"
```

## 一键全量验证

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
source /data/npu_env.sh
python -m pip install -r requirements-npu.txt
bash benchmarks/scripts/run_npu_full_verify.sh
```

`run_full_verify.sh` 为兼容别名，内部调用同一脚本。可选 `SKIP_XLARGE=1` 跳过 131072 候选 + `npu-smi` 能效采样以缩短耗时。报告与 `npu_env_snapshot.log` 默认写入 `benchmarks/reports/`。

> Windows/WSL 不能运行本脚本；NPU 复验仅在 Ascend Linux 服务器执行。Windows 侧 Nexent/DataMate/Neo4j 部署见 [初步准备与部署记录](preparation.md)。

## 分步命令

除一键脚本外，可按需单独运行基准测试（完整参数见 `npu_optimization.md`）：

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
source /data/npu_env.sh
PY=/usr/local/python3.11.14/bin/python3

# 任务二关系张量（4k / 65k）
$PY benchmarks/task2_relation_tensor_benchmark.py \
  --candidate-count 65536 --feature-dim 256 --relation-count 5 \
  --iterations 20 --prefer-device npu --benchmark-modes all --profile-breakdown \
  --report benchmarks/reports/task2_topk_65k.json

# 任务三图张量 + 中心性
$PY benchmarks/task3_graph_tensor_benchmark.py \
  --nodes 5000 --edges 50000 --iterations 20 --prefer-device npu \
  --benchmark-modes all --profile-breakdown \
  --report benchmarks/reports/task3_graph_tensor_ascend_910b2c_large.json

$PY benchmarks/task3_centrality_benchmark.py \
  --nodes 5000 --edges 50000 --iterations 20 --prefer-device npu \
  --benchmark-modes all --multi-type \
  --report benchmarks/reports/task3_centrality_5k.json

# 关系级质量（NPU）
$PY benchmarks/task2_relation_quality_benchmark.py --backend npu \
  --report benchmarks/reports/task2_relation_quality_ascend_910b2c_npu.json
```

NPU 复跑后若 pytest 数量或基准测试数值变化，需同步更新本文、`npu_optimization.md` 与 `benchmarks/reports/ascend_910b2c_experiment_summary.md` 的 NPU 口径。
