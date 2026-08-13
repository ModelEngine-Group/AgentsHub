#!/usr/bin/env bash
# ============================================================================
# 昇腾 910B 上的 NPU 基准一键编排脚本（在服务器上的项目根目录执行）
# 用法：bash scripts/run_npu_all.sh 2>&1 | tee outputs/bench/npu/run_all.log
# 产物：outputs/bench/npu/ 下的 json/md/csv 全套
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONUTF8=1
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
mkdir -p outputs/bench/npu

step() { echo ""; echo "=================================================="; echo "== $1"; echo "=================================================="; }

step "0/8 环境体检"
npu-smi info || { echo "[FATAL] npu-smi 不可用，确认这是昇腾环境"; exit 1; }
$PY -V
$PY - <<'EOF'
import torch
print("torch:", torch.__version__)
try:
    import torch_npu
    print("torch_npu:", torch_npu.__version__, "| npu available:", torch_npu.npu.is_available())
    x = torch.ones(2, 2).to("npu:0")
    print("npu matmul sanity:", (x @ x).sum().item())
except Exception as e:
    raise SystemExit(f"[FATAL] torch_npu 不可用: {e}\n请先 source CANN 的 set_env.sh 再重跑")
EOF
[ $? -ne 0 ] && exit 1

step "1/8 Python 依赖（缺则装）"
$PY -c "import dotenv" 2>/dev/null || $PY -m pip install python-dotenv
$PY -c "import transformers" 2>/dev/null || $PY -m pip install "transformers>=4.40,<5"
# HF 直连不通时走国内镜像（已下载过则本地缓存直接命中）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
$PY - <<'EOF'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("bert-base-chinese")
print("模型权重就绪（bert-base-chinese）")
EOF
[ $? -ne 0 ] && { echo "[FATAL] 模型下载失败：手工执行交接文档『模型离线导入』一节"; exit 1; }

ITERS=${ITERS:-50}
BATCHES=${BATCHES:-8,32}

step "2/8 CPU 基线（batch ${BATCHES} FP32）"
$PY -m dki.bench.npu.run_npu_bench --device cpu --batch "$BATCHES" --iters $ITERS || exit 1

step "3/8 NPU FP32（batch ${BATCHES}，静态分桶）"
$PY -m dki.bench.npu.run_npu_bench --device npu --batch "$BATCHES" --iters $ITERS || exit 1
cp outputs/bench/npu/ner_bench_npu.json outputs/bench/npu/ner_bench_npu_fp32.json

step "4/8 NPU FP16（batch ${BATCHES}，静态分桶）"
$PY -m dki.bench.npu.run_npu_bench --device npu --batch "$BATCHES" --iters $ITERS --half || exit 1
cp outputs/bench/npu/ner_bench_npu.json outputs/bench/npu/ner_bench_npu_fp16.json

step "5/8 消融：NPU FP16 动态 shape（关闭分桶）"
$PY -m dki.bench.npu.run_npu_bench --device npu --batch "$BATCHES" --iters $ITERS --half --dynamic-shape || exit 1

step "6/8 功耗采样轮（FP16 bs32 复跑 + 并行采样 → 能效）"
$PY scripts/sample_npu_power.py --out outputs/bench/npu/power_fp16_bs32.csv --duration 90 &
SAMPLER=$!
sleep 2
$PY -m dki.bench.npu.run_npu_bench --device npu --batch 32 --iters $ITERS --half
wait $SAMPLER

step "7/8 生成对比报告"
$PY -m dki.bench.npu.run_npu_bench --compare \
    outputs/bench/npu/ner_bench_cpu.json outputs/bench/npu/ner_bench_npu_fp16.json
$PY -m dki.bench.npu.run_npu_bench --compare-ablation \
    outputs/bench/npu/ner_bench_npu_fp16.json outputs/bench/npu/ner_bench_npu_dynamic.json

step "8/8 完成——回传清单"
ls -l outputs/bench/npu/
echo ""
echo "请把 outputs/bench/npu/ 整个目录 scp 回本地项目同路径，并按交接文档更新 docs/npu_optimization.md"
