#!/usr/bin/env bash
# One-shot NPU verification on Ascend 910B2C (openEuler / Linux only).
#
# Prerequisites:
#   1. CANN + torch_npu installed; `source /data/npu_env.sh` (or project-local npu_env.sh).
#   2. Repo checked out at $DIR (default /data/nexent-dkm-agent/nexent-dkm-agent).
#   3. Python deps: `python -m pip install -r requirements-npu.txt`
#
# Usage:
#   source /data/npu_env.sh
#   cd /data/nexent-dkm-agent/nexent-dkm-agent
#   bash benchmarks/scripts/run_npu_full_verify.sh
#
# Optional env:
#   DIR=/path/to/nexent-dkm-agent   project root
#   PY=/usr/local/python3.11.14/bin/python3
#   REPORT_DIR=benchmarks/reports     JSON output directory
#   SKIP_XLARGE=1                     skip 131072-candidate + monitor run (faster)
#   SKIP_REACHABILITY=1               skip service_reachability_probe
#   SKIP_ENV_SNAPSHOT=1               skip collect_env.sh snapshot

set -euo pipefail

PY="${PY:-/usr/local/python3.11.14/bin/python3}"
DIR="${DIR:-/data/nexent-dkm-agent/nexent-dkm-agent}"
REPORT_DIR="${REPORT_DIR:-benchmarks/reports}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOTAL_STEPS=16
STEP=0

step() {
  STEP=$((STEP + 1))
  echo ""
  echo "=========================================="
  echo "  [$STEP/$TOTAL_STEPS] $1"
  echo "=========================================="
}

run_tail() {
  # Run a command and show the last N lines for compact logs.
  local tail_lines="${1:-5}"
  shift
  "$@" 2>&1 | tail -n "$tail_lines"
}

cd "$DIR"
mkdir -p "$REPORT_DIR"

step "Preflight: NPU runtime and project root"
$PY -c "import torch; import torch_npu; t=torch.randn(64,64,device='npu:0'); print('  NPU OK:', t.device)"
echo "  project: $(pwd)"
echo "  python:  $($PY --version 2>&1)"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>&1 | head -n 8 || true
else
  echo "  npu-smi: not in PATH (optional for compute path)"
fi

if [[ "${SKIP_ENV_SNAPSHOT:-0}" != "1" ]]; then
  step "Environment snapshot (collect_env.sh)"
  bash "$SCRIPT_DIR/collect_env.sh" | tee "$REPORT_DIR/npu_env_snapshot.log" | tail -n 20
fi

step "Task 1 demo (CSV baseline)"
run_tail 5 $PY demos/task1_demo.py

step "Task 2 demo (rule relation backend)"
run_tail 8 $PY demos/task2_demo.py --output-dir outputs/npu/task2_rule

step "Task 2 demo (NPU relation backend)"
run_tail 8 $PY demos/task2_demo.py --relation-backend npu --output-dir outputs/npu/task2

step "Task 3 demo (graph analytics; expect top_hubs_backend=torch_npu when NPU available)"
run_tail 8 $PY demos/task3_demo.py --output-dir outputs/npu/task3

step "Full pytest"
$PY -m pytest -q 2>&1 | tee "$REPORT_DIR/npu_pytest.log" | tail -n 5

step "NPU-focused pytest subset"
$PY -m pytest tests/test_npu_kg_tensor_ops.py tests/test_npu_graph_tensor_ops.py tests/test_task2_benchmark.py tests/test_task3_centrality_benchmark.py -q 2>&1 | tail -n 5

step "Task 1 data quality benchmark"
run_tail 3 $PY benchmarks/task1_data_quality_benchmark.py \
  --iterations 3 \
  --report "$REPORT_DIR/task1_data_quality.json"

step "Task 2 relation tensor benchmark (4k, all modes, NPU preferred)"
run_tail 3 $PY benchmarks/task2_relation_tensor_benchmark.py \
  --candidate-count 4096 --feature-dim 256 --relation-count 5 \
  --iterations 20 --prefer-device npu --benchmark-modes all \
  --report "$REPORT_DIR/task2_topk_4k.json"

step "Task 2 relation tensor benchmark (65k, profile breakdown)"
run_tail 3 $PY benchmarks/task2_relation_tensor_benchmark.py \
  --candidate-count 65536 --feature-dim 256 --relation-count 5 \
  --iterations 20 --prefer-device npu --benchmark-modes all \
  --profile-breakdown \
  --report "$REPORT_DIR/task2_topk_65k.json"

if [[ "${SKIP_XLARGE:-0}" != "1" ]]; then
  step "Task 2 relation tensor benchmark (131072 xlarge + NPU monitor)"
  run_tail 5 $PY benchmarks/task2_relation_tensor_benchmark.py \
    --candidate-count 131072 --feature-dim 768 --relation-count 16 \
    --iterations 30 --prefer-device npu --benchmark-modes all \
    --monitor-npu --monitor-interval 0.2 \
    --report "$REPORT_DIR/task2_relation_tensor_ascend_910b2c_xlarge.json"
else
  echo ""
  echo "  (skipped xlarge benchmark: SKIP_XLARGE=1)"
fi

step "Task 2 relation quality benchmark (NPU backend)"
run_tail 3 $PY benchmarks/task2_relation_quality_benchmark.py --backend npu \
  --report "$REPORT_DIR/task2_relation_quality_ascend_910b2c_npu.json"

step "Task 3 graph tensor benchmark (5k/50k, NPU preferred)"
run_tail 3 $PY benchmarks/task3_graph_tensor_benchmark.py \
  --nodes 5000 --edges 50000 --iterations 20 \
  --prefer-device npu --benchmark-modes all \
  --profile-breakdown --amortized-runs 1,2,5,10,20 \
  --report "$REPORT_DIR/task3_graph_tensor_ascend_910b2c_large.json"

step "Task 3 centrality benchmark (5k/50k, cached NPU path)"
run_tail 3 $PY benchmarks/task3_centrality_benchmark.py \
  --nodes 5000 --edges 50000 --iterations 20 \
  --prefer-device npu --benchmark-modes all --multi-type \
  --report "$REPORT_DIR/task3_centrality_5k.json"

if [[ "${SKIP_REACHABILITY:-0}" != "1" ]]; then
  step "Service reachability probe (NPU node; external services may be unavailable)"
  run_tail 8 $PY benchmarks/service_reachability_probe.py \
    --host-label ascend_910b2c \
    --neo4j-uri none --datamate-url none --nexent-url none \
    --report "$REPORT_DIR/service_reachability_ascend_910b2c.json" || true
fi

echo ""
echo "=========================================="
echo "  NPU full verification completed."
echo "  Reports: $REPORT_DIR/"
echo "=========================================="
