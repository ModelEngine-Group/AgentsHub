#!/usr/bin/env bash
# Run 30-case relation quality benchmark on Ascend NPU for competition NPU bonus evidence.
# Usage (on Ascend host with CANN + torch_npu):
#   bash benchmarks/scripts/run_npu_30case_relation_quality.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
REPORT="${1:-benchmarks/reports/task2_relation_quality_ascend_910b3_30case.json}"
python benchmarks/task2_relation_quality_benchmark.py \
  --backend npu \
  --report "$REPORT"
echo "Wrote $REPORT"
