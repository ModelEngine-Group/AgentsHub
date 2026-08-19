#!/usr/bin/env bash
# Backward-compatible entry for Ascend 910B2C one-shot verification.
# Delegates to the expanded NPU script.
#
# Usage:
#   source /data/npu_env.sh
#   cd /data/nexent-dkm-agent/nexent-dkm-agent
#   bash benchmarks/scripts/run_full_verify.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_npu_full_verify.sh" "$@"
