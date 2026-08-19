#!/usr/bin/env bash
# One-time / refresh environment setup for Ascend NPU server (works without NPU card).
set -euo pipefail

REPO_ROOT="/data/nexent-dkm-agent/nexent-dkm-agent"
PROJECT_ROOT="${REPO_ROOT}/nexent-dkm-agent"
PY="${PY:-/usr/local/python3.11.14/bin/python3}"

echo "==> Sourcing NPU/CANN environment"
# shellcheck disable=SC1091
source /data/npu_env.sh

echo "==> Project root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

echo "==> Installing Python dependencies (Tsinghua mirror via ~/.pip/pip.conf)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
"${PY}" -m pip install -r requirements-npu.txt -r requirements.txt -r requirements-dev.txt modelscope -q

echo "==> Creating output directories"
mkdir -p "${PROJECT_ROOT}/outputs/npu" "${PROJECT_ROOT}/benchmarks/reports" "${PROJECT_ROOT}/.local"
mkdir -p /data/modelscope_cache /data/huggingface_home

if [[ ! -f "${PROJECT_ROOT}/.local/llm_deepseek_v4.env" ]]; then
  echo "==> Creating DeepSeek LLM config template at .local/llm_deepseek_v4.env"
  cat > "${PROJECT_ROOT}/.local/llm_deepseek_v4.env" <<'EOF'
OPENAI_API_KEY=<your-deepseek-api-key>
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
OPENAI_THINKING=disabled
EOF
  chmod 600 "${PROJECT_ROOT}/.local/llm_deepseek_v4.env"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "==> Preflight checks"
"${PY}" --version
"${PY}" -c "import torch; print('torch', torch.__version__)"
"${PY}" -c "import torch_npu; print('torch_npu import OK')" 2>/dev/null || echo "torch_npu: not available"
"${PY}" -c "import modelscope; print('modelscope', modelscope.__version__)" 2>/dev/null || echo "modelscope: not installed"

echo "==> NPU device probe (expected to fail without /dev/davinci0)"
if "${PY}" -c "import torch; import torch_npu; t=torch.randn(4,4,device='npu:0'); print('NPU tensor OK:', t.device)" 2>/dev/null; then
  echo "NPU hardware: READY"
else
  echo "NPU hardware: NOT READY (software stack OK; insert card / map devices and re-run)"
fi

echo "==> CPU pytest smoke (excludes hardware NPU tensor tests when no card)"
"${PY}" -m pytest -q \
  --ignore=tests/test_npu_kg_tensor_ops.py \
  --ignore=tests/test_npu_graph_tensor_ops.py \
  2>&1 | tail -n 3

echo ""
echo "Environment setup complete."
echo "  cd ${REPO_ROOT}"
echo "  source /data/npu_env.sh"
echo "  export PYTHONPATH=${PROJECT_ROOT}/src:\$PYTHONPATH"
echo "  # After NPU card is available:"
echo "  bash benchmarks/scripts/run_npu_full_verify.sh"
