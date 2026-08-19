#!/bin/bash
# NPU Environment Version Collection Script
# Run this on the Ascend 910B2C server

echo "=========================================="
echo "  Ascend 910B2C Server Environment Report"
echo "=========================================="
echo ""

echo "=== OS ==="
cat /etc/os-release 2>/dev/null | head -5
echo ""

echo "=== Kernel ==="
uname -r
echo ""

echo "=== CPU ==="
lscpu 2>/dev/null | grep -E "Model name|Architecture|CPU\(s\):|Thread|Core|Socket"
echo ""

echo "=== Memory ==="
free -h 2>/dev/null | head -2
echo ""

echo "=== Disk (/data) ==="
df -h /data 2>/dev/null
echo ""

echo "=== Python ==="
/usr/local/python3.11.14/bin/python3 --version 2>&1
echo ""

echo "=== pip key packages ==="
/usr/local/python3.11.14/bin/pip list 2>/dev/null | grep -iE "torch|npu|ascend|numpy|cann|transformers|pydantic|fastapi|flask"
echo ""

echo "=== NPU-SMI ==="
npu-smi info 2>&1
echo ""

echo "=== CANN Version ==="
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo "version.cfg not found"
ls -la /usr/local/Ascend/ 2>/dev/null | head -10
echo ""

echo "=== torch ==="
/usr/local/python3.11.14/bin/python3 -c "import torch; print(f'torch version: {torch.__version__}')" 2>&1
echo ""

echo "=== torch_npu ==="
/usr/local/python3.11.14/bin/python3 -c "import torch_npu; print(f'torch_npu version: {torch_npu.__version__}')" 2>&1
echo ""

echo "=== acl ==="
/usr/local/python3.11.14/bin/python3 -c "import acl; print('acl: OK')" 2>&1
echo ""

echo "=== NPU compute test ==="
/usr/local/python3.11.14/bin/python3 -c "
import torch
import torch_npu
t = torch.randn(1000, 1000, device='npu:0')
result = t @ t
print(f'NPU compute test: OK (device={t.device}, shape={result.shape})')
" 2>&1
echo ""

echo "=== CANN environment vars ==="
env | grep -iE "ASCEND|CANN|NPU|LD_LIBRARY|PYTHONPATH" 2>/dev/null | sort
echo ""

echo "=== /data/npu_env.sh contents ==="
cat /data/npu_env.sh 2>/dev/null || echo "not found"
echo ""

echo "=== Project status ==="
cd /data/nexent-dkm-agent/nexent-dkm-agent 2>/dev/null && git log --oneline -5 2>/dev/null || echo "project dir not found"
echo ""

echo "=========================================="
echo "  Collection complete"
echo "=========================================="
