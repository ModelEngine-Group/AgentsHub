"""TL 预检 2：确认 base 模型缓存完整 + 磁盘容量足够，避免多小时训练中途失败。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _ssh_npu  # noqa: E402

PROBE = r'''
set -uo pipefail
cd /root/dki-agents 2>/dev/null || true
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
echo "== base model cache completeness =="
python3 - <<'PY'
from transformers import AutoTokenizer, AutoModel
import time
t=time.time()
try:
    tok = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    m = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    nparam = sum(p.numel() for p in m.parameters())
    print(f"OK base model loaded params={nparam/1e6:.1f}M in {time.time()-t:.1f}s hidden={m.config.hidden_size}")
except Exception as e:
    print("BASE_MODEL_LOAD_FAIL", type(e).__name__, str(e)[:300])
PY
echo "== disk mounts =="
df -h / /root /data 2>/dev/null | sort -u
echo "== where does dki-agents live =="
df -h /root/dki-agents/outputs 2>/dev/null | tail -1
echo "== aoe help sanity =="
aoe --help 2>&1 | head -3 || echo "aoe_help_fail"
echo "PREFLIGHT2_DONE"
'''


def main() -> int:
    return _ssh_npu.run(PROBE, timeout=300)


if __name__ == "__main__":
    raise SystemExit(main())
