"""TL 预检：确认目标机具备 CBLUE 训练 + AOE 所需条件（无 PowerShell 引号困扰）。

从本地运行：python scripts/_tl_preflight.py
仅只读探测，不改动目标机。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _ssh_npu  # noqa: E402

PROBE = r'''
set -uo pipefail
cd /root/dki-agents 2>/dev/null || { echo "NO_DKI_DIR"; exit 0; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
echo "== python/torch stack =="
python3 - <<'PY'
import importlib, json
info = {}
for m in ("numpy", "torch", "torch_npu", "transformers", "onnx"):
    try:
        mod = importlib.import_module(m)
        info[m] = getattr(mod, "__version__", "?")
    except Exception as e:
        info[m] = f"MISSING ({type(e).__name__})"
try:
    import torch, torch_npu
    info["npu_available"] = bool(torch_npu.npu.is_available())
    info["npu_count"] = torch_npu.npu.device_count()
except Exception as e:
    info["npu_available"] = f"ERR {e}"
print(json.dumps(info, ensure_ascii=False))
PY
echo "== HF mirror reachability (hfl/chinese-roberta-wwm-ext config) =="
echo "HF_ENDPOINT=${HF_ENDPOINT:-<unset>}"
curl -s -o /dev/null -w "config.json http_code=%{http_code}\n" \
  --max-time 20 "${HF_ENDPOINT:-https://hf-mirror.com}/hfl/chinese-roberta-wwm-ext/resolve/main/config.json" || echo "curl_fail"
echo "== local model cache for base model =="
python3 - <<'PY'
import os, glob
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
hits = glob.glob(os.path.join(hf_home, "**", "*chinese-roberta-wwm-ext*"), recursive=True)
print("HF_HOME=", hf_home)
print("cached_hits=", len(hits))
for h in hits[:5]:
    print("  ", h)
PY
echo "== AOE / ATC availability =="
command -v aoe && aoe --version 2>&1 | head -1 || echo "aoe_MISSING"
command -v atc && echo "atc_OK" || echo "atc_MISSING"
echo "== existing artifacts for AOE step =="
ls -la outputs/bench/npu_int8/ner_ft_bs64_seq256_fp32.onnx 2>/dev/null || echo "no_fp32_onnx"
ls -la models/medical_ner_bert 2>/dev/null | head -5 || echo "no_medical_ner_bert_model"
echo "== disk =="
df -h /root 2>/dev/null | tail -1
echo "== data on VM (if any) =="
ls -la data/cblue 2>/dev/null || echo "no_cblue_dir_yet"
echo "PREFLIGHT_DONE"
'''


def main() -> int:
    return _ssh_npu.run(PROBE, timeout=180)


if __name__ == "__main__":
    raise SystemExit(main())
