"""上传最新代码 + CBLUE 数据到目标机并做小样本冒烟（NER/RE 各限 64 样本 1 epoch）。

目的：在启动多小时脱管流水线前，先在 NPU 上快速验证整条链路
（tokenizer/模型/NPU 前反向/解码/指标落盘），避免脱管跑到一半才失败。

用法：python scripts/_tl_upload_and_smoke.py
"""
from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _ssh_npu  # noqa: E402


def _make_code_tgz(dst: Path):
    include = ["dki", "tests", "scripts", "configs", "pyproject.toml",
               "run.sh", "run.ps1", "README.md", "CONTRACTS.md", ".env.example",
               "outputs/remote_run_theoretical_limit_20260721.sh",
               "outputs/remote_aoe_bench_20260721.py"]
    skip = {"__pycache__", ".pytest_cache", ".venv"}
    with tarfile.open(dst, "w:gz") as tar:
        for item in include:
            p = ROOT / item
            if not p.exists():
                continue
            tar.add(p, arcname=item,
                    filter=lambda ti: None if any(s in ti.name.split("/") for s in skip) else ti)


def _make_data_tgz(dst: Path):
    cblue = ROOT / "data" / "cblue"
    with tarfile.open(dst, "w:gz") as tar:
        for sub in ("CMeEE-V2", "CMeIE-V2"):
            d = cblue / sub
            if d.is_dir():
                tar.add(d, arcname=sub)


SMOKE = r'''
set -uo pipefail
cd /root/dki-agents
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONUTF8=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -d /data/huggingface_home ] && export HF_HOME=/data/huggingface_home
export TOKENIZERS_PARALLELISM=false
echo "== extract code + data =="
[ -f /root/dki_code_sync.tgz ] && tar xzf /root/dki_code_sync.tgz -C /root/dki-agents
mkdir -p data/cblue outputs/cblue
[ -f /root/cblue_norm.tgz ] && tar xzf /root/cblue_norm.tgz -C data/cblue
echo "cblue dirs:"; ls data/cblue
echo "train sizes:"; wc -l data/cblue/CMeEE-V2/train.jsonl data/cblue/CMeIE-V2/train.jsonl
echo "== SMOKE NER (limit 64, 1 epoch) =="
python3 scripts/train_ner_cmeee.py --device npu --epochs 1 --batch-size 8 \
    --max-len 128 --limit-train 64 2>&1 | tail -12 && echo "SMOKE_NER_OK" || echo "SMOKE_NER_FAIL rc=$?"
echo "== SMOKE RE (limit 64, 1 epoch) =="
python3 scripts/train_re_cmeie.py --device npu --epochs 1 --batch-size 8 \
    --max-len 160 --limit-train 64 2>&1 | tail -12 && echo "SMOKE_RE_OK" || echo "SMOKE_RE_FAIL rc=$?"
echo "SMOKE_DONE"
'''


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tl_smoke_"))
    code_tgz = tmp / "dki_code_sync.tgz"
    data_tgz = tmp / "cblue_norm.tgz"
    _make_code_tgz(code_tgz)
    _make_data_tgz(data_tgz)
    print(f"code tgz {code_tgz.stat().st_size} B; data tgz {data_tgz.stat().st_size} B", flush=True)
    _ssh_npu.put(str(code_tgz), "/root/dki_code_sync.tgz")
    _ssh_npu.put(str(data_tgz), "/root/cblue_norm.tgz")
    return _ssh_npu.run(SMOKE, timeout=900)


if __name__ == "__main__":
    raise SystemExit(main())
