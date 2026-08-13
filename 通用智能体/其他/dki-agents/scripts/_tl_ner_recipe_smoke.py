"""上传修正后的 NER 训练脚本 + globalpointer，跑一个有意义的小规模验证：
512 训练样本 / 3 epoch / dev 前 300，确认差异化学习率下 F1 明显 > 0（脱离塌陷）。
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
    include = ["dki/nlp/globalpointer.py", "scripts/train_ner_cmeee.py",
               "scripts/train_re_cmeie.py",
               "outputs/remote_run_theoretical_limit_20260721.sh"]
    with tarfile.open(dst, "w:gz") as tar:
        for item in include:
            p = ROOT / item
            if p.exists():
                tar.add(p, arcname=item)


SMOKE = r'''
set -uo pipefail
cd /root/dki-agents
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONUTF8=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -d /data/huggingface_home ] && export HF_HOME=/data/huggingface_home
export TOKENIZERS_PARALLELISM=false
tar xzf /root/dki_code_sync.tgz -C /root/dki-agents
echo "== A) head-warmup 2 + JOINT enc-lr 3e-6 constant, encoder-eval: 2000/500 =="
python3 scripts/train_ner_cmeee.py --device npu --epochs 10 --batch-size 16 \
    --max-len 160 --limit-train 2000 --limit-dev 500 --head-warmup-epochs 2 \
    --lr 3e-6 --head-lr 1e-3 --no-sched --encoder-eval \
    --sweep-threshold 2>&1 | grep -E 'align|logit|thr=|micro_f1|\[joint\]|\[head\]' | tail -22
echo "NER_RECIPE_SMOKE_DONE"
'''


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tl_nerrecipe_"))
    code_tgz = tmp / "dki_code_sync.tgz"
    _make_code_tgz(code_tgz)
    print(f"code tgz {code_tgz.stat().st_size} B", flush=True)
    _ssh_npu.put(str(code_tgz), "/root/dki_code_sync.tgz")
    return _ssh_npu.run(SMOKE, timeout=900)


if __name__ == "__main__":
    raise SystemExit(main())
