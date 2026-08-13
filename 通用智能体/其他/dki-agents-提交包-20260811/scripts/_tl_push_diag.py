"""上传 GP 单元诊断 + 训练脚本并在 NPU 跑单元过拟合诊断。"""
from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _ssh_npu  # noqa: E402


def _tgz(dst: Path):
    include = ["dki/nlp/globalpointer.py", "scripts/train_ner_cmeee.py",
               "scripts/_gp_unit_diag.py", "scripts/_encoder_probe.py"]
    with tarfile.open(dst, "w:gz") as tar:
        for item in include:
            p = ROOT / item
            if p.exists():
                tar.add(p, arcname=item)


RUN = r'''
set -uo pipefail
cd /root/dki-agents
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONUTF8=1
tar xzf /root/dki_code_sync.tgz -C /root/dki-agents
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -d /data/huggingface_home ] && export HF_HOME=/data/huggingface_home
echo "== encoder probe (real RoBERTa hidden + GP overfit on NPU) =="
python3 scripts/_encoder_probe.py --device npu 2>&1 | grep -vE 'transfer_to_npu|warnings.warn|ImportWarning|RuntimeWarning|^\s+\*|The torch|The backend|The device|^torch\.|Loading weights|LOAD REPORT|UNEXPECTED|MISSING|Notes:|can be ignored|newly initialized|^Key|^---|^cls\.|^classifier|^-------' | tail -20
echo "DIAG_DONE"
'''


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tl_diag_"))
    tgz = tmp / "dki_code_sync.tgz"
    _tgz(tgz)
    _ssh_npu.put(str(tgz), "/root/dki_code_sync.tgz")
    return _ssh_npu.run(RUN, timeout=300)


if __name__ == "__main__":
    raise SystemExit(main())
