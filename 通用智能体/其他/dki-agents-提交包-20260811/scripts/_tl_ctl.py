"""TL 流水线远端控制器（避免 PowerShell 引号地狱，全部用 raw bash 字符串）。

用法：
  python scripts/_tl_ctl.py poll      # 看日志尾 + 步骤进度 + 指标 + NPU 占用 + 进程状态
  python scripts/_tl_ctl.py kill      # 杀掉正在跑的 TL 流水线
  python scripts/_tl_ctl.py ps        # 只看进程
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _ssh_npu  # noqa: E402

POLL = r'''
set -uo pipefail
cd /root/dki-agents
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
echo "== steps =="
grep -nE 'STEP:|STEP_OK|STEP_FAIL|epoch |TL_DONE|AOE_TUNE_DONE|Traceback|Error:' outputs/tl_run.log 2>/dev/null | tail -30
echo "== log tail =="
tail -4 outputs/tl_run.log 2>/dev/null
echo "== metrics =="
for f in outputs/cblue/cmeee_metrics.json outputs/cblue/cmeie_metrics.json outputs/remote_acceptance_20260721_aoe.json; do
  if [ -f "$f" ]; then echo "--- $f ---"; python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:d[k] for k in d if k in ('micro_f1','precision','recall','tp','fp','fn','train_seconds','results','passed')}, ensure_ascii=False))" "$f"; fi
done
echo "== npu =="
npu-smi info 2>/dev/null | grep -E '910B2C|AICore' | head -3
echo "== procs =="
ps -eo pid,etime,comm,args 2>/dev/null | grep -E 'train_ner|train_re|theoretical|aoe|remote_aoe' | grep -v grep || echo "no_tl_procs"
echo "POLL_DONE"
'''

KILL = r'''
set -uo pipefail
pkill -9 -f remote_run_theoretical_limit 2>/dev/null || true
pkill -9 -f train_ner_cmeee 2>/dev/null || true
pkill -9 -f train_re_cmeie 2>/dev/null || true
pkill -9 -f aoe_tune_om 2>/dev/null || true
pkill -9 -f remote_aoe_bench 2>/dev/null || true
sleep 2
LEFT=$(ps -eo args 2>/dev/null | grep -E 'train_ner|train_re|theoretical' | grep -v grep | wc -l)
echo "remaining_tl_procs=$LEFT"
echo "KILL_DONE"
'''

PS = r'''
ps -eo pid,etime,args 2>/dev/null | grep -E 'train_ner|train_re|theoretical|aoe|python3 outputs' | grep -v grep || echo "no_tl_procs"
'''


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "poll"
    script = {"poll": POLL, "kill": KILL, "ps": PS}.get(mode)
    if script is None:
        raise SystemExit(f"unknown mode {mode}")
    return _ssh_npu.run(script, timeout=180)


if __name__ == "__main__":
    raise SystemExit(main())
