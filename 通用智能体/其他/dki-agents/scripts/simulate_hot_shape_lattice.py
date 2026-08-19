"""生成 hot-shape lattice 仿真报告（CPU 确定性；非 NPU 实机成绩）。

输入优先级：
1. `--lengths-json`：一个 JSON 数组，元素为真实 token 长度（如 profile_ner_workload 产物）；
2. 否则用 `outputs/clean/*.txt` 的字符长度作为**近似 token 长度**（中文 1 字≈1 token，
   报告里显式标注为 proxy）。

产出同 trace 下 fixed-seq256 与 2–4 节点 lattice 的对照（open/closed loop）：
padding 浪费、等待分位、吞吐（sim-time 单位）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.hot_shape import compare_fixed_vs_lattice, plan_lattice  # noqa: E402
from dki.paths import OUTPUT_DIR  # noqa: E402


def _token_lengths(args) -> tuple[list[int], str]:
    if args.lengths_json:
        data = json.loads(Path(args.lengths_json).read_text(encoding="utf-8"))
        lengths = [int(v) for v in (data if isinstance(data, list) else data["token_lengths"])]
        return lengths, "real_token_lengths"
    clean = OUTPUT_DIR / "clean"
    files = sorted(clean.glob("*.txt")) if clean.is_dir() else []
    if not files:
        raise SystemExit("no token lengths: pass --lengths-json or run demo-all to populate outputs/clean")
    lengths = [len(f.read_text(encoding="utf-8")) for f in files]
    return lengths, "char_length_proxy(outputs/clean/*.txt)"


def main() -> int:
    parser = argparse.ArgumentParser(description="hot-shape lattice 仿真（CPU）")
    parser.add_argument("--lengths-json", default=None)
    parser.add_argument("--sequence-lengths", default="64,128,256")
    parser.add_argument("--fixed-seq", type=int, default=256)
    parser.add_argument("--max-nodes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--arrival-interval", type=float, default=4.0)
    parser.add_argument("--per-token-cost", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "verification" / "hot_shape_lattice_sim.json")
    args = parser.parse_args()

    lengths, source = _token_lengths(args)
    seqs = tuple(int(v) for v in args.sequence_lengths.split(","))
    plan = plan_lattice(lengths, sequence_lengths=seqs, max_nodes=args.max_nodes)
    report = {
        "schema_version": 1,
        "length_source": source,
        "sequence_lengths": list(seqs),
        "fixed_seq": args.fixed_seq,
        "lattice_plan": plan.to_dict(),
        "simulation": compare_fixed_vs_lattice(
            lengths, sequence_lengths=seqs, fixed_seq=args.fixed_seq,
            max_nodes=args.max_nodes, batch_size=args.batch_size,
            arrival_interval=args.arrival_interval, per_token_cost=args.per_token_cost,
        ),
        "promotion_state": "cpu_simulation_only_not_npu_measured",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    open_loop = report["simulation"]["runs"]["open_loop"]
    print(json.dumps({
        "length_source": source, "n_docs": len(lengths),
        "lattice_nodes": plan.node_lengths(), "overflow": plan.overflow,
        "open_loop_padding_reduction": open_loop["padding_waste_reduction"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
