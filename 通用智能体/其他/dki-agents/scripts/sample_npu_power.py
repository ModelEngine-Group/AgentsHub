"""NPU 功耗采样器（在昇腾服务器上运行；纯标准库）。

原理：周期性执行 `npu-smi info` 并解析 Power(W) 列，写 CSV。
与基准并行运行后，能效 = 吞吐(样本/s) / 平均功耗(W) = 样本/焦耳。

用法（服务器上）：
    # 终端A（或后台）：
    python3 scripts/sample_npu_power.py --out outputs/bench/npu/power_npu_fp16_bs32.csv --duration 120
    # 终端B：立刻启动基准
    python3 -m dki.bench.npu.run_npu_bench --device npu --batch 32 --iters 50 --half
"""

import argparse
import csv
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

# npu-smi info 表格中功耗列形如 " 93.6 " 且行含芯片名（910B…）；
# 不同驱动版本表头略有差异，这里取每行第一个形如 xx.x 的瓦数字段。
_POWER_RE = re.compile(r"\b(\d{1,4}\.\d)\b")


def read_power_watts() -> list:
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
    watts = []
    for line in out.stdout.splitlines():
        if "910" in line or "Ascend" in line:
            m = _POWER_RE.search(line)
            if m:
                watts.append(float(m.group(1)))
    return watts


def integrate_joules(rows, start_mono=None, end_mono=None, idle_watts=0.0):
    """Trapezoid-integrate synchronized power samples.

    Rows are ``(monotonic_seconds, watts)``. Missing samples are not
    interpolated beyond adjacent observations; coverage is reported so sparse
    sampling cannot silently look like complete evidence.
    """
    selected = [
        (float(ts), float(watts)) for ts, watts in rows
        if (start_mono is None or ts >= start_mono) and (end_mono is None or ts <= end_mono)
    ]
    selected.sort()
    if len(selected) < 2:
        return {"joules": 0.0, "idle_subtracted_joules": 0.0, "covered_s": 0.0, "samples": len(selected)}
    joules = idle_subtracted = covered = 0.0
    for (t0, p0), (t1, p1) in zip(selected, selected[1:]):
        dt = max(0.0, t1 - t0)
        joules += dt * (p0 + p1) / 2
        idle_subtracted += dt * (max(0.0, p0 - idle_watts) + max(0.0, p1 - idle_watts)) / 2
        covered += dt
    return {
        "joules": round(joules, 3),
        "idle_subtracted_joules": round(idle_subtracted, 3),
        "covered_s": round(covered, 3),
        "samples": len(selected),
    }


def power_statistics(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def percentile(fraction):
        index = min(
            len(ordered) - 1,
            max(0, round(fraction * (len(ordered) - 1))),
        )
        return ordered[index]

    return {
        "minimum": round(ordered[0], 3),
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p95": round(percentile(0.95), 3),
        "maximum": round(ordered[-1], 3),
    }


def _read_marker(path):
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bench/npu/power.csv")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=int, default=120, help="采样时长（秒）")
    ap.add_argument("--marker", type=Path, help="benchmark 写入 start_mono/end_mono 的 JSON marker")
    ap.add_argument("--summary", type=Path, help="同步积分 summary JSON；默认 <out>.summary.json")
    ap.add_argument("--idle-watts", type=float, default=0.0, help="预先测得的 idle 功率，用于另报 idle-subtracted joules")
    ap.add_argument("--device-index", type=int, default=0, help="按 npu-smi 解析顺序选择目标 NPU")
    ap.add_argument(
        "--stop-after-marker",
        action="store_true",
        help="marker 出现且超过 end_mono 后提前结束；duration 仍是硬上限",
    )
    ap.add_argument("--post-marker-seconds", type=float, default=1.0)
    ap.add_argument(
        "--measurement-state",
        choices=["benchmark", "idle", "unspecified"],
        default="benchmark",
    )
    args = ap.parse_args()

    out = Path(args.out)
    summary_path = args.summary or out.with_suffix(out.suffix + ".summary.json")
    existing = [str(path) for path in (out, summary_path) if path.exists()]
    if existing:
        ap.error(f"power evidence outputs are create-once: {existing}")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    t_end = time.monotonic() + args.duration
    rows = []
    integration_rows = []
    print(f"开始功耗采样 {args.duration}s → {out}")
    while time.monotonic() < t_end:
        sampled_mono = time.monotonic()
        try:
            watts = read_power_watts()
        except Exception as e:
            print(f"[!] npu-smi 读取失败: {e}")
            watts = []
        rows.append([round(time.time(), 3), round(sampled_mono, 6)] + watts)
        if len(watts) > args.device_index:
            integration_rows.append((sampled_mono, watts[args.device_index]))
        time.sleep(args.interval)
        if args.stop_after_marker:
            live_marker = _read_marker(args.marker)
            marker_end = live_marker.get("end_mono")
            if (
                isinstance(marker_end, (int, float))
                and time.monotonic() >= marker_end + args.post_marker_seconds
            ):
                break

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "monotonic_s"] + [f"npu{i}_watts" for i in range(max((len(r) - 2) for r in rows) if rows else 0)])
        w.writerows(rows)
    selected_powers = [watts for _, watts in integration_rows]
    if selected_powers:
        marker = _read_marker(args.marker)
        marker_start = marker.get("start_mono")
        marker_end = marker.get("end_mono")
        summary = integrate_joules(
            integration_rows,
            marker_start,
            marker_end,
            args.idle_watts,
        )
        marker_bound = isinstance(marker_start, (int, float)) and isinstance(
            marker_end, (int, float)
        )
        expected = (
            max(0.0, marker_end - marker_start)
            if marker_bound
            else float(args.duration)
        )
        window_powers = [
            watts
            for sampled_mono, watts in integration_rows
            if not marker_bound
            or marker_start <= sampled_mono <= marker_end
        ]
        summary.update({
            "schema_version": 1,
            "method": "monotonic-marker trapezoid integration",
            "measurement_state": args.measurement_state,
            "marker_bound": marker_bound,
            "idle_watts": args.idle_watts,
            "expected_window_s": expected,
            "coverage_ratio": (
                None
                if not expected
                else round(min(1.0, summary["covered_s"] / expected), 4)
            ),
            "interval_s": args.interval,
            "csv": str(out),
            "marker": str(args.marker) if args.marker else None,
            "device_index": args.device_index,
            "power_watts": power_statistics(window_powers),
        })
        n_docs = marker.get("n_docs")
        if isinstance(n_docs, int) and n_docs > 0:
            summary["n_docs"] = n_docs
            summary["joules_per_document"] = round(summary["joules"] / n_docs, 6)
            summary["idle_subtracted_joules_per_document"] = round(
                summary["idle_subtracted_joules"] / n_docs,
                6,
            )
        coverage_ok = (
            summary["samples"] >= 2
            and summary["coverage_ratio"] is not None
            and summary["coverage_ratio"] >= 0.9
        )
        summary["passed"] = bool(
            coverage_ok
            and summary["power_watts"].get("median", 0.0) > 0.0
            and (
                args.measurement_state == "idle"
                or (
                    marker_bound
                    and summary.get("joules_per_document", 0.0) > 0.0
                )
            )
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"目标设备平均功耗: {summary['power_watts']['mean']:.1f} W"
            f"（样本数 {len(window_powers)}）"
        )
        print(f"同步积分: {summary['joules']:.1f} J，覆盖 {summary['covered_s']:.2f}s → {summary_path}")
    else:
        print("[!] 未解析到功耗数据，请人工核对 `npu-smi info` 输出格式")


if __name__ == "__main__":
    main()
