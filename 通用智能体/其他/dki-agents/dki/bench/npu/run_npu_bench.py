"""CPU/NPU NER 推理基准脚本。

用法：
    python -m dki.bench.npu.run_npu_bench --device cpu --batch 8 --iters 20
    python -m dki.bench.npu.run_npu_bench --device npu --batch 8,32 --iters 50   # 昇腾环境
    python -m dki.bench.npu.run_npu_bench --compare outputs/bench/npu/ner_bench_cpu.json outputs/bench/npu/ner_bench_npu.json
"""

import argparse
import json
import sys
from pathlib import Path

from dki.paths import OUTPUT_DIR, RAW_DATA_DIR

OUT_DIR = OUTPUT_DIR / "bench" / "npu"

_FALLBACK_TEXTS = [
    "患者主诉胸痛3天，伴胸闷、气短。诊断：冠心病。给予阿司匹林口服治疗。",
    "患者出现多饮多尿，体重下降明显。诊断：2型糖尿病。处方二甲双胍，嘱按时服药。",
    "反复头痛头晕1个月。行动态血压监测，明确诊断为高血压。给予硝苯地平口服。",
    "发热咳嗽咳痰1周，行胸部CT提示肺炎。予左氧氟沙星治疗，症状好转。",
] * 8


def _load_texts(n: int = 32):
    files = sorted(RAW_DATA_DIR.glob("*.txt"))
    if files:
        texts = [f.read_text(encoding="utf-8")[:250] for f in files]
    else:
        texts = _FALLBACK_TEXTS
    if len(texts) >= n:
        return texts[:n]
    # 大 batch sweep 需要满批样本；原始语料不足时循环复用，避免尾批补齐影响吞吐结论。
    return [texts[i % len(texts)] for i in range(n)]


def run_device_bench(device: str, batches, iters: int, model: str, half: bool,
                     bucketing: bool = True):
    from dki.bench.npu.bert_ner_infer import BertNerBackend

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for bs in batches:
        backend = BertNerBackend(model_name=model, device=device, half=half,
                                 batch_size=bs, bucketing=bucketing)
        texts = _load_texts(bs * 4)
        res = backend.benchmark(texts, iters=iters)
        res["model"] = model
        res["half"] = half
        print(json.dumps(res, ensure_ascii=False, indent=2))
        all_results.append(res)
    suffix = "" if bucketing else "_dynamic"
    out = OUT_DIR / f"ner_bench_{device}{suffix}.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[√] 结果 → {out}")
    return out


def compare(cpu_json: Path, npu_json: Path):
    cpu = json.loads(Path(cpu_json).read_text(encoding="utf-8"))
    npu = json.loads(Path(npu_json).read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BERT NER 推理：CPU vs 昇腾 NPU 对比", "",
        "| batch | CPU 吞吐(样本/s) | NPU 吞吐(样本/s) | 加速比 | CPU p50(ms) | NPU p50(ms) | NPU warmup(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cpu_by_bs = {r["batch_size"]: r for r in cpu}
    for n in npu:
        c = cpu_by_bs.get(n["batch_size"])
        if not c:
            continue
        speedup = round(n["throughput_samples_s"] / max(c["throughput_samples_s"], 1e-9), 2)
        lines.append(
            f"| {n['batch_size']} | {c['throughput_samples_s']} | {n['throughput_samples_s']} "
            f"| {speedup}x | {c['batch_latency_p50_ms']} | {n['batch_latency_p50_ms']} "
            f"| {n['warmup_s']} |"
        )
    lines += [
        "",
        "- warmup 时间为静态分桶预热 + 编译缓存构建，已从吞吐计时中排除；",
        "- 能效（samples/J）依赖功耗采集（npu-smi info watch / ipmitool），按需补充。",
    ]
    out = OUT_DIR / "ner_cpu_vs_npu.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(f"[√] 对比报告 → {out}")


def compare_ablation(static_json: Path, dynamic_json: Path):
    """静态分桶 vs 动态 shape 消融对比（量化"分桶优化"本身的收益）。"""
    static = json.loads(Path(static_json).read_text(encoding="utf-8"))
    dynamic = json.loads(Path(dynamic_json).read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 消融实验：静态 shape 分桶 vs 动态 shape（同设备/同精度/同数据）", "",
        "| batch | 动态shape吞吐(样本/s) | 静态分桶吞吐(样本/s) | 分桶收益 | 动态p95(ms) | 静态p95(ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    dyn_by_bs = {r["batch_size"]: r for r in dynamic}
    for s in static:
        d = dyn_by_bs.get(s["batch_size"])
        if not d:
            continue
        gain = round(s["throughput_samples_s"] / max(d["throughput_samples_s"], 1e-9), 2)
        lines.append(
            f"| {s['batch_size']} | {d['throughput_samples_s']} | {s['throughput_samples_s']} "
            f"| {gain}x | {d['batch_latency_p95_ms']} | {s['batch_latency_p95_ms']} |"
        )
    lines += [
        "",
        "- 动态 shape 模式按批内最长 padding，形状随批变化，NPU 上会反复触发图编译（体现在吞吐下降与 p95 长尾）；",
        "- 静态分桶把形状收敛到 64/128/256 三档，预热后计时期零编译；",
        "- CPU 上两种模式差异应很小（无图编译开销），可作为对照佐证收益来自 NPU 编译特性。",
    ]
    out = OUT_DIR / "ner_bucket_ablation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(f"[√] 消融报告 → {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="BERT NER CPU/NPU 基准")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu"])
    ap.add_argument("--batch", default="8", help="批大小，逗号分隔，如 8,32")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--half", action="store_true", help="FP16（NPU/GPU 建议开启）")
    ap.add_argument("--dynamic-shape", action="store_true",
                    help="消融模式：关闭静态分桶，按批内最长动态 padding")
    ap.add_argument("--compare", nargs=2, metavar=("CPU_JSON", "NPU_JSON"))
    ap.add_argument("--compare-ablation", nargs=2, metavar=("STATIC_JSON", "DYNAMIC_JSON"))
    args = ap.parse_args(argv)

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
        return
    if args.compare_ablation:
        compare_ablation(Path(args.compare_ablation[0]), Path(args.compare_ablation[1]))
        return

    batches = [int(x) for x in str(args.batch).split(",") if x.strip()]
    try:
        run_device_bench(args.device, batches, args.iters, args.model, args.half,
                         bucketing=not args.dynamic_shape)
    except RuntimeError as e:
        print(f"[!] {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
