"""INT8 W8A8 PTQ 量化微调 NER 模型（msmodelslim onnx squant_ptq）。

在昇腾 910B 上把 FP16/FP32 ONNX 量化为 INT8，配合 atc 转 INT8 OM，冲击
cube INT8 峰值吞吐（≈FP16 的 1.4~1.8×）。量化后须配合 scripts/eval_om_ner.py
验证实体级 F1 掉点（不编造，真实实测）。

工具链要点（已实测 CANN 8.5.1 / 910B2C）：
- CANN 自带 tools/msmodelslim 是编译 .so stub、无 Python API；须 pip 装 wheel：
      pip install 'onnx==1.16.2' onnxruntime 'msmodelslim==8.2.1'
  （onnx 必须 ≤1.17：msmodelslim 依赖已被 onnx 1.18 删除的 onnx.mapping；
   故用隔离 venv /root/quant_venv，勿降级系统 onnx）
- API：OnnxCalibrator(input_model, QuantConfig(...), calib_data).run() +
       .export_quant_onnx(save_path)

校准集：复用 dki.bench.npu.run_npu_bench._load_texts 的真实病历，tokenize 到
固定 (batch, seq) —— 与 bench/OM 的静态 shape 完全一致。

用法（服务器，隔离 venv）：
    /root/quant_venv/bin/python scripts/quantize_ner_int8.py \
        --onnx outputs/bench/npu_int8/ner_ft_bs64_seq256_fp32.onnx \
        --output outputs/bench/npu_int8/ner_ft_bs64_seq256_int8.onnx \
        --model-name models/medical_ner_bert --batch 64 --seq-len 256 --calib-batches 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_calib_data(model_name: str, batch: int, seq_len: int, calib_batches: int) -> List[List[np.ndarray]]:
    """构造校准数据：真实病历 tokenize 到定长，返回 [[input_ids, attention_mask, token_type_ids], ...]。

    输入顺序须与 ONNX 图输入一致（input_ids, attention_mask, token_type_ids）。
    """
    from transformers import AutoTokenizer

    from dki.bench.npu.run_npu_bench import _load_texts

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = _load_texts(batch * calib_batches)
    calib = []
    for i in range(calib_batches):
        chunk = texts[i * batch:(i + 1) * batch]
        if len(chunk) < batch:
            chunk = chunk + [chunk[-1]] * (batch - len(chunk))
        enc = tokenizer(
            chunk, truncation=True, padding="max_length", max_length=seq_len,
            return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        ttype = enc.get("token_type_ids", np.zeros_like(ids)).astype(np.int64)
        calib.append([ids, mask, ttype])
    return calib


def _initializer_consumers(model) -> Dict[str, List[str]]:
    consumers: Dict[str, List[str]] = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node.name)
    return consumers


def _live_values(model) -> set[str]:
    producers = {output: node for node in model.graph.node for output in node.output}
    live = {value.name for value in model.graph.output}
    pending = list(live)
    while pending:
        value = pending.pop()
        node = producers.get(value)
        if node is None:
            continue
        for input_value in node.input:
            if input_value and input_value not in live:
                live.add(input_value)
                pending.append(input_value)
    return live


def _upstream_nodes(model, start_values: Sequence[str]) -> set[str]:
    producers = {output: node for node in model.graph.node for output in node.output}
    result: set[str] = set()
    pending = list(start_values)
    visited = set()
    while pending:
        value = pending.pop()
        if not value or value in visited:
            continue
        visited.add(value)
        node = producers.get(value)
        if node is None:
            continue
        if node.name:
            result.add(node.name)
        pending.extend(node.input)
    return result


def _quantized_source_nodes(model) -> set[str]:
    """Return source node names with weights consumed by quantization operators.

    msmodelslim exports can use either Q/DQ nodes or operator-oriented integer
    nodes.  The proof therefore follows graph edges from quantization nodes to
    their initializer inputs instead of trusting filenames or configuration.
    """
    initializer_names = {value.name for value in model.graph.initializer}
    producer = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    quant_ops = {
        "QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear",
        "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger",
    }
    live = _live_values(model)
    result: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in quant_ops or not any(output in live for output in node.output):
            continue
        result.update(_upstream_nodes(model, node.input))
        for value in node.input:
            if value in initializer_names:
                result.add(value)
        if node.name:
            result.add(node.name)
    return result


def verify_selective_quantization(input_path: Path, output_path: Path, requested: Sequence[str]) -> Mapping[str, object]:
    """Prove requested graph components exist and remain outside quantized paths."""
    import onnx

    before = onnx.load(str(input_path))
    after = onnx.load(str(output_path))
    before_names = {node.name for node in before.graph.node if node.name}
    before_initializers = {value.name for value in before.graph.initializer}
    after_names = {node.name for node in after.graph.node if node.name}
    after_initializers = {value.name for value in after.graph.initializer}
    known_before = before_names | before_initializers
    missing = sorted(name for name in requested if name not in known_before)
    if missing:
        raise RuntimeError(f"disable_names not present in input ONNX graph: {missing}")

    quantized = _quantized_source_nodes(after)
    live = _live_values(after)
    after_by_name = {node.name: node for node in after.graph.node if node.name}
    still_present = sorted(name for name in requested if name in after_names or name in after_initializers)
    absent_after = sorted(set(requested) - set(still_present))
    direct_quant_ops = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"}
    quantized_requested = set(requested) & quantized
    for name in requested:
        node = after_by_name.get(name)
        if node is None:
            continue
        if node.op_type in direct_quant_ops:
            quantized_requested.add(name)
            continue
        upstream = _upstream_nodes(after, node.input)
        if any(after_by_name[parent].op_type in direct_quant_ops for parent in upstream if parent in after_by_name):
            quantized_requested.add(name)
    quantized_requested = sorted(quantized_requested)
    quant_nodes = [
        node.name or node.op_type
        for node in after.graph.node
        if node.op_type in direct_quant_ops and any(output in live for output in node.output)
    ]
    if not quant_nodes:
        raise RuntimeError("output ONNX contains no recognized quantization operators")
    if absent_after:
        raise RuntimeError(f"cannot verify disabled graph components after export: {absent_after}")
    if quantized_requested:
        raise RuntimeError(f"requested disabled components are still quantized: {quantized_requested}")
    quantizable_ops = {"Conv", "Gemm", "MatMul"}
    requested_quantizable = sorted(
        name for name in requested
        if name in after_by_name and after_by_name[name].op_type in quantizable_ops
    )
    if requested and not requested_quantizable:
        raise RuntimeError("disable_names contains no quantization-eligible compute nodes")
    return {
        "verified": bool(requested_quantizable),
        "requested_present_before": sorted(requested),
        "requested_present_after": still_present,
        "requested_quantizable_compute": requested_quantizable,
        "recognized_quantization_nodes": len(quant_nodes),
        "quantized_source_count": len(quantized),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="INT8 W8A8 PTQ 量化 NER ONNX")
    ap.add_argument("--onnx", required=True, help="输入 ONNX（建议 FP32，量化更稳）")
    ap.add_argument("--output", required=True, help="输出量化 ONNX 路径")
    ap.add_argument("--model-name", default="bert-base-chinese", help="tokenizer 来源")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--calib-batches", type=int, default=4)
    ap.add_argument("--disable-names", nargs="*", default=None,
                    help="保持 FP16 不量化的节点名（掉精度过大时对敏感层回退）")
    ap.add_argument("--manifest", help="输出量化审计 manifest；默认 <output>.manifest.json")
    args = ap.parse_args()

    # 内容哈希级门：量化校准语料（RAW/clean EMR）绝不能是 final3/blind。
    from dki.evidence.final3_guard import guard_dir
    from dki.paths import OUTPUT_DIR, RAW_DATA_DIR
    guard_dir(RAW_DATA_DIR, "quantization")
    guard_dir(OUTPUT_DIR / "clean", "quantization")

    from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig

    calib = _build_calib_data(args.model_name, args.batch, args.seq_len, args.calib_batches)
    print(f"[calib] {len(calib)} batches x {args.batch} x {args.seq_len}")

    requested_disable_names = args.disable_names or []
    cfg_kwargs = dict(w_bit=8, a_bit=8, disable_names=requested_disable_names,
                      act_method=1, keep_acc=True)
    try:
        cfg = QuantConfig(**cfg_kwargs)
    except TypeError as exc:
        if requested_disable_names:
            raise RuntimeError(
                "installed msmodelslim does not support disable_names; refusing to silently produce full INT8"
            ) from exc
        cfg = QuantConfig(w_bit=8, a_bit=8)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    calibrator = OnnxCalibrator(args.onnx, cfg, calib)
    calibrator.run()
    calibrator.export_quant_onnx(str(out))
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError("msmodelslim did not create a non-empty quantized ONNX")

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    input_path = Path(args.onnx)
    if digest(input_path) == digest(out):
        raise RuntimeError("quantized ONNX is byte-identical to input; refusing unverifiable profile")
    graph_verification = verify_selective_quantization(input_path, out, requested_disable_names)
    manifest = {
        "schema_version": 1,
        "input_onnx": str(input_path),
        "input_sha256": digest(input_path),
        "output_onnx": str(out),
        "output_sha256": digest(out),
        "model_name": args.model_name,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "calib_batches": args.calib_batches,
        "calibration_samples": args.batch * args.calib_batches,
        "disable_names_requested": requested_disable_names,
        "mixed_precision_verified": graph_verification["verified"],
        "graph_verification": graph_verification,
        "config": cfg_kwargs,
    }
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] INT8 量化 ONNX → {out}")
    print(f"[OK] 量化审计 → {manifest_path}")


if __name__ == "__main__":
    main()
