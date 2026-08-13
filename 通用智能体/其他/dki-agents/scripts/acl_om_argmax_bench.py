"""Benchmark a static Ascend OM model with tokenizer, copies, execute, and argmax.

This is a stronger probe than acl_om_forward_bench.py: each timed iteration
tokenizes the input texts, copies inputs to device, executes the OM model,
copies logits back, and performs CPU argmax. BIO span reconstruction is still
optional because the benchmark model is the upstream bert-base-chinese token
classification head, not a domain-finetuned medical NER checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from transformers import AutoConfig, AutoTokenizer

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(ret, action: str) -> None:
    if isinstance(ret, tuple):
        ret = ret[-1]
    if ret != 0:
        raise RuntimeError(f"{action} failed, ret={ret}")


def _p95(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def _select_input_name(raw_name: str) -> str:
    name = raw_name.split(":")[0]
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in name:
            return key
    return name


class StaticOmRunner:
    def __init__(self, model_path: Path, batch_size: int, seq_len: int, device_id: int = 0):
        import acl

        self.acl = acl
        self.model_path = model_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_ptrs: Dict[str, int] = {}
        self.input_nbytes: Dict[str, int] = {}
        self.output_ptrs: List[int] = []
        self.output_nbytes: List[int] = []
        self.input_buffers = []
        self.output_buffers = []

        _check(acl.init(), "acl.init")
        _check(acl.rt.set_device(device_id), "acl.rt.set_device")
        self.context, ret = acl.rt.create_context(device_id)
        _check(ret, "acl.rt.create_context")
        self.model_id, ret = acl.mdl.load_from_file(str(model_path))
        _check(ret, "acl.mdl.load_from_file")
        self.desc = acl.mdl.create_desc()
        _check(acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc")
        self._prepare_datasets()

    def _prepare_datasets(self) -> None:
        acl = self.acl
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()

        for i in range(acl.mdl.get_num_inputs(self.desc)):
            raw_name = acl.mdl.get_input_name_by_index(self.desc, i)
            name = _select_input_name(raw_name)
            size = acl.mdl.get_input_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            _check(ret, f"acl.rt.malloc(input:{name})")
            buf = acl.create_data_buffer(ptr, size)
            _check(acl.mdl.add_dataset_buffer(self.input_dataset, buf), f"add_dataset_buffer(input:{name})")
            self.input_ptrs[name] = ptr
            self.input_nbytes[name] = size
            self.input_buffers.append(buf)

        for i in range(acl.mdl.get_num_outputs(self.desc)):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            _check(ret, f"acl.rt.malloc(output:{i})")
            buf = acl.create_data_buffer(ptr, size)
            _check(acl.mdl.add_dataset_buffer(self.output_dataset, buf), f"add_dataset_buffer(output:{i})")
            self.output_ptrs.append(ptr)
            self.output_nbytes.append(size)
            self.output_buffers.append(buf)

    def execute(self, arrays: Dict[str, np.ndarray], output_dtype=np.float16) -> List[np.ndarray]:
        acl = self.acl
        for name, ptr in self.input_ptrs.items():
            arr = np.ascontiguousarray(arrays[name])
            if arr.nbytes != self.input_nbytes[name]:
                raise ValueError(f"{name} has {arr.nbytes} bytes, expected {self.input_nbytes[name]}")
            _check(
                acl.rt.memcpy(
                    ptr,
                    arr.nbytes,
                    acl.util.numpy_to_ptr(arr),
                    arr.nbytes,
                    ACL_MEMCPY_HOST_TO_DEVICE,
                ),
                f"memcpy H2D {name}",
            )

        _check(acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), "acl.mdl.execute")

        outputs = []
        for ptr, size in zip(self.output_ptrs, self.output_nbytes):
            host = np.empty(size // np.dtype(output_dtype).itemsize, dtype=output_dtype)
            _check(
                acl.rt.memcpy(
                    acl.util.numpy_to_ptr(host),
                    size,
                    ptr,
                    size,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
                "memcpy D2H output",
            )
            outputs.append(host)
        return outputs

    def close(self) -> None:
        acl = self.acl
        for buf in self.input_buffers:
            acl.destroy_data_buffer(buf)
        for buf in self.output_buffers:
            acl.destroy_data_buffer(buf)
        if self.input_dataset is not None:
            acl.mdl.destroy_dataset(self.input_dataset)
        if self.output_dataset is not None:
            acl.mdl.destroy_dataset(self.output_dataset)
        for ptr in self.input_ptrs.values():
            acl.rt.free(ptr)
        for ptr in self.output_ptrs:
            acl.rt.free(ptr)
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        # In this mixed tokenizers + ACL process, some CANN Python builds can
        # segfault during interpreter shutdown after acl.finalize(). All model,
        # dataset, buffer, context, and device resources have already been
        # released above; process exit will clean up the remaining ACL runtime
        # state more reliably on the tested CANN 8.5.1 environment.


def _batches(texts: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if len(batch) < batch_size:
            batch = batch + [batch[-1]] * (batch_size - len(batch))
        yield batch


def _decode_loop(pred_ids: np.ndarray, offsets: np.ndarray, id2label: Dict[int, str]) -> int:
    """Run a minimal BIO span loop and return entity count to prevent dead work."""
    n_entities = 0
    for row_ids, row_offsets in zip(pred_ids, offsets):
        in_entity = False
        cur_tag = None
        for pid, (start, end) in zip(row_ids.tolist(), row_offsets.tolist()):
            if start == end:
                continue
            label = str(id2label.get(int(pid), "O"))
            if label.startswith("B-"):
                n_entities += 1
                in_entity = True
                cur_tag = label[2:]
            elif label.startswith("I-") and in_entity and cur_tag == label[2:]:
                continue
            else:
                in_entity = False
                cur_tag = None
    return n_entities


def run_bench(args: argparse.Namespace) -> dict:
    from dki.bench.npu.run_npu_bench import _load_texts

    texts = _load_texts(args.batch * args.text_multiplier)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    id2label = AutoConfig.from_pretrained(args.model_name).id2label
    runner = StaticOmRunner(args.om, args.batch, args.seq_len, args.device)
    checksum = 0
    decode_entities = 0
    try:
        for _ in range(args.warmup):
            for batch in _batches(texts, args.batch):
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=args.seq_len,
                    return_offsets_mapping=args.decode,
                    return_tensors="np",
                )
                offsets = enc.pop("offset_mapping", None)
                arrays = {
                    "input_ids": enc["input_ids"].astype(np.int64, copy=False),
                    "attention_mask": enc["attention_mask"].astype(np.int64, copy=False),
                    "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64, copy=False),
                }
                logits = runner.execute(arrays, output_dtype=np.float16)[0]
                num_labels = logits.size // (args.batch * args.seq_len)
                pred_ids = logits.reshape(args.batch, args.seq_len, num_labels).argmax(axis=-1)
                checksum += int(pred_ids[:, 0].sum())
                if args.decode and offsets is not None:
                    decode_entities += _decode_loop(pred_ids, offsets, id2label)

        lat_ms = []
        total_samples = 0
        t_all = time.perf_counter()
        for _ in range(args.iters):
            t0 = time.perf_counter()
            for batch in _batches(texts, args.batch):
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=args.seq_len,
                    return_offsets_mapping=args.decode,
                    return_tensors="np",
                )
                offsets = enc.pop("offset_mapping", None)
                arrays = {
                    "input_ids": enc["input_ids"].astype(np.int64, copy=False),
                    "attention_mask": enc["attention_mask"].astype(np.int64, copy=False),
                    "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64, copy=False),
                }
                logits = runner.execute(arrays, output_dtype=np.float16)[0]
                num_labels = logits.size // (args.batch * args.seq_len)
                pred_ids = logits.reshape(args.batch, args.seq_len, num_labels).argmax(axis=-1)
                checksum += int(pred_ids[:, 0].sum())
                if args.decode and offsets is not None:
                    decode_entities += _decode_loop(pred_ids, offsets, id2label)
            lat_ms.append((time.perf_counter() - t0) * 1000)
            total_samples += len(texts)
        total_s = time.perf_counter() - t_all
    finally:
        runner.close()

    return {
        "backend": "acl_om_tokenize_copy_execute_copy_argmax",
        "mode": "decode" if args.decode else "argmax",
        "batch_size": args.batch,
        "seq_len": args.seq_len,
        "n_texts_per_iter": len(texts),
        "iters": args.iters,
        "throughput_samples_s": round(total_samples / total_s, 2),
        "iteration_latency_p50_ms": round(sorted(lat_ms)[len(lat_ms) // 2], 2),
        "iteration_latency_p95_ms": round(_p95(lat_ms), 2),
        "checksum": checksum,
        "decode_entities": decode_entities,
        "note": "includes tokenizer, H2D copy, OM execute, D2H copy, CPU argmax"
                + (" and BIO span loop" if args.decode else "; excludes BIO span decode"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ACL OM benchmark including tokenizer/copies/argmax")
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--text-multiplier", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--decode", action="store_true", help="also run a CPU BIO span loop with tokenizer offsets")
    args = parser.parse_args()

    result = run_bench(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
