"""Forward-only benchmark for a static-shape Ascend OM model.

This probe measures the lower-level OM/ACL model execution ceiling. It is not a
replacement for the end-to-end BERT NER benchmark, because tokenization,
argmax, BIO decode, and per-request host/device transfers are intentionally
excluded after the initial static input upload.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1


def _check(ret: int, action: str) -> None:
    if isinstance(ret, tuple):
        ret = ret[-1]
    if ret != 0:
        raise RuntimeError(f"{action} failed, ret={ret}")


class OmForwardBench:
    def __init__(self, model_path: Path, device_id: int):
        import acl

        self.acl = acl
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_ptrs: list[int] = []
        self.output_ptrs: list[int] = []
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

    def _add_input(self, array: np.ndarray) -> None:
        acl = self.acl
        contiguous = np.ascontiguousarray(array)
        ptr, ret = acl.rt.malloc(contiguous.nbytes, ACL_MEM_MALLOC_HUGE_FIRST)
        _check(ret, "acl.rt.malloc(input)")
        _check(
            acl.rt.memcpy(
                ptr,
                contiguous.nbytes,
                acl.util.numpy_to_ptr(contiguous),
                contiguous.nbytes,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "acl.rt.memcpy(input)",
        )
        buf = acl.create_data_buffer(ptr, contiguous.nbytes)
        _check(acl.mdl.add_dataset_buffer(self.input_dataset, buf), "acl.mdl.add_dataset_buffer(input)")
        self.input_ptrs.append(ptr)
        self.input_buffers.append(buf)

    def _add_output(self, size: int) -> None:
        acl = self.acl
        ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
        _check(ret, "acl.rt.malloc(output)")
        buf = acl.create_data_buffer(ptr, size)
        _check(acl.mdl.add_dataset_buffer(self.output_dataset, buf), "acl.mdl.add_dataset_buffer(output)")
        self.output_ptrs.append(ptr)
        self.output_buffers.append(buf)

    def prepare_static_io(self, batch_size: int, seq_len: int) -> None:
        acl = self.acl
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()

        self._add_input(np.ones((batch_size, seq_len), dtype=np.int64))
        self._add_input(np.ones((batch_size, seq_len), dtype=np.int64))
        self._add_input(np.zeros((batch_size, seq_len), dtype=np.int64))

        n_outputs = acl.mdl.get_num_outputs(self.desc)
        for i in range(n_outputs):
            self._add_output(acl.mdl.get_output_size_by_index(self.desc, i))

    def run(self, batch_size: int, seq_len: int, warmup: int, iters: int) -> dict:
        self.prepare_static_io(batch_size, seq_len)
        for _ in range(warmup):
            _check(self.acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), "warmup execute")

        lat_ms = []
        start = time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            _check(self.acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), "execute")
            lat_ms.append((time.perf_counter() - t0) * 1000)
        total_s = time.perf_counter() - start
        lat_sorted = sorted(lat_ms)
        return {
            "backend": "acl_om_forward_only",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "iters": iters,
            "throughput_samples_s": round((batch_size * iters) / total_s, 2),
            "execute_latency_p50_ms": round(lat_sorted[len(lat_sorted) // 2], 2),
            "execute_latency_p95_ms": round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))], 2),
            "note": "forward-only static OM; excludes tokenizer/decode and per-iteration host/device copies",
        }

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
        for ptr in self.input_ptrs:
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
        acl.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a static Ascend OM model via Python ACL")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    bench = OmForwardBench(args.model, args.device)
    try:
        result = bench.run(args.batch, args.seq_len, args.warmup, args.iters)
    finally:
        bench.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
