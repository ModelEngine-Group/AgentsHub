"""Pipelined ACL OM end-to-end NER benchmark (tokenize / execute / decode overlap).

Builds on acl_om_argmax_bench.py (serialized probe). Two further optimizations:

1. Vectorized BIO decode: the serialized probe walks every token in a Python
   loop; here argmax + validity masking + span anchoring are computed with
   numpy array ops, so decode cost is per-entity instead of per-token.
2. Three-stage pipeline: tokenizer worker threads, one NPU executor thread
   (owns the ACL context; H2D + execute + D2H), and decode worker threads are
   connected with bounded queues, so host-side work overlaps device execution.
   HF fast tokenizers (Rust) and numpy release the GIL, which is what makes
   thread-level overlap effective here.

`--stages serial` runs the same vectorized-decode code path single-threaded,
isolating the vectorization gain from the pipelining gain.

Same caveat as the other probes: the benchmark model is upstream
bert-base-chinese with an untuned token-classification head, so this measures
throughput and link viability, not extraction quality.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# transformers 延迟导入（见 _load_all / worker 函数）——使 VectorizedBioDecoder
# 等纯 numpy 组件可在无 transformers 的环境（如本机 .venv）被单元测试直接导入。

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

_SENTINEL = object()


def _check(ret, action: str) -> None:
    if isinstance(ret, tuple):
        ret = ret[-1]
    if ret != 0:
        raise RuntimeError(f"{action} failed, ret={ret}")


def _select_input_name(raw_name: str) -> str:
    name = raw_name.split(":")[0]
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in name:
            return key
    return name


class LegacyStaticOmRunner:
    """Same ACL lifecycle as acl_om_argmax_bench.StaticOmRunner, plus
    bind_thread() so a worker thread can own the context."""

    def __init__(self, model_path: Path, device_id: int = 0):
        import acl

        self.acl = acl
        self.device_id = device_id
        self.input_ptrs: Dict[str, int] = {}
        self.input_nbytes: Dict[str, int] = {}
        self.output_ptrs: List[int] = []
        self.output_nbytes: List[int] = []
        self.input_buffers = []
        self.output_buffers = []
        self.context = None
        self.model_id = None
        self.desc = None
        self.input_dataset = None
        self.output_dataset = None
        self._device_set = False
        self._closed = False

        try:
            _check(acl.init(), "acl.init")
            _check(acl.rt.set_device(device_id), "acl.rt.set_device")
            self._device_set = True
            self.context, ret = acl.rt.create_context(device_id)
            _check(ret, "acl.rt.create_context")
            self.model_id, ret = acl.mdl.load_from_file(str(model_path))
            _check(ret, "acl.mdl.load_from_file")
            self.desc = acl.mdl.create_desc()
            _check(acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc")

            self.input_dataset = acl.mdl.create_dataset()
            self.output_dataset = acl.mdl.create_dataset()
            for i in range(acl.mdl.get_num_inputs(self.desc)):
                name = _select_input_name(acl.mdl.get_input_name_by_index(self.desc, i))
                if name in self.input_ptrs:
                    raise RuntimeError(f"duplicate normalized ACL input name: {name}")
                size = acl.mdl.get_input_size_by_index(self.desc, i)
                ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
                _check(ret, f"acl.rt.malloc(input:{name})")
                self.input_ptrs[name] = ptr
                self.input_nbytes[name] = size
                buf = acl.create_data_buffer(ptr, size)
                self.input_buffers.append(buf)
                _check(acl.mdl.add_dataset_buffer(self.input_dataset, buf), f"add_dataset_buffer(input:{name})")
            for i in range(acl.mdl.get_num_outputs(self.desc)):
                size = acl.mdl.get_output_size_by_index(self.desc, i)
                ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
                _check(ret, f"acl.rt.malloc(output:{i})")
                self.output_ptrs.append(ptr)
                self.output_nbytes.append(size)
                buf = acl.create_data_buffer(ptr, size)
                self.output_buffers.append(buf)
                _check(acl.mdl.add_dataset_buffer(self.output_dataset, buf), f"add_dataset_buffer(output:{i})")

            dims, ret = acl.mdl.get_output_dims(self.desc, 0)
            _check(ret, "acl.mdl.get_output_dims")
            # host buffers can be alignment-padded beyond prod(dims); slice before reshape
            self.output_shape: Tuple[int, ...] = tuple(int(d) for d in dims["dims"])
            # 输出数据类型随 OM 精度模式变化：force_fp16 的 OM 其 IO 常为 FP32；
            # 动态读取而非硬编码 FP16，否则按错误 itemsize 解析 host 缓冲区 → argmax 全乱。
            _ACL_DT = {
                0: np.float32, 1: np.float16, 3: np.int32,
                9: np.int64, 11: np.float64, 12: np.bool_,
            }
            dt_enum = acl.mdl.get_output_data_type(self.desc, 0)
            if isinstance(dt_enum, tuple):
                dt_enum = dt_enum[0]
            if int(dt_enum) not in _ACL_DT:
                raise RuntimeError(
                    f"unsupported ACL output dtype enum {dt_enum}; refusing to guess FP16"
                )
            self.output_dtype = _ACL_DT[int(dt_enum)]
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def bind_thread(self) -> None:
        _check(self.acl.rt.set_context(self.context), "acl.rt.set_context")

    def execute(self, arrays: Dict[str, np.ndarray], output_dtype=None) -> List[np.ndarray]:
        acl = self.acl
        if output_dtype is None:
            output_dtype = self.output_dtype
        for name, ptr in self.input_ptrs.items():
            arr = np.ascontiguousarray(arrays[name])
            if arr.nbytes != self.input_nbytes[name]:
                raise ValueError(f"{name} has {arr.nbytes} bytes, expected {self.input_nbytes[name]}")
            _check(
                acl.rt.memcpy(ptr, arr.nbytes, acl.util.numpy_to_ptr(arr), arr.nbytes, ACL_MEMCPY_HOST_TO_DEVICE),
                f"memcpy H2D {name}",
            )
        _check(acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), "acl.mdl.execute")
        outputs = []
        for ptr, size in zip(self.output_ptrs, self.output_nbytes):
            host = np.empty(size // np.dtype(output_dtype).itemsize, dtype=output_dtype)
            _check(
                acl.rt.memcpy(acl.util.numpy_to_ptr(host), size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST),
                "memcpy D2H output",
            )
            outputs.append(host)
        return outputs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        acl = self.acl
        for buf in reversed(self.input_buffers + self.output_buffers):
            try:
                acl.destroy_data_buffer(buf)
            except BaseException:
                pass
        for dataset in (self.output_dataset, self.input_dataset):
            if dataset is not None:
                try:
                    acl.mdl.destroy_dataset(dataset)
                except BaseException:
                    pass
        for ptr in reversed(list(self.input_ptrs.values()) + self.output_ptrs):
            try:
                acl.rt.free(ptr)
            except BaseException:
                pass
        if self.desc is not None:
            try:
                acl.mdl.destroy_desc(self.desc)
            except BaseException:
                pass
        if self.model_id is not None:
            try:
                acl.mdl.unload(self.model_id)
            except BaseException:
                pass
        if self.context is not None:
            try:
                acl.rt.destroy_context(self.context)
            except BaseException:
                pass
        if self._device_set:
            try:
                acl.rt.reset_device(self.device_id)
            except BaseException:
                pass
        # acl.finalize() intentionally skipped: see note in acl_om_argmax_bench.py


from dki.bench.npu.static_om_runner import StaticOmRunner
from dki.nlp.bio_decode import BioDecoder


class VectorizedBioDecoder(BioDecoder):
    """Backward-compatible name for the canonical shared BIO decoder.

    The implementation intentionally uses the reference state machine for all
    legal and malformed BIO transitions.  In production the OM forward pass is
    dominant; correctness and PyTorch/OM parity take precedence over the old
    sub-millisecond numpy shortcut.
    """


def _load_all(args) -> Tuple[List[List[str]], object, VectorizedBioDecoder]:
    from transformers import AutoConfig, AutoTokenizer

    from dki.bench.npu.run_npu_bench import _load_texts

    texts = _load_texts(args.batch * args.text_multiplier)
    batches: List[List[str]] = []
    for i in range(0, len(texts), args.batch):
        b = texts[i:i + args.batch]
        if len(b) < args.batch:
            b = b + [b[-1]] * (args.batch - len(b))
        batches.append(b)
    decoder = VectorizedBioDecoder(AutoConfig.from_pretrained(args.model_name).id2label)
    return batches, AutoTokenizer.from_pretrained(args.model_name), decoder


def _tokenize(tokenizer, batch: List[str], seq_len: int, with_offsets: bool = True) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    enc = tokenizer(
        batch,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_offsets_mapping=with_offsets,
        return_tensors="np",
    )
    offsets = enc.pop("offset_mapping") if with_offsets else None
    arrays = {
        "input_ids": enc["input_ids"].astype(np.int64, copy=False),
        "attention_mask": enc["attention_mask"].astype(np.int64, copy=False),
        "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64, copy=False),
    }
    return arrays, offsets


# ---- multiprocessing workers (fork context, started before ACL init) ----


def _tok_worker_proc(model_name: str, seq_len: int, batches, q_work, q_tok, q_ready, with_offsets: bool = True) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _tokenize(tokenizer, batches[0], seq_len, with_offsets)  # trigger rayon pool before signaling ready
    q_ready.put("tok")
    while True:
        item = q_work.get()
        if item is None:
            break
        t0 = time.perf_counter()
        idx = item % len(batches)
        arrays, offsets = _tokenize(tokenizer, batches[idx], seq_len, with_offsets)
        # fast-tokenize：不算 offset，把 batch 索引传给 decode worker 自行补算（decode 侧空闲）
        q_tok.put((arrays, offsets, idx, time.perf_counter() - t0))


def _dec_worker_proc(model_name: str, seq_len: int, batches, q_dec, q_res, q_ready) -> None:
    from transformers import AutoConfig, AutoTokenizer

    decoder = VectorizedBioDecoder(AutoConfig.from_pretrained(model_name).id2label)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 仅 fast-tokenize 补算 offset 用
    q_ready.put("dec")
    while True:
        item = q_dec.get()
        if item is None:
            break
        pred, offsets, idx = item
        t0 = time.perf_counter()
        if offsets is None:  # fast-tokenize：decode 侧补算 offset（利用其空闲算力）
            enc = tokenizer(batches[idx], truncation=True, padding="max_length",
                            max_length=seq_len, return_offsets_mapping=True, return_tensors="np")
            offsets = enc["offset_mapping"]
        n, _spans = decoder.decode(pred, offsets)
        q_res.put((n, int(pred[:, 0].sum()), time.perf_counter() - t0))


def run_pipeline(args, batches, tokenizer, decoder, runner) -> dict:
    """Bounded thread-pipeline ablation retained by the public CLI.

    ACL execution remains on the owner/main thread. Tokenization and decoding
    run in bounded worker queues; worker exceptions are surfaced rather than
    being mistaken for a slow pipeline.
    """
    if args.tokenizer_workers <= 0 or args.decode_workers <= 0:
        raise ValueError("pipeline stages require positive tokenizer-workers and decode-workers")
    if args.queue_depth <= 0:
        raise ValueError("queue-depth must be positive")
    if not batches:
        raise ValueError("pipeline requires at least one input batch")
    n_batches = args.iters * len(batches)
    work = queue.Queue(maxsize=args.queue_depth)
    tokenized = queue.Queue(maxsize=args.queue_depth)
    decode_work = queue.Queue(maxsize=args.queue_depth)
    results = queue.Queue(maxsize=args.queue_depth)
    errors = queue.Queue(maxsize=args.tokenizer_workers + args.decode_workers + 1)
    stop = threading.Event()

    def put_bounded(target, value):
        while not stop.is_set():
            try:
                target.put(value, timeout=0.2)
                return
            except queue.Full:
                continue
        raise RuntimeError("pipeline stopped")

    def token_worker():
        try:
            while not stop.is_set():
                try:
                    item = work.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    return
                t0 = time.perf_counter()
                idx = item % len(batches)
                arrays, offsets = _tokenize(tokenizer, batches[idx], args.seq_len)
                put_bounded(tokenized, (arrays, offsets, idx, time.perf_counter() - t0))
        except BaseException as exc:
            errors.put(exc)
            stop.set()

    def decode_worker():
        try:
            while not stop.is_set():
                try:
                    item = decode_work.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    return
                pred, offsets = item
                t0 = time.perf_counter()
                n_entities, _ = decoder.decode(pred, offsets)
                put_bounded(results, (n_entities, int(pred[:, 0].sum()), time.perf_counter() - t0))
        except BaseException as exc:
            errors.put(exc)
            stop.set()

    token_threads = [threading.Thread(target=token_worker, daemon=True) for _ in range(args.tokenizer_workers)]
    decode_threads = [threading.Thread(target=decode_worker, daemon=True) for _ in range(args.decode_workers)]
    threads = token_threads + decode_threads
    for thread in threads:
        thread.start()

    shape = runner.output_shape
    n_elems = int(np.prod(shape))
    for i in range(args.warmup_batches):
        arrays, _ = _tokenize(tokenizer, batches[i % len(batches)], args.seq_len)
        runner.execute(arrays)

    stage_busy = {"tokenize": 0.0, "execute": 0.0, "decode": 0.0}
    entities = checksum = collected = 0
    def feed_thread_work():
        for i in range(n_batches):
            put_bounded(work, i)
        for _ in token_threads:
            put_bounded(work, _SENTINEL)

    feeder = threading.Thread(target=feed_thread_work, daemon=True)
    t_all = time.perf_counter()
    feeder.start()
    try:
        for _ in range(n_batches):
            if not errors.empty():
                raise errors.get()
            arrays, offsets, _idx, tok_s = tokenized.get(timeout=60)
            stage_busy["tokenize"] += tok_s
            t0 = time.perf_counter()
            logits = runner.execute(arrays)[0]
            stage_busy["execute"] += time.perf_counter() - t0
            pred = logits[:n_elems].reshape(shape).argmax(axis=-1)
            put_bounded(decode_work, (pred, offsets))
            # Keep the result queue truly bounded: drain completed decode work
            # while NPU submissions continue instead of retaining all batches.
            while True:
                try:
                    n_entities, chk, dec_s = results.get_nowait()
                except queue.Empty:
                    break
                entities += n_entities
                checksum += chk
                stage_busy["decode"] += dec_s
                collected += 1
        while collected < n_batches:
            if not errors.empty():
                raise errors.get()
            n_entities, chk, dec_s = results.get(timeout=60)
            entities += n_entities
            checksum += chk
            stage_busy["decode"] += dec_s
            collected += 1
        for _ in decode_threads:
            put_bounded(decode_work, _SENTINEL)
        total_s = time.perf_counter() - t_all
    finally:
        stop.set()
        feeder.join(timeout=5)
        for thread in threads:
            thread.join(timeout=5)

    total_samples = n_batches * args.batch
    return {
        "backend": "acl_om_pipeline",
        "stages": "pipeline",
        "tokenizer_workers": args.tokenizer_workers,
        "decode_workers": args.decode_workers,
        "queue_depth": args.queue_depth,
        "batch_size": args.batch,
        "seq_len": args.seq_len,
        "n_batches": n_batches,
        "throughput_samples_s": round(total_samples / total_s, 2),
        "wall_s": round(total_s, 3),
        "stage_busy_s": {key: round(value, 3) for key, value in stage_busy.items()},
        "device_occupancy": round(stage_busy["execute"] / total_s, 3),
        "entities": entities,
        "checksum": checksum,
        "note": "bounded thread pipeline/GIL ablation; ACL remains on owner thread",
    }


def run_mp(args, batches, tokenizer) -> dict:
    """Process-based pipeline. The GIL serializes the thread version (pyACL and
    fp16 numpy hold it for most of their runtime), so tokenize and decode move
    to forked worker processes while the main process keeps exclusive ACL
    ownership. Workers are forked BEFORE acl.init() (CANN and fork do not mix)
    and are fully warmed before the timed window opens."""
    import multiprocessing as mp

    if args.tokenizer_workers <= 0 or args.decode_workers <= 0:
        raise ValueError("mp stages require positive tokenizer-workers and decode-workers")
    if args.queue_depth <= 0:
        raise ValueError("queue-depth must be positive")

    ctx = mp.get_context("fork")
    n_batches = args.iters * len(batches)
    q_work = ctx.Queue(maxsize=args.queue_depth)
    q_tok = ctx.Queue(maxsize=args.queue_depth)
    q_dec = ctx.Queue(maxsize=args.queue_depth)
    q_res = ctx.Queue(maxsize=args.queue_depth)
    q_ready = ctx.Queue(maxsize=max(1, args.tokenizer_workers + args.decode_workers))

    with_offsets = not getattr(args, "fast_tokenize", False)
    procs = [
        ctx.Process(target=_tok_worker_proc, daemon=True,
                    args=(args.model_name, args.seq_len, batches, q_work, q_tok, q_ready, with_offsets))
        for _ in range(args.tokenizer_workers)
    ] + [
        ctx.Process(target=_dec_worker_proc, daemon=True,
                    args=(args.model_name, args.seq_len, batches, q_dec, q_res, q_ready))
        for _ in range(args.decode_workers)
    ]
    for proc in procs:
        proc.start()
    for _ in procs:
        q_ready.get(timeout=180)

    runner = None
    feeder_stop = threading.Event()

    def feed_work():
        try:
            for i in range(n_batches):
                while not feeder_stop.is_set():
                    try:
                        q_work.put(i, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            for _ in range(args.tokenizer_workers):
                while not feeder_stop.is_set():
                    try:
                        q_work.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except BaseException:
            feeder_stop.set()

    def get_with_liveness(source, workers, stage, timeout_s=60):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                return source.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                dead = [proc for proc in workers if not proc.is_alive() and proc.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(
                        f"{stage} worker exited early: {[proc.exitcode for proc in dead]}"
                    )
        raise TimeoutError(f"timed out waiting for {stage} worker output")

    feeder = threading.Thread(target=feed_work, daemon=True)
    try:
        runner = StaticOmRunner(args.om, args.device)
        shape = runner.output_shape
        n_elems = int(np.prod(shape))
        warm_arrays, warm_offsets = _tokenize(tokenizer, batches[0], args.seq_len)
        warm_logits = runner.execute(warm_arrays)[0]
        for _ in range(max(0, args.warmup_batches - 1)):
            warm_logits = runner.execute(warm_arrays)[0]
        warm_prediction = warm_logits[:n_elems].reshape(shape).argmax(axis=-1)
        for _ in range(args.decode_workers):
            warm_worker_offsets = None if getattr(args, "fast_tokenize", False) else warm_offsets
            q_dec.put((warm_prediction, warm_worker_offsets, 0), timeout=10)
        for _ in range(args.decode_workers):
            get_with_liveness(q_res, procs[args.tokenizer_workers:], "decode warmup")

        stage_busy = {"tokenize": 0.0, "execute": 0.0, "decode": 0.0}
        entities = 0
        checksum = 0
        collected = 0

        t_all = time.perf_counter()
        feeder.start()
        for _ in range(n_batches):
            arrays, offsets, idx, tok_s = get_with_liveness(
                q_tok, procs[:args.tokenizer_workers], "tokenizer"
            )
            stage_busy["tokenize"] += tok_s
            t0 = time.perf_counter()
            logits = runner.execute(arrays)[0]
            stage_busy["execute"] += time.perf_counter() - t0
            prediction = logits[:n_elems].reshape(shape).argmax(axis=-1)
            while True:
                try:
                    q_dec.put((prediction, offsets, idx), timeout=0.5)
                    break
                except queue.Full:
                    if not any(proc.is_alive() for proc in procs[args.tokenizer_workers:]):
                        raise RuntimeError("all decode workers exited")
                    try:
                        n, chk, dec_s = q_res.get_nowait()
                    except queue.Empty:
                        continue
                    entities += n
                    checksum += chk
                    stage_busy["decode"] += dec_s
                    collected += 1
            while collected < n_batches:
                try:
                    n, chk, dec_s = q_res.get_nowait()
                except queue.Empty:
                    break
                entities += n
                checksum += chk
                stage_busy["decode"] += dec_s
                collected += 1
        while collected < n_batches:
            n, chk, dec_s = get_with_liveness(
                q_res, procs[args.tokenizer_workers:], "decode"
            )
            entities += n
            checksum += chk
            stage_busy["decode"] += dec_s
            collected += 1
        total_s = time.perf_counter() - t_all
    finally:
        feeder_stop.set()
        if feeder.is_alive():
            feeder.join(timeout=2)
        for _ in range(args.decode_workers):
            try:
                q_dec.put_nowait(None)
            except queue.Full:
                break
        for proc in procs:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        if runner is not None:
            runner.close()

    total_samples = n_batches * args.batch
    return {
        "backend": "acl_om_pipeline",
        "stages": "mp",
        "tokenizer_workers": args.tokenizer_workers,
        "decode_workers": args.decode_workers,
        "queue_depth": args.queue_depth,
        "fast_tokenize": bool(getattr(args, "fast_tokenize", False)),
        "batch_size": args.batch,
        "seq_len": args.seq_len,
        "n_batches": n_batches,
        "throughput_samples_s": round(total_samples / total_s, 2),
        "wall_s": round(total_s, 3),
        "stage_busy_s": {k: round(v, 3) for k, v in stage_busy.items()},
        "device_occupancy": round(stage_busy["execute"] / total_s, 3),
        "entities": entities,
        "checksum": checksum,
        "note": "tokenize/decode in forked worker processes, main-process argmax, ACL exclusive to main; vectorized BIO decode",
    }


def run_serial(args, batches, tokenizer, decoder, runner) -> dict:
    n_batches = args.iters * len(batches)
    for i in range(args.warmup_batches):
        arrays, offsets = _tokenize(tokenizer, batches[i % len(batches)], args.seq_len)
        runner.execute(arrays)

    stage_busy = {"tokenize": 0.0, "execute": 0.0, "decode": 0.0}
    entities = 0
    checksum = 0
    t_all = time.perf_counter()
    for i in range(n_batches):
        t0 = time.perf_counter()
        arrays, offsets = _tokenize(tokenizer, batches[i % len(batches)], args.seq_len)
        t1 = time.perf_counter()
        logits = runner.execute(arrays)[0]
        t2 = time.perf_counter()
        shape = runner.output_shape
        pred = logits[: int(np.prod(shape))].reshape(shape).argmax(axis=-1)
        n, _spans = decoder.decode(pred, offsets)
        entities += n
        checksum += int(pred[:, 0].sum())
        t3 = time.perf_counter()
        stage_busy["tokenize"] += t1 - t0
        stage_busy["execute"] += t2 - t1
        stage_busy["decode"] += t3 - t2
    total_s = time.perf_counter() - t_all

    total_samples = n_batches * args.batch
    return {
        "backend": "acl_om_pipeline",
        "stages": "serial",
        "batch_size": args.batch,
        "seq_len": args.seq_len,
        "n_batches": n_batches,
        "throughput_samples_s": round(total_samples / total_s, 2),
        "wall_s": round(total_s, 3),
        "stage_busy_s": {k: round(v, 3) for k, v in stage_busy.items()},
        "device_occupancy": round(stage_busy["execute"] / total_s, 3),
        "entities": entities,
        "checksum": checksum,
        "note": "same vectorized BIO decode, stages run sequentially (pipelining ablation)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipelined ACL OM end-to-end NER benchmark")
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--text-multiplier", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--iters", type=int, default=25, help="passes over the batch set")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stages", choices=["pipeline", "serial", "mp"], default="mp")
    parser.add_argument("--tokenizer-workers", type=int, default=2)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--queue-depth", type=int, default=4)
    parser.add_argument("--fast-tokenize", action="store_true",
                        help="tokenize 不算 offset（省 ~53%%），offset 移到空闲的 decode worker 补算——"
                             "当 tokenize 成为瓶颈（如小模型/INT8）时显著提速")
    args = parser.parse_args()
    if args.batch <= 0 or args.text_multiplier <= 0 or args.iters <= 0:
        parser.error("batch, text-multiplier and iters must be positive")
    if args.warmup_batches < 0:
        parser.error("warmup-batches must be non-negative")
    if args.queue_depth <= 0:
        parser.error("queue-depth must be positive")
    if args.stages in {"pipeline", "mp"} and (
        args.tokenizer_workers <= 0 or args.decode_workers <= 0
    ):
        parser.error("pipeline/mp stages require positive worker counts")

    batches, tokenizer, decoder = _load_all(args)
    if not batches:
        parser.error("input corpus produced no batches")
    if args.stages == "mp":
        result = run_mp(args, batches, tokenizer)
    else:
        runner = StaticOmRunner(args.om, args.device)
        try:
            if args.stages == "pipeline":
                result = run_pipeline(args, batches, tokenizer, decoder, runner)
            else:
                result = run_serial(args, batches, tokenizer, decoder, runner)
        finally:
            runner.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
