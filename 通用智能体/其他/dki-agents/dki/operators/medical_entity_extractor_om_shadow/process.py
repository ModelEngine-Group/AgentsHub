"""Independent opt-in OM shadow operator.

The signed primary backend remains authoritative. Shadow inference is submitted
best-effort to an injected runtime and only pseudonymous telemetry is written;
all shadow failures are isolated from the primary sample.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from dki.dmcompat import Mapper, OPERATORS

from dki.operators.medical_entity_extractor.process import MedicalEntityExtractor


def _load_factory(dotted_path: str) -> Callable[..., Any]:
    module_name, separator, attribute = dotted_path.rpartition(".")
    if not separator or not module_name or not attribute:
        raise ValueError("shadowRuntimeFactory must be a dotted module attribute")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError(f"shadow runtime factory is not callable: {dotted_path}")
    return factory


# Registration is done lazily in __init__.py via a string module_path (mirrors
# the five catalog operators), so importing this module does not double-register.
class MedicalEntityExtractorOmShadow(Mapper):
    def __init__(self, *args, shadow_runtime=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_backend = kwargs.get("primaryBackend", "dict")
        self.sidecar_path = kwargs.get("sidecarPath", "")
        self.queue_depth = max(1, int(kwargs.get("queueDepth", 4)))
        self.max_batch_size = min(
            self.queue_depth,
            max(1, int(kwargs.get("maxBatchSize", self.queue_depth))),
        )
        self.max_wait_s = max(0.0, float(kwargs.get("maxWaitMs", 10)) / 1000)
        self.shadow_timeout_s = max(0.001, float(kwargs.get("shadowTimeoutMs", 1000)) / 1000)
        self.enabled = bool(kwargs.get("shadowEnabled", True))
        runtime_factory_path = str(kwargs.get("shadowRuntimeFactory", "") or "")
        runtime_config = kwargs.get("shadowRuntimeConfig", None)
        if runtime_config is None:
            runtime_config = {}
        if not isinstance(runtime_config, dict):
            raise ValueError("shadowRuntimeConfig must be a mapping")
        if shadow_runtime is not None and runtime_factory_path:
            raise ValueError("provide either shadow_runtime or shadowRuntimeFactory, not both")
        if self.enabled and shadow_runtime is None and not runtime_factory_path:
            raise ValueError(
                "shadowEnabled=true requires shadow_runtime or shadowRuntimeFactory; "
                "refusing to run an inert OM shadow"
            )
        self._runtime_factory_path = runtime_factory_path if self.enabled else ""
        self._runtime_config = dict(runtime_config) if self.enabled else {}
        self._runtime = shadow_runtime if self.enabled else None
        if self._runtime is not None and not callable(getattr(self._runtime, "execute_batch", None)):
            raise ValueError("shadow runtime must provide execute_batch(texts, timeout_s=...)")
        secret_name = str(kwargs.get("shadowHashSecretEnv", "DKI_SHADOW_HASH_KEY"))
        secret = os.environ.get(secret_name, "") if self.enabled else ""
        if self.enabled and not secret:
            raise ValueError(f"enabled shadow requires pseudonymization secret in {secret_name}")
        self._hash_key = secret.encode("utf-8")
        self._primary = MedicalEntityExtractor(
            backend=self.primary_backend,
            modelPath=kwargs.get("modelPath"),
            dictPath=kwargs.get("dictPath"),
            # The wrapper owns DataMate's first/last lifecycle. Delegating these
            # flags would read/export twice and loses the raw text needed by the
            # asynchronous shadow request.
            is_first_op=False,
            is_last_op=False,
            accelerator=kwargs.get("accelerator", "cpu"),
            text_key=self.text_key,
            data_key=self.data_key,
            fileName_key=self.filename_key,
            fileType_key=self.filetype_key,
            fileId_key=self.fileid_key,
            filePath_key=self.filepath_key,
            fileSize_key=self.filesize_key,
            ext_params_key=self.ext_params_key,
        )
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_depth)
        self._stop = threading.Event()
        self._worker = None
        self._runtime_ready = threading.Event()
        self._runtime_error = None
        self._dropped = 0
        if self.enabled:
            self._worker = threading.Thread(target=self._shadow_loop, daemon=True)
            self._worker.start()
            if not self._runtime_ready.wait(timeout=10):
                self._stop.set()
                raise RuntimeError("shadow runtime initialization timed out")
            if self._runtime_error is not None:
                raise ValueError(f"shadow runtime initialization failed: {self._runtime_error}") from self._runtime_error

    @property
    def dropped(self) -> int:
        return self._dropped

    def _extract_primary(self, text: str) -> List[Dict[str, Any]]:
        return self._primary._extract_backend(text)

    def execute(self, sample):
        # The wrapper, not the nested implementation detail, owns DataMate's
        # file-loading boundary when it is configured as the first operator.
        self.read_file_first(sample)
        # Run the unmodified primary extraction semantics so section parsing,
        # output ordering and ext_params stay byte/structure compatible.
        original_text = sample[self.text_key]
        primary_output = self._primary.execute(sample)
        authoritative = json.loads(primary_output[self.text_key])
        entities = authoritative.get("entities", [])
        sections = authoritative.get("sections", {}) or {}
        if self.enabled:
            shadow_text = "\n".join(str(value) for value in sections.values()) if sections else original_text
            request = {
                "request_id": str(sample.get("fileId") or sample.get("fileName") or ""),
                "text": shadow_text,
                "primary": entities,
                "submitted_mono": time.monotonic(),
            }
            try:
                self._queue.put_nowait(request)
            except queue.Full:
                self._dropped += 1
        return primary_output

    def _close_runtime_isolated(self):
        close = getattr(self._runtime, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as exc:
                logger.warning("OM shadow runtime close isolated: {}", exc)

    def _shadow_loop(self):
        try:
            try:
                if self._runtime is None:
                    self._runtime = _load_factory(self._runtime_factory_path)(**self._runtime_config)
                if not callable(getattr(self._runtime, "execute_batch", None)):
                    raise ValueError(
                        "shadow runtime must provide execute_batch(texts, timeout_s=...)"
                    )
            except BaseException as exc:
                self._runtime_error = exc
                return
            finally:
                self._runtime_ready.set()
            while True:
                try:
                    request = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._stop.is_set():
                        return
                    continue
                if request is None:
                    self._queue.task_done()
                    return
                batch = [request]
                self._drain_shadow_batch(batch)
                try:
                    predictions = self._runtime.execute_batch(
                        [item["text"] for item in batch], timeout_s=self.shadow_timeout_s,
                    )
                    if len(predictions) != len(batch):
                        raise RuntimeError(
                            f"shadow runtime returned {len(predictions)} results for {len(batch)} requests"
                        )
                    for item, prediction in zip(batch, predictions):
                        self._write_sidecar(item, prediction, None)
                except BaseException as exc:
                    error_kind = getattr(getattr(exc, "kind", None), "value", None) or type(exc).__name__
                    for item in batch:
                        self._write_sidecar(item, None, error_kind)
                    logger.warning("OM shadow batch inference isolated: {}", error_kind)
                finally:
                    for _ in batch:
                        self._queue.task_done()
        finally:
            # The same thread that first owns pyACL must release it.
            self._close_runtime_isolated()

    def _drain_shadow_batch(self, batch) -> None:
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                request = self._queue.get(timeout=remaining)
            except queue.Empty:
                return
            if request is None:
                self._stop.set()
                self._queue.task_done()
                return
            batch.append(request)

    def _pseudonym(self, value: str) -> str:
        return hmac.new(self._hash_key, value.encode("utf-8"), hashlib.sha256).hexdigest()

    def _write_sidecar(self, request, prediction, error):
        if not self.sidecar_path:
            return
        record = {
            "schema_version": 1,
            "request_hash": self._pseudonym(request["request_id"]),
            "text_hmac_sha256": self._pseudonym(request["text"]),
            "primary_count": len(request["primary"]),
            "shadow_count": None if prediction is None else len(prediction),
            "error": error,
            "latency_ms": round((time.monotonic() - request["submitted_mono"]) * 1000, 3),
        }
        try:
            path = Path(self.sidecar_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("OM shadow sidecar write isolated: {}", exc)

    def flush(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self):
        if self._stop.is_set():
            return
        self.flush()
        self._stop.set()
        if self._worker is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=2)
            if self._worker.is_alive():
                logger.warning("OM shadow worker did not stop; runtime remains owned and is not closed unsafely")
        else:
            self._close_runtime_isolated()
