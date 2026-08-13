"""Fail-closed helpers for a single-owner ACL OM runtime.

The concrete pyACL adapter is injected so CPU CI can exhaustively exercise
lifecycle behavior without importing ACL.  Production adapters must implement
``open()``, ``execute(payload)`` and idempotent ``close()``.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import pickle
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Mapping, Optional


class FailureKind(str, Enum):
    CONFIG = "config"
    COMPATIBILITY = "compatibility"
    QUEUE_FULL = "queue_full"
    TIMEOUT = "timeout"
    WORKER_CRASH = "worker_crash"
    ACL_FATAL = "acl_fatal"


class OmRuntimeError(RuntimeError):
    def __init__(self, kind: FailureKind, message: str):
        super().__init__(message)
        self.kind = kind


class ResourceStack:
    """Own resources and release each acquired item exactly once in reverse."""

    def __init__(self):
        self._closers: List[Callable[[], None]] = []
        self._closed = False

    def push(self, closer: Callable[[], None]) -> None:
        if self._closed:
            raise RuntimeError("resource stack is already closed")
        self._closers.append(closer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures = []
        while self._closers:
            closer = self._closers.pop()
            try:
                closer()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise OmRuntimeError(FailureKind.ACL_FATAL, f"{len(failures)} resource close operation(s) failed")


class OwnerThreadGuard:
    def __init__(self):
        self._owner = None
        self._lock = threading.Lock()

    def check(self) -> None:
        current = threading.get_ident()
        with self._lock:
            if self._owner is None:
                # Ownership belongs to the thread that first opens/uses ACL,
                # not necessarily the thread that assembled the object graph.
                self._owner = current
            elif current != self._owner:
                raise OmRuntimeError(FailureKind.COMPATIBILITY, "ACL runtime called from a non-owner thread")


@dataclass(frozen=True)
class RuntimeHealth:
    state: str
    failures: int
    restarts: int
    circuit_open: bool


class ProcessOmRuntime:
    """Spawn-owned adapter runtime with enforceable deadlines.

    ``adapterFactory`` is a dotted import path so the spawn child never needs a
    pickled ACL object.  The imported callable receives ``adapterConfig`` and
    returns an object implementing open/execute/close.  CPU processes can import
    this module without importing pyACL; only the configured child factory does.
    """

    def __init__(
        self,
        adapterFactory: str,
        adapterConfig: Optional[Mapping[str, Any]] = None,
        *,
        queueDepth: int = 4,
        maxRestarts: int = 1,
        startTimeoutS: float = 30.0,
        mpStartMethod: str = "spawn",
        maxBatchItems: int = 256,
        maxPayloadBytes: int = 16 * 1024 * 1024,
    ):
        if not adapterFactory or "." not in adapterFactory:
            raise ValueError("adapterFactory must be a dotted module attribute")
        if (queueDepth <= 0 or maxRestarts not in {0, 1} or startTimeoutS <= 0
                or maxBatchItems <= 0 or maxPayloadBytes <= 0):
            raise ValueError("process runtime requires positive limits and maxRestarts of zero or one")
        self._factory_path = adapterFactory
        self._config = dict(adapterConfig or {})
        self._queue_depth = queueDepth
        self._max_restarts = maxRestarts
        self._start_timeout_s = startTimeoutS
        self._max_batch_items = int(maxBatchItems)
        self._max_payload_bytes = int(maxPayloadBytes)
        self._context = mp.get_context(mpStartMethod)
        self._commands = None
        self._results = None
        self._process = None
        self._sequence = 0
        self._failures = 0
        self._restarts = 0
        self._circuit_open = False
        self._closed = False
        self._lock = threading.Lock()

    @property
    def health(self) -> RuntimeHealth:
        alive = self._process is not None and self._process.is_alive()
        state = "closed" if self._closed else "open" if self._circuit_open else "ready" if alive else "new"
        return RuntimeHealth(state, self._failures, self._restarts, self._circuit_open)

    def _start(self, timeout_s: float) -> None:
        if self._closed:
            raise OmRuntimeError(FailureKind.CONFIG, "runtime is closed")
        # Fail fast in the parent too: a bad/absent manifest must not even spawn
        # a child that would load an unverified OM. No-op unless manifestPath set.
        try:
            from dki.bench.npu.om_manifest import enforce_preload_manifest

            enforce_preload_manifest(self._config)
        except OmRuntimeError:
            raise
        except Exception as exc:
            raise OmRuntimeError(FailureKind.COMPATIBILITY, f"OM manifest gate failed: {exc}") from exc
        if self._process is not None or self._commands is not None or self._results is not None:
            self._terminate()
        commands = self._context.Queue(maxsize=self._queue_depth)
        results = self._context.Queue(maxsize=self._queue_depth)
        process = self._context.Process(
            target=_process_worker,
            args=(self._factory_path, self._config, commands, results),
            daemon=True,
        )
        self._commands, self._results, self._process = commands, results, process
        try:
            process.start()
            wait_s = min(self._start_timeout_s, timeout_s)
            message = results.get(timeout=wait_s)
        except queue.Empty as exc:
            kind = FailureKind.WORKER_CRASH if process.exitcode is not None else FailureKind.TIMEOUT
            self._terminate()
            raise OmRuntimeError(kind, "ACL worker failed during initialization") from exc
        except BaseException:
            self._terminate()
            raise
        if message.get("kind") != "ready":
            self._terminate()
            raise OmRuntimeError(FailureKind.ACL_FATAL, "ACL worker initialization failed")

    def execute(self, payload: Any, *, timeout_s: float = 30.0) -> Any:
        return self._execute_request("execute", payload, timeout_s=timeout_s)

    def execute_batch(self, payloads: List[Any], *, timeout_s: float = 30.0) -> List[Any]:
        """Send one ordered microbatch through a single IPC command."""
        if isinstance(payloads, (str, bytes, Mapping)) or payloads is None:
            raise TypeError("payloads must be a sequence")
        batch = list(payloads)
        if len(batch) > self._max_batch_items:
            raise OmRuntimeError(
                FailureKind.CONFIG,
                f"batch has {len(batch)} items, limit is {self._max_batch_items}",
            )

        def validate(result):
            if not isinstance(result, list) or len(result) != len(batch):
                raise OmRuntimeError(
                    FailureKind.COMPATIBILITY,
                    "ACL batch response cardinality does not match request",
                )
            return result

        return self._execute_request(
            "execute_batch", batch, timeout_s=timeout_s, response_validator=validate,
        )

    def _execute_request(
        self, command_kind: str, payload: Any, *, timeout_s: float,
        response_validator: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        if timeout_s <= 0:
            raise OmRuntimeError(FailureKind.TIMEOUT, "request deadline already expired")
        deadline = time.monotonic() + timeout_s
        if not self._lock.acquire(timeout=timeout_s):
            raise OmRuntimeError(FailureKind.TIMEOUT, "ACL request expired in admission queue")
        try:
            if self._circuit_open:
                raise OmRuntimeError(FailureKind.ACL_FATAL, "ACL circuit is open")
            self._sequence += 1
            request_id = self._sequence
            command = {"kind": command_kind, "id": request_id, "payload": payload}
            try:
                encoded_command = pickle.dumps(
                    command, protocol=pickle.HIGHEST_PROTOCOL,
                )
            except BaseException as exc:
                raise OmRuntimeError(
                    FailureKind.CONFIG, "ACL request payload is not serializable",
                ) from exc
            if len(encoded_command) > self._max_payload_bytes:
                raise OmRuntimeError(
                    FailureKind.CONFIG,
                    f"ACL request payload exceeds {self._max_payload_bytes} bytes",
                )
            if time.monotonic() >= deadline:
                raise OmRuntimeError(
                    FailureKind.TIMEOUT, "ACL request expired during serialization",
                )
            for attempt in range(self._max_restarts + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._circuit_open = True
                    raise OmRuntimeError(FailureKind.TIMEOUT, "ACL request deadline expired")
                if self._process is None or not self._process.is_alive():
                    try:
                        self._start(remaining)
                    except OmRuntimeError:
                        self._failures += 1
                        if attempt >= self._max_restarts:
                            self._circuit_open = True
                            raise
                        self._restarts += 1
                        continue
                try:
                    self._commands.put_nowait(encoded_command)
                except queue.Full as exc:
                    raise OmRuntimeError(FailureKind.QUEUE_FULL, "ACL admission queue is full") from exc
                remaining = deadline - time.monotonic()
                response = None
                while remaining > 0:
                    try:
                        response = self._results.get(timeout=min(0.05, remaining))
                        break
                    except queue.Empty:
                        if self._process is None or self._process.exitcode is not None:
                            break
                        remaining = deadline - time.monotonic()
                if response is None:
                    self._failures += 1
                    kind = (
                        FailureKind.WORKER_CRASH
                        if self._process is None or self._process.exitcode is not None
                        else FailureKind.TIMEOUT
                    )
                    self._terminate()
                    if attempt >= self._max_restarts:
                        self._circuit_open = True
                        raise OmRuntimeError(kind, "ACL worker did not return a result")
                    self._restarts += 1
                    continue
                if response.get("id") != request_id:
                    self._failures += 1
                    self._terminate()
                    self._circuit_open = True
                    raise OmRuntimeError(FailureKind.COMPATIBILITY, "ACL worker returned an out-of-order response")
                if response.get("kind") == "result":
                    value = response.get("value")
                    if response_validator is None:
                        return value
                    try:
                        return response_validator(value)
                    except OmRuntimeError:
                        self._failures += 1
                        self._terminate()
                        self._circuit_open = True
                        raise
                self._failures += 1
                process = self._process
                if process is not None:
                    process.join(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
                self._terminate()
                if attempt >= self._max_restarts:
                    self._circuit_open = True
                    raise OmRuntimeError(FailureKind.ACL_FATAL, "ACL worker execute failed")
                self._restarts += 1
            raise OmRuntimeError(FailureKind.ACL_FATAL, "ACL runtime exhausted restart budget")
        finally:
            self._lock.release()

    def _terminate(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            started = getattr(process, "_popen", None) is not None
            if started:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            try:
                process.close()
            except (ValueError, AssertionError):
                pass
        for channel_name in ("_commands", "_results"):
            channel = getattr(self, channel_name)
            setattr(self, channel_name, None)
            if channel is not None:
                try:
                    channel.cancel_join_thread()
                except (ValueError, AssertionError):
                    pass
                channel.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is not None and process.is_alive():
                try:
                    self._commands.put(
                        pickle.dumps(
                            {"kind": "close"}, protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                        timeout=0.2,
                    )
                    process.join(timeout=5)
                except BaseException:
                    pass
            self._terminate()


def _load_dotted(path: str):
    module_name, _, attribute = path.rpartition(".")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"adapter factory is not callable: {path}")
    return factory


def _process_worker(factory_path: str, config: Mapping[str, Any], commands, results) -> None:
    adapter = None
    try:
        # Hard promotion gate: verify artifact hashes + SoC/CANN BEFORE the
        # untrusted OM is ever loaded. Opt-in by manifestPath in config, so
        # fake-adapter tests and non-OM adapters are unaffected; when present,
        # any mismatch raises here and is reported as init_error (fail closed).
        from dki.bench.npu.om_manifest import enforce_preload_manifest

        enforce_preload_manifest(config)
        adapter = _load_dotted(factory_path)(**dict(config))
        adapter.open()
        results.put({"kind": "ready"})
        while True:
            encoded_command = commands.get()
            if not isinstance(encoded_command, bytes):
                raise TypeError("ACL worker command must be serialized bytes")
            command = pickle.loads(encoded_command)
            if command.get("kind") == "close":
                return
            request_id = command.get("id")
            try:
                if command.get("kind") == "execute_batch":
                    execute_batch = getattr(adapter, "execute_batch", None)
                    if not callable(execute_batch):
                        raise TypeError("adapter does not implement execute_batch")
                    value = execute_batch(command.get("payload"))
                elif command.get("kind") == "execute":
                    value = adapter.execute(command.get("payload"))
                else:
                    raise ValueError("unknown ACL worker command")
                response = {"kind": "result", "id": request_id, "value": value}
                try:
                    pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                except BaseException:
                    raise TypeError("adapter returned a non-serializable result")
                results.put(response)
            except BaseException as exc:
                # Never transfer exception text or traceback: adapters may include
                # raw medical text in an error message.
                error_type = type(exc).__name__
                try:
                    adapter.close()
                finally:
                    adapter = None
                results.put({"kind": "error", "id": request_id, "error_type": error_type})
                return
    except BaseException as exc:
        results.put({"kind": "init_error", "error_type": type(exc).__name__})
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except BaseException:
                pass


def create_process_runtime(**config) -> ProcessOmRuntime:
    """Dotted factory suitable for ``shadowRuntimeFactory`` configuration."""
    return ProcessOmRuntime(**config)


class OmRuntime:
    """Bounded single-owner runtime with one whole-adapter restart.

    This class deliberately serializes adapter execution. The outer microbatch
    scheduler may be concurrent, but pyACL ownership stays here.
    """

    def __init__(
        self,
        adapter_factory: Callable[[], Any],
        *,
        queue_depth: int = 4,
        max_restarts: int = 1,
        preload_config: Optional[Mapping[str, Any]] = None,
    ):
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        self._factory = adapter_factory
        self._preload_config = dict(preload_config) if preload_config else None
        self._admission = queue.Queue(maxsize=queue_depth)
        self._max_restarts = max_restarts
        self._adapter = None
        self._guard = OwnerThreadGuard()
        self._failures = 0
        self._restarts = 0
        self._circuit_open = False
        self._closed = False

    @property
    def health(self) -> RuntimeHealth:
        state = "closed" if self._closed else "open" if self._circuit_open else "ready" if self._adapter else "new"
        return RuntimeHealth(state, self._failures, self._restarts, self._circuit_open)

    def open(self) -> None:
        self._guard.check()
        if self._closed:
            raise OmRuntimeError(FailureKind.CONFIG, "runtime is closed")
        if self._adapter is not None:
            return
        if self._preload_config is not None:
            from dki.bench.npu.om_manifest import enforce_preload_manifest

            try:
                enforce_preload_manifest(self._preload_config)
            except OmRuntimeError:
                raise
            except Exception as exc:
                raise OmRuntimeError(FailureKind.COMPATIBILITY, f"OM manifest gate failed: {exc}") from exc
        adapter = None
        try:
            adapter = self._factory()
            adapter.open()
            self._adapter = adapter
        except BaseException as exc:
            if adapter is not None:
                try:
                    adapter.close()
                except BaseException:
                    pass
            raise OmRuntimeError(FailureKind.ACL_FATAL, f"ACL adapter open failed: {exc}") from exc

    def execute(self, payload: Any, *, timeout_s: float = 30.0) -> Any:
        return self._execute_command("execute", payload, timeout_s=timeout_s)

    def execute_batch(self, payloads: List[Any], *, timeout_s: float = 30.0) -> List[Any]:
        if isinstance(payloads, (str, bytes, Mapping)) or payloads is None:
            raise TypeError("payloads must be a sequence")
        batch = list(payloads)
        result = self._execute_command("execute_batch", batch, timeout_s=timeout_s)
        if not isinstance(result, list) or len(result) != len(batch):
            self._circuit_open = True
            raise OmRuntimeError(
                FailureKind.COMPATIBILITY,
                "ACL batch response cardinality does not match request",
            )
        return result

    def _execute_command(self, command: str, payload: Any, *, timeout_s: float) -> Any:
        self._guard.check()
        if self._circuit_open:
            raise OmRuntimeError(FailureKind.ACL_FATAL, "ACL circuit is open")
        if timeout_s <= 0:
            raise OmRuntimeError(FailureKind.TIMEOUT, "request deadline already expired")
        try:
            self._admission.put_nowait(object())
        except queue.Full as exc:
            raise OmRuntimeError(FailureKind.QUEUE_FULL, "ACL admission queue is full") from exc
        try:
            self.open()
            started = time.monotonic()
            try:
                operation = getattr(self._adapter, command, None)
                if not callable(operation):
                    raise TypeError(f"ACL adapter does not implement {command}")
                result = operation(payload)
            except BaseException as first_exc:
                self._failures += 1
                if self._restarts >= self._max_restarts:
                    self._circuit_open = True
                    raise OmRuntimeError(FailureKind.ACL_FATAL, f"ACL {command} failed: {first_exc}") from first_exc
                try:
                    self._replace_adapter()
                    operation = getattr(self._adapter, command, None)
                    if not callable(operation):
                        raise TypeError(f"ACL adapter does not implement {command}")
                    result = operation(payload)
                except BaseException as retry_exc:
                    self._failures += 1
                    self._circuit_open = True
                    raise OmRuntimeError(
                        FailureKind.ACL_FATAL,
                        f"ACL {command} failed before and after adapter restart: {retry_exc}",
                    ) from retry_exc
            if time.monotonic() - started > timeout_s:
                raise OmRuntimeError(FailureKind.TIMEOUT, "ACL execution exceeded deadline")
            return result
        finally:
            self._admission.get_nowait()
            self._admission.task_done()

    def _replace_adapter(self) -> None:
        previous, self._adapter = self._adapter, None
        try:
            if previous is not None:
                previous.close()
        finally:
            self._restarts += 1
        self.open()

    def close(self) -> None:
        self._guard.check()
        if self._closed:
            return
        self._closed = True
        adapter, self._adapter = self._adapter, None
        if adapter is not None:
            adapter.close()
