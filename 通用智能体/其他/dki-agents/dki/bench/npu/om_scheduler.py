"""服务级 NPU 推理微批调度器（本地语义演示，非平台成绩）。

在 process-owned ACL Runtime 之上，补齐一层请求级调度语义：

- **绝对 deadline**：入队即冻结 `now + deadline_s`；排队等待计入端到端 deadline，
  超时请求在成批前被丢弃（记 `dropped_deadline`），不占用 NPU。
- **maxBatchSize / maxWaitMs**：攒够 batch 或等待窗口到期即成批，二者取先到。
- **stable reorder**：批内严格按提交序号排序，输出按请求回填（顺序无关正确）。
- **cancellation**：请求执行前可取消（记 `cancelled`）。
- **tail padding + mask**：当底层 runtime 需要定长 batch（如静态 OM bs=64）时，
  用哨兵补齐尾批并屏蔽 padding 输出，只回填真实请求。
- **kill switch**：一键停机，拒绝新请求、取消在队请求、优雅收尾。
- **telemetry**：计数（submitted/batched/completed/dropped_deadline/cancelled/failed/
  circuit_open）与每请求 queue_wait_ms / exec_ms，均**不含**任何原始文本。

诚实边界：这是**本地服务语义**，证明 deadline/reorder/cancel/tail/telemetry 行为，
**不等于**真实 DataMate `RayDataset.process` 的跨文档微批（`formal_microbatch=false`）。
底层 `batch_runtime` 需实现 `execute_batch(payloads) -> results`（一一对应、同序）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


class RequestState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEADLINE = "deadline"
    KILLED = "killed"


class SchedulerError(RuntimeError):
    pass


@dataclass
class _Request:
    seq: int
    payload: Any
    enqueue_mono: float
    deadline_mono: float
    event: threading.Event = field(default_factory=threading.Event)
    state: RequestState = RequestState.PENDING
    result: Any = None
    error: Optional[str] = None
    queue_wait_ms: float = 0.0
    exec_ms: float = 0.0
    _cancel: bool = False


class RequestHandle:
    """提交回执：可取消、可等待结果。不暴露其它请求的任何信息。"""

    def __init__(self, request: _Request):
        self._request = request

    @property
    def seq(self) -> int:
        return self._request.seq

    @property
    def state(self) -> RequestState:
        return self._request.state

    def cancel(self) -> bool:
        """执行前取消成功返回 True；已在执行/完成返回 False。"""
        if self._request.state == RequestState.PENDING:
            self._request._cancel = True
            return True
        return False

    def result(self, timeout_s: Optional[float] = None) -> Any:
        if not self._request.event.wait(timeout=timeout_s):
            raise TimeoutError("request result not ready")
        if self._request.state == RequestState.DONE:
            return self._request.result
        raise SchedulerError(f"request {self._request.seq} ended in {self._request.state.value}: {self._request.error}")


@dataclass
class SchedulerTelemetry:
    submitted: int = 0
    batched: int = 0
    batches: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    dropped_deadline: int = 0
    killed: int = 0
    circuit_open: bool = False
    queue_wait_ms_sum: float = 0.0
    exec_ms_sum: float = 0.0

    def snapshot(self) -> dict:
        completed = max(1, self.completed)
        return {
            "submitted": self.submitted,
            "batched": self.batched,
            "batches": self.batches,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "dropped_deadline": self.dropped_deadline,
            "killed": self.killed,
            "circuit_open": self.circuit_open,
            "avg_queue_wait_ms": round(self.queue_wait_ms_sum / completed, 3),
            "avg_exec_ms": round(self.exec_ms_sum / completed, 3),
            "avg_batch_fill": round(self.batched / max(1, self.batches), 3),
        }


class NpuInferenceScheduler:
    def __init__(
        self,
        batch_runtime: Any,
        *,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
        static_batch_size: Optional[int] = None,
        pad_payload: Any = "",
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        if static_batch_size is not None:
            if static_batch_size <= 0:
                raise ValueError("static_batch_size must be positive")
            if max_batch_size > static_batch_size:
                raise ValueError("max_batch_size cannot exceed static_batch_size")
        if not callable(getattr(batch_runtime, "execute_batch", None)):
            raise ValueError("batch_runtime must provide execute_batch(payloads) -> results")
        self._runtime = batch_runtime
        self._max_batch = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._static_batch = static_batch_size
        self._pad_payload = pad_payload
        self._clock = clock
        self._pending: List[_Request] = []
        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._seq = 0
        self._killed = False
        self._circuit_open = False
        self.telemetry = SchedulerTelemetry()
        self._worker = threading.Thread(target=self._loop, name="npu-scheduler", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ submit
    def submit(self, payload: Any, *, deadline_s: float = 30.0) -> RequestHandle:
        if deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        with self._lock:
            if self._killed:
                raise SchedulerError("scheduler is stopped")
            if self._circuit_open:
                raise SchedulerError("scheduler circuit is open")
            now = self._clock()
            self._seq += 1
            request = _Request(
                seq=self._seq, payload=payload,
                enqueue_mono=now, deadline_mono=now + deadline_s,
            )
            self._pending.append(request)
            self.telemetry.submitted += 1
            self._wakeup.notify()
        return RequestHandle(request)

    # ------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._pending and not self._killed:
                    self._wakeup.wait(timeout=0.05)
                if self._killed and not self._pending:
                    return
                batch = self._collect_batch_locked()
            if batch:
                self._run_batch(batch)

    def _collect_batch_locked(self) -> List[_Request]:
        # Wait until batch is full or the oldest live request's wait window
        # (or its deadline) expires. Runs under self._lock via condition wait.
        while True:
            self._drop_expired_and_cancelled_locked()
            if not self._pending:
                if self._killed:
                    return []
                return []
            oldest = self._pending[0]
            now = self._clock()
            live = [r for r in self._pending if r.state == RequestState.PENDING]
            if len(live) >= self._max_batch:
                break
            window_deadline = oldest.enqueue_mono + self._max_wait_s
            wait_until = min(window_deadline, oldest.deadline_mono)
            remaining = wait_until - now
            if remaining <= 0 or self._killed:
                break
            self._wakeup.wait(timeout=remaining)
        batch, keep = [], []
        for request in self._pending:
            if request.state != RequestState.PENDING:
                continue
            if len(batch) < self._max_batch:
                request.state = RequestState.RUNNING
                request.queue_wait_ms = (self._clock() - request.enqueue_mono) * 1000.0
                batch.append(request)
            else:
                keep.append(request)
        self._pending = keep
        return batch

    def _drop_expired_and_cancelled_locked(self) -> None:
        survivors = []
        now = self._clock()
        for request in self._pending:
            if request._cancel:
                self._finish_locked(request, RequestState.CANCELLED, error="cancelled")
                self.telemetry.cancelled += 1
            elif now >= request.deadline_mono:
                self._finish_locked(request, RequestState.DEADLINE, error="deadline exceeded in queue")
                self.telemetry.dropped_deadline += 1
            else:
                survivors.append(request)
        self._pending = survivors

    def _finish_locked(self, request: _Request, state: RequestState, *, result=None, error=None) -> None:
        request.state = state
        request.result = result
        request.error = error
        request.event.set()

    def _run_batch(self, batch: List[_Request]) -> None:
        batch.sort(key=lambda r: r.seq)  # stable order independent of arrival jitter
        payloads = [r.payload for r in batch]
        real_count = len(payloads)
        if self._static_batch is not None:
            payloads = payloads + [self._pad_payload] * (self._static_batch - real_count)
        started = self._clock()
        try:
            outputs = self._runtime.execute_batch(payloads)
        except BaseException as exc:
            error_type = type(exc).__name__  # never surface raw text/traceback
            with self._lock:
                for request in batch:
                    self._finish_locked(request, RequestState.FAILED, error=error_type)
                    self.telemetry.failed += 1
                self._circuit_open = True
                self.telemetry.circuit_open = True
            return
        exec_ms = (self._clock() - started) * 1000.0
        if len(outputs) < real_count:
            with self._lock:
                for request in batch:
                    self._finish_locked(request, RequestState.FAILED, error="short batch output")
                    self.telemetry.failed += 1
                self._circuit_open = True
                self.telemetry.circuit_open = True
            return
        with self._lock:
            self.telemetry.batches += 1
            self.telemetry.batched += real_count
            for offset, request in enumerate(batch):
                request.exec_ms = exec_ms
                self._finish_locked(request, RequestState.DONE, result=outputs[offset])
                self.telemetry.completed += 1
                self.telemetry.queue_wait_ms_sum += request.queue_wait_ms
                self.telemetry.exec_ms_sum += exec_ms

    # -------------------------------------------------------------- kill
    def kill(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            if self._killed:
                return
            self._killed = True
            for request in self._pending:
                self._finish_locked(request, RequestState.KILLED, error="scheduler killed")
                self.telemetry.killed += 1
            self._pending = []
            self._wakeup.notify_all()
        self._worker.join(timeout=timeout_s)
        close = getattr(self._runtime, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                pass
