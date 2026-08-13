"""LocalPipelineRunner —— 复刻 DataMate RayDataset.process 线性算子链语义的本地执行器。

与 DataMate 平台执行器（datamate/core/dataset.py + wrappers/datamate_executor.py）的对应关系：

| DataMate 平台                          | 本运行器                                   |
|----------------------------------------|--------------------------------------------|
| process.yaml 的 process 列表           | 相同结构（[{OpName: {参数}}]）             |
| OPERATORS 注册表 + dotted path 懒加载  | 相同（直接复用 datamate 的 Registry）      |
| is_first_op / is_last_op / instance_id | 相同注入规则                               |
| Ray map / flat_map / filter            | 顺序 map / flat_map / filter（无 Ray 依赖）|
| Mapper.__call__ 失败样本隔离           | 相同语义（execute_result=False 继续流转）  |
| PostgreSQL 任务状态                    | TaskReport JSON（outputs/tasks/<id>.json） |

额外提供平台没有的本地能力：逐算子吞吐/延迟指标、可配置重试、任务注册表。
"""

import copy
import importlib
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dki import dmcompat
from dki.dmcompat import OPERATORS, Fields, Filter, Mapper, Slicer
from dki.paths import TASK_DIR
from dki.pipeline.recovery import RecoveryAction, RecoveryDecision, decide_recovery
from dki.pipeline.state import OpMetrics, TaskReport, TaskStatus

logger = logging.getLogger(__name__)

_OP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODULE_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def resolve_operator(op_name: str):
    """按 DataMate RayDataset.load_ops_module 的规则解析算子类。"""
    if not op_name or not _OP_NAME_RE.match(op_name):
        raise ValueError(f"非法算子名: {op_name!r}")
    registry_content = OPERATORS.modules.get(op_name)
    if registry_content is None:
        raise KeyError(
            f"算子 {op_name} 未注册。已注册算子: {sorted(OPERATORS.modules)[:40]} ..."
        )
    if isinstance(registry_content, str):
        if not _MODULE_PATH_RE.match(registry_content):
            raise ValueError(f"算子 {op_name} 的模块路径非法: {registry_content}")
        submodule = importlib.import_module(registry_content)
        cls = getattr(submodule, op_name, None)
        if cls is None:
            raise ImportError(f"模块 {registry_content} 中不存在类 {op_name}")
        return cls
    return registry_content


def _op_kind(cls) -> str:
    # LLM 是 Mapper 子类，先判 LLM
    if issubclass(cls, dmcompat.LLM):
        return "LLM"
    if issubclass(cls, Slicer):
        return "Slicer"
    if issubclass(cls, Filter):
        return "Filter"
    if issubclass(cls, Mapper):
        return "Mapper"
    return "BaseOp"


class TaskRegistry:
    """任务注册表：内存 + JSON 落盘，供智能体的 task_status 工具查询。"""

    def __init__(self, task_dir: Path = TASK_DIR):
        self.task_dir = Path(task_dir)
        self._tasks: Dict[str, TaskReport] = {}
        self._results: Dict[str, "PipelineResult"] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def register(self, report: TaskReport):
        with self._lock:
            self._tasks[report.task_id] = report
            report.save(self.task_dir)

    def attach_worker(self, task_id: str, worker: threading.Thread) -> None:
        with self._lock:
            self._workers[task_id] = worker

    def set_result(self, task_id: str, result: "PipelineResult") -> None:
        with self._lock:
            self._results[task_id] = result

    def get_result(self, task_id: str) -> Optional["PipelineResult"]:
        with self._lock:
            return self._results.get(task_id)

    def wait(self, task_id: str, timeout_s: float) -> None:
        from dki.pipeline.state import _validate_task_id

        task_id = _validate_task_id(task_id)
        with self._lock:
            worker = self._workers.get(task_id)
        if worker is not None:
            worker.join(timeout=max(0.0, timeout_s))

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(task_id)
        return bool(worker and worker.is_alive())

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        # Validate before consulting memory too, so filesystem-backed and live
        # task lookups expose one identical public contract.
        from dki.pipeline.state import _validate_task_id

        task_id = _validate_task_id(task_id)
        with self._lock:
            if task_id in self._tasks:
                return self._tasks[task_id].to_dict()
        return TaskReport.load(self.task_dir, task_id)

    def get_report(self, task_id: str) -> Optional[TaskReport]:
        with self._lock:
            return self._tasks.get(task_id)

    def link_child(self, parent_task_id: str, child_task_id: str) -> None:
        from dki.pipeline.state import _validate_task_id

        parent_task_id = _validate_task_id(parent_task_id)
        child_task_id = _validate_task_id(child_task_id)
        with self._lock:
            parent = self._tasks.get(parent_task_id)
            if parent is not None:
                if child_task_id not in parent.child_task_ids:
                    parent.child_task_ids.append(child_task_id)
                    parent.save(self.task_dir)
                return

        path = self.task_dir / f"{parent_task_id}.json"
        if not path.is_file():
            raise ValueError(f"父任务不存在: {parent_task_id}")
        import json
        import uuid

        payload = json.loads(path.read_text(encoding="utf-8"))
        children = payload.setdefault("child_task_ids", [])
        if child_task_id not in children:
            children.append(child_task_id)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(path)

    def list_ids(self) -> List[str]:
        with self._lock:
            ids = set(self._tasks)
        if self.task_dir.exists():
            ids.update(p.stem for p in self.task_dir.glob("*.json"))
        return sorted(ids)


TASKS = TaskRegistry()


class LocalPipelineRunner:
    """顺序执行一条算子链，产出 TaskReport 与处理后的样本列表。"""

    def __init__(
        self,
        load_builtin_ops: bool = True,
        register_dki_ops: bool = True,
        registry: TaskRegistry = TASKS,
    ):
        if load_builtin_ops:
            dmcompat.load_builtin_ops()
        if register_dki_ops:
            dmcompat.register_dki_ops()
        self.registry = registry

    # ------------------------------------------------------------------
    def run(
        self,
        process: List[Dict[str, Any]],
        samples: List[Dict[str, Any]],
        name: str = "",
        export_path: str = "",
        preloaded: Optional[bool] = None,
        retry: int = 0,
        keep_failed: bool = True,
        _report: Optional[TaskReport] = None,
    ) -> "PipelineResult":
        """执行流水线。

        参数
        ----
        process: [{OpName: {参数}}, ...]，与 DataMate process.yaml 的 process 字段同构。
        samples: 输入样本（make_sample 构造）。
        preloaded: 样本文本是否已预加载（None=自动判断：全部 filePath 为空则视为已预加载）。
        retry: 单样本算子失败后的重试次数（对 LLM 类瞬时错误有效）。
        keep_failed: 是否在输出中保留失败样本（False 则丢弃）。
        """
        report = _report or TaskReport.new(name=name, pipeline=copy.deepcopy(process))
        if _report is None:
            report.n_input = len(samples)
            report.export_path = export_path
        report.mark_running()
        self.registry.register(report)

        if preloaded is None:
            preloaded = all(not s.get("filePath") for s in samples)

        current = [copy.deepcopy(s) for s in samples]
        for s in current:
            s.setdefault(Fields.instance_id, report.task_id)
            s[Fields.instance_id] = s.get(Fields.instance_id) or report.task_id
            if export_path:
                s["export_path"] = export_path

        try:
            op_instances = self._build_ops(process, report.task_id, preloaded)
            report.op_metrics = [m for _, m in op_instances]
        except Exception as e:
            report.error = f"算子解析失败: {e}"
            report.mark_finished()
            report.status = TaskStatus.FAILED
            self.registry.register(report)
            logger.exception("算子解析失败")
            return PipelineResult(report, [])

        runtime_kwargs = {"ext_params": {}, "failed_reason": {}, "target_type": None}

        try:
            for op, metrics in op_instances:
                t0 = time.perf_counter()
                metrics.n_in = len(current)
                nxt: List[Dict[str, Any]] = []
                active_positions = [
                    index for index, sample in enumerate(current)
                    if sample.get(Fields.result) is not False
                ]
                execute_batch = getattr(op, "execute_batch", None)
                if callable(execute_batch) and isinstance(op, Mapper) and not isinstance(op, (Filter, Slicer)):
                    active = [current[index] for index in active_positions]
                    batch_outputs = self._apply_batch(op, metrics, active, runtime_kwargs, retry, report)
                    by_position = dict(zip(active_positions, batch_outputs))
                    for index, sample in enumerate(current):
                        out = sample if sample.get(Fields.result) is False else by_position[index]
                        if out is not None:
                            nxt.append(out)
                        else:
                            metrics.n_filtered += 1
                else:
                    for sample in current:
                        if sample.get(Fields.result) is False:
                            # 前序已失败：与平台一致，跳过处理但保留流转
                            nxt.append(sample)
                            continue
                        out = self._apply_op(op, metrics, sample, runtime_kwargs, retry, report)
                        if out is not None:
                            nxt.append(out)
                        else:
                            metrics.n_filtered += 1
                flush = getattr(op, "flush", None)
                if callable(flush):
                    try:
                        if flush() is False:
                            logger.warning("算子 %s flush 超时；主任务输出不受 sidecar 影响", metrics.op_name)
                    except Exception:
                        # Optional lifecycle hooks (notably shadow sidecars) are
                        # isolated from the authoritative sample projection.
                        logger.exception("算子 flush 失败（主输出保留）: %s", metrics.op_name)
                metrics.wall_s = time.perf_counter() - t0
                metrics.n_out = len(nxt)
                current = nxt
                report.save(self.registry.task_dir)
        except Exception as exc:
            report.error = f"流水线执行失败: {exc}"
            logger.exception("流水线执行失败")
        finally:
            for op, _ in reversed(op_instances):
                close = getattr(op, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logger.exception("算子 close 失败: %s", type(op).__name__)

        report.n_failed = sum(1 for s in current if s.get(Fields.result) is False)
        ok_samples = [s for s in current if s.get(Fields.result) is not False]
        report.n_output = len(ok_samples)
        report.n_filtered = sum(m.n_filtered for m in report.op_metrics)
        report.mark_finished()
        self.registry.register(report)

        result_samples = current if keep_failed else ok_samples
        return PipelineResult(report, result_samples)

    def submit(
        self,
        process: List[Dict[str, Any]],
        samples: List[Dict[str, Any]],
        name: str = "",
        export_path: str = "",
        preloaded: Optional[bool] = None,
        retry: int = 0,
        keep_failed: bool = True,
        parent_task_id: str = "",
    ) -> TaskReport:
        """Submit a non-blocking task and return its durable PENDING report."""
        report = TaskReport.new(name=name, pipeline=copy.deepcopy(process))
        report.n_input = len(samples)
        report.export_path = export_path
        report.parent_task_id = parent_task_id
        if parent_task_id and self.registry.get(parent_task_id) is None:
            raise ValueError(f"父任务不存在: {parent_task_id}")
        self.registry.register(report)
        if parent_task_id:
            self.registry.link_child(parent_task_id, report.task_id)

        def _worker() -> None:
            result = self.run(
                process, samples, name=name, export_path=export_path,
                preloaded=preloaded, retry=retry, keep_failed=keep_failed,
                _report=report,
            )
            self.registry.set_result(report.task_id, result)

        worker = threading.Thread(target=_worker, name=f"dki-task-{report.task_id}", daemon=True)
        self.registry.attach_worker(report.task_id, worker)
        worker.start()
        return report

    # ------------------------------------------------------------------
    def _build_ops(self, process: List[Dict[str, Any]], instance_id: str, preloaded: bool):
        instances = []
        n = len(process)
        try:
            for index, item in enumerate(process):
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError(f"process 第 {index} 项必须是 {{算子名: 参数}} 结构: {item!r}")
                op_name, init_kwargs = list(item.items())[0]
                init_kwargs = dict(init_kwargs) if init_kwargs else {}
                init_kwargs["op_name"] = op_name
                if index == 0 and not preloaded:
                    init_kwargs["is_first_op"] = True
                if index == n - 1:
                    init_kwargs["is_last_op"] = True
                init_kwargs["instance_id"] = instance_id
                cls = resolve_operator(op_name)
                op = cls(**init_kwargs)
                metrics = OpMetrics(op_name=op_name, op_kind=_op_kind(cls))
                instances.append((op, metrics))
            return instances
        except BaseException:
            # A later constructor may fail after earlier operators acquired
            # model/device/file resources. They are still owned here, so unwind.
            for op, _ in reversed(instances):
                close = getattr(op, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException:
                        logger.exception("部分构造回滚时算子 close 失败: %s", type(op).__name__)
            raise

    def _apply_batch(
        self,
        op,
        metrics: OpMetrics,
        samples: List[Dict[str, Any]],
        runtime_kwargs: Dict[str, Any],
        retry: int,
        report: TaskReport,
        allow_batch_reduction: bool = True,
    ) -> List[Optional[Dict[str, Any]]]:
        """Apply an opt-in mapper batch while preserving per-sample isolation.

        The contract is intentionally local: DataMate platform microbatch support
        must be established independently before claiming formal platform batching.
        A batch implementation returns one output/exception per input in the same
        order.  Whole-batch exceptions fall back to ordinary per-sample execution.
        """
        if not samples:
            return []
        for sample in samples:
            op.fill_sample_params(sample, **copy.deepcopy(runtime_kwargs))
        attempts = max(max(0, retry) + 1, 3)
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            # Batch implementations may partially mutate inputs before raising;
            # always isolate them so the per-sample fallback starts clean.
            work = copy.deepcopy(samples)
            t0 = time.perf_counter()
            try:
                outputs = list(op.execute_batch(work))
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if len(outputs) != len(samples):
                    raise ValueError(
                        f"execute_batch returned {len(outputs)} outputs for {len(samples)} inputs"
                    )
                if any(
                    output is not None
                    and not isinstance(output, (dict, BaseException))
                    for output in outputs
                ):
                    raise TypeError("execute_batch outputs must be mappings, None, or exceptions")
                metrics.per_sample_ms.extend([elapsed_ms] * len(samples))
                normalized: List[Optional[Dict[str, Any]]] = []
                for sample, output in zip(samples, outputs):
                    if isinstance(output, BaseException):
                        try:
                            raise output
                        except BaseException as item_exc:
                            # DataMate derives failure metadata from the active
                            # traceback, so materialize returned exceptions here.
                            op.create_failure_sample(sample, metrics.op_name, item_exc)
                        metrics.n_failed += 1
                        metrics.errors.append({
                            "fileId": str(sample.get("fileId")),
                            "fileName": str(sample.get("fileName")),
                            "reason": str(output)[:500],
                        })
                        normalized.append(sample)
                    elif output is None or isinstance(output, dict):
                        if isinstance(output, dict) and output.get(Fields.result) is False:
                            metrics.n_failed += 1
                            reason = output.get("failed_reason", {})
                            metrics.errors.append({
                                "fileId": str(output.get("fileId")),
                                "fileName": str(output.get("fileName")),
                                "reason": str(reason.get("reason", reason))[:500],
                            })
                        normalized.append(output)
                return normalized
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                metrics.per_sample_ms.extend([elapsed_ms] * len(samples))
                last_exc = exc
                decision = decide_recovery(str(exc), attempt=attempt, idempotent=True)
                if (
                    decision.action == RecoveryAction.REDUCE_BATCH_ONCE
                    and allow_batch_reduction
                    and len(samples) > 1
                ):
                    midpoint = max(1, len(samples) // 2)
                    self._record_recovery(
                        report,
                        decision,
                        op_name=metrics.op_name,
                        attempt=attempt,
                        scope="batch",
                        applied=True,
                        detail=f"batch {len(samples)} -> {midpoint}+{len(samples) - midpoint}",
                    )
                    metrics.n_retried += len(samples)
                    left = self._apply_batch(
                        op, metrics, samples[:midpoint], runtime_kwargs, retry=0,
                        report=report, allow_batch_reduction=False,
                    )
                    right = self._apply_batch(
                        op, metrics, samples[midpoint:], runtime_kwargs, retry=0,
                        report=report, allow_batch_reduction=False,
                    )
                    return left + right
                can_retry = (
                    decision.action == RecoveryAction.RETRY_BACKOFF
                    and attempt < decision.max_attempts
                    and attempt < attempts - 1
                )
                self._record_recovery(
                    report,
                    decision,
                    op_name=metrics.op_name,
                    attempt=attempt,
                    scope="batch",
                    applied=can_retry,
                    detail="bounded backoff" if can_retry else "quarantined or delegated",
                )
                if can_retry:
                    metrics.n_retried += len(samples)
                    time.sleep(0.05 * (attempt + 1))
                    continue
                break
        logger.warning(
            "算子 %s 批处理失败，回退逐样本隔离: %s", metrics.op_name, last_exc
        )
        return [
            self._apply_op(op, metrics, sample, runtime_kwargs, retry=0, report=report)
            for sample in samples
        ]

    # ------------------------------------------------------------------
    def _apply_op(
        self,
        op,
        metrics: OpMetrics,
        sample: Dict[str, Any],
        runtime_kwargs: Dict[str, Any],
        retry: int,
        report: TaskReport,
    ) -> Optional[Dict[str, Any]]:
        """对单样本应用算子，复刻 Mapper/Filter/Slicer.__call__ 语义（无 DB 依赖）。

        返回 None 表示样本被 Filter 过滤。
        """
        op.fill_sample_params(sample, **copy.deepcopy(runtime_kwargs))
        attempts = max(max(0, retry) + 1, 3)
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            # 开启重试时在副本上执行，避免失败中途污染原样本
            work = copy.deepcopy(sample) if attempts > 1 else sample
            t0 = time.perf_counter()
            try:
                if isinstance(op, Slicer):
                    chunks = op.execute(work)
                    op.load_sample_to_sample(work, chunks)
                    out: Optional[Dict[str, Any]] = work
                elif isinstance(op, Filter):
                    s2 = op.execute(work)
                    dropped = s2[op.text_key] == "" and s2[op.data_key] == b""
                    out = None if dropped else s2
                else:  # Mapper / LLM
                    out = op.execute(work)
                metrics.per_sample_ms.append((time.perf_counter() - t0) * 1000)
                return out
            except Exception as e:
                metrics.per_sample_ms.append((time.perf_counter() - t0) * 1000)
                last_exc = e
                decision = decide_recovery(str(e), attempt=attempt, idempotent=True)
                can_retry = (
                    decision.action == RecoveryAction.RETRY_BACKOFF
                    and attempt < decision.max_attempts
                    and attempt < attempts - 1
                )
                self._record_recovery(
                    report,
                    decision,
                    op_name=metrics.op_name,
                    attempt=attempt,
                    scope="sample",
                    applied=can_retry,
                    detail=(
                        "bounded backoff"
                        if can_retry
                        else "single-sample failures cannot reduce a batch or restart this process"
                    ),
                )
                if can_retry:
                    metrics.n_retried += 1
                    logger.warning(
                        "算子 %s 处理 %s 失败（第 %d 次），重试: %s",
                        metrics.op_name, sample.get("fileId"), attempt + 1, e,
                    )
                    time.sleep(0.05 * (attempt + 1))
                    continue
                break
        # 最终失败：失败样本隔离，不阻断流水线（与平台一致）
        op.create_failure_sample(sample, metrics.op_name, last_exc)
        metrics.n_failed += 1
        metrics.errors.append(
            {
                "fileId": str(sample.get("fileId")),
                "fileName": str(sample.get("fileName")),
                "reason": str(last_exc)[:500],
            }
        )
        logger.error("算子 %s 处理 %s 失败: %s", metrics.op_name, sample.get("fileId"), last_exc)
        return sample

    def _record_recovery(
        self,
        report: TaskReport,
        decision: RecoveryDecision,
        *,
        op_name: str,
        attempt: int,
        scope: str,
        applied: bool,
        detail: str,
    ) -> None:
        """Persist both the selected policy and whether this runner applied it."""
        report.failure_category = decision.category.value
        report.recovery_action = decision.action.value
        report.recovery_events.append({
            "timestamp": time.time(),
            "op_name": op_name,
            "attempt": attempt + 1,
            "scope": scope,
            "category": decision.category.value,
            "action": decision.action.value,
            "max_attempts": decision.max_attempts,
            "applied": bool(applied),
            "detail": detail,
        })
        report.save(self.registry.task_dir)

    # 汇总每个算子的 metrics 到 report（_build_ops 中创建的 metrics 挂到 report）
    # 注：_build_ops 返回 (op, metrics) 对，run() 中逐个执行时已引用同一对象。


class PipelineResult:
    def __init__(self, report: TaskReport, samples: List[Dict[str, Any]]):
        self.report = report
        self.samples = samples

    @property
    def ok_samples(self) -> List[Dict[str, Any]]:
        return [s for s in self.samples if s.get(Fields.result) is not False]
