"""流水线任务状态与执行报告（状态机对齐 DataMate CleaningTaskStatus）。"""

import json
import re
import statistics
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_task_id(task_id: str) -> str:
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError("invalid task_id")
    return task_id


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass
class OpMetrics:
    """单算子执行指标（吞吐/延迟/成败统计）。"""

    op_name: str
    op_kind: str  # Mapper / Filter / Slicer / LLM
    n_in: int = 0
    n_out: int = 0
    n_failed: int = 0
    n_filtered: int = 0
    n_retried: int = 0
    wall_s: float = 0.0
    per_sample_ms: List[float] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def latency_stats(self) -> Dict[str, float]:
        xs = self.per_sample_ms
        if not xs:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
        xs_sorted = sorted(xs)
        p95_idx = min(len(xs_sorted) - 1, max(0, round(0.95 * len(xs_sorted)) - 1))
        return {
            "p50_ms": round(statistics.median(xs_sorted), 3),
            "p95_ms": round(xs_sorted[p95_idx], 3),
            "mean_ms": round(statistics.fmean(xs_sorted), 3),
            "max_ms": round(xs_sorted[-1], 3),
        }

    def throughput_per_s(self) -> float:
        return round(self.n_in / self.wall_s, 2) if self.wall_s > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_name": self.op_name,
            "op_kind": self.op_kind,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "n_failed": self.n_failed,
            "n_filtered": self.n_filtered,
            "n_retried": self.n_retried,
            "wall_s": round(self.wall_s, 4),
            "throughput_per_s": self.throughput_per_s(),
            **self.latency_stats(),
            "errors": self.errors[:20],
        }


@dataclass
class TaskReport:
    """一次流水线任务的完整执行报告（供智能体查询任务状态/诊断异常）。"""

    task_id: str
    name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    pipeline: List[Dict[str, Any]] = field(default_factory=list)
    n_input: int = 0
    n_output: int = 0
    n_failed: int = 0
    n_filtered: int = 0
    export_path: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    op_metrics: List[OpMetrics] = field(default_factory=list)
    error: str = ""
    schema_version: int = 3
    parent_task_id: str = ""
    child_task_ids: List[str] = field(default_factory=list)
    failure_category: str = ""
    recovery_action: str = ""
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def new(name: str = "", pipeline: Optional[List[Dict[str, Any]]] = None) -> "TaskReport":
        return TaskReport(task_id=uuid.uuid4().hex[:12], name=name, pipeline=pipeline or [])

    def mark_running(self):
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def mark_finished(self):
        self.finished_at = time.time()
        if self.status == TaskStatus.STOPPED:
            return
        if self.error:
            self.status = TaskStatus.FAILED
        elif self.n_failed > 0 and self.n_output == 0:
            self.status = TaskStatus.FAILED
        elif self.n_failed > 0:
            self.status = TaskStatus.PARTIAL_SUCCESS
        else:
            self.status = TaskStatus.COMPLETED

    @property
    def duration_s(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return round(end - self.started_at, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "pipeline": self.pipeline,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_failed": self.n_failed,
            "n_filtered": self.n_filtered,
            "export_path": self.export_path,
            "duration_s": self.duration_s,
            "op_metrics": [m.to_dict() for m in self.op_metrics],
            "error": self.error,
            "schema_version": self.schema_version,
            "parent_task_id": self.parent_task_id,
            "child_task_ids": list(self.child_task_ids),
            "failure_category": self.failure_category,
            "recovery_action": self.recovery_action,
            "recovery_events": list(self.recovery_events),
        }

    def summary(self) -> str:
        """面向智能体/终端的紧凑摘要。"""
        lines = [
            f"任务 {self.task_id}（{self.name}）状态: {self.status.value}",
            f"输入 {self.n_input} | 输出 {self.n_output} | 失败 {self.n_failed} | 过滤 {self.n_filtered} | 耗时 {self.duration_s}s",
        ]
        for m in self.op_metrics:
            st = m.latency_stats()
            lines.append(
                f"  - {m.op_name}[{m.op_kind}] in={m.n_in} out={m.n_out} fail={m.n_failed} "
                f"吞吐={m.throughput_per_s()}/s p50={st['p50_ms']}ms p95={st['p95_ms']}ms"
            )
            for err in m.errors[:3]:
                lines.append(f"      ! {err.get('fileId')}: {str(err.get('reason'))[:120]}")
        if self.error:
            lines.append(f"  任务级错误: {self.error[:300]}")
        if self.parent_task_id:
            lines.append(f"  父任务: {self.parent_task_id}")
        if self.child_task_ids:
            lines.append(f"  子任务: {', '.join(self.child_task_ids)}")
        for event in self.recovery_events[-3:]:
            applied = "已执行" if event.get("applied") else "未执行"
            lines.append(
                f"  恢复: {event.get('category')}/{event.get('action')} "
                f"({applied}, {event.get('scope')})"
            )
        return "\n".join(lines)

    def save(self, task_dir: Path) -> Path:
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / f"{self.task_id}.json"
        temporary = task_dir / f".{self.task_id}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    @staticmethod
    def load(task_dir: Path, task_id: str) -> Optional[Dict[str, Any]]:
        task_id = _validate_task_id(task_id)
        path = task_dir / f"{task_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
