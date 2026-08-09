"""算子基础接口。

与 DataMate 算子规范对齐（metadata + process），同时可注册为 Nexent 工具。
每个算子自描述（name / description / inputs / output_type），
既可被智能体自动编排，也可单独复用。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OperatorMetadata:
    name: str
    description: str
    modality: str = "text"
    op_type: str = "mapper"  # mapper / filter / generator / validator
    version: str = "0.1.0"
    inputs: dict[str, str] = field(default_factory=dict)
    output_type: str = "list"


@dataclass
class OperatorResult:
    output: Any
    elapsed_ms: float
    num_in: int
    num_out: int

    @property
    def throughput(self) -> float:
        """条/秒。"""
        if self.elapsed_ms <= 0:
            return float("inf")
        return self.num_in / (self.elapsed_ms / 1000.0)


class Operator(ABC):
    """所有算子的基类：批式处理，内置计时，便于性能评测与编排。"""

    metadata: OperatorMetadata

    @abstractmethod
    def process(self, batch: list[Any], **kwargs: Any) -> list[Any]:
        """处理一批数据，返回一批结果。"""

    def run(self, batch: list[Any], **kwargs: Any) -> OperatorResult:
        start = time.perf_counter()
        output = self.process(batch, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return OperatorResult(
            output=output,
            elapsed_ms=elapsed_ms,
            num_in=len(batch),
            num_out=len(output) if isinstance(output, list) else 1,
        )

    def describe(self) -> dict[str, Any]:
        m = self.metadata
        return {
            "name": m.name,
            "description": m.description,
            "modality": m.modality,
            "type": m.op_type,
            "inputs": m.inputs,
            "output_type": m.output_type,
        }


class OperatorRegistry:
    """算子注册表：智能体通过它发现并编排算子。"""

    def __init__(self) -> None:
        self._ops: dict[str, Operator] = {}

    def register(self, op: Operator) -> Operator:
        self._ops[op.metadata.name] = op
        return op

    def get(self, name: str) -> Operator:
        if name not in self._ops:
            raise KeyError(f"未注册的算子: {name}，可用: {list(self._ops)}")
        return self._ops[name]

    def names(self) -> list[str]:
        return list(self._ops)

    def catalog(self) -> list[dict[str, Any]]:
        return [op.describe() for op in self._ops.values()]


registry = OperatorRegistry()
