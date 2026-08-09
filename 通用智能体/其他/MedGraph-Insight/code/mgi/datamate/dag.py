"""任务一：DAG 算子编排 + 数据质量指标 + 血缘追踪。

在线性流水线之上提供有向无环图（DAG）编排能力：
- 每个节点是一个 DataMate 算子，可声明多个上游依赖（支持分叉/汇合）；
- 拓扑排序后按依赖顺序执行，支持多输入合并（union）；
- 每步记录数据质量指标（DQ）与血缘（lineage），可导出为报告。

设计与 DataMate「数据处理编排」理念一致：算子即节点、数据集即边。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .client import DataMateClient, OperatorInstance

_PHONE = re.compile(r"1[3-9]\d{9}")
_IDCARD = re.compile(r"\d{17}[\dXx]|\d{15}")
_HTML_TAG = re.compile(r"<[^>]+>")


def data_quality(records: list[str]) -> dict[str, Any]:
    """计算一批文本记录的数据质量指标。"""
    n = len(records)
    if n == 0:
        return {"count": 0, "avg_len": 0.0, "empty_rate": 0.0,
                "dup_rate": 0.0, "pii_leak_rate": 0.0, "html_rate": 0.0}
    empties = sum(1 for r in records if not r.strip())
    uniq = len(set(records))
    pii = sum(1 for r in records if _PHONE.search(r) or _IDCARD.search(r))
    html = sum(1 for r in records if _HTML_TAG.search(r))
    return {
        "count": n,
        "avg_len": round(sum(len(r) for r in records) / n, 2),
        "empty_rate": round(empties / n, 4),
        "dup_rate": round(1 - uniq / n, 4),
        "pii_leak_rate": round(pii / n, 4),
        "html_rate": round(html / n, 4),
    }


@dataclass
class OpNode:
    """DAG 节点：算子名 + 参数 + 上游依赖节点 id。"""

    id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)


@dataclass
class NodeResult:
    id: str
    op: str
    backend: str
    rows_in: int
    rows_out: int
    dq_in: dict[str, Any]
    dq_out: dict[str, Any]
    inputs: list[str]
    elapsed_ms: float


class OperatorDAG:
    """算子有向无环图编排器。"""

    def __init__(self, client: DataMateClient | None = None) -> None:
        self.client = client or DataMateClient()
        self.nodes: dict[str, OpNode] = {}

    def add(self, node_id: str, op: str, inputs: list[str] | None = None,
            **params: Any) -> "OperatorDAG":
        if node_id in self.nodes:
            raise ValueError(f"重复节点 id: {node_id}")
        self.nodes[node_id] = OpNode(id=node_id, op=op, params=params, inputs=inputs or [])
        return self

    # ---- 拓扑排序（Kahn 算法，检测环） --------------------------------

    def topo_order(self) -> list[str]:
        indeg = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for up in node.inputs:
                if up not in self.nodes:
                    raise ValueError(f"节点 {node.id} 引用了不存在的上游 {up}")
                indeg[node.id] += 1
        queue = sorted(nid for nid, d in indeg.items() if d == 0)
        order: list[str] = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            children = sorted(n.id for n in self.nodes.values() if cur in n.inputs)
            for ch in children:
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    queue.append(ch)
            queue.sort()
        if len(order) != len(self.nodes):
            raise ValueError("DAG 中存在环，无法拓扑排序")
        return order

    # ---- 执行 -----------------------------------------------------------

    def run(self, source: list[str]) -> dict[str, Any]:
        """执行 DAG。无上游依赖的节点以 source 为输入；多上游节点取并集。"""
        order = self.topo_order()
        outputs: dict[str, list[str]] = {}
        results: list[NodeResult] = []
        for nid in order:
            node = self.nodes[nid]
            if not node.inputs:
                data_in = list(source)
            else:
                merged: list[str] = []
                for up in node.inputs:
                    merged.extend(outputs[up])
                data_in = merged
            dq_in = data_quality(data_in)
            t0 = time.perf_counter()
            res = self.client.run_cleaning(data_in, [OperatorInstance(node.op, node.params)])
            elapsed = (time.perf_counter() - t0) * 1000
            data_out = res["output"]
            outputs[nid] = data_out
            results.append(NodeResult(
                id=nid, op=node.op, backend=res["backend"],
                rows_in=len(data_in), rows_out=len(data_out),
                dq_in=dq_in, dq_out=data_quality(data_out),
                inputs=list(node.inputs), elapsed_ms=round(elapsed, 2),
            ))
        sinks = [nid for nid in self.nodes
                 if not any(nid in n.inputs for n in self.nodes.values())]
        return {
            "order": order,
            "outputs": outputs,
            "results": results,
            "sinks": sinks,
            "final": outputs[sinks[-1]] if sinks else source,
        }

    # ---- 血缘 -----------------------------------------------------------

    def lineage(self) -> dict[str, list[str]]:
        """返回每个节点的上游血缘链（递归展开到源）。"""
        cache: dict[str, list[str]] = {}

        def trace(nid: str) -> list[str]:
            if nid in cache:
                return cache[nid]
            chain: list[str] = []
            for up in self.nodes[nid].inputs:
                chain.extend(trace(up))
                chain.append(up)
            # 去重保序
            seen: set[str] = set()
            uniq = [x for x in chain if not (x in seen or seen.add(x))]
            cache[nid] = uniq
            return uniq

        return {nid: trace(nid) for nid in self.nodes}
