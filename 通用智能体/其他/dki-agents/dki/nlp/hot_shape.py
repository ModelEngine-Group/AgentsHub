"""Hot-shape lattice 规划与到达回放（CPU 确定性仿真，非 NPU 实机成绩）。

目标（对应差距报告 §4.4）：把“固定 bs64/seq256 单点”升级为 2–4 个静态 shape 节点的
lattice，并给出可复现的对照证据：

- `plan_lattice`：按真实 token 长度分布选 ≤max_nodes 个 sequence 节点 + 一个 overflow 节点，
  每节点带 shape 签名（manifest stub）、命中计数、padding 浪费（Σ(node_len - required)）。
- `simulate_arrivals`：离散事件仿真，open-loop（固定到达率）与 closed-loop（有界在队/背压）
  两种模式，回放到 fixed-shape 与 lattice 两种调度，产出 p95/p99 等待、padding 浪费、吞吐，
  同一 trace 对照。**纯 CPU、确定性（seed）**，用于论证 lattice 相对固定 shape 的取舍，
  不声称等于 NPU 实测 p95/p99。

诚实边界：service_latency 单位为“仿真时间单位”，execute 成本由 `per_token_cost` 线性建模
（GEMM≈O(batch·seq)）；真实到达 p95/p99、能效 Pareto 必须在目标机另行实测（promotion=false）。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class LatticeNode:
    seq_len: int
    hits: int
    padding_waste: int  # Σ(seq_len - required_tokens) over docs routed here

    @property
    def shape_signature(self) -> str:
        return f"seq{self.seq_len}"


@dataclass
class LatticePlan:
    nodes: List[LatticeNode]
    overflow: int
    total: int
    special_tokens: int

    def node_lengths(self) -> List[int]:
        return [node.seq_len for node in self.nodes]

    def total_padding_waste(self) -> int:
        return sum(node.padding_waste for node in self.nodes)

    def to_dict(self) -> dict:
        return {
            "sequence_nodes": self.node_lengths(),
            "nodes": [
                {"seq_len": n.seq_len, "hits": n.hits, "padding_waste": n.padding_waste,
                 "shape_signature": n.shape_signature}
                for n in self.nodes
            ],
            "overflow": self.overflow,
            "total": self.total,
            "total_padding_waste": self.total_padding_waste(),
            "special_tokens": self.special_tokens,
        }


def _required(tokens: int, special_tokens: int) -> int:
    return int(tokens) + special_tokens


def choose_node(required: int, node_lengths: Sequence[int]) -> Optional[int]:
    """最小可容纳节点长度；都放不下返回 None（overflow）。"""
    for length in sorted(node_lengths):
        if length >= required:
            return length
    return None


def plan_lattice(
    token_lengths: Sequence[int],
    *,
    sequence_lengths: Sequence[int] = (64, 128, 256),
    max_nodes: int = 3,
    special_tokens: int = 2,
) -> LatticePlan:
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    candidates = sorted({int(v) for v in sequence_lengths})
    if not candidates:
        raise ValueError("sequence_lengths must be non-empty")
    # 先统计每个候选节点作为“最小可容纳”被命中的次数。
    hit_counter: Counter = Counter()
    overflow = 0
    for tokens in token_lengths:
        chosen = choose_node(_required(tokens, special_tokens), candidates)
        if chosen is None:
            overflow += 1
        else:
            hit_counter[chosen] += 1
    # 取命中最多的 max_nodes 个节点；被裁掉的节点其文档回落到更大的保留节点或 overflow。
    ranked = sorted(hit_counter, key=lambda length: (-hit_counter[length], length))[:max_nodes]
    kept = sorted(ranked)
    waste: Dict[int, int] = {length: 0 for length in kept}
    counts: Dict[int, int] = {length: 0 for length in kept}
    overflow = 0
    for tokens in token_lengths:
        required = _required(tokens, special_tokens)
        chosen = choose_node(required, kept)
        if chosen is None:
            overflow += 1
            continue
        counts[chosen] += 1
        waste[chosen] += chosen - required
    nodes = [LatticeNode(seq_len=length, hits=counts[length], padding_waste=waste[length])
             for length in kept if counts[length] > 0]
    return LatticePlan(nodes=nodes, overflow=overflow, total=len(list(token_lengths)),
                       special_tokens=special_tokens)


# ----------------------------------------------------------------------------
# 离散事件到达回放（fixed-shape vs lattice 同 trace 对照）
# ----------------------------------------------------------------------------

@dataclass
class SimResult:
    schedule: str  # "fixed" | "lattice"
    mode: str      # "open_loop" | "closed_loop"
    completed: int
    dropped: int
    throughput: float          # requests / sim-time
    padding_waste_tokens: int
    wait_p50: float
    wait_p95: float
    wait_p99: float
    exec_time_units: float
    makespan: float

    def to_dict(self) -> dict:
        return {
            "schedule": self.schedule, "mode": self.mode,
            "completed": self.completed, "dropped": self.dropped,
            "throughput_req_per_unit": round(self.throughput, 6),
            "padding_waste_tokens": self.padding_waste_tokens,
            "wait_p50": round(self.wait_p50, 4),
            "wait_p95": round(self.wait_p95, 4),
            "wait_p99": round(self.wait_p99, 4),
            "exec_time_units": round(self.exec_time_units, 4),
            "makespan": round(self.makespan, 4),
        }


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[idx]


def _batch_seq_for_fixed(fixed_seq: int, required: int) -> Optional[int]:
    return fixed_seq if fixed_seq >= required else None


def simulate_arrivals(
    token_lengths: Sequence[int],
    *,
    schedule: str,
    mode: str = "open_loop",
    sequence_lengths: Sequence[int] = (64, 128, 256),
    fixed_seq: int = 256,
    max_nodes: int = 3,
    special_tokens: int = 2,
    batch_size: int = 8,
    per_token_cost: float = 1.0,
    arrival_interval: float = 4.0,
    max_inflight: int = 64,
    seed: int = 20260720,
) -> SimResult:
    """回放到达并批处理；返回等待分位、padding 浪费、吞吐（确定性）。

    - schedule="fixed"：所有请求 pad 到 `fixed_seq`（放不下则丢弃）。
    - schedule="lattice"：请求按最小可容纳节点分组，各节点独立按 batch 成批。
    - mode="open_loop"：请求按固定间隔到达，队列无限。
    - mode="closed_loop"：在队上限 `max_inflight`，满则丢弃（背压）。
    execute 成本 = batch_size_effective 无关的 per-batch = seq_len * per_token_cost
    （模型前向≈O(seq)；同批共享，故按批计一次）。
    """
    if schedule not in {"fixed", "lattice"}:
        raise ValueError("schedule must be 'fixed' or 'lattice'")
    if mode not in {"open_loop", "closed_loop"}:
        raise ValueError("mode must be 'open_loop' or 'closed_loop'")
    if batch_size <= 0 or arrival_interval <= 0:
        raise ValueError("batch_size and arrival_interval must be positive")

    rng = random.Random(seed)
    order = list(range(len(token_lengths)))
    rng.shuffle(order)
    node_lengths = sorted({int(v) for v in sequence_lengths})
    if schedule == "lattice":
        plan = plan_lattice(token_lengths, sequence_lengths=sequence_lengths,
                            max_nodes=max_nodes, special_tokens=special_tokens)
        active_nodes = plan.node_lengths()
    else:
        active_nodes = [fixed_seq]

    # 每个到达：(arrival_time, required_tokens)
    arrivals = []
    t = 0.0
    for idx in order:
        arrivals.append((t, _required(int(token_lengths[idx]), special_tokens)))
        t += arrival_interval

    # 分桶排队
    queues: Dict[int, List[tuple]] = {length: [] for length in active_nodes}
    dropped = 0
    inflight = 0

    def route(required: int) -> Optional[int]:
        if schedule == "fixed":
            return _batch_seq_for_fixed(fixed_seq, required)
        return choose_node(required, active_nodes)

    device_free_at = 0.0
    waits: List[float] = []
    padding_waste = 0
    exec_total = 0.0
    completed = 0

    # 事件推进：按到达序处理，成满批或在最后 flush。
    pending = list(arrivals)
    pending_index = 0

    def flush_batch(node_len: int, now: float):
        nonlocal device_free_at, padding_waste, exec_total, completed, inflight
        q = queues[node_len]
        if not q:
            return
        take = q[:batch_size]
        del q[:batch_size]
        start = max(now, device_free_at)
        exec_time = node_len * per_token_cost
        device_free_at = start + exec_time
        exec_total += exec_time
        for arrival_time, required in take:
            waits.append(device_free_at - arrival_time)
            padding_waste += node_len - required
            completed += 1
            inflight -= 1

    for arrival_time, required in pending:
        node_len = route(required)
        if node_len is None:
            dropped += 1
            continue
        if mode == "closed_loop" and inflight >= max_inflight:
            dropped += 1
            continue
        queues[node_len].append((arrival_time, required))
        inflight += 1
        if len(queues[node_len]) >= batch_size:
            flush_batch(node_len, arrival_time)
    # flush 残批
    last_arrival = pending[-1][0] if pending else 0.0
    for node_len in active_nodes:
        while queues[node_len]:
            flush_batch(node_len, last_arrival)

    makespan = device_free_at
    throughput = completed / makespan if makespan > 0 else 0.0
    return SimResult(
        schedule=schedule, mode=mode, completed=completed, dropped=dropped,
        throughput=throughput, padding_waste_tokens=padding_waste,
        wait_p50=_percentile(waits, 0.50), wait_p95=_percentile(waits, 0.95),
        wait_p99=_percentile(waits, 0.99), exec_time_units=exec_total, makespan=makespan,
    )


def compare_fixed_vs_lattice(token_lengths: Sequence[int], **kwargs) -> dict:
    """同 trace 下 fixed 与 lattice 的对照（open + closed loop）。"""
    out = {"n_docs": len(list(token_lengths)), "runs": {}}
    for mode in ("open_loop", "closed_loop"):
        fixed = simulate_arrivals(token_lengths, schedule="fixed", mode=mode, **kwargs)
        lattice = simulate_arrivals(token_lengths, schedule="lattice", mode=mode, **kwargs)
        padding_reduction = (
            1.0 - lattice.padding_waste_tokens / fixed.padding_waste_tokens
            if fixed.padding_waste_tokens else 0.0
        )
        out["runs"][mode] = {
            "fixed": fixed.to_dict(),
            "lattice": lattice.to_dict(),
            "padding_waste_reduction": round(padding_reduction, 4),
        }
    out["note"] = "CPU 确定性仿真（sim-time 单位）；真实 p95/p99 与能效须目标机实测，promotion=false。"
    return out
