"""DAG executor for data-processing pipelines.

Takes a DAG (nodes = operator invocations, edges = data dependencies), runs nodes
in topological order with per-node status tracking, retries and timing, and
produces a processing report + per-document lineage.

A node is: {"id": str, "op": <operator name>, "args": {...}, "deps": [node_ids]}
The executor threads a `payload` dict between nodes; each operator reads/writes
known keys (text, chunks, entities, triples, valid, ...).
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from medigraph.operators.base import get_operator


class NodeStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


def topological_order(nodes: list[dict]) -> list[str]:
    """Kahn's algorithm. Raises on cycle / missing dependency."""
    ids_in_order = [str(node.get("id", "")) for node in nodes]
    if not ids_in_order or any(not node_id for node_id in ids_in_order):
        raise ValueError("DAG nodes must have non-empty ids")
    if len(set(ids_in_order)) != len(ids_in_order):
        raise ValueError("DAG node ids must be unique")
    ids = set(ids_in_order)
    unknown = sorted(
        {
            str(dependency)
            for node in nodes
            for dependency in node.get("deps", [])
            if str(dependency) not in ids
        }
    )
    if unknown:
        raise ValueError(f"DAG references unknown dependencies: {unknown}")
    deps = {str(n["id"]): {str(d) for d in n.get("deps", [])} for n in nodes}
    indeg = {nid: len(deps[nid]) for nid in ids_in_order}
    queue = [nid for nid in ids_in_order if indeg[nid] == 0]
    order: list[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nid in ids_in_order:
            if cur in deps[nid]:
                deps[nid].discard(cur)
                indeg[nid] -= 1
                if indeg[nid] == 0:
                    queue.append(nid)
    if len(order) != len(ids):
        raise ValueError("DAG has a cycle")
    return order


def topological_layers(nodes: list[dict]) -> list[list[str]]:
    """Group nodes into dependency layers (Kahn, level-by-level).

    Every node in a returned layer has all its dependencies satisfied by earlier
    layers and none by its own layer, so a layer can run concurrently. Ordering
    within a layer follows the declaration order, keeping execution reports stable.

    Validation is delegated to `topological_order`, so both entry points reject the
    same malformed DAGs (missing ids, duplicate ids, unknown deps, cycles).
    """
    topological_order(nodes)  # validate; raises on cycle / unknown dependency
    ids_in_order = [str(node["id"]) for node in nodes]
    remaining = {
        str(node["id"]): {str(dep) for dep in node.get("deps", [])} for node in nodes
    }
    layers: list[list[str]] = []
    done: set[str] = set()
    while remaining:
        layer = [nid for nid in ids_in_order if nid in remaining and not (remaining[nid] - done)]
        if not layer:  # pragma: no cover - topological_order already rejects cycles
            raise ValueError("DAG has a cycle")
        layers.append(layer)
        done.update(layer)
        for nid in layer:
            remaining.pop(nid)
    return layers


class DAGExecutor:
    """Executes an operator DAG layer by layer, concurrently within each layer.

    Concurrency
        Nodes are grouped by `topological_layers`; a layer's nodes are independent by
        construction, so they run on a thread pool. Operators here are IO-bound
        (LLM calls dominate), which is what makes threads the right primitive
        despite the GIL.

        Every node in a layer reads the *same* payload snapshot, and outputs are
        merged only once the layer completes ("collect in layer, merge at layer
        end"). Without that, two concurrent nodes writing the same payload key
        would race and the result would depend on completion order.

    Timeouts
        `timeout` (executor default, overridable per node via ``"timeout"``) bounds a
        single attempt. Caveat, stated plainly: Python cannot force-kill a thread, so
        a timeout abandons the worker rather than stopping it -- the DAG stops waiting
        and the node follows the normal failure path, but a genuinely wedged operator
        keeps occupying a pool thread. Bounding that properly needs process isolation,
        which is out of scope here.
    """

    def __init__(
        self,
        max_retries: int = 2,
        max_workers: int = 4,
        timeout: float | None = None,
        parallel: bool = True,
    ):
        self.max_retries = max_retries
        self.max_workers = max(1, max_workers)
        self.timeout = timeout
        self.parallel = parallel

    def run(self, dag: list[dict], initial_payload: dict, verbose: bool = True) -> dict:
        layers = topological_layers(dag)
        node_by_id = {n["id"]: n for n in dag}
        payload = dict(initial_payload)
        node_outputs: dict[str, dict] = {"input": dict(initial_payload)}
        states: dict[str, dict] = {n["id"]: {"status": NodeStatus.PENDING} for n in dag}
        lineage: list[dict] = []
        recoveries: list[dict] = []

        for depth, layer in enumerate(layers):
            runnable: list[str] = []
            for nid in layer:
                node = node_by_id[nid]
                failed_dependencies = [
                    dep
                    for dep in node.get("deps", [])
                    if states.get(dep, {}).get("status")
                    in {NodeStatus.FAILED, NodeStatus.SKIPPED}
                ]
                if failed_dependencies:
                    states[nid].update(
                        status=NodeStatus.SKIPPED,
                        error=f"blocked by failed dependencies: {failed_dependencies}",
                    )
                    lineage.append(
                        {
                            "node": nid,
                            "op": node["op"],
                            "status": NodeStatus.SKIPPED,
                            "blocked_by": failed_dependencies,
                        }
                    )
                else:
                    states[nid]["status"] = NodeStatus.RUNNING
                    runnable.append(nid)

            if not runnable:
                continue
            if verbose:
                names = ", ".join(f"{nid}:{node_by_id[nid]['op']}" for nid in runnable)
                print(f"  [layer {depth}] {names}", flush=True)

            # Run the layer. A single node stays on this thread: no pool, no context
            # switch, and tracebacks/profiles stay readable for the common linear DAG.
            if len(runnable) == 1 or not self.parallel:
                results = [self._run_node(node_by_id[nid], payload, node_outputs, verbose) for nid in runnable]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(runnable)),
                    thread_name_prefix="dag",
                ) as pool:
                    futures = [
                        pool.submit(self._run_node, node_by_id[nid], payload, node_outputs, verbose)
                        for nid in runnable
                    ]
                    results = [future.result() for future in futures]

            # Merge after the whole layer finishes -- see the class docstring.
            for nid, outcome in zip(runnable, results):
                if outcome["ok"]:
                    payload.update(outcome["output"])
                    node_outputs[nid] = dict(outcome["output"])
                    states[nid].update(
                        status=NodeStatus.SUCCESS,
                        seconds=outcome["seconds"],
                        attempts=outcome["attempts"],
                        output_keys=outcome["output_keys"],
                    )
                    lineage.append(
                        {
                            "node": nid,
                            "op": outcome["op_name"],
                            "status": "success",
                            "seconds": outcome["seconds"],
                            "layer": depth,
                        }
                    )
                else:
                    states[nid].update(
                        status=NodeStatus.FAILED,
                        seconds=outcome["seconds"],
                        attempts=outcome["attempts"],
                        error=outcome["error"],
                    )
                    lineage.append(
                        {
                            "node": nid,
                            "op": outcome["op_name"],
                            "status": "failed",
                            "error": outcome["error"],
                            "layer": depth,
                        }
                    )
                recoveries.extend(outcome["recoveries"])

        report = self._build_report(states, payload)
        report["recoveries"] = len(recoveries)
        report["layers"] = len(layers)
        report["max_layer_width"] = max((len(layer) for layer in layers), default=0)
        return {
            "payload": payload,
            "states": states,
            "lineage": lineage,
            "recoveries": recoveries,
            "layers": layers,
            "report": report,
        }

    # ------------------------------------------------------------------ #
    def _run_node(
        self,
        node: dict,
        payload: dict,
        node_outputs: dict[str, dict],
        verbose: bool,
    ) -> dict:
        """Execute one node with retries, timeout and fallback. Pure w.r.t. `payload`.

        Returns an outcome dict instead of mutating shared state, so it is safe to
        call concurrently for every node of a layer.
        """
        nid = node["id"]
        op_name = node["op"]
        recoveries: list[dict] = []
        op = get_operator(op_name)
        inputs = {**payload, **self._resolve_args(node.get("args", {}), payload, node_outputs)}
        node_timeout = node.get("timeout", self.timeout)

        attempt, ok, err, out = 0, False, None, {}
        started = time.time()
        while attempt <= self.max_retries and not ok:
            attempt += 1
            try:
                retry_args = node.get("retry_args", [])
                attempt_inputs = dict(inputs)
                if attempt > 1 and isinstance(retry_args, list) and len(retry_args) >= attempt - 1:
                    adjustment = retry_args[attempt - 2]
                    if isinstance(adjustment, dict):
                        attempt_inputs.update(adjustment)
                        recoveries.append(
                            {
                                "node": nid,
                                "strategy": "retry_with_adjusted_args",
                                "attempt": attempt,
                                "adjustment_keys": sorted(adjustment),
                            }
                        )
                out = self._call(op, attempt_inputs, node_timeout)
                self._validate_output(op, out)
                ok = True
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                if verbose:
                    print(f"    ! [{nid}] attempt {attempt} failed: {exc}")
                if attempt <= self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 2))

        if not ok:
            fallback = node.get("on_error", {})
            if isinstance(fallback, dict) and fallback.get("fallback_op"):
                try:
                    fallback_name = str(fallback["fallback_op"])
                    fallback_operator = get_operator(fallback_name)
                    fallback_inputs = {
                        **inputs,
                        **(
                            fallback.get("args", {})
                            if isinstance(fallback.get("args"), dict)
                            else {}
                        ),
                    }
                    out = self._call(fallback_operator, fallback_inputs, node_timeout)
                    self._validate_output(fallback_operator, out)
                    op_name = fallback_name
                    op = fallback_operator
                    ok = True
                    recoveries.append(
                        {
                            "node": nid,
                            "strategy": "fallback_operator",
                            "fallback_op": fallback_name,
                        }
                    )
                except Exception as fallback_exc:  # noqa: BLE001
                    err = f"{err}; fallback failed: {fallback_exc}"

        seconds = round(time.time() - started, 3)
        if verbose:
            if ok:
                print(f"    [OK] [{nid}] {seconds}s (attempts={attempt})")
            else:
                print(f"    [X] [{nid}] failed after {attempt} attempts: {err}")
        return {
            "ok": ok,
            "op_name": op_name,
            "output": out if ok else {},
            "output_keys": self._output_keys(op) if ok else [],
            "seconds": seconds,
            "attempts": attempt,
            "error": err,
            "recoveries": recoveries,
        }

    @staticmethod
    def _call(op, inputs: dict, timeout: float | None):
        """Run an operator, optionally bounded by `timeout` seconds."""
        if not timeout or timeout <= 0:
            return op.run(inputs)
        # One-off worker so the wait can be abandoned. See the timeout caveat in the
        # class docstring: this stops *waiting*, it cannot stop the operator.
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dag-timeout")
        future = pool.submit(op.run, inputs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(
                f"operator '{op.meta.name}' exceeded timeout of {timeout}s"
            ) from exc
        finally:
            # wait=False: never block the DAG on an abandoned worker.
            pool.shutdown(wait=False)

    @classmethod
    def _resolve_args(cls, args: dict, payload: dict, node_outputs: dict[str, dict]) -> dict:
        """Resolve lightweight Jinja-style references emitted by LLM planners.

        The planner is allowed to produce args such as ``{"text": "{{input_text}}"}``
        or ``{"entities": "{{n2.entities}}"}``. Older code treated those strings
        literally, which could overwrite the real document text. This resolver
        keeps normal constants intact, resolves known placeholders, and drops
        unresolved placeholders so the threaded payload can still provide values.
        """
        if not isinstance(args, dict):
            return {}
        out: dict[str, Any] = {}
        for key, value in args.items():
            resolved, ok = cls._resolve_value(value, payload, node_outputs)
            if ok:
                out[key] = resolved
        return out

    @classmethod
    def _resolve_value(cls, value: Any, payload: dict, node_outputs: dict[str, dict]) -> tuple[Any, bool]:
        if isinstance(value, dict):
            resolved = {}
            for k, v in value.items():
                rv, ok = cls._resolve_value(v, payload, node_outputs)
                if ok:
                    resolved[k] = rv
            return resolved, True
        if isinstance(value, list):
            items = []
            for v in value:
                rv, ok = cls._resolve_value(v, payload, node_outputs)
                if ok:
                    items.append(rv)
            return items, True
        if not isinstance(value, str):
            return value, True

        match = re.fullmatch(r"\s*\{\{\s*([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?\s*\}\}\s*", value)
        if not match:
            return value, True

        source, field = match.group(1), match.group(2)
        if source in {"input", "payload"}:
            if field:
                return payload.get(field), field in payload
            return payload, True
        if source == "input_text":
            return payload.get("text"), "text" in payload
        if field and source in node_outputs and field in node_outputs[source]:
            return node_outputs[source][field], True
        if field and field in payload:
            return payload[field], True
        # Unresolved placeholder: do not let it clobber a real payload value.
        return None, False

    @staticmethod
    def _output_keys(op) -> list[str]:
        props = (op.meta.output_schema or {}).get("properties", {})
        return list(props.keys())

    @staticmethod
    def _validate_output(op, output: Any) -> None:
        if not isinstance(output, dict):
            raise TypeError(f"operator '{op.meta.name}' returned {type(output).__name__}, expected dict")
        schema = op.meta.output_schema or {}
        required = schema.get("required", [])
        missing = [key for key in required if key not in output]
        if missing:
            raise ValueError(f"operator '{op.meta.name}' missing required outputs: {missing}")

    @staticmethod
    def _build_report(states: dict, payload: dict) -> dict:
        n_success = sum(1 for s in states.values() if s["status"] == NodeStatus.SUCCESS)
        n_failed = sum(1 for s in states.values() if s["status"] == NodeStatus.FAILED)
        n_skipped = sum(1 for s in states.values() if s["status"] == NodeStatus.SKIPPED)
        total_seconds = round(sum(s.get("seconds", 0) for s in states.values()), 3)
        return {
            "nodes_total": len(states),
            "nodes_success": n_success,
            "nodes_failed": n_failed,
            "nodes_skipped": n_skipped,
            "total_seconds": total_seconds,
            "produced_chunks": len(payload.get("chunks", []) or []),
            "produced_entities": len(payload.get("entities", []) or []),
            "produced_valid_triples": len(payload.get("valid", []) or []),
        }
