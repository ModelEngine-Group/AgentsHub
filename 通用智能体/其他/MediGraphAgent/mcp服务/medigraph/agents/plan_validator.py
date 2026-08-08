"""Static validation of a planned operator DAG, before any operator runs.

The planner is a language model, so its output is a *proposal*. `DataProcAgent`
already whitelists operator names and `topological_layers` rejects cycles and
unknown dependencies, but neither checks whether each node's declared inputs can
actually be satisfied. Without that, an unsatisfiable plan is only discovered
part-way through execution -- after earlier nodes have already spent LLM calls and
written artefacts.

This module closes that gap by walking the DAG in dependency order and propagating
the set of payload keys that *will* exist at each node, comparing it against the
node's ``input_schema.required``. Everything wrong with the plan is reported at
once, as structured errors, so the re-plan prompt can fix all of it in one round
instead of converging one failure at a time.

Deliberately conservative: it only rejects what it can prove unsatisfiable from
declared schemas. Operators whose `input_schema` declares no `required` keys are
accepted, so an under-annotated operator degrades to today's behaviour rather than
blocking a valid plan.

    report = validate_plan(dag, available={"paths"})
    if not report.ok:
        ...  # re-plan using report.feedback()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medigraph.agents.dag_executor import topological_layers
from medigraph.operators.base import OP_REGISTRY, catalog

#: Payload keys the executor provides before any node runs. `_resolve_args` also
#: accepts these as placeholder targets.
DEFAULT_AVAILABLE: frozenset[str] = frozenset({"text", "paths", "documents"})

#: Budget guard-rails. A planner loop that emits hundreds of nodes is a bug, and
#: an unbounded plan is an unbounded bill.
MAX_NODES = 32
MAX_DEPTH = 16


@dataclass
class PlanError:
    """One reason a plan cannot run, addressed to whoever must fix the plan."""

    node: str
    code: str
    message: str
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = {"node": self.node, "code": self.code, "message": self.message}
        if self.missing:
            out["missing"] = self.missing
        return out


@dataclass
class PlanReport:
    ok: bool
    errors: list[PlanError] = field(default_factory=list)
    layers: list[list[str]] = field(default_factory=list)
    node_count: int = 0
    produced_keys: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "node_count": self.node_count,
            "depth": len(self.layers),
            "layers": self.layers,
            "errors": [error.as_dict() for error in self.errors],
        }

    def feedback(self) -> str:
        """Chinese-language summary for the planner's re-plan prompt."""
        if self.ok:
            return ""
        lines = ["计划静态校验未通过，请修正后重新输出 DAG："]
        for error in self.errors:
            detail = f"（缺少输入：{', '.join(error.missing)}）" if error.missing else ""
            lines.append(f"- 节点 {error.node or '-'}[{error.code}]：{error.message}{detail}")
        lines.append(
            "可用算子及其输入/输出契约见 toolbox；请确保每个节点所需的输入"
            "由其上游节点产出，或由初始输入提供。"
        )
        return "\n".join(lines)


def _required_inputs(op_name: str) -> list[str]:
    op = OP_REGISTRY.get(op_name)
    if op is None:
        return []
    schema = op.meta.input_schema or {}
    required = schema.get("required", [])
    return [str(key) for key in required] if isinstance(required, list) else []


def _produced_outputs(op_name: str) -> list[str]:
    op = OP_REGISTRY.get(op_name)
    if op is None:
        return []
    schema = op.meta.output_schema or {}
    properties = schema.get("properties", {})
    return list(properties.keys()) if isinstance(properties, dict) else []


def _explicit_arg_keys(node: dict) -> set[str]:
    """Keys the plan supplies directly via `args` (constants or placeholders)."""
    args = node.get("args", {})
    return set(map(str, args)) if isinstance(args, dict) else set()


def validate_plan(
    dag: list[dict],
    available: set[str] | frozenset[str] | None = None,
    max_nodes: int = MAX_NODES,
    max_depth: int = MAX_DEPTH,
) -> PlanReport:
    """Check a plan without executing it.

    `available` is the set of payload keys present before the first node (e.g.
    ``{"paths"}`` when the caller passes file paths, ``{"text"}`` for raw text).
    """
    errors: list[PlanError] = []

    if not dag:
        return PlanReport(ok=False, errors=[PlanError("", "empty_plan", "计划为空，没有任何算子节点。")])

    if len(dag) > max_nodes:
        errors.append(
            PlanError(
                "",
                "budget_nodes",
                f"计划包含 {len(dag)} 个节点，超过上限 {max_nodes}。",
            )
        )

    known = {entry["name"] for entry in catalog()}
    for node in dag:
        op_name = str(node.get("op", ""))
        if op_name not in known:
            errors.append(
                PlanError(
                    str(node.get("id", "")),
                    "unknown_operator",
                    f"算子 '{op_name}' 不在可用算子清单中。",
                )
            )

    # Structural validation reuses the executor's own rules, so a plan accepted here
    # cannot be rejected later for a different structural reason.
    try:
        layers = topological_layers(dag)
    except ValueError as exc:
        errors.append(PlanError("", "invalid_structure", str(exc)))
        return PlanReport(ok=False, errors=errors, node_count=len(dag))

    if len(layers) > max_depth:
        errors.append(
            PlanError(
                "",
                "budget_depth",
                f"计划深度 {len(layers)} 超过上限 {max_depth}。",
            )
        )

    # Dataflow validation: propagate the keys that will exist, layer by layer.
    node_by_id = {str(node["id"]): node for node in dag}
    reachable: set[str] = set(DEFAULT_AVAILABLE if available is None else available)
    produced_keys: dict[str, list[str]] = {}

    for layer in layers:
        # Peers in a layer cannot see each other's outputs (the executor merges only
        # at layer end), so validate the whole layer against the same snapshot.
        snapshot = set(reachable)
        for nid in layer:
            node = node_by_id[nid]
            op_name = str(node.get("op", ""))
            if op_name not in known:
                continue  # already reported
            satisfied = snapshot | _explicit_arg_keys(node)
            missing = [key for key in _required_inputs(op_name) if key not in satisfied]
            if missing:
                errors.append(
                    PlanError(
                        nid,
                        "unsatisfied_input",
                        f"算子 '{op_name}' 所需输入在执行到该节点时不存在。",
                        missing=missing,
                    )
                )
            produced_keys[nid] = _produced_outputs(op_name)
        for nid in layer:
            reachable.update(produced_keys.get(nid, []))

    return PlanReport(
        ok=not errors,
        errors=errors,
        layers=layers,
        node_count=len(dag),
        produced_keys=produced_keys,
    )


def initial_available_keys(payload: dict[str, Any]) -> set[str]:
    """Payload keys that count as present (empty values are still keys)."""
    return {str(key) for key in payload}
