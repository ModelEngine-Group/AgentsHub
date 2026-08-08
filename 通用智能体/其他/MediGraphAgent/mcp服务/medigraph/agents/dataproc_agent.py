"""DataProc Agent (Task 1): natural-language -> operator DAG -> execution.

The agent reads the live operator catalog (name + description + schema) and asks
the LLM to plan a valid, executable operator DAG for the user's goal. It then
runs the DAG with the DAGExecutor (status tracking + retries) and, if a node
fails, performs one ReAct-style re-plan using the error feedback.

This shows task understanding + autonomous planning + multi-operator scheduling +
exception handling, decoupled from the operators themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import OUTPUTS_DIR
from medigraph.agents.dag_executor import DAGExecutor
from medigraph.agents.plan_validator import initial_available_keys, validate_plan
from medigraph.operators.base import catalog, load_default_operators

_SYSTEM = "你是数据处理流程编排专家，只输出 JSON 描述的算子 DAG，不要解释。"

_PLAN_PROMPT = """你可以使用以下数据处理算子（toolbox）：

{catalog}

用户的数据处理目标：
"{goal}"

请规划一个**合法且可执行**的算子 DAG 来完成该目标。规则：
1. 每个节点格式：{{"id":"n1","op":"<算子名>","args":{{}},"deps":[]}}。
2. op 只能取上面 toolbox 里的算子名。
3. deps 用前置节点的 id 表示数据依赖；执行器会按拓扑序把上游输出(text/chunks/entities/triples/valid)透传给下游。
4. 典型知识构建顺序：document_loader -> text_clean -> data_quality -> pii_redact -> chunker
   -> medical_ner -> entity_linker -> medical_re -> triple_validator（按目标裁剪，不需要的算子不要加）。
5. 涉及病例/患者原文时应加入 pii_redact；涉及图谱节点合并时应加入 entity_linker。
6. 只输出 JSON：{{"dag":[ ...nodes... ]}}。

{feedback}"""


class DataProcAgent:
    def __init__(self, llm: Any | None = None, local_planner: Any | None = None):
        """local_planner: optional callable(goal)->dag (e.g. the fine-tuned <1B
        orchestrator's `LocalOrchestrator.plan`). When set, planning runs on the
        local small model instead of the big API model -- cheaper, faster, offline.
        """
        if llm is None:
            from medigraph.llm.client import LLMClient
            llm = LLMClient()
        self.llm = llm
        self.local_planner = local_planner
        # ensure operators are registered (idempotent)
        load_default_operators(llm=llm)
        self.executor = DAGExecutor(max_retries=2)

    # ------------------------------------------------------------------ #
    def plan(self, goal: str, feedback: str = "") -> list[dict]:
        # Use the fine-tuned local orchestrator when available (skip feedback loop).
        if self.local_planner is not None and not feedback:
            try:
                return self._sanitize(self.local_planner(goal))
            except Exception as exc:  # noqa: BLE001 - fall back to the API planner
                print(f"[DataProcAgent] local planner failed ({exc}); using API planner")
        prompt = _PLAN_PROMPT.format(
            catalog=json.dumps(catalog(), ensure_ascii=False, indent=2),
            goal=goal,
            feedback=feedback,
        )
        data = self.llm.chat_json(prompt, system=_SYSTEM, default={"dag": []})
        dag = data.get("dag", []) if isinstance(data, dict) else []
        return self._sanitize(dag)

    def _sanitize(self, dag: list) -> list[dict]:
        valid_ops = {c["name"] for c in catalog()}
        out = []
        used_ids: set[str] = set()
        for i, node in enumerate(dag):
            if not isinstance(node, dict) or node.get("op") not in valid_ops:
                continue
            node_id = str(node.get("id") or f"n{i+1}")
            if node_id in used_ids:
                suffix = 2
                while f"{node_id}_{suffix}" in used_ids:
                    suffix += 1
                node_id = f"{node_id}_{suffix}"
            used_ids.add(node_id)
            out.append(
                {
                    "id": node_id,
                    "op": node["op"],
                    "args": node.get("args", {}) if isinstance(node.get("args"), dict) else {},
                    "deps": [str(d) for d in node.get("deps", []) if isinstance(node.get("deps"), list)],
                    "retry_args": node.get("retry_args", []) if isinstance(node.get("retry_args"), list) else [],
                    "on_error": node.get("on_error", {}) if isinstance(node.get("on_error"), dict) else {},
                }
            )
        valid_ids = {node["id"] for node in out}
        for node in out:
            node["deps"] = [
                dependency
                for dependency in node["deps"]
                if dependency in valid_ids and dependency != node["id"]
            ]
        return out

    @staticmethod
    def reflect(goal: str, result: dict) -> dict:
        """Check that execution produced the artifacts implied by the user goal."""
        goal_lower = goal.lower()
        payload = result.get("payload", {})
        expectations = [
            (("清洗", "clean"), "text"),
            (("切块", "分块", "chunk"), "chunks"),
            (("实体", "ner"), "entities"),
            (("关系", "三元组", "图谱", "relation"), "valid"),
        ]
        required = [
            output
            for keywords, output in expectations
            if any(keyword in goal_lower for keyword in keywords)
        ]
        missing = [
            output
            for output in required
            if output not in payload or payload.get(output) in (None, [], "")
        ]
        return {
            "satisfied": not missing,
            "required_outputs": required,
            "missing_outputs": missing,
        }

    # ------------------------------------------------------------------ #
    def _validate(self, dag: list[dict], payload: dict) -> Any:
        return validate_plan(dag, available=initial_available_keys(payload))

    @staticmethod
    def _record_rejected_plan(goal: str, dag: list[dict], report: Any) -> None:
        """Persist a statically-rejected plan for later inspection.

        Purely an observability artefact -- these files are for debugging the planner,
        not a training-data pipeline.
        """
        try:
            directory = Path(OUTPUTS_DIR) / "badcases"
            directory.mkdir(parents=True, exist_ok=True)
            existing = len(list(directory.glob("rejected_plan_*.json")))
            path = directory / f"rejected_plan_{existing + 1:04d}.json"
            path.write_text(
                json.dumps(
                    {"goal": goal, "dag": dag, "validation": report.as_dict()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:  # pragma: no cover - never fail a run over a debug artefact
            pass

    # ------------------------------------------------------------------ #
    def run(self, goal: str, payload: dict, verbose: bool = True) -> dict:
        """Plan, statically validate, then execute. Re-plans once on failure.

        Validation runs *before* the executor, so an unsatisfiable plan costs nothing
        to reject: the planner is asked again with the full error list instead of the
        DAG failing part-way through and leaving partial artefacts behind.
        """
        if verbose:
            print(f"\n[DataProcAgent] goal: {goal}")
            print("[DataProcAgent] planning operator DAG ...")
        dag = self.plan(goal)
        if not dag:
            raise RuntimeError("planner produced an empty/invalid DAG; check LLM/API config")

        validation = self._validate(dag, payload)
        planning_history = [{"reason": "initial", "dag": dag, "validation": validation.as_dict()}]
        if not validation.ok:
            if verbose:
                print("[DataProcAgent] plan rejected by static validation:")
                for error in validation.errors:
                    print(f"    - {error.node or '-'} [{error.code}] {error.message} {error.missing}")
                print("[DataProcAgent] re-planning without executing ...")
            self._record_rejected_plan(goal, dag, validation)
            repaired = self.plan(goal, feedback=validation.feedback())
            if repaired:
                revalidated = self._validate(repaired, payload)
                planning_history.append(
                    {
                        "reason": "static_validation",
                        "dag": repaired,
                        "validation": revalidated.as_dict(),
                    }
                )
                if revalidated.ok:
                    dag, validation = repaired, revalidated
                else:
                    self._record_rejected_plan(goal, repaired, revalidated)

        if verbose:
            print("[DataProcAgent] planned DAG:")
            for n in dag:
                print(f"    {n['id']}: {n['op']}  deps={n['deps']}  args={n['args']}")
            print(f"[DataProcAgent] validation: ok={validation.ok} depth={len(validation.layers)}")

        result = self.executor.run(dag, payload, verbose=verbose)
        executed_dag = dag

        reflection = self.reflect(goal, result)
        if result["report"]["nodes_failed"] > 0 or not reflection["satisfied"]:
            failed = [s for s in result["states"].values() if s["status"] == "failed"]
            feedback = (
                "上一次执行未完全满足目标，请修正 DAG（可调整参数、替换算子或补充步骤）。"
                f"失败信息：{json.dumps(failed, ensure_ascii=False)[:500]}；"
                f"产物自检：{json.dumps(reflection, ensure_ascii=False)}"
            )
            if verbose:
                print("[DataProcAgent] re-planning after failure ...")
            dag2 = self.plan(goal, feedback=feedback)
            if dag2:
                result = self.executor.run(dag2, payload, verbose=verbose)
                result["replanned"] = True
                executed_dag = dag2
                planning_history.append({"reason": "failure_or_reflection", "dag": dag2})
        result["dag"] = executed_dag
        result["planning_history"] = planning_history
        result["validation"] = validation.as_dict()
        result["reflection"] = self.reflect(goal, result)
        return result
