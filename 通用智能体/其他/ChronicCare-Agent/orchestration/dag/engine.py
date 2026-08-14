from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "dag_runs"
CONTRACTS_PATH = ROOT / "configs" / "operator_contracts" / "contracts.json"

DEPS = {
    "chronic_file_ingest": [],
    "chronic_table_clean": ["chronic_file_ingest"],
    "chronic_field_normalize": ["chronic_table_clean"],
    "chronic_text_split": ["chronic_field_normalize"],
    "chronic_entity_extract": ["chronic_field_normalize", "chronic_text_split"],
    "chronic_entity_extract_model_npu": ["chronic_field_normalize", "chronic_text_split"],
    "chronic_relation_extract": ["ENTITY"],
    "chronic_relation_extract_model_npu": ["ENTITY"],
    "chronic_triple_validate": ["ENTITY", "RELATION"],
    "chronic_kg_build": ["chronic_triple_validate"],
    "chronic_sqlite_loader": ["chronic_field_normalize"],
    "chronic_nl2sql_analyze": ["chronic_sqlite_loader"],
    "chronic_report_pack": ["chronic_kg_build", "chronic_nl2sql_analyze"],
}

GOAL_ALIASES = {
    "clean": "clean", "清洗": "clean", "只清洗数据": "clean", "数据清洗": "clean",
    "kg": "kg", "graph": "kg", "知识图谱": "kg", "只重建知识图谱": "kg", "重建知识图谱": "kg",
    "sqlite": "sqlite", "数据库": "sqlite", "只刷新分析库": "sqlite", "刷新分析库": "sqlite",
    "analysis": "analysis", "分析": "analysis", "只运行分析": "analysis", "运行分析": "analysis",
    "full": "full", "完整主线": "full", "全链路": "full", "全部": "full",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hash_value(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def path_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in (path.rglob("*") if path.is_dir() else [path]) if item.is_file()) if path.exists() else []
    for item in files:
        relative = item.relative_to(path).as_posix() if path.is_dir() else item.name
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_contracts() -> dict[str, Any]:
    return json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))


def normalize_goal(goal: str) -> str:
    normalized = "".join(str(goal).strip().lower().split())
    if normalized in GOAL_ALIASES:
        return GOAL_ALIASES[normalized]
    for alias, canonical in GOAL_ALIASES.items():
        if "".join(alias.lower().split()) in normalized:
            return canonical
    raise ValueError(f"unsupported goal: {goal}")


def profile_input(input_path: str | None = None, goals: list[str] | None = None, use_npu: bool = False) -> dict[str, Any]:
    path = Path(input_path).resolve() if input_path else (ROOT / "data" / "raw").resolve()
    files = sorted(item for item in (path.rglob("*") if path.is_dir() else [path]) if item.is_file()) if path.exists() else []
    structured = [item for item in files if item.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}]
    text = [item for item in files if item.suffix.lower() in {".txt", ".md"} or "text" in item.parts]
    manifest_path = path / "data_manifest.json" if path.is_dir() else Path("")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return {
        "input_path": str(path),
        "exists": path.exists(),
        "file_types": sorted({item.suffix.lower() for item in files}),
        "encoding": "utf-8/utf-8-sig (profiled at ingest)",
        "structured_ratio": round(len(structured) / max(1, len(files)), 4),
        "text_file_count": len(text),
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
        "primary_key_candidates": ["patient_id", "visit_id"],
        "foreign_key_candidates": ["patient_id", "visit_id"],
        "missing_rate": "computed by clean operator",
        "duplicate_rate": "computed by clean operator",
        "anomaly_types": ["missing", "duplicate", "type", "range"],
        "contains_text": bool(text),
        "goals": goals or ["full"],
        "needs_graph": any(item in {"full", "kg"} for item in (goals or ["full"])),
        "needs_sqlite": any(item in {"full", "analysis", "sqlite"} for item in (goals or ["full"])),
        "needs_npu": use_npu,
        "data_version": manifest.get("data_version"),
        "patient_count": manifest.get("patient_count"),
        "manifest_content_sha256": manifest.get("content_sha256"),
        "input_hash": path_content_hash(path),
    }


def build_plan(goal: str = "full", input_path: str | None = None, use_npu: bool = False) -> dict[str, Any]:
    canonical = normalize_goal(goal)
    entity = "chronic_entity_extract_model_npu" if use_npu else "chronic_entity_extract"
    relation = "chronic_relation_extract_model_npu" if use_npu else "chronic_relation_extract"
    clean = ["chronic_file_ingest", "chronic_table_clean", "chronic_field_normalize"]
    kg = [*clean, "chronic_text_split", entity, relation, "chronic_triple_validate", "chronic_kg_build"]
    sqlite = [*clean, "chronic_sqlite_loader"]
    analysis = [*sqlite, "chronic_nl2sql_analyze"]
    full = [*kg, "chronic_sqlite_loader", "chronic_nl2sql_analyze", "chronic_report_pack"]
    selected_by_goal = {"clean": clean, "kg": kg, "sqlite": sqlite, "analysis": analysis, "full": full}
    selected = selected_by_goal[canonical]
    selected_set = set(selected)
    contracts = load_contracts()
    defaults = contracts["defaults"]
    nodes = []
    for name in selected:
        dependencies = [entity if item == "ENTITY" else relation if item == "RELATION" else item for item in DEPS[name]]
        dependencies = [item for item in dependencies if item in selected_set]
        contract = {**defaults, **contracts["operators"][name]}
        resources = {**defaults.get("resource_requirements", {}), **contract.get("resource_requirements", {})}
        params: dict[str, Any] = {}
        if name.endswith("_model_npu"):
            params = {"use_npu": True, "fallback": True, "model": "bge-small-zh-v1.5"}
        nodes.append({
            "name": name,
            "depends_on": dependencies,
            "reason": f"selected for {canonical}",
            "state": "planned",
            "operator_version": contract["version"],
            "contract_version": contracts["version"],
            "input_schema": contract["input_schema"],
            "output_schema": contract["output_schema"],
            "input_artifacts": contract["input_artifacts"],
            "output_artifacts": contract["output_artifacts"],
            "preconditions": contract.get("preconditions", []),
            "postconditions": contract.get("postconditions", []),
            "idempotent": contract.get("idempotent", True),
            "supports_resume": contract.get("supports_resume", True),
            "retry_policy": contract["retry_policy"],
            "timeout_seconds": contract["timeout_seconds"],
            "resource": resources,
            "params": params,
        })
    completed: set[str] = set()
    while len(completed) < len(nodes):
        ready = [item["name"] for item in nodes if item["name"] not in completed and set(item["depends_on"]) <= completed]
        if not ready:
            raise ValueError("DAG_CYCLE_OR_MISSING_DEPENDENCY")
        completed.update(ready)
    profile = profile_input(input_path, [canonical], use_npu)
    signature = [(item["name"], item["depends_on"], item["operator_version"], item["params"]) for item in nodes]
    return {
        "status": "planned",
        "goal": canonical,
        "requested_goal": goal,
        "use_npu": use_npu,
        "profile": profile,
        "nodes": nodes,
        "dag_hash": hash_value({"contracts": contracts["version"], "nodes": signature}),
        "skipped": [item for item in DEPS if item not in selected_set],
        "risks": [] if profile["exists"] else ["input_missing"],
        "estimated_resources": {"cpu": max((item["resource"].get("cpu", 1) for item in nodes), default=1), "npu": 1 if use_npu else 0},
        "dry_run": True,
        "validation": {"acyclic": True, "contracts_loaded": len(nodes), "input_exists": profile["exists"]},
    }


def get_run(run_id: str) -> dict[str, Any]:
    path = RUNS / run_id / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_found", "state": "not_found", "run_id": run_id}


class DagEngine:
    def __init__(self, runner: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None):
        self.runner = runner

    @staticmethod
    def _persist(run_dir: Path, report: dict[str, Any], plan: dict[str, Any]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (run_dir / "dag.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(
        self,
        plan: dict[str, Any],
        *,
        dry_run: bool = False,
        resume_run_id: str | None = None,
        resume_from: str | None = None,
        fail_node: str | None = None,
        fail_attempts: int | None = None,
    ) -> dict[str, Any]:
        if dry_run:
            return {**plan, "status": "validated", "state": "validated", "dry_run": True, "writes_performed": False, "cache_hits": []}
        if not plan["profile"]["exists"]:
            raise FileNotFoundError(plan["profile"]["input_path"])
        run_id = resume_run_id or f"dag_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = RUNS / run_id
        previous = get_run(run_id) if resume_run_id else {}
        prior = {item["name"]: item for item in previous.get("nodes", [])}
        runner = self.runner
        if runner is None:
            from .datamate_runner import RealDataMateRunner
            runner = RealDataMateRunner(input_path=plan["profile"]["input_path"], use_npu=plan["use_npu"])
        report: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "state": "running",
            "goal": plan["goal"],
            "dag_hash": plan["dag_hash"],
            "profile": plan["profile"],
            "nodes": [],
            "created_at": previous.get("created_at") or now_iso(),
            "started_at": previous.get("started_at") or now_iso(),
            "updated_at": now_iso(),
            "resume_from": resume_from,
            "degraded": False,
            "cache_hits": [],
        }
        self._persist(run_dir, report, plan)
        force_downstream = False
        injection_counts: dict[str, int] = {}
        for spec in plan["nodes"]:
            name = spec["name"]
            dependency_outputs = [(dep, next((item.get("output_hash") for item in report["nodes"] if item["name"] == dep), None)) for dep in spec["depends_on"]]
            input_hash = hash_value({
                "profile_input_hash": plan["profile"]["input_hash"],
                "dependencies": dependency_outputs,
                "operator_version": spec["operator_version"],
                "params": spec["params"],
            })
            old = prior.get(name)
            if resume_from and name == resume_from:
                force_downstream = True
            cache_valid = bool(
                old and not force_downstream and old.get("state") in {"succeeded", "skipped"}
                and old.get("input_hash") == input_hash
                and old.get("operator_version") == spec["operator_version"]
                and old.get("params") == spec["params"]
                and old.get("output_hash")
            )
            if cache_valid:
                cached = {**old, "state": "skipped", "cache_hit": True, "cache_checked_at": now_iso()}
                report["nodes"].append(cached)
                report["cache_hits"].append(name)
                report["updated_at"] = now_iso()
                self._persist(run_dir, report, plan)
                continue
            item = {**spec, "state": "queued", "input_hash": input_hash, "attempts": 0, "cache_hit": False, "started_at": now_iso()}
            policy = spec["retry_policy"]
            max_attempts = int(policy.get("max_attempts", 1))
            retryable_names = set(policy.get("retryable_errors", []))
            while item["attempts"] < max_attempts:
                item["attempts"] += 1
                item["state"] = "running"
                report["nodes"] = [*report["nodes"], item]
                report["updated_at"] = now_iso()
                self._persist(run_dir, report, plan)
                report["nodes"].pop()
                try:
                    if name == fail_node:
                        injection_counts[name] = injection_counts.get(name, 0) + 1
                        requested = max_attempts if fail_attempts is None else fail_attempts
                        if injection_counts[name] <= requested:
                            raise TimeoutError("acceptance injected retryable failure")
                    result = runner(name, {
                        "input_hash": input_hash,
                        "profile_input_hash": plan["profile"]["input_hash"],
                        "run_id": run_id,
                        "timeout_seconds": spec["timeout_seconds"],
                        "operator_version": spec["operator_version"],
                        "params": spec["params"],
                    })
                    if result.get("status") not in {"success", "degraded", "completed"}:
                        raise RuntimeError(f"operator returned non-success status: {result.get('status')}")
                    item["result"] = result
                    item["output_hash"] = result.get("artifact_hash") or hash_value(result)
                    item["state"] = "succeeded"
                    item["degraded"] = result.get("status") == "degraded" or bool(result.get("fallback_used"))
                    report["degraded"] = report["degraded"] or item["degraded"]
                    break
                except Exception as exc:
                    category = getattr(exc, "category", type(exc).__name__)
                    retryable = bool(getattr(exc, "retryable", category in retryable_names or type(exc).__name__ in retryable_names))
                    item["error"] = {"type": category, "message": str(exc), "retryable": retryable}
                    can_retry = retryable and item["attempts"] < max_attempts
                    item["state"] = "retrying" if can_retry else "failed"
                    if can_retry:
                        time.sleep(float(policy.get("base_delay_seconds", 0.1)) * (2 ** (item["attempts"] - 1)))
                    else:
                        break
            item["ended_at"] = now_iso()
            report["nodes"].append(item)
            report["updated_at"] = now_iso()
            self._persist(run_dir, report, plan)
            if item["state"] == "failed":
                break
        succeeded = len(report["nodes"]) == len(plan["nodes"]) and all(item["state"] in {"succeeded", "skipped"} for item in report["nodes"])
        report["status"] = "succeeded" if succeeded else "failed"
        report["state"] = report["status"]
        if succeeded and hasattr(runner, "finalize"):
            report["materialization"] = runner.finalize(run_id)
        report["ended_at"] = now_iso()
        report["updated_at"] = now_iso()
        self._persist(run_dir, report, plan)
        return report
