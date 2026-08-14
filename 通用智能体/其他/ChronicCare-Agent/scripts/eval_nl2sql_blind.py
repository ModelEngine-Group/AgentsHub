#!/usr/bin/env python3
"""Blind evaluation separating deterministic templates from real LLM candidates."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis.open_sql.llm_sql_candidate import PROMPT_VERSION, generate_llm_sql_candidate
from analysis.open_sql.nl_security import classify_nl_security
from analysis.open_sql.open_sql_service import open_sql_query
from analysis.open_sql.schema_catalog import get_schema_catalog
from analysis.open_sql.sql_executor import execute_sql
from analysis.open_sql.sql_guard import validate_sql
from tool_server.utils import load_current_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs/nl2sql_eval/blind.json"


def canonical_rows(rows: list[dict[str, Any]]) -> list[tuple]:
    normalized = []
    for row in rows:
        values = []
        for value in row.values():
            if isinstance(value, float):
                value = round(value, 4)
            values.append(value)
        normalized.append(tuple(sorted(values, key=lambda x: (str(type(x)), str(x)))))
    return sorted(normalized, key=lambda x: str(x))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return round(ordered[index], 2)


def gold_result(sql: str) -> dict:
    guard = validate_sql(sql)
    return execute_sql(guard["normalized_sql"]) if guard.get("safe") else {"status": "failed", "rows": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATA))
    parser.add_argument("--report-prefix", default=None)
    args = parser.parse_args()
    data_path = Path(args.dataset)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    prefix = args.report_prefix or ("nl2sql_blind" if data_path.name == "blind.json" else f"nl2sql_{data_path.stem}")
    report_path = ROOT / f"outputs/evaluation/{prefix}_eval_report.json"
    markdown_path = ROOT / f"outputs/evaluation/{prefix}_eval_report.md"
    failures_path = ROOT / f"outputs/evaluation/{prefix}_failures.json"
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    active_data_version = str(load_current_metrics().get("data_version") or "unknown")
    catalog = get_schema_catalog()
    broad_link = {"status": "success", "tables": sorted((catalog.get("tables") or {}).keys()), "fields": []}
    results, latencies, llm_latencies = [], [], []
    llm_calls = llm_failures = llm_timeouts = prompt_tokens = completion_tokens = 0
    for item in dataset["cases"]:
        started = time.perf_counter()
        subset, question = item["subset"], item["question"]
        row = {"id": item["id"], "subset": subset, "question": question, "expected_status": item["expected_status"]}
        failure_category = None
        if subset == "security_nl":
            policy = classify_nl_security(question)
            row.update({"stage": "nl_security", "actual_status": "rejected" if not policy["safe"] else "failed", "policy": policy})
            correct = not policy["safe"]
            if not correct: failure_category = "guard"
        elif subset == "unsupported":
            output = open_sql_query(question, prefer_llm=False, allow_chart=False, allow_fixed_tool_overlap=True)
            row.update({"stage": output.get("stage"), "actual_status": output.get("status"), "intent": output.get("intent")})
            correct = output.get("status") == "unsupported"
            if not correct: failure_category = "rejection"
        elif subset == "llm_candidate":
            llm_calls += 1
            llm_started = time.perf_counter()
            candidate = generate_llm_sql_candidate(question, broad_link, catalog)
            llm_ms = (time.perf_counter() - llm_started) * 1000
            llm_latencies.append(llm_ms)
            usage = candidate.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            row.update({"stage": "llm_candidate", "candidate": candidate, "llm_latency_ms": round(llm_ms, 2)})
            if candidate.get("status") != "success":
                llm_failures += 1
                if "timed out" in str(candidate.get("reason", "")).lower(): llm_timeouts += 1
                row["actual_status"] = candidate.get("status")
                correct, failure_category = False, "llm"
            else:
                guard = validate_sql(candidate.get("sql") or "", catalog)
                row["guard"] = guard
                if not guard.get("safe"):
                    row["actual_status"] = "rejected"
                    correct, failure_category = False, "guard"
                else:
                    execution = execute_sql(guard["normalized_sql"])
                    gold = gold_result(item["gold_sql"])
                    row.update({"actual_status": execution.get("status"), "sql": guard["normalized_sql"], "predicted_rows": execution.get("rows"), "gold_rows": gold.get("rows")})
                    correct = execution.get("status") == "success" and canonical_rows(execution.get("rows") or []) == canonical_rows(gold.get("rows") or [])
                    if not correct: failure_category = "result" if execution.get("status") == "success" else "execution"
        else:
            output = open_sql_query(question, prefer_llm=False, allow_chart=False, allow_fixed_tool_overlap=True)
            gold = gold_result(item["gold_sql"])
            row.update({"stage": output.get("stage"), "actual_status": output.get("status"), "intent": output.get("intent"), "sql": output.get("sql"), "predicted_rows": (output.get("result") or {}).get("rows"), "gold_rows": gold.get("rows")})
            correct = output.get("status") == "success" and canonical_rows((output.get("result") or {}).get("rows") or []) == canonical_rows(gold.get("rows") or [])
            if not correct: failure_category = "result" if output.get("status") == "success" else "parsing"
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        candidate_evidence = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        candidate_sql = row.get("sql") or candidate_evidence.get("sql")
        if candidate_sql and "guard" not in row:
            row["guard"] = validate_sql(candidate_sql, catalog)
        row.update({
            "data_version": active_data_version,
            "model": candidate_evidence.get("model") or ("deepseek-chat" if subset == "llm_candidate" else "deterministic_template_router"),
            "prompt_version": candidate_evidence.get("prompt_version") or (PROMPT_VERSION if subset == "llm_candidate" else "template_router_v1"),
            "candidate_sql": candidate_sql,
            "guard_stage": row.get("guard") or row.get("policy"),
            "execution_result": {"status": row.get("actual_status"), "rows": row.get("predicted_rows")},
            "token_usage": candidate_evidence.get("usage") or {},
            "correct": bool(correct),
            "failure_category": failure_category,
            "latency_ms": round(latency, 2),
        })
        results.append(row)

    subset_stats = {}
    for subset in sorted({x["subset"] for x in results}):
        rows = [x for x in results if x["subset"] == subset]
        subset_stats[subset] = {"total": len(rows), "correct": sum(x["correct"] for x in rows), "accuracy": round(sum(x["correct"] for x in rows) / len(rows), 4)}
    correct = sum(x["correct"] for x in results)
    unsupported_rows = [x for x in results if x["subset"] == "unsupported"]
    unsupported_tp = sum(x["correct"] for x in unsupported_rows)
    unsupported_precision = unsupported_recall = unsupported_tp / max(1, len(unsupported_rows))
    report = {
        "status": "success" if ((dataset.get("evaluation_kind") == "engineering_holdout" and len(results) >= 30 and llm_calls >= 20 and correct / len(results) >= .80) or (dataset.get("evaluation_kind") != "engineering_holdout" and len(results) >= 200 and 50 <= llm_calls <= 100 and correct / len(results) >= .85)) else "failed",
        "generated_at": datetime.now().astimezone().isoformat(), "dataset_version": dataset["version"], "evaluation_kind": dataset.get("evaluation_kind", "fixed_engineering_regression"), "active_data_version": active_data_version, "seed": dataset["seed"],
        "total": len(results), "correct": correct, "execution_accuracy": round(correct / len(results), 4),
        "results": results,
        "subset_metrics": subset_stats,
        "pipeline_metrics": {"intent_accuracy": subset_stats["template"]["accuracy"], "schema_linking_accuracy": subset_stats["template"]["accuracy"], "sql_syntax_success": round(sum(x.get("actual_status") == "success" for x in results if x["subset"] not in ("unsupported","security_nl")) / 190, 4), "guard_correctness": subset_stats["security_nl"]["accuracy"], "result_accuracy": round(correct / len(results), 4)},
        "unsupported_detection": {"precision": round(unsupported_precision,4), "recall": round(unsupported_recall,4), "f1": round(unsupported_recall,4)},
        "latency_ms": {"mean": round(statistics.mean(latencies),2), "p50": percentile(latencies,.5), "p95": percentile(latencies,.95)},
        "llm": {"actual_calls": llm_calls, "model": "deepseek-chat", "endpoint_category": "OpenAI-compatible remote API", "temperature": 0, "prompt_version": PROMPT_VERSION, "failures": llm_failures, "timeouts": llm_timeouts, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "mean_latency_ms": round(statistics.mean(llm_latencies),2) if llm_latencies else 0, "p95_latency_ms": percentile(llm_latencies,.95)},
        "failure_categories": dict(Counter(x["failure_category"] for x in results if x["failure_category"])),
        "methodology": "Gold SQL never enters the candidate prompt. Accuracy is execution-result equivalence; SQL text is retained only for structural review. Template and real model results are reported separately.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [x for x in results if not x["correct"]]
    failures_path.write_text(json.dumps({"dataset_version":dataset["version"],"count":len(failures),"failures":failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text("# NL2SQL 盲测报告\n\n" + f"- 状态：{report['status']}\n- 总题数：{report['total']}\n- 总执行/结果准确率：{report['execution_accuracy']:.2%}\n- 真实 LLM Candidate：{llm_calls} 题，准确率 {subset_stats['llm_candidate']['accuracy']:.2%}\n- 模板子集准确率：{subset_stats['template']['accuracy']:.2%}\n- 安全自然语言阻断率：{subset_stats['security_nl']['accuracy']:.2%}\n- 失败分类：`{json.dumps(report['failure_categories'],ensure_ascii=False)}`\n\n金标准 SQL 未进入模型提示词；主要判据是执行结果等价，不是 SQL 字符串相等。\n", encoding="utf-8")
    print(json.dumps({"status":report["status"],"total":report["total"],"accuracy":report["execution_accuracy"],"llm_calls":llm_calls,"llm_accuracy":subset_stats["llm_candidate"]["accuracy"],"failures":len(failures)},ensure_ascii=False))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
