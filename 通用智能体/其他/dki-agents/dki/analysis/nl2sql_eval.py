"""NL2SQL 执行准确率评测（结果集等价比较）。

- evaluate(engine, db_path)：在基础 44 题集上评测（向后兼容，规则引擎主指标）。
- evaluate_set(...)：任意评测集 + 任意 engine（rule/agent；兼容 auto 别名）。
  agent 调用 NL2SQL.query，预测可并发预取，但结果比较始终串行执行。
- evaluate_layered(...)：合并基础集 + 难题集，按 difficulty 分层报 EX（Spider/BIRD 口径）。
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dki.analysis.database import run_sql
from dki.analysis.nl2sql_capabilities import (
    classify_nl2sql_capabilities,
    normalise_capabilities,
)
from dki.analysis.nl2sql_protocol import execute_complete_readonly, validate_gold_sql
from dki.analysis.nl2sql import NL2SQL
from dki.paths import PROJECT_ROOT

EVAL_SET = PROJECT_ROOT / "data" / "nl2sql_eval.jsonl"
HARD_EVAL_SET = PROJECT_ROOT / "data" / "nl2sql_eval_hard.jsonl"
RESULT_ORDER_POLICY = "question_explicit_final_order_v1"


_EXPLICIT_RESULT_ORDER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # The order is explicitly attached to the returned/presented rows.
        r"(?:最终|返回的|输出的|展示的|列出的|查询的)?(?:结果|记录|行|列表|输出)"
        r".{0,24}?(?:按照|按|依照|根据|以).{0,32}?(?:升序|降序|排序|排列|顺序)",
        r"(?:按照|按|依照|根据|以).{0,32}?(?:升序|降序|排序|排列)"
        r".{0,24}?(?:展示|输出|列出)(?:最终)?(?:结果|记录|行|列表)?",
        r"(?:final\s+)?(?:result|output|returned\s+rows?|result\s+rows?)"
        r".{0,32}?(?:order(?:ed)?\s+by|sort(?:ed)?\s+by|ascending|descending)",
        r"(?:return|output|display|list).{0,24}?(?:rows?|results?)"
        r".{0,24}?(?:in|by).{0,12}?(?:ascending|descending|sorted|order)",
    )
)

_GENERAL_ORDER_LANGUAGE = re.compile(
    r"(?:升序|降序|从小到大|从大到小|ascending|descending|"
    r"order(?:ed)?\s+by|sort(?:ed)?\s+by)",
    re.IGNORECASE,
)
_OUTPUT_LANGUAGE = re.compile(
    r"(?:返回|输出|展示|列出|给出|result|output|return|display|list)",
    re.IGNORECASE,
)
_SELECTION_OR_INTERNAL_ORDER = re.compile(
    r"(?:前\s*\d+|后\s*\d+|top\s*\d+|bottom\s*\d+|最高|最低|最大|最小|"
    r"最近|最早|最新|排名|名次|编号|row_number|dense_rank|\brank\b|limit\s+\d+)",
    re.IGNORECASE,
)


def _normalize_result(columns: List[str], rows: List, ordered: bool) -> List:
    """结果集规范化：列名无关；浮点 round(4)；无 ORDER BY 语义时行序无关。"""
    norm_rows = []
    for r in rows:
        norm_rows.append(tuple(
            round(v, 4) if isinstance(v, float) else v for v in r
        ))
    return norm_rows if ordered else sorted(norm_rows, key=repr)


def _question_requires_ordered_result(question: str) -> bool:
    """Whether natural language explicitly constrains final row presentation.

    SQL ``ORDER BY`` can select a Top-K set, define a window value, or merely
    make a gold result deterministic. Those uses must not turn row presentation
    into answer semantics. We therefore require explicit question evidence.
    """

    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    if not normalized:
        return False
    if any(pattern.search(normalized) for pattern in _EXPLICIT_RESULT_ORDER_PATTERNS):
        return True
    return bool(
        _GENERAL_ORDER_LANGUAGE.search(normalized)
        and _OUTPUT_LANGUAGE.search(normalized)
        and not _SELECTION_OR_INTERNAL_ORDER.search(normalized)
    )


def _is_ordered(
    gold_sql: str,
    question: str = "",
    override: Optional[bool] = None,
) -> bool:
    """Return final-result order sensitivity under the versioned policy.

    ``gold_sql`` remains in the signature for backward compatibility and audit
    call sites, but SQL text alone is intentionally not treated as evidence of
    presentation order.
    """

    del gold_sql
    if override is not None:
        if not isinstance(override, bool):
            raise ValueError("result_order_sensitive override must be boolean")
        return override
    return _question_requires_ordered_result(question)


def _compare_result_sets(
    gold_columns: List[str],
    gold_rows: List,
    pred_columns: List[str],
    pred_rows: List,
    *,
    ordered: bool,
    strict: bool = False,
) -> Dict[str, Any]:
    """Compare two executed result sets using the evaluator's legacy contract.

    The non-strict trailing-column tolerance is intentionally preserved exactly:
    when row counts match and the predicted rows are wider, only the gold-width
    prefix is retried.  Strict mode always requires matching result arity.
    """

    gold_norm = _normalize_result(gold_columns, gold_rows, ordered)
    pred_norm = _normalize_result(pred_columns, pred_rows, ordered)
    column_count_matches = len(gold_columns) == len(pred_columns)
    correct = column_count_matches and gold_norm == pred_norm
    used_trailing_column_tolerance = False
    if (
        not strict
        and not correct
        and len(gold_norm) == len(pred_norm)
        and gold_norm
        and len(gold_norm[0]) != len(pred_norm[0])
    ):
        gold_width = len(gold_norm[0])
        pred_cut = [row[:gold_width] for row in pred_norm]
        comparable_pred = pred_cut if ordered else sorted(pred_cut, key=repr)
        correct = gold_norm == comparable_pred
        used_trailing_column_tolerance = correct
    return {
        "correct": correct,
        "gold_normalized": gold_norm,
        "pred_normalized": pred_norm,
        "gold_column_count": len(gold_columns),
        "pred_column_count": len(pred_columns),
        "column_count_matches": column_count_matches,
        "used_trailing_column_tolerance": used_trailing_column_tolerance,
    }


def compare_sql_results(
    gold_sql: str,
    pred_sql: str,
    db_path: Path,
    *,
    question: str = "",
    result_order_sensitive: Optional[bool] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Execute and compare one gold/predicted SQL pair without scoring an item."""

    executor = execute_complete_readonly if strict else run_sql
    gold_columns, gold_rows = executor(gold_sql, db_path)
    if strict:
        validate_gold_sql(pred_sql)
    pred_columns, pred_rows = executor(pred_sql, db_path)
    ordered = _is_ordered(gold_sql, question, result_order_sensitive)
    comparison = _compare_result_sets(
        gold_columns,
        gold_rows,
        pred_columns,
        pred_rows,
        ordered=ordered,
        strict=strict,
    )
    comparison.update(
        result_order_sensitive=ordered,
        result_order_policy=RESULT_ORDER_POLICY,
    )
    return comparison


def _capabilities_for_item(item: Dict[str, Any]) -> tuple[str, ...]:
    """Prefer explicit labels; otherwise classify from question and gold AST."""

    explicit = item.get("capabilities")
    if explicit is None:
        explicit = item.get("capability")
    if explicit is not None:
        return normalise_capabilities(explicit)
    return classify_nl2sql_capabilities(
        str(item.get("question") or ""),
        item.get("gold_sql"),
    )


def _private_query_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Copy private Agent diagnostics without interpreting or publishing them."""

    diagnostics: Dict[str, Any] = {}
    if "llm_meta" in result:
        diagnostics["llm_meta"] = result.get("llm_meta")
    for key in (
        "candidate_diagnostics",
        "candidate_diagnostic",
        "candidate_details",
        "candidates",
        "candidate_records",
        "candidate_sqls",
        "candidate_set",
        "selected_candidate",
        "selected_candidate_index",
        "selected_index",
        "selector_selected_index",
        "selector_diagnostics",
    ):
        if key in result:
            diagnostics[key] = result.get(key)
    return diagnostics


def _base_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item["question"],
        "gold_sql": item["gold_sql"],
        "difficulty": item.get("difficulty"),
        "category": item.get("category"),
        "capabilities": _capabilities_for_item(item),
    }


def _load_items(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score_item(nl: NL2SQL, item: Dict[str, Any], engine: str, db_path: Path, *, strict: bool = False) -> Dict[str, Any]:
    """对单题打分。engine='rule' 只认规则命中；engine='agent' 调产品 query。"""
    q, gold_sql = item["question"], item["gold_sql"]
    if engine == "auto":
        engine = "agent"
    if engine not in {"rule", "agent"}:
        raise ValueError(f"unsupported NL2SQL evaluation engine: {engine}")
    record = _base_record(item)
    if engine == "rule":
        # 仅规则引擎（绝不触发 LLM，保证确定性与离线可评测）
        pred_sql = nl.rule_sql(q)
        used = "rule" if pred_sql else "none"
        if not pred_sql:
            record.update(
                engine=used,
                correct=False,
                reason="规则引擎未命中",
                pred_sql="",
            )
            return record
    else:
        res = nl.query(q)
        record.update(_private_query_diagnostics(res))
        pred_sql, used = res.get("sql", ""), res.get("engine")
        if not pred_sql:
            record.update(
                pred_sql="",
                engine=used or "none",
                correct=False,
                reason=res.get("disposition") or res.get("error") or "agent_abstained",
                terminal=bool(res.get("terminal", False)),
                llm_failure_stages=tuple(
                    (res.get("llm_meta") or {}).get("failure_stages", ())
                ),
            )
            return record
    record.update(pred_sql=pred_sql, engine=used)
    try:
        comparison = compare_sql_results(
            gold_sql,
            pred_sql,
            db_path,
            question=q,
            result_order_sensitive=item.get("result_order_sensitive"),
            strict=strict,
        )
        record.update(
            result_order_sensitive=comparison["result_order_sensitive"],
            result_order_policy=comparison["result_order_policy"],
        )
        if strict:
            record.update(
                gold_column_count=comparison["gold_column_count"],
                pred_column_count=comparison["pred_column_count"],
                column_count_matches=comparison["column_count_matches"],
            )
            if not comparison["column_count_matches"]:
                record["reason"] = "strict_result_column_count_mismatch"
        record["correct"] = comparison["correct"]
        if not comparison["correct"]:
            record.update(
                gold_head=comparison["gold_normalized"][:3],
                pred_head=comparison["pred_normalized"][:3],
            )
    except Exception as e:
        record.update(correct=False, reason=str(e))
    return record


def _score_items(
    items: List[Dict[str, Any]],
    engine: str,
    db_path: Path,
    nl: Optional[NL2SQL],
    *,
    strict: bool,
    nl_factory: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Score every item, optionally prefetching predictions concurrently.

    Agent evaluation is dominated by per-item LLM round trips, so when
    ``NL2SQL_EVAL_WORKERS`` exceeds 1 the *prediction* phase runs in a thread
    pool.  Scoring itself stays sequential on purpose: ``bounded_execution``
    enforces a wall-clock budget per query, and running comparisons
    concurrently makes healthy queries miss that budget and be recorded as
    ``interrupted`` -- silently turning correct answers into failures.

    ``NL2SQL`` also keeps per-query mutable state, so each worker builds its own
    engine via ``nl_factory``.  Without a factory we cannot isolate that state
    and therefore stay fully sequential.  Results always keep input order.
    """

    effective_engine = "agent" if engine == "auto" else engine
    try:
        workers = int(os.environ.get("NL2SQL_EVAL_WORKERS", "1"))
    except ValueError:
        workers = 1
    workers = max(1, min(workers, len(items)))

    engine_instance = nl or NL2SQL(db_path=db_path)
    if workers == 1 or nl_factory is None or effective_engine != "agent":
        return [
            _score_item(engine_instance, it, effective_engine, db_path, strict=strict)
            for it in items
        ]

    import threading
    from concurrent.futures import ThreadPoolExecutor

    local = threading.local()

    def predict(item: Dict[str, Any]):
        """Warm the LLM result for one item; never scores it."""
        instance = getattr(local, "nl", None)
        if instance is None:
            instance = nl_factory()
            local.nl = instance
        try:
            return instance.query(item["question"])
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        prefetched = list(pool.map(predict, items))

    # Replay each prediction through the untouched sequential scorer so the
    # recorded verdict is produced exactly as it is without concurrency.
    details: List[Dict[str, Any]] = []
    for item, cached in zip(items, prefetched):
        if cached is None:
            details.append(_score_item(engine_instance, item, effective_engine, db_path, strict=strict))
            continue
        details.append(
            _score_item(
                _ReplayEngine(engine_instance, item["question"], cached),
                item,
                effective_engine,
                db_path,
                strict=strict,
            )
        )
    return details


class _ReplayEngine:
    """Serve one already-computed ``query`` result, else defer to the engine.

    Keeping the scorer untouched means the strict contract, ordering rules and
    record shape are byte-for-byte what a sequential run produces.
    """

    def __init__(self, engine: NL2SQL, question: str, result: Dict[str, Any]) -> None:
        self._engine = engine
        self._question = question
        self._result = result

    def query(self, question: str) -> Dict[str, Any]:
        if question == self._question:
            return self._result
        return self._engine.query(question)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


def _aggregate_breakdown(
    details: List[Dict[str, Any]],
    key: str,
    *,
    multi_value: bool = False,
    include_coverage: bool = True,
    include_unspecified: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate accuracy and optional coverage for one report dimension."""

    buckets: Dict[str, Dict[str, int]] = {}
    for detail in details:
        raw_values = detail.get(key)
        if multi_value:
            if isinstance(raw_values, str):
                values = (raw_values,)
            else:
                values = tuple(raw_values or ())
        elif raw_values in (None, ""):
            values = ("unspecified",) if include_unspecified else ()
        else:
            values = (raw_values,)
        for raw_value in values:
            value = str(raw_value).strip() or "unspecified"
            bucket = buckets.setdefault(
                value,
                {"n": 0, "correct": 0, "answered": 0},
            )
            bucket["n"] += 1
            bucket["correct"] += 1 if detail.get("correct") else 0
            bucket["answered"] += (
                1 if detail.get("engine") not in (None, "none") else 0
            )
    result: Dict[str, Dict[str, Any]] = {}
    for value in sorted(buckets):
        bucket = buckets[value]
        summary: Dict[str, Any] = {
            "n": bucket["n"],
            "correct": bucket["correct"],
            "accuracy": round(bucket["correct"] / max(1, bucket["n"]), 4),
        }
        if include_coverage:
            summary.update(
                answered=bucket["answered"],
                coverage=round(bucket["answered"] / max(1, bucket["n"]), 4),
            )
        result[value] = summary
    return result


def _candidate_payloads(query_result: Dict[str, Any]) -> tuple[Any, ...]:
    for key in (
        "candidate_diagnostics",
        "candidate_diagnostic",
        "candidate_details",
        "candidates",
        "candidate_records",
        "candidate_sqls",
        "candidate_set",
    ):
        value = query_result.get(key)
        if isinstance(value, (list, tuple)):
            return tuple(value)
    meta = query_result.get("llm_meta")
    if isinstance(meta, dict):
        for key in (
            "candidate_diagnostics",
            "candidate_diagnostic",
            "candidates",
            "candidate_records",
        "candidate_sqls",
            "candidate_set",
        ):
            value = meta.get(key)
            if isinstance(value, (list, tuple)):
                return tuple(value)
    return ()


def _candidate_sql(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("sql", "candidate_sql", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _selected_candidate_index(
    query_result: Dict[str, Any],
    candidates: tuple[Any, ...],
) -> Optional[int]:
    containers = [query_result]
    meta = query_result.get("llm_meta")
    if isinstance(meta, dict):
        containers.append(meta)
    for container in containers:
        for key in (
            "selected_candidate_index",
            "selected_index",
            "selector_selected_index",
        ):
            value = container.get(key)
            if type(value) is int and 0 <= value < len(candidates):
                return value
    selected_sql = str(query_result.get("sql") or "").strip()
    if selected_sql:
        for index, payload in enumerate(candidates):
            if _candidate_sql(payload) == selected_sql:
                return index
    return None


def diagnose_candidate_set_with_gold(
    query_result: Dict[str, Any],
    gold_sql: str,
    db_path: Path,
    *,
    question: str = "",
    result_order_sensitive: Optional[bool] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Development-only oracle diagnostic for an explicitly supplied gold SQL.

    Normal evaluation never calls this helper.  Gold-derived candidate verdicts
    therefore cannot appear in regular ``evaluate_set`` output unless development
    code deliberately opts in by calling this function.
    """

    candidates = _candidate_payloads(query_result)
    selected_index = _selected_candidate_index(query_result, candidates)
    selected_sql = str(query_result.get("sql") or "").strip()
    valid_count = 0
    unique_candidates: set[tuple[str, Any]] = set()
    candidate_hits: List[bool] = []
    candidate_validity: List[bool] = []

    for payload in candidates:
        sql = _candidate_sql(payload)
        if not sql:
            candidate_validity.append(False)
            candidate_hits.append(False)
            continue
        try:
            if strict:
                validate_gold_sql(sql)
            executor = execute_complete_readonly if strict else run_sql
            executor(sql, db_path)
        except Exception:
            candidate_validity.append(False)
            candidate_hits.append(False)
            continue
        valid_count += 1
        try:
            candidate_result = compare_sql_results(
                gold_sql,
                sql,
                db_path,
                question=question,
                result_order_sensitive=result_order_sensitive,
                strict=strict,
            )
            result_signature = repr(candidate_result["pred_normalized"])
            hit = bool(candidate_result["correct"])
        except Exception:
            result_signature = re.sub(
                r"\s+", " ", sql.strip().rstrip(";")
            ).casefold()
            hit = False
        unique_candidates.add(("result", result_signature))
        candidate_validity.append(True)
        candidate_hits.append(hit)

    oracle_hit = any(candidate_hits)
    if selected_index is not None:
        selected_hit = candidate_hits[selected_index]
    elif selected_sql:
        try:
            selected_hit = bool(
                compare_sql_results(
                    gold_sql,
                    selected_sql,
                    db_path,
                    question=question,
                    result_order_sensitive=result_order_sensitive,
                    strict=strict,
                )["correct"]
            )
        except Exception:
            selected_hit = False
    else:
        selected_hit = False

    return {
        "oracle_hit": oracle_hit,
        "selected_hit": selected_hit,
        "selector_regret": bool(oracle_hit and not selected_hit),
        "candidate_count": len(candidates),
        "valid_candidate_count": valid_count,
        "unique_candidate_count": len(unique_candidates),
        "selected_candidate_index": selected_index,
        "candidate_validity": tuple(candidate_validity),
        "candidate_hits": tuple(candidate_hits),
    }


# Explicit name makes the gold boundary obvious at call sites.
development_candidate_set_diagnostics = diagnose_candidate_set_with_gold


def evaluate_set(items: List[Dict[str, Any]], engine: str, db_path: Path,
                 nl: Optional[NL2SQL] = None, *, strict: bool = False,
                 nl_factory: Optional[Any] = None) -> Dict[str, Any]:
    details = _score_items(
        items, engine, db_path, nl, strict=strict, nl_factory=nl_factory
    )
    n_correct = sum(1 for d in details if d.get("correct"))
    n_answered = sum(1 for d in details if d.get("engine") not in (None, "none"))
    by_difficulty = _aggregate_breakdown(
        details,
        "difficulty",
        include_coverage=False,
    )
    by_category = _aggregate_breakdown(
        details,
        "category",
        include_unspecified=False,
    )
    by_capability = _aggregate_breakdown(
        details,
        "capabilities",
        multi_value=True,
    )
    return {
        "engine": engine, "n": len(items), "n_correct": n_correct,
        "accuracy": round(n_correct / max(1, len(items)), 4),
        "n_answered": n_answered,
        "n_abstained": len(items) - n_answered,
        "coverage": round(n_answered / max(1, len(items)), 4),
        "execution_accuracy_answered": round(n_correct / max(1, n_answered), 4),
        "strict_result_contract": strict,
        "result_order_policy": RESULT_ORDER_POLICY,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "by_capability": by_capability,
        "details": details,
    }


def evaluate(engine: str = "rule", db_path: Path = None) -> Dict[str, Any]:
    """基础 44 题集评测（向后兼容，规则引擎主指标）。"""
    assert db_path is not None, "db_path 必填"
    return evaluate_set(_load_items(EVAL_SET), engine, db_path)


def evaluate_layered(engine: str = "auto", db_path: Path = None,
                     include_hard: bool = True) -> Dict[str, Any]:
    """合并基础集 + 难题集，按 difficulty 分层报 EX。

    engine='rule' 报规则引擎在全集上的命中率（难题多为规则未覆盖，属预期）；
    engine='auto'（兼容别名）报规则+LLM SOTA 管线的综合执行准确率（需配置 LLM）。
    """
    assert db_path is not None, "db_path 必填"
    base = _load_items(EVAL_SET)
    for it in base:
        it.setdefault("difficulty", "easy")
        it.setdefault("category", "basic")
    items = list(base)
    if include_hard:
        items += _load_items(HARD_EVAL_SET)
    return evaluate_set(items, engine, db_path)
