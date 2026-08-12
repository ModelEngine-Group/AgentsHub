# -*- coding: utf-8 -*-
"""
任务二知识图谱构建服务。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from core.llm_client import LLMClient
from core.medical_extraction_service import (
    ExtractionBundle,
    extract_medical_knowledge,
    normalize_backend,
)
from core.task2_cascade import (
    apply_cascade_merge,
    build_gap_review_candidates,
    count_skipped_offline_candidates,
    dedupe_review_candidates,
    prepare_cascade_targets,
    select_auto_accepted_gap_candidate_ids,
)
from core.task2_verifier import extract_gap_facts_batch, review_candidates_parallel
from mcp_server.config import KG_DB, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from mcp_server.datamate.resolver import (
    _task2_read_datamate_dataset,
    _task2_resolve_datamate_dataset,
)
from mcp_server.kg.analytics_refresh import refresh_task3_analytics
from mcp_server.kg.persistence import ensure_source, persist_triples
from mcp_server.kg.schema import _task2_ensure_kg_schema
from mcp_server.shared.parsing import parse_files
from mcp_server.shared.sqlite_utils import connect_kg
from mcp_server.task2.reporting import (
    format_stage_duration,
    summarize_analytics_refresh,
    summarize_source_files,
)
from mcp_server.task2.selection import select_balanced_records


_LLM: LLMClient | None = None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _get_task2_llm() -> LLMClient:
    global _LLM
    if _LLM is None:
        _LLM = LLMClient(base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY or None)
    return _LLM


def _llm_for_backend(backend: str) -> LLMClient | None:
    selected = normalize_backend(backend)
    return _get_task2_llm() if selected in {"llm", "hybrid"} else None


def _backend_label(backend: str) -> str:
    return {
        "offline": "本地知识抽取链",
        "hybrid": "本地抽取增强链",
        "llm": "语义抽取链",
    }.get(backend, "本地知识抽取链")


def _triple_value(triple, key: str, default=""):
    if isinstance(triple, dict):
        return triple.get(key, default)
    return getattr(triple, key, default)


def _classify_generated_triples(triples: list) -> dict[str, int]:
    """Classify generated triples without writing quality-issue rows."""

    counts = {"accepted_high": 0, "candidate": 0, "rejected": 0, "invalid": 0}
    for triple in triples:
        reliability = str(_triple_value(triple, "reliability_level") or "").strip().lower()
        method = str(
            _triple_value(triple, "extraction_method")
            or _triple_value(triple, "method")
            or "unknown"
        ).strip().lower()
        if reliability not in {"high", "medium", "low"}:
            reliability = "medium" if method == "llm" else "low"
        if reliability == "high":
            counts["accepted_high"] += 1
        elif reliability == "medium":
            counts["candidate"] += 1
        else:
            counts["rejected"] += 1
    return counts


_MAX_BATCH_REVIEW_CANDIDATES = 256
_MAX_BATCH_GAP_SEGMENTS = 96


def _round_robin_candidates(per_record: list[dict], source_key: str, kind: str) -> list:
    """Interleave records so a batch cap cannot starve later records."""

    groups = [
        [candidate for candidate in item[source_key] if candidate.kind == kind]
        for item in per_record
    ]
    result: list = []
    offset = 0
    while any(offset < len(group) for group in groups):
        for group in groups:
            if offset < len(group):
                result.append(group[offset])
        offset += 1
    return result


def _cascade_max_gap_segments_total() -> int:
    """Bound the total number of gap segments sent to the external model.

    The per-record cascade already has a gap limit.  A second, run-level limit
    is needed for a dataset pipeline: otherwise a large dataset can create an
    unbounded number of model requests even though every individual record is
    small.  Records are interleaved so the budget does not starve later files.
    """

    raw = os.getenv(
        "CCF_TASK2_CASCADE_MAX_GAP_SEGMENTS_TOTAL",
        str(_MAX_BATCH_GAP_SEGMENTS),
    )
    try:
        return max(1, min(512, int(raw)))
    except ValueError:
        return _MAX_BATCH_GAP_SEGMENTS


def _cascade_max_review_candidates() -> int:
    """Bound the candidate review queue for one dataset pipeline run."""

    raw = os.getenv(
        "CCF_TASK2_CASCADE_MAX_REVIEW_CANDIDATES",
        str(_MAX_BATCH_REVIEW_CANDIDATES),
    )
    try:
        return max(1, min(512, int(raw)))
    except ValueError:
        return _MAX_BATCH_REVIEW_CANDIDATES


def _round_robin_gap_segments(per_record: list[dict], limit: int) -> dict[int, list]:
    """Select a bounded, record-balanced subset of gap segments."""

    selected: dict[int, list] = {item["index"]: [] for item in per_record}
    groups = [(item["index"], item["gap_segments"]) for item in per_record]
    chosen = 0
    offset = 0
    while chosen < limit and any(offset < len(segments) for _, segments in groups):
        for record_index, segments in groups:
            if chosen >= limit:
                break
            if offset < len(segments):
                selected[record_index].append(segments[offset])
                chosen += 1
        offset += 1
    return selected


def _cascade_gap_workers(record_count: int) -> int:
    raw = os.getenv("CCF_TASK2_CASCADE_GAP_WORKERS", "8")
    try:
        configured = int(raw)
    except ValueError:
        configured = 8
    return max(1, min(8, configured, record_count))


def _cascade_gap_batch_size() -> int:
    raw = os.getenv("CCF_TASK2_CASCADE_GAP_BATCH_SIZE", "6")
    try:
        configured = int(raw)
    except ValueError:
        configured = 6
    return max(1, min(8, configured))


def _batch_cascade_precompute(
    selected_records: list[dict],
    llm: LLMClient | None,
    kg_db_path: str,
) -> dict[int, ExtractionBundle]:
    """Run hybrid extraction in two phases with record-scoped candidates."""

    phase_start = time.perf_counter()
    per_record: list[dict] = []
    for record_index, record in enumerate(selected_records):
        text = record.get("text", "")
        if not text.strip():
            continue
        bundle = extract_medical_knowledge(
            text,
            backend="offline",
            kg_db_path=kg_db_path,
            apply_offline_gate=False,
        )
        scope = f"r{record_index}"
        gap_segments, candidates = prepare_cascade_targets(
            text,
            bundle.entities,
            bundle.relations,
            candidate_scope=scope,
        )
        per_record.append(
            {
                "index": record_index,
                "text": text,
                "offline_bundle": bundle,
                "scope": scope,
                "gap_segments": gap_segments,
                "offline_candidates": candidates,
                "offline_review_skipped_count": count_skipped_offline_candidates(
                    bundle.entities,
                    bundle.relations,
                    candidates,
                    candidate_scope=scope,
                ),
                "gap_entities": [],
                "gap_relations": [],
                "gap_candidates": [],
                "reviewable_gap_candidates": [],
                "auto_accepted_candidate_ids": set(),
                "gap_error": "",
                "gap_budget_skipped_count": 0,
                "review_budget_skipped_count": 0,
            }
        )

    if not per_record:
        return {}

    gap_results: dict[int, tuple[list, list]] = {}
    gap_segments_for_llm = _round_robin_gap_segments(
        per_record,
        _cascade_max_gap_segments_total(),
    )
    for item in per_record:
        skipped = len(item["gap_segments"]) - len(gap_segments_for_llm[item["index"]])
        if skipped > 0:
            item["gap_budget_skipped_count"] = skipped
    gap_records = [
        (
            item["index"],
            item["text"],
            gap_segments_for_llm[item["index"]],
            item["offline_bundle"].entities,
        )
        for item in per_record
        if gap_segments_for_llm[item["index"]]
    ]
    if llm is not None and gap_records:
        batch_size = _cascade_gap_batch_size()
        gap_batches = [
            gap_records[offset : offset + batch_size]
            for offset in range(0, len(gap_records), batch_size)
        ]
        max_workers = _cascade_gap_workers(len(gap_batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(extract_gap_facts_batch, batch, llm): batch
                for batch in gap_batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    gap_results.update(future.result())
                except Exception as exc:
                    error = (
                        f"gap extraction failed: {type(exc).__name__}: {str(exc)[:200]}"
                    )
                    for record_index, _, _, _ in batch:
                        for item in per_record:
                            if item["index"] == record_index:
                                item["gap_error"] = (
                                    f"{item['gap_error']}; {error}"
                                    if item["gap_error"]
                                    else error
                                )
                                break
    else:
        for item in per_record:
            if item["gap_segments"]:
                item["gap_error"] = "LLM client is not configured"

    for item in per_record:
        gap_entities, gap_relations = gap_results.get(item["index"], ([], []))
        item["gap_entities"] = gap_entities
        item["gap_relations"] = gap_relations
        item["gap_candidates"] = build_gap_review_candidates(
            gap_entities,
            gap_relations,
            item["gap_segments"],
            candidate_scope=item["scope"],
        )
        item["auto_accepted_candidate_ids"] = select_auto_accepted_gap_candidate_ids(
            item["gap_candidates"]
        )
        item["reviewable_gap_candidates"] = [
            candidate
            for candidate in item["gap_candidates"]
            if candidate.candidate_id not in item["auto_accepted_candidate_ids"]
        ]

    # Relation augmentation has the largest recall gap.  Interleave records in
    # every tier so early records cannot consume the whole batch safety cap.
    # Medium/high offline facts never enter this queue because the merge policy
    # preserves them regardless of a review decision.
    all_review_candidates = dedupe_review_candidates(
        _round_robin_candidates(per_record, "reviewable_gap_candidates", "relation")
        + _round_robin_candidates(per_record, "offline_candidates", "relation")
        + _round_robin_candidates(per_record, "reviewable_gap_candidates", "entity")
        + _round_robin_candidates(per_record, "offline_candidates", "entity")
    )
    review_queue = all_review_candidates[:_cascade_max_review_candidates()]
    reviewed_candidate_ids = {candidate.candidate_id for candidate in review_queue}
    all_review_candidate_ids = {candidate.candidate_id for candidate in all_review_candidates}
    for item in per_record:
        record_candidate_ids = {
            candidate.candidate_id
            for candidate in [
                *item["reviewable_gap_candidates"],
                *item["offline_candidates"],
            ]
        }
        item["review_budget_skipped_count"] = len(
            (record_candidate_ids & all_review_candidate_ids) - reviewed_candidate_ids
        )
    decisions: dict = {}
    review_error = ""
    review_skipped = len(all_review_candidates) - len(review_queue)
    if review_queue and llm is not None:
        try:
            decisions = review_candidates_parallel(
                llm,
                review_queue,
                batch_size=64,
                max_workers=min(4, _cascade_gap_workers(len(per_record))),
            )
        except Exception as exc:
            error = f"candidate review failed: {type(exc).__name__}: {str(exc)[:200]}"
            review_error = f"{review_error}; {error}" if review_error else error
            reviewed_candidate_ids = set()
    elif review_queue:
        error = "LLM client is not configured"
        review_error = f"{review_error}; {error}" if review_error else error
        reviewed_candidate_ids = set()

    elapsed_per_record = (time.perf_counter() - phase_start) / max(1, len(per_record))
    result: dict[int, ExtractionBundle] = {}
    for item in per_record:
        offline_bundle = item["offline_bundle"]
        try:
            cascade = apply_cascade_merge(
                text=item["text"],
                entities=offline_bundle.entities,
                relations=offline_bundle.relations,
                gap_segments=item["gap_segments"],
                review_candidates=item["offline_candidates"],
                decisions=decisions,
                gap_entities=item["gap_entities"],
                gap_relations=item["gap_relations"],
                candidate_scope=item["scope"],
                reviewed_candidate_ids=reviewed_candidate_ids,
                auto_accepted_candidate_ids=item["auto_accepted_candidate_ids"],
                offline_review_skipped_count=item["offline_review_skipped_count"],
            )
            errors = [value for value in (item["gap_error"], review_error) if value]
            result[item["index"]] = ExtractionBundle(
                entities=cascade.entities,
                relations=cascade.relations,
                triples=cascade.triples,
                backend="hybrid",
                elapsed_seconds=round(elapsed_per_record, 4),
                llm_error="; ".join(errors),
                gap_segment_count=cascade.gap_segment_count,
                gap_candidate_count=cascade.gap_candidate_count,
                reviewed_candidate_count=cascade.reviewed_candidate_count,
                auto_accepted_candidate_count=cascade.auto_accepted_candidate_count,
                review_skipped_candidate_count=cascade.review_skipped_candidate_count,
                offline_filtered_candidate_count=cascade.offline_filtered_candidate_count,
                rejected_candidate_count=cascade.rejected_candidate_count,
                llm_added_count=cascade.llm_added_count,
                llm_added_entity_count=cascade.llm_added_entity_count,
                llm_added_relation_count=cascade.llm_added_relation_count,
                gap_budget_skipped_count=item["gap_budget_skipped_count"],
                review_budget_skipped_count=item["review_budget_skipped_count"],
            )
        except Exception as exc:
            fallback = extract_medical_knowledge(
                item["text"],
                backend="offline",
                kg_db_path=kg_db_path,
            )
            result[item["index"]] = ExtractionBundle(
                entities=fallback.entities,
                relations=fallback.relations,
                triples=fallback.triples,
                backend="hybrid",
                elapsed_seconds=round(
                    elapsed_per_record + fallback.elapsed_seconds, 4
                ),
                llm_error=f"cascade merge failed: {type(exc).__name__}: {str(exc)[:240]}",
                offline_filtered_candidate_count=fallback.offline_filtered_candidate_count,
            )
    return result


def run_kg_pipeline_service(
    *,
    dataset_id: str,
    task_name: str = "",
    max_records: int = 0,
    dry_run: bool = False,
    persist: bool = True,
    refresh_analytics: bool = False,
    backend: str = "offline",
) -> dict:
    """执行任务二知识图谱流水线并返回结构化 MCP 结果。"""
    t0 = time.time()
    selected_backend = normalize_backend(backend)
    backend_label = _backend_label(selected_backend)
    progress_log: list[dict] = []
    tool_call_trace: list[dict] = []
    persistence_enabled = bool(persist and not dry_run)
    analytics_enabled = bool(refresh_analytics and not dry_run)

    if not dataset_id:
        return {"status": "error", "error": "需要 dataset_id"}

    stage_start = time.time()
    progress_log.append({"step": "resolve", "label": "数据集解析", "status": "running", "time": _now()})
    try:
        ds, _ = _task2_resolve_datamate_dataset(dataset_id)
        progress_log[-1].update(
            {
                "status": "done",
                "dataset_name": ds.get("name", ""),
                "duration_seconds": round(time.time() - stage_start, 4),
            }
        )
    except Exception as exc:
        progress_log[-1].update({"status": "error", "duration_seconds": round(time.time() - stage_start, 4)})
        return {"status": "error", "error": str(exc), "progress_log": progress_log}

    stage_start = time.time()
    progress_log.append({"step": "read_files", "label": "文件读取", "status": "running", "time": _now()})
    try:
        _, files = _task2_read_datamate_dataset(ds["id"])
        progress_log[-1].update(
            {"status": "done", "file_count": len(files), "duration_seconds": round(time.time() - stage_start, 4)}
        )
    except Exception as exc:
        progress_log[-1].update({"status": "error", "duration_seconds": round(time.time() - stage_start, 4)})
        return {"status": "error", "error": str(exc), "progress_log": progress_log}

    stage_start = time.time()
    progress_log.append({"step": "parse", "label": "记录解析", "status": "running", "time": _now()})
    records, stats = parse_files(files)
    selected_records = select_balanced_records(records, int(max_records or 0))
    progress_log[-1].update(
        {
            "status": "done",
            "record_count": len(records),
            "selected_record_count": len(selected_records),
            "format_stats": stats,
            "duration_seconds": round(time.time() - stage_start, 4),
        }
    )
    tool_call_trace.append(
        {
            "tool": "parse_files",
            "input_count": len(files),
            "output_count": len(records),
            "selected_count": len(selected_records),
        }
    )

    conn = connect_kg() if persistence_enabled else None
    if conn:
        _task2_ensure_kg_schema(conn)

    source_id = None
    record_results: list[dict] = []
    generated_triple_count = 0
    inserted_triple_count = 0
    candidate_triple_count = 0
    rejected_triple_count = 0
    accepted_high_triple_count = 0
    deduplicated_triple_count = 0
    invalid_triple_count = 0
    extraction_errors: list[dict] = []
    extraction_elapsed_total = 0.0
    persistence_elapsed_total = 0.0
    llm_degraded_count = 0
    cascade_gap_segment_count = 0
    cascade_gap_candidate_count = 0
    cascade_reviewed_candidate_count = 0
    cascade_auto_accepted_candidate_count = 0
    cascade_review_skipped_candidate_count = 0
    cascade_offline_filtered_candidate_count = 0
    cascade_rejected_candidate_count = 0
    cascade_llm_added_count = 0
    cascade_llm_added_entity_count = 0
    cascade_llm_added_relation_count = 0
    cascade_gap_budget_skipped_count = 0
    cascade_review_budget_skipped_count = 0
    llm = _llm_for_backend(selected_backend)

    precomputed_bundles: dict[int, ExtractionBundle] = {}
    if selected_backend == "hybrid" and llm is not None:
        precomputed_bundles = _batch_cascade_precompute(selected_records, llm, KG_DB)

    stage_start = time.time()
    progress_log.append({"step": "extract", "label": "实体关系三元组生成", "status": "running", "time": _now()})
    for index, record in enumerate(selected_records):
        text = record.get("text", "")
        if not text.strip():
            continue
        try:
            if precomputed_bundles:
                bundle = precomputed_bundles[index]
            else:
                bundle = extract_medical_knowledge(text, backend=selected_backend, kg_db_path=KG_DB, llm=llm)
            extraction_elapsed_total += bundle.elapsed_seconds
            if bundle.llm_error:
                llm_degraded_count += 1
            cascade_gap_segment_count += bundle.gap_segment_count
            cascade_gap_candidate_count += bundle.gap_candidate_count
            cascade_reviewed_candidate_count += bundle.reviewed_candidate_count
            cascade_auto_accepted_candidate_count += bundle.auto_accepted_candidate_count
            cascade_review_skipped_candidate_count += bundle.review_skipped_candidate_count
            cascade_offline_filtered_candidate_count += bundle.offline_filtered_candidate_count
            cascade_rejected_candidate_count += bundle.rejected_candidate_count
            cascade_llm_added_count += bundle.llm_added_count
            cascade_llm_added_entity_count += bundle.llm_added_entity_count
            cascade_llm_added_relation_count += bundle.llm_added_relation_count
            cascade_gap_budget_skipped_count += bundle.gap_budget_skipped_count
            cascade_review_budget_skipped_count += bundle.review_budget_skipped_count

            triples = bundle.triples
            generated_triple_count += len(triples)
            record_result = {
                "record": index,
                "source_file": record.get("source_file", ""),
                "backend": bundle.backend,
                "entities": len(bundle.entities),
                "relations": len(bundle.relations),
                "triples": len(triples),
                "elapsed_seconds": bundle.elapsed_seconds,
                "gap_segments": bundle.gap_segment_count,
                "gap_candidates": bundle.gap_candidate_count,
                "reviewed_candidates": bundle.reviewed_candidate_count,
                "auto_accepted_candidates": bundle.auto_accepted_candidate_count,
                "review_skipped_candidates": bundle.review_skipped_candidate_count,
                "offline_filtered_candidates": bundle.offline_filtered_candidate_count,
                "rejected_candidates": bundle.rejected_candidate_count,
                "llm_added_facts": bundle.llm_added_count,
                "llm_added_entities": bundle.llm_added_entity_count,
                "llm_added_relations": bundle.llm_added_relation_count,
                "inserted_triples": 0,
                "candidate_triples": 0,
                "rejected_triples": 0,
                "accepted_high_triples": 0,
                "deduplicated_triples": 0,
                "invalid_triples": 0,
            }
            if bundle.llm_error:
                record_result["llm_error"] = bundle.llm_error

            if persistence_enabled and conn and triples:
                if source_id is None:
                    source_id = ensure_source(conn, ds, len(selected_records))
                persist_start = time.time()
                persistence_result = persist_triples(
                    conn,
                    triples,
                    record.get("source_file", ""),
                    source_id,
                    source_record_id=record.get("record_id", ""),
                    return_details=True,
                    include_quality_metrics=True,
                )
                persistence_elapsed_total += time.time() - persist_start
                record_result["inserted_triples"] = persistence_result["inserted"]
                record_result["candidate_triples"] = persistence_result["candidate"]
                record_result["rejected_triples"] = persistence_result["rejected"]
                record_result["accepted_high_triples"] = persistence_result.get("accepted_high", 0)
                record_result["deduplicated_triples"] = persistence_result.get("deduplicated", 0)
                record_result["invalid_triples"] = persistence_result.get("invalid", 0)
                inserted_triple_count += persistence_result["inserted"]
                candidate_triple_count += persistence_result["candidate"]
                rejected_triple_count += persistence_result["rejected"]
                accepted_high_triple_count += persistence_result.get("accepted_high", 0)
                deduplicated_triple_count += persistence_result.get("deduplicated", 0)
                invalid_triple_count += persistence_result.get("invalid", 0)
            else:
                quality_counts = _classify_generated_triples(triples)
                record_result["candidate_triples"] = quality_counts["candidate"]
                record_result["rejected_triples"] = quality_counts["rejected"]
                record_result["accepted_high_triples"] = quality_counts["accepted_high"]
                record_result["invalid_triples"] = quality_counts["invalid"]
                candidate_triple_count += quality_counts["candidate"]
                rejected_triple_count += quality_counts["rejected"]
                accepted_high_triple_count += quality_counts["accepted_high"]
                invalid_triple_count += quality_counts["invalid"]

            record_results.append(record_result)
            tool_call_trace.append({"tool": f"record_{index}", **record_result})
        except Exception as exc:
            error = {"record": index, "source_file": record.get("source_file", ""), "error": str(exc)}
            extraction_errors.append(error)
            tool_call_trace.append({"tool": f"record_{index}", **error})

    progress_log[-1].update(
        {
            "status": "partial" if extraction_errors else "done",
            "processed_record_count": len(record_results),
            "error_count": len(extraction_errors),
            "duration_seconds": round(time.time() - stage_start, 4),
        }
    )

    commit_elapsed = 0.0
    if conn:
        commit_start = time.time()
        conn.commit()
        commit_elapsed = time.time() - commit_start
        conn.close()
    progress_log.append(
        {
            "step": "persist",
            "label": "三元组入库",
            "status": "done" if persistence_enabled else "skipped",
            "reason": "dry_run" if dry_run else ("persist=false" if not persist else ""),
            "inserted_triple_count": inserted_triple_count,
            "candidate_triple_count": candidate_triple_count,
            "rejected_triple_count": rejected_triple_count,
            "accepted_high_triple_count": accepted_high_triple_count,
            "deduplicated_triple_count": deduplicated_triple_count,
            "invalid_triple_count": invalid_triple_count,
            "duration_seconds": round(persistence_elapsed_total + commit_elapsed, 4),
        }
    )

    analytics_refresh_result = {"status": "skipped", "reason": "dry_run" if dry_run else "refresh_analytics=false"}
    if analytics_enabled:
        stage_start = time.time()
        if inserted_triple_count > 0:
            try:
                analytics_refresh_result = refresh_task3_analytics()
            except Exception as exc:
                analytics_refresh_result = {"status": "error", "error": str(exc)}
        else:
            analytics_refresh_result = {"status": "skipped", "reason": "no newly inserted triples"}
        analytics_refresh_result["duration_seconds"] = round(time.time() - stage_start, 4)
    progress_log.append(
        {
            "step": "refresh_analytics",
            "label": "分析库刷新",
            "status": analytics_refresh_result.get("status", "skipped"),
            "duration_seconds": analytics_refresh_result.get("duration_seconds", 0),
        }
    )

    if dry_run and not extraction_errors:
        status = "dry_run"
    elif extraction_errors and not record_results:
        status = "error"
    elif extraction_errors:
        status = "partial_success"
    else:
        status = "success"
    status_label = {
        "success": "\u5b8c\u6210",
        "dry_run": "\u8bd5\u8fd0\u884c\u5b8c\u6210",
        "partial_success": "\u90e8\u5206\u5b8c\u6210",
        "error": "\u5931\u8d25",
    }.get(status, status)


    elapsed = round(time.time() - t0, 1)
    processed_records = len(record_results)
    avg_latency = round(extraction_elapsed_total / processed_records, 4) if processed_records else 0.0
    throughput = round(processed_records / extraction_elapsed_total, 4) if extraction_elapsed_total else 0.0
    source_file_summary = summarize_source_files(records, selected_records, record_results)
    analytics_summary = summarize_analytics_refresh(analytics_refresh_result)
    source_format_summary = dict(Counter(item.get("source_format", "unknown") for item in records))

    return {
        "status": status,
        "error": "all records failed during extraction or persistence" if status == "error" else "",
        "elapsed_seconds": elapsed,
        "backend": selected_backend,
        "backend_label": backend_label,
        "dataset": {"id": ds["id"], "name": ds.get("name", "")},
        "file_count": len(files),
        "record_count": len(records),
        "selected_record_count": len(selected_records),
        "source_format_summary": source_format_summary,
        "source_file_summary": source_file_summary,
        "unprocessed_record_count": max(0, len(records) - len(selected_records)),
        "processed_record_count": processed_records,
        "entity_count": sum(item.get("entities", 0) for item in record_results),
        "relation_count": sum(item.get("relations", 0) for item in record_results),
        "generated_triple_count": generated_triple_count,
        "inserted_triple_count": inserted_triple_count,
        "candidate_triple_count": candidate_triple_count,
        "rejected_triple_count": rejected_triple_count,
        "accepted_high_triple_count": accepted_high_triple_count,
        "deduplicated_triple_count": deduplicated_triple_count,
        "invalid_triple_count": invalid_triple_count,
        "triple_count": inserted_triple_count,
        "cascade": {
            "gap_segment_count": cascade_gap_segment_count,
            "gap_candidate_count": cascade_gap_candidate_count,
            "reviewed_candidate_count": cascade_reviewed_candidate_count,
            "auto_accepted_candidate_count": cascade_auto_accepted_candidate_count,
            "review_skipped_candidate_count": cascade_review_skipped_candidate_count,
            "offline_filtered_candidate_count": cascade_offline_filtered_candidate_count,
            "rejected_candidate_count": cascade_rejected_candidate_count,
            "llm_added_count": cascade_llm_added_count,
            "llm_added_entity_count": cascade_llm_added_entity_count,
            "llm_added_relation_count": cascade_llm_added_relation_count,
            "gap_budget_skipped_count": cascade_gap_budget_skipped_count,
            "review_budget_skipped_count": cascade_review_budget_skipped_count,
        },
        "performance": {
            "extractor_backend": selected_backend,
            "extractor_label": backend_label,
            "extraction_elapsed_seconds": round(extraction_elapsed_total, 4),
            "avg_record_latency_seconds": avg_latency,
            "throughput_records_per_second": throughput,
            "llm_degraded_records": llm_degraded_count,
        },
        "extraction_errors": extraction_errors,
        "progress_log": progress_log,
        "tool_call_trace": tool_call_trace,
        "report_markdown": (
            f"任务二知识图谱构建{status_label}：使用{backend_label}，解析 {len(records)} 条记录，"
            f"处理 {processed_records} 条"
            f"{'（跨文件均衡抽样）' if len(selected_records) < len(records) else ''}，"
            f"覆盖 {len(source_file_summary)} 个来源文件，生成 {generated_triple_count} 条三元组，"
            f"高可靠通过 {accepted_high_triple_count} 条（新入库 {inserted_triple_count} 条，"
            f"已存在 {deduplicated_triple_count} 条），中可靠待复核 {candidate_triple_count} 条，"
            f"低可靠候选过滤 {cascade_offline_filtered_candidate_count} 条，"
            f"LLM 复核拒绝 {cascade_rejected_candidate_count} 条，级联实际复核 {cascade_reviewed_candidate_count} 条，"
            f"证据门禁直接通过 {cascade_auto_accepted_candidate_count} 条，无需或未进入复核 {cascade_review_skipped_candidate_count} 条，"
            f"LLM 补充实体 {cascade_llm_added_entity_count} 个、关系 {cascade_llm_added_relation_count} 条，"
            f"总耗时 {format_stage_duration(elapsed)}，"
            f"抽取吞吐 {throughput} records/s。"
        ),
        "analytics_summary": analytics_summary,
        "refresh_analytics": analytics_refresh_result,
    }
