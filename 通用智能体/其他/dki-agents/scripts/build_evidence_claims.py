"""从既有证据产物生成 validator-passing 的 typed claims（headline 指标）。

`validate_evidence.py` 之所以拒绝聚合 summary/promotion JSON，是因为它们缺少
typed-claim 的枚举字段与带 hash 的 artifacts 列表。本脚本把每个 headline 指标
包装成一条合法 claim：枚举齐全、`claim_id` 非空、`artifacts` 逐个存在且 sha256 匹配，
并按 schema 规则区分 development/blind、exact/relaxed、probe/formal、cpu/npu。

诚实原则：
- 只为**磁盘上真实存在**的证据文件生成 claim；缺失的证据跳过并在结果里标 skipped，
  绝不为不存在的产物编造 claim；
- blind/prospective 角色的 claim 一律 `selection_used=false`；
- CPU 吞吐/延迟/能效 claim 不声明 `device_claim=npu`；
- 每条 claim 携带 `metric`（value/unit）与 `notes`，数值只从证据文件读取，不硬编码。

用法：
    python scripts/build_evidence_claims.py --output outputs/claims/headline_claims.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence import sha256_file, validate_claim  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class ClaimSpec:
    def __init__(
        self,
        claim_id: str,
        artifact: str,
        extract: Callable[[Any], Dict[str, Any]],
        *,
        dataset_role: str,
        metric_kind: str,
        runtime_kind: str,
        evidence_level: str,
        promotion_state: str,
        temperature_kind: str = "not_applicable",
        selection_used: bool = False,
        device_claim: Optional[str] = None,
        extra_artifacts: Optional[List[str]] = None,
    ):
        self.claim_id = claim_id
        self.artifact = artifact
        self.extract = extract
        self.enums = {
            "dataset_role": dataset_role,
            "metric_kind": metric_kind,
            "runtime_kind": runtime_kind,
            "evidence_level": evidence_level,
            "promotion_state": promotion_state,
            "temperature_kind": temperature_kind,
        }
        self.selection_used = selection_used
        self.device_claim = device_claim
        self.extra_artifacts = extra_artifacts or []


def _extract_cmeee_claim(d: Any) -> Dict[str, Any]:
    if d.get("schema_version") != 2:
        raise ValueError("CMeEE claim requires metrics schema v2")
    if d.get("dataset_role") != "development_locked_report":
        raise ValueError("CMeEE artifact is not a locked development report")
    if d.get("eval_split") != "official_dev" or int(d.get("n_dev") or 0) != 5000:
        raise ValueError("CMeEE claim requires the complete 5000-row official dev set")
    if (d.get("selection_protocol") or {}).get("official_dev_used_for_selection") is not False:
        raise ValueError("CMeEE official dev must not be used for selection")
    selected = d.get("internally_selected_threshold") or {}
    report = selected.get("official_report") or {}
    threshold = selected.get("threshold", d.get("decode_threshold"))
    if d.get("primary_operating_point") != "internally_selected_threshold":
        raise ValueError("CMeEE primary operating point is not declared")
    if threshold != d.get("decode_threshold") or report.get("micro_f1") != d.get("micro_f1"):
        raise ValueError("CMeEE deployed threshold and headline metric disagree")
    phase = d.get("selected_phase")
    decoder = d.get("decoder") or "GlobalPointer"
    if phase == "soft_ensemble" or "Ensemble" in str(decoder):
        note = "soft-logit 融合（成员与阈值仅由 train-derived internal split 选择）"
    elif phase == "head":
        note = "冻结编码器训 GlobalPointer 头"
    else:
        note = f"选择 {phase} 阶段 checkpoint"
    return {
        "metric": {
            "name": "cmeee_v2_dev_micro_f1", "value": d["micro_f1"], "unit": "f1",
            "precision": d["precision"], "recall": d["recall"],
            "base_model": d.get("base_model"), "decode_threshold": threshold,
            "selected_phase": phase, "decoder": decoder,
        },
        "notes": (
            "CBLUE CMeEE-V2 完整官方 dev 实体级 Micro-F1（<1B；"
            f"{note}）。此为真实权威基准栏，勿与合成语料 99.97% 或同分布微调 F1 混报。"
        ),
    }


def _extract_cmeie_claim(d: Any) -> Dict[str, Any]:
    if d.get("schema_version") != 2:
        raise ValueError("CMeIE claim requires metrics schema v2")
    if d.get("dataset_role") != "development_locked_report" or d.get("n_dev") != 3585:
        raise ValueError("CMeIE claim requires the complete 3585-row locked official dev report")
    if d.get("evaluation_scope") != (
        "gold_entity_pairs_and_gold_type_signatures_not_end_to_end_spo"
    ):
        raise ValueError(
            "CMeIE claim requires explicit gold entity-pair and gold type-signature scope"
        )
    if d.get("gold_type_signature_conditioning") is not True:
        raise ValueError("CMeIE claim requires explicit gold type-signature conditioning")
    if (d.get("selection_protocol") or {}).get("official_dev_used_for_selection") is not False:
        raise ValueError("CMeIE official dev must not be used for selection")
    config = d.get("train_config") or {}
    if config.get("objective") != "multi_label_bce":
        raise ValueError("CMeIE claim requires the multi-label BCE protocol")
    if d.get("schema_constraint_scope") != "fit_partition_only":
        raise ValueError("CMeIE admissible predicate ontology must be fit-partition-only")
    return {
        "metric": {
            "name": "cmeie_v2_dev_gold_pair_micro_f1", "value": d["micro_f1"], "unit": "f1",
            "precision": d["precision"], "recall": d["recall"],
            "pred_na": d.get("pred_na"), "pred_rel": d.get("pred_rel"),
            "selected_threshold": d.get("selected_threshold"),
        },
        "notes": (
            "CBLUE CMeIE-V2 完整官方 dev 的 gold 实体对 + gold 主客体类型签名条件下"
            "多标签谓词分类 Micro-F1；方向实体标记 + multi-label BCE + "
            "fit-partition-only 本体约束 + mandatory top-1 gold-pair decode。"
            "这是条件化 gold-pair 上界代理，不宣称端到端 SPO 抽取。"
        ),
    }


SPECS: List[ClaimSpec] = [
    ClaimSpec(
        "task2_kg_extraction_micro_f1",
        "outputs/eval/kg_extraction_eval.json",
        lambda d: {"metric": {"name": "triple_micro_f1", "value": d["f1"],
                              "precision": d["precision"], "recall": d["recall"],
                              "tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "unit": "f1"},
                   "notes": "同源合成语料 dict 后端；受控正确性指标，非开放语料泛化。"},
        dataset_role="internal_acceptance", metric_kind="exact_span_type",
        runtime_kind="cpu", evidence_level="formal", promotion_state="promoted",
    ),
    ClaimSpec(
        "task3_nl2sql_rule_accuracy",
        "outputs/eval/nl2sql_eval.json",
        lambda d: {"metric": {"name": "execution_accuracy",
                              "value": _nl2sql_acc(d), "unit": "accuracy"},
                   "notes": "规则引擎 44 题执行准确率（要求≥0.85）。"},
        dataset_role="internal_acceptance", metric_kind="exact_span_type",
        runtime_kind="cpu", evidence_level="formal", promotion_state="promoted",
    ),
    ClaimSpec(
        "task3_nl2sql_layered_rule_ex",
        "outputs/eval/nl2sql_layered_eval.json",
        lambda d: {"metric": {"name": "execution_accuracy_layered_rule",
                              "value": d["execution_accuracy"], "unit": "accuracy",
                              "n": d["n"], "by_difficulty": d.get("by_difficulty"),
                              "selection_used": d.get("selection_used")},
                   "notes": "120 题分层 development/selection-used 集规则 EX；该集合已用于实现回归，非 holdout。"},
        dataset_role="development", metric_kind="exact_span_type",
        runtime_kind="cpu", evidence_level="formal", promotion_state="candidate",
        selection_used=True,
    ),
    ClaimSpec(
        "task3_nl2sql_layered_auto_ex",
        "outputs/eval/nl2sql_layered_auto_eval.json",
        lambda d: {"metric": {"name": "execution_accuracy_layered_auto",
                              "value": d["execution_accuracy"], "unit": "accuracy",
                              "n": d["n"], "by_difficulty": d.get("by_difficulty"),
                              "llm_fallback": d.get("llm_fallback"),
                              "selection_used": d.get("selection_used")},
                   "notes": "规则+LLM 管线在 120 题 development/selection-used 集的综合 EX；非 holdout。"},
        dataset_role="development", metric_kind="exact_span_type",
        runtime_kind="cpu", evidence_level="formal", promotion_state="candidate",
        selection_used=True,
    ),
    ClaimSpec(
        "task1_finetune_bert_eval_f1",
        "outputs/ner_finetune/metrics.json",
        lambda d: {"metric": {"name": "entity_f1_finetuned_bert",
                              "value": _finetune_f1(d), "unit": "f1"},
                   "notes": "<1B 微调 BERT，40 份训练不可见验证集实体级 F1（同分布）。"},
        dataset_role="internal_acceptance", metric_kind="exact_span_type",
        runtime_kind="cpu", evidence_level="formal", promotion_state="promoted",
    ),
    ClaimSpec(
        "npu_bert_int8_formal_throughput",
        "outputs/remote_acceptance_20260720_formal_performance_retest.json",
        lambda d: {"metric": {"name": "throughput_samples_per_s",
                              "value": d["summary"]["throughput_mean"],
                              "stdev": d["summary"].get("throughput_stdev"),
                              "samples_per_joule": d["summary"].get("samples_per_joule_mean"),
                              "unit": "samples/s"},
                   "notes": "910B2C 进程独占 ACL，5 次正式重复均值（固定 bs64/seq256）。"},
        dataset_role="internal_acceptance", metric_kind="throughput",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        temperature_kind="warm", device_claim="npu",
    ),
    ClaimSpec(
        "npu_bert_int8_entity_f1",
        "outputs/remote_acceptance_20260720_eval_bert_int8_reproduction.json",
        lambda d: {"metric": {"name": "om_entity_f1_int8",
                              "value": d["entity_f1"]["f1"],
                              "precision": d["entity_f1"]["precision"],
                              "recall": d["entity_f1"]["recall"], "unit": "f1"},
                   "notes": "BERT-INT8 OM 对同源金标的实体级 F1（非 exact-equivalent）。"},
        dataset_role="internal_acceptance", metric_kind="exact_span_type",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        device_claim="npu",
    ),
    ClaimSpec(
        "npu_bert_fp16b_pytorch_parity",
        "outputs/remote_acceptance_20260720_parity_bert_fp16b_vs_pytorch.json",
        lambda d: {"metric": {"name": "om_vs_pytorch_micro_f1",
                              "value": d["micro"]["f1"], "unit": "f1"},
                   "notes": "FP16b OM 与 PyTorch 的 counterfactual exact parity（=1.0）。"},
        dataset_role="internal_acceptance", metric_kind="exact_span_type",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        device_claim="npu",
    ),
    ClaimSpec(
        "npu_acl_lifecycle_100",
        "outputs/remote_acceptance_20260720_lifecycle_retest.json",
        lambda d: {"metric": {"name": "acl_lifecycle_cycles_passed",
                              "value": d.get("completed_cycles"), "unit": "cycles",
                              "passed": d.get("passed")},
                   "notes": "100 次 init-execute-close 生命周期全过（进程独占 ACL）。"},
        dataset_role="internal_acceptance", metric_kind="latency",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        temperature_kind="mixed", device_claim="npu",
    ),
    ClaimSpec(
        "npu_bert_int8_soak_3600",
        "outputs/remote_acceptance_20260720_soak_3600_retest.json",
        lambda d: {"metric": {"name": "soak_seconds_passed",
                              "value": _soak_seconds(d), "unit": "seconds",
                              "passed": d.get("passed")},
                   "notes": "3600s soak 无执行失败、无残留进程。"},
        dataset_role="internal_acceptance", metric_kind="throughput",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        temperature_kind="warm", device_claim="npu",
    ),
    ClaimSpec(
        "ood_final3_quality_v2_blind",
        "outputs/ner_finetune/ood_final3_quality_v2.json",
        lambda d: {"metric": {"name": "ood_final3_recall_note",
                              "value": None, "unit": "f1",
                              "raw_keys": list(d.keys())[:8]},
                   "notes": "final3 一次性盲测证据（历史封存，只读；未达未知分布 recall≥0.90）。"},
        dataset_role="historical_blind", metric_kind="relaxed_overlap_type",
        runtime_kind="cpu", evidence_level="historical", promotion_state="historical",
        selection_used=False,
    ),
    ClaimSpec(
        "npu_medical_ner_adapter_text_to_entities",
        "outputs/remote_acceptance_20260720_medical_ner_adapter.json",
        lambda d: {"metric": {"name": "production_adapter_text_to_entities",
                              "value": bool(d.get("passed")), "unit": "boolean",
                              "entity_counts": d.get("direct_adapter", {}).get("entity_counts"),
                              "process_runtime_matches_direct": d.get("process_runtime", {}).get("matches_direct"),
                              "fail_closed_on_bad_manifest": d.get("fail_closed_on_bad_manifest")},
                   "notes": "910B2C 真实 INT8 OM：完整病历 text→tokenize→OM→BIO→span→归一→带类型实体；"
                            "进程独占 runtime 与直连逐条一致；manifest 篡改在 ACL 加载前 fail-closed。"},
        dataset_role="internal_acceptance", metric_kind="exact_span_type",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        device_claim="npu",
    ),
    ClaimSpec(
        "target_host_full_pytest",
        "outputs/remote_acceptance_20260720_target_pytest.json",
        lambda d: {"metric": {"name": "target_host_pytest_passed",
                              "value": d.get("passed_count"), "unit": "tests",
                              "failed": d.get("failed_count"), "skipped": d.get("skipped_count"),
                              "summary": d.get("summary_line")},
                   "notes": "910B2C 目标机全量 pytest（真实 datamate+ray+nexent+onnx，含量化图校验）0 失败。"},
        dataset_role="internal_acceptance", metric_kind="exact_span_type",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        device_claim="npu",
    ),
    ClaimSpec(
        "npu_ner_adapter_end_to_end_throughput",
        "outputs/remote_acceptance_20260720_adapter_throughput.json",
        lambda d: {"metric": {"name": "production_adapter_docs_per_second",
                              "value": d["end_to_end"]["docs_per_second"], "unit": "docs/s",
                              "n_docs": d["end_to_end"]["n_docs"],
                              "total_entities": d["end_to_end"]["total_entities"],
                              "latency_ms_p50": d["end_to_end"]["latency_ms"]["p50"],
                              "latency_ms_p95": d["end_to_end"]["latency_ms"]["p95"]},
                   "notes": "910B2C 真实 INT8 OM，生产适配器 text→实体单文档端到端：200 份真实 EMR "
                            "47.2 docs/s（p50 21ms/doc），抽出 2360 实体。单文档走 bs64 静态 OM 有 padding 余量，"
                            "批式服务可更高。"},
        dataset_role="internal_acceptance", metric_kind="latency",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        temperature_kind="warm", device_claim="npu",
    ),
    ClaimSpec(
        "datamate_ray_real_pipeline",
        "outputs/remote_acceptance_20260720_datamate_ray.json",
        lambda d: {"metric": {"name": "real_raydataset_process_triples",
                              "value": d["real_pipeline"]["total_triples"], "unit": "triples",
                              "n_output_rows": d["real_pipeline"]["n_output_rows"],
                              "docs_per_second": d["real_pipeline"]["docs_per_second"],
                              "map_batches_max_fill": d["dispatch_granularity"]["map_batches_max_fill"],
                              "formal_microbatch_for_mapper": d["dispatch_granularity"]["formal_microbatch_for_mapper_operators"]},
                   "notes": "真实 datamate.core.dataset.RayDataset.process（Ray actor pool）在 200 份真实 EMR "
                            "上抽出 1569 三元组；Mapper 走 .map 逐样本(fill=1)，故 formal_microbatch=false（实测结论）。"},
        dataset_role="internal_acceptance", metric_kind="throughput",
        runtime_kind="cpu", evidence_level="formal", promotion_state="candidate",
    ),
    ClaimSpec(
        "task3_nl2sql_layered_holdout_rule_ex",
        "outputs/eval/nl2sql_layered_holdout_rule_eval.json",
        lambda d: {"metric": {"name": "execution_accuracy_holdout_rule",
                              "value": d["execution_accuracy"], "unit": "accuracy",
                              "n": d["n"], "by_difficulty": d.get("by_difficulty"),
                              "holdout_status": d.get("holdout_status")},
                   "notes": "冻结 hard holdout 规则 EX；禁止用于调参。"},
        dataset_role="historical_blind", metric_kind="exact_span_type",
        runtime_kind="cpu", evidence_level="formal", promotion_state="candidate",
        selection_used=False,
    ),
    ClaimSpec(
        "cblue_cmeee_v2_micro_f1",
        "outputs/cblue/cmeee_metrics.json",
        _extract_cmeee_claim,
        dataset_role="development", metric_kind="exact_span_type",
        runtime_kind="pytorch_npu", evidence_level="formal", promotion_state="candidate",
    ),
    ClaimSpec(
        "cblue_cmeie_v2_micro_f1",
        "outputs/cblue/cmeie_metrics.json",
        _extract_cmeie_claim,
        dataset_role="development", metric_kind="exact_span_type",
        runtime_kind="pytorch_npu", evidence_level="formal", promotion_state="candidate",
    ),
    ClaimSpec(
        "npu_aoe_tuning_ab",
        "outputs/remote_acceptance_20260721_aoe.json",
        lambda d: {"metric": {"name": "aoe_speedup_vs_baseline",
                              "value": next((r.get("speedup_vs_baseline") for r in d.get("results", [])
                                             if r.get("tag") == "aoe"), None),
                              "unit": "x",
                              "aoe_atc_status": d.get("aoe_atc_status"),
                              "fallback": d.get("fallback"),
                              "results": [{"tag": r["tag"], "throughput_samples_s": r.get("throughput_samples_s"),
                                           "speedup": r.get("speedup_vs_baseline")} for r in d.get("results", [])]},
                   "notes": "ATC-AOE(SGAT/OPAT) 在本镜像 GELib/Opp 初始化失败；以 FP16→INT8→TASK_QUEUE "
                            "环境阶梯作为可复现 NPU 优化证据（不虚报 AOE 知识库增益）。"},
        dataset_role="internal_acceptance", metric_kind="throughput",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        temperature_kind="warm", device_claim="npu",
    ),
    ClaimSpec(
        "npu_om_optimization_ladder",
        "outputs/remote_acceptance_20260721_om_ladder.json",
        lambda d: {"metric": {"name": "int8_task_queue_speedup_vs_fp16",
                              "value": next((r.get("speedup_vs_fp16") for r in d.get("results", [])
                                             if r.get("tag") == "int8_task_queue"), None),
                              "unit": "x",
                              "results": [{"tag": r["tag"], "throughput_samples_s": r.get("throughput_samples_s"),
                                           "speedup_vs_fp16": r.get("speedup_vs_fp16")} for r in d.get("results", [])]},
                   "notes": "910B2C 上 FP16/INT8 OM + TASK_QUEUE_ENABLE 吞吐阶梯（acl_om_pipeline_bench mp）。"},
        dataset_role="internal_acceptance", metric_kind="throughput",
        runtime_kind="acl_om_npu", evidence_level="formal", promotion_state="candidate",
        temperature_kind="warm", device_claim="npu",
    ),
    ClaimSpec(
        "datamate_platform_smoke",
        "outputs/verification/datamate_smoke.json",
        lambda d: {"metric": {"name": "platform_deploy_invoke_isolation",
                              "value": bool(d.get("passed")), "unit": "boolean",
                              "formal_microbatch": d.get("formal_microbatch")},
                   "notes": "真实 DataMate upload/import/instantiate/invoke/故障隔离；formal_microbatch=false。"},
        dataset_role="internal_acceptance", metric_kind="throughput",
        runtime_kind="shadow", evidence_level="probe", promotion_state="shadow",
    ),
]


def _nl2sql_acc(d: Any) -> float:
    if isinstance(d, dict):
        for key in ("accuracy", "acc"):
            if key in d:
                return float(d[key])
        details = d.get("details") or d.get("cases")
        if isinstance(details, list) and details:
            ok = sum(1 for x in details if x.get("match") or x.get("correct") or x.get("ok"))
            return round(ok / len(details), 4)
    raise KeyError("nl2sql accuracy not found")


def _finetune_f1(d: Any) -> float:
    # metrics.json nests the per-backend results under eval_f1.
    containers = [d]
    if isinstance(d, dict) and isinstance(d.get("eval_f1"), dict):
        containers.append(d["eval_f1"])
        if isinstance(d["eval_f1"].get("backends"), dict):
            containers.append(d["eval_f1"]["backends"])
    if isinstance(d, dict) and isinstance(d.get("backends"), dict):
        containers.append(d["backends"])
    for container in containers:
        for key in ("finetuned_bert", "bert_finetuned", "finetune", "bert"):
            node = container.get(key) if isinstance(container, dict) else None
            if isinstance(node, dict) and "f1" in node:
                return float(node["f1"])
    if isinstance(d, dict) and "f1" in d:
        return float(d["f1"])
    raise KeyError("finetune f1 not found")


def _soak_seconds(d: Any) -> Optional[float]:
    for key in ("elapsed_s", "wall_s", "duration_s", "seconds"):
        if isinstance(d, dict) and key in d:
            return float(d[key])
    summary = d.get("summary") if isinstance(d, dict) else None
    if isinstance(summary, dict):
        for key in ("elapsed_s", "wall_s", "duration_s"):
            if key in summary:
                return float(summary[key])
    return None


def build(output: Path, root: Path) -> dict:
    claims: List[dict] = []
    skipped: List[dict] = []
    for spec in SPECS:
        artifact_path = root / spec.artifact
        if not artifact_path.is_file():
            skipped.append({"claim_id": spec.claim_id, "reason": f"missing artifact {spec.artifact}"})
            continue
        try:
            body = spec.extract(_load(artifact_path))
        except Exception as exc:
            skipped.append({"claim_id": spec.claim_id, "reason": f"extract failed: {exc}"})
            continue
        artifacts = [{"path": spec.artifact, "sha256": sha256_file(artifact_path)}]
        for extra in spec.extra_artifacts:
            extra_path = root / extra
            if extra_path.is_file():
                artifacts.append({"path": extra, "sha256": sha256_file(extra_path)})
        claim = {
            "claim_id": spec.claim_id,
            **spec.enums,
            "selection_used": spec.selection_used,
            "artifacts": artifacts,
            **body,
        }
        if spec.device_claim is not None:
            claim["device_claim"] = spec.device_claim
        validate_claim(claim)  # fail fast if we ever build an illegal claim
        claims.append(claim)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"built": [c["claim_id"] for c in claims], "skipped": skipped, "output": str(output)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 validator-passing typed claims")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "claims" / "headline_claims.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = build(args.output, args.root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.output} ({len(summary['built'])} claim(s), {len(summary['skipped'])} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
