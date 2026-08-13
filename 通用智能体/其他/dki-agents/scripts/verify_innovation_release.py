"""Drive the CPU-verifiable Release A/B innovation contracts end to end.

This is intentionally a public CLI rather than a test module. It exercises the
operator catalog, a real shadow operator invocation, evidence manifest checks,
strict evaluation, overflow reconciliation, bounded recovery, calibrated
routing and create-once prospective receipts without requiring an NPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.agents.tools.data_tools import ListOperatorsTool
from dki.evidence import verify_protected_manifest
from dki.nlp.adaptive_router import AdaptiveRouter, RoutingPolicy
from dki.nlp.confidence import TemperatureCalibrator, calibration_metrics, fit_temperature
from dki.nlp.evaluation import evaluate_both
from dki.nlp.windowing import merge_window_entities, plan_token_windows
from dki.operators.medical_entity_extractor_om_shadow.process import MedicalEntityExtractorOmShadow
from dki.pipeline.recovery import decide_recovery, recovery_patch
from scripts.blind_protocol import consume, seal, verify


class _ShadowProbeRuntime:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def execute_batch(self, texts, timeout_s: float = 1.0):
        if self.fail:
            raise RuntimeError("simulated NPU unavailable")
        return [
            [{"span": [0, min(2, len(text))], "type": "Disease"}]
            for text in texts
        ]

    def close(self):
        return None


def _sample():
    payload = {"sections": {"诊断": "高血压", "治疗方案": "给予阿司匹林"}}
    return {
        "fileId": "patient-1",
        "fileName": "demo.txt",
        "text": json.dumps(payload, ensure_ascii=False),
        "data": b"",
        "ext_params": {},
    }


def _shadow_probe(workdir: Path) -> dict:
    previous_secret = os.environ.get("DKI_SHADOW_HASH_KEY")
    os.environ["DKI_SHADOW_HASH_KEY"] = "cpu-verification-only-not-production"
    try:
        baseline = MedicalEntityExtractorOmShadow(shadowEnabled=False)
        baseline_output = baseline.execute(copy.deepcopy(_sample()))
        baseline.close()

        success = MedicalEntityExtractorOmShadow(
            shadow_runtime=_ShadowProbeRuntime(),
            sidecarPath=str(workdir / "shadow_success.jsonl"),
        )
        success_output = success.execute(copy.deepcopy(_sample()))
        success.flush()
        success.close()

        failure = MedicalEntityExtractorOmShadow(
            shadow_runtime=_ShadowProbeRuntime(fail=True),
            sidecarPath=str(workdir / "shadow_failure.jsonl"),
        )
        failure_output = failure.execute(copy.deepcopy(_sample()))
        failure.flush()
        failure.close()
    finally:
        if previous_secret is None:
            os.environ.pop("DKI_SHADOW_HASH_KEY", None)
        else:
            os.environ["DKI_SHADOW_HASH_KEY"] = previous_secret

    parsed = json.loads(baseline_output["text"])
    return {
        "primary_no_impact": baseline_output["text"] == success_output["text"] == failure_output["text"],
        "section_semantics": all(
            entity.get("span_base") == "section" and entity.get("section")
            for entity in parsed["entities"]
        ),
        "primary_entities": len(parsed["entities"]),
        "success_sidecar": json.loads((workdir / "shadow_success.jsonl").read_text(encoding="utf-8")),
        "failure_sidecar": json.loads((workdir / "shadow_failure.jsonl").read_text(encoding="utf-8")),
    }


def run_verification() -> dict:
    catalog = json.loads(ListOperatorsTool().forward(category="medical"))
    catalog_names = sorted(item["op"] for item in catalog["medical_operators"])
    expected_catalog = sorted([
        "MedicalTextCleaner", "MedicalSectionSplitter", "MedicalEntityExtractor",
        "MedicalRelationExtractor", "TripleBuilderValidator",
    ])

    protected_path = ROOT / "evidence" / "manifests" / "protected_baseline.json"
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    verify_protected_manifest(protected, ROOT)

    exact = evaluate_both(
        {"d": [(2, 4, "Disease"), (0, 1, "Disease")]},
        {"d": [(0, 3, "Disease"), (3, 6, "Disease")]},
    )

    offsets = [(index, index + 1) for index in range(300)]
    windows = plan_token_windows(offsets, capacity=254, overlap=32)
    merged = merge_window_entities([
        (windows[1], [{"span": [280, 284], "type": "Disease", "mention": "同名", "anchor_token": 280}]),
        (windows[0], [{"span": [220, 224], "type": "Disease", "mention": "同名", "anchor_token": 220}]),
    ])

    recovery = decide_recovery("NPU out of memory", idempotent=True)
    logits, labels = [5.0, 4.0, -1.0, -0.5], [1, 0, 0, 1]
    before = calibration_metrics(TemperatureCalibrator(1.0).predict(logits), labels)
    calibrator = fit_temperature(logits, labels)
    after = calibration_metrics(calibrator.predict(logits), labels)
    router = AdaptiveRouter(RoutingPolicy(max_upgrade_rate=1.0), calibration_valid=True, promotion_valid=True)
    route = router.decide([{"type": "Treatment"}], calibrated_confidence=0.99)

    with tempfile.TemporaryDirectory() as raw_dir:
        workdir = Path(raw_dir)
        shadow = _shadow_probe(workdir)
        protocol = workdir / "protocol.json"
        candidate = workdir / "candidate.json"
        seal_path = workdir / "seal.json"
        receipt = workdir / "receipt.json"
        protocol.write_text('{"metric":"exact_span_type"}', encoding="utf-8")
        candidate.write_text('{"promotion_state":"candidate"}', encoding="utf-8")
        seal(protocol, [candidate], seal_path)
        verified_seal = verify(seal_path)
        consumed = consume(seal_path, receipt)

    result = {
        "schema_version": 1,
        "runtime_scope": "cpu_local_no_npu_claim",
        "checks": {
            "default_catalog_unchanged": catalog_names == expected_catalog,
            "protected_manifest_verified": True,
            "shadow_primary_no_impact": shadow["primary_no_impact"],
            "shadow_failure_isolated": shadow["failure_sidecar"]["error"] == "RuntimeError",
            "shadow_section_semantics": shadow["section_semantics"],
            "strict_exact_separate_from_relaxed": (
                exact["primary_exact"]["tp"] == 0
                and exact["relaxed_diagnostic"]["tp"] == 2
            ),
            "long_text_positions_preserved": [item["span"] for item in merged] == [[220, 224], [280, 284]],
            "oom_recovery_bounded": recovery.max_attempts == 1,
            "calibration_nll_nonworse": after["nll"] <= before["nll"],
            "treatment_route_promoted": route["route"] == "quality",
            "prospective_receipt_created": consumed["status"] == "consumed",
        },
        "observations": {
            "medical_catalog": catalog_names,
            "shadow": shadow,
            "strict_metrics": exact,
            "window_count": len(windows),
            "merged_spans": [item["span"] for item in merged],
            "recovery": {
                "category": recovery.category.value,
                "action": recovery.action.value,
                "max_attempts": recovery.max_attempts,
                "patch": recovery_patch(recovery, {"batch": 64}),
            },
            "calibration": {
                "temperature": calibrator.temperature,
                "before": {key: before[key] for key in ("nll", "brier", "ece")},
                "after": {key: after[key] for key in ("nll", "brier", "ece")},
            },
            "route": route,
            "prospective_threat_model": verified_seal["threat_model"],
        },
        "npu_only_unverified": [
            "real pyACL lifecycle and ABI compatibility",
            "OM output parity and Treatment mixed-precision recovery",
            "formal DataMate cross-document microbatch",
            "hot-shape OM throughput, latency and synchronized joules",
        ],
    }
    result["ok"] = all(result["checks"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_verification()
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
