"""证据托管清单：对交付证据做 SHA-256 + 大小 + mtime 归档，附来源/环境 provenance、
逐指标复跑命令、性能 bootstrap CI，并用 HMAC 签名 + ISO8601 时间戳做防篡改。

不追求外部 WORM/第三方 custodian（那是研究级理想态），但提供：
- 本地可复验的内容哈希清单；
- 签名（`DKI_EVIDENCE_HMAC_KEY`）+ 时间戳，离线也能证明清单未被改动；
- 大 OM 的受控 URI + SHA + 复跑命令占位（大文件不入库，只登记指纹）；
- 正式性能重复的 mean/std/median/p95 + bootstrap 95% CI；
- request-latency（调度器 queue_wait）与 batch-execution-latency 的口径说明。

用法：
    $env:DKI_EVIDENCE_HMAC_KEY = "..."   # 可选；缺省用 unsigned 占位并标注
    python scripts/build_evidence_custody.py --output outputs/verification/evidence_custody_20260720.json
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence import canonical_hash, sha256_file  # noqa: E402
from dki.evidence.stats import summarize  # noqa: E402

# 纳入托管的证据（存在才登记；缺失记 missing，绝不编造）。
EVIDENCE_FILES = [
    "outputs/claims/headline_claims.json",
    "outputs/eval/kg_extraction_eval.json",
    "outputs/eval/nl2sql_eval.json",
    "outputs/ner_finetune/metrics.json",
    "outputs/verification/datamate_smoke.json",
    "outputs/verification/phase2_release_a_status_20260720.md",
    "outputs/remote_acceptance_20260720_promotion.json",
    "outputs/remote_acceptance_20260720_formal_performance_retest.json",
    "outputs/remote_acceptance_20260720_lifecycle_retest.json",
    "outputs/remote_acceptance_20260720_soak_3600_retest.json",
    "outputs/remote_acceptance_20260720_parity_bert_fp16b_vs_pytorch.json",
    "outputs/remote_acceptance_20260720_eval_bert_int8_reproduction.json",
    "outputs/remote_acceptance_20260720_medical_ner_adapter.json",
    "outputs/remote_acceptance_20260720_target_pytest.json",
    "outputs/remote_acceptance_20260720_datamate_ray.json",
    "outputs/remote_acceptance_20260720_adapter_throughput.json",
    "outputs/remote_acceptance_20260720_soak_3600_retest.json",
    "outputs/cblue/cmeee_metrics.json",
    "outputs/cblue/cmeie_metrics.json",
    "outputs/remote_acceptance_20260721_aoe.json",
    "outputs/remote_acceptance_20260721_om_ladder.json",
    "evidence/manifests/protected_baseline.json",
]

# 大工件（OM/ONNX 等）：不入库，只登记受控 URI + 指纹 + 复跑命令。
LARGE_ARTIFACTS = [
    {
        "artifact": "bert_ner_bs64_seq256_int8.om",
        "controlled_uri": "npu-host:/root/dki-agents/outputs/bench/npu_int8/ner_ft_bs64_seq256_int8.om",
        "sha256_source": "outputs/remote_acceptance_20260720_lifecycle_retest.json",  # 内含 om_sha256
        "sha256_field": "om_sha256",
        "reproduce": [
            "python3 scripts/export_bert_ner_onnx.py --model models/medical_ner_bert --output $F/ner_ft_bs64_seq256_fp32.onnx --batch 64 --seq-len 256",
            "python scripts/quantize_ner_int8.py --onnx $F/ner_ft_bs64_seq256_fp32.onnx --output $F/ner_ft_bs64_seq256_int8.onnx --model-name models/medical_ner_bert --batch 64 --seq-len 256 --calib-batches 4",
            "atc --framework=5 --model=$F/ner_ft_bs64_seq256_int8.onnx --output=$F/ner_ft_bs64_seq256_int8 --input_format=ND --input_shape=\"input_ids:64,256;attention_mask:64,256;token_type_ids:64,256\" --soc_version=Ascend910B2C",
        ],
    },
]

REPRODUCE_COMMANDS = {
    "task2_kg_extraction_micro_f1": ".\\run.ps1 task2 --eval",
    "task3_nl2sql_rule_accuracy": ".\\run.ps1 eval-nl2sql",
    "task1_finetune_bert_eval_f1": "python scripts/train_ner.py --epochs 5 --device cpu",
    "npu_bert_int8_formal_throughput": "python outputs/remote_formal_performance_20260720.py --script <bench> --om <int8.om> --output <json> --telemetry-dir <dir> --repeats 5",
    "datamate_platform_smoke": "python scripts/datamate_smoke.py",
}

LATENCY_KINDS = {
    "request_latency": "调度器端到端：入队到结果，含 queue_wait（dki/bench/npu/om_scheduler.py 遥测）。",
    "batch_execution_latency": "单次 OM execute 的批处理延迟：formal performance 的 latency_ms（不含排队）。",
    "note": "两者口径不同，不可混用；吞吐能效来自 batch execution，服务级延迟来自 request latency。",
}


def _fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _environment() -> Dict[str, Any]:
    versions = {}
    for module in ("numpy", "transformers", "torch", "onnx", "datamate", "paramiko"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _source_provenance(root: Path) -> Dict[str, Any]:
    git_dir = root / ".git"
    if git_dir.exists():
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        return {"vcs": "git", "head": head}
    # 无 git：登记关键源码文件的合并哈希，作为轻量 source-revision 指纹。
    tracked = sorted(
        [p for p in (root / "dki").rglob("*.py")] + [p for p in (root / "scripts").glob("*.py")]
    )
    digest = hashlib.sha256()
    for path in tracked:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("utf-8"))
    return {"vcs": "none", "source_tree_sha256": digest.hexdigest(), "n_source_files": len(tracked)}


def _performance_stats(root: Path) -> Dict[str, Any]:
    perf_path = root / "outputs" / "remote_acceptance_20260720_formal_performance_retest.json"
    if not perf_path.is_file():
        return {"available": False}
    data = json.loads(perf_path.read_text(encoding="utf-8"))
    repeats = data.get("repeats", [])
    throughputs = [r["throughput_samples_s"] for r in repeats if "throughput_samples_s" in r]
    joules = [r["samples_per_joule"] for r in repeats if r.get("samples_per_joule") is not None]
    p99s = [r["latency_ms"]["p99"] for r in repeats if isinstance(r.get("latency_ms"), dict)]
    out: Dict[str, Any] = {"available": True, "n_repeats": len(throughputs)}
    if throughputs:
        out["throughput_samples_s"] = summarize(throughputs)
    if joules:
        out["samples_per_joule"] = summarize(joules)
    if p99s:
        out["batch_p99_latency_ms"] = summarize(p99s)
    out["latency_kinds"] = LATENCY_KINDS
    out["energy_note"] = (
        "joules 为并行功率采样梯形积分，含 worker 启停窗口（保守口径）；"
        "如需 idle-subtracted 需另采 idle 基线再扣除。"
    )
    return out


def _large_artifacts(root: Path) -> List[Dict[str, Any]]:
    registered = []
    for item in LARGE_ARTIFACTS:
        entry = dict(item)
        src = root / item["sha256_source"]
        entry["om_sha256"] = None
        if src.is_file():
            try:
                entry["om_sha256"] = json.loads(src.read_text(encoding="utf-8")).get(item["sha256_field"])
            except Exception:
                entry["om_sha256"] = None
        registered.append(entry)
    return registered


def main() -> int:
    parser = argparse.ArgumentParser(description="构建签名+时间戳的证据托管清单")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "verification" / "evidence_custody_20260720.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root

    files: Dict[str, Any] = {}
    missing: List[str] = []
    for rel in EVIDENCE_FILES:
        path = root / rel
        if path.is_file():
            files[rel] = _fingerprint(path)
        else:
            missing.append(rel)

    body = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_provenance": _source_provenance(root),
        "environment": _environment(),
        "evidence_files": files,
        "missing_files": missing,
        "large_artifacts": _large_artifacts(root),
        "performance_statistics": _performance_stats(root),
        "reproduce_commands": REPRODUCE_COMMANDS,
        "custody_note": (
            "本清单为本地可审计内容指纹 + 签名/时间戳；未做外部 WORM/第三方保管，"
            "不据此声称第三方公证。大工件仅登记受控 URI 与指纹。"
        ),
    }
    content_hash = canonical_hash(body)
    key = os.environ.get("DKI_EVIDENCE_HMAC_KEY", "")
    if key:
        signature = hmac.new(key.encode("utf-8"), content_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        signature_kind = "hmac_sha256"
    else:
        signature = None
        signature_kind = "unsigned (set DKI_EVIDENCE_HMAC_KEY to sign)"
    manifest = {
        **body,
        "content_hash": content_hash,
        "signature_kind": signature_kind,
        "signature": signature,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "n_evidence_files": len(files),
        "n_missing": len(missing),
        "signature_kind": signature_kind,
        "content_hash": content_hash,
        "throughput_ci95": body["performance_statistics"].get("throughput_samples_s", {}).get("ci95"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
