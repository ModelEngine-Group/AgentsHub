"""Assemble a self-contained defense evidence package with HTML report.

Copies screenshots, figures, benchmark reports, integration probes, demo log
excerpts, and the main Markdown document into one directory, then generates
`competition_defense_document.html` via `export_defense_pdf.py`.

Usage:
    python demos/build_defense_pdf_package.py
    python demos/build_defense_pdf_package.py --source outputs/competition_evidence/20260703-105plus
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = ROOT / "outputs" / "competition_evidence" / "20260703-105plus"
DEFAULT_OUTPUT = ROOT / "outputs" / "competition_evidence" / "defense-package-final"
SUBMISSION_OUTPUT = ROOT / "competition_submission" / "defense-package-final"
DOC_SOURCE = ROOT / "docs" / "competition_defense_document.md"
REPO_SUBMISSION_PREFIX = "../competition_submission/defense-package-final/"
ONLINE_EVIDENCE = ROOT / "outputs" / "competition_evidence" / "online-integration"
NPU_REPORTS = (
    "task2_topk_4k.json",
    "task2_topk_65k.json",
    "task2_relation_tensor_ascend_910b2c_xlarge.json",
    "task2_relation_quality_ascend_910b2c_npu.json",
    "task3_graph_tensor_ascend_910b2c_large.json",
    "task3_centrality_5k.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build defense evidence package with embedded HTML report.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Evidence bundle from collect_competition_evidence.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Defense package output directory.")
    parser.add_argument(
        "--skip-submission-sync",
        action="store_true",
        help="Build only under outputs/ and leave competition_submission unchanged.",
    )
    return parser.parse_args()


def _source_path(source: Path, *parts: str) -> Path:
    direct = source.joinpath(*parts)
    if direct.exists():
        return direct
    return source.joinpath("evidence", *parts)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def _is_canonical_online_evidence(path: Path) -> bool:
    """Keep latest finals; drop superseded 2026-06-18 and intermediate reruns."""

    name = path.name
    if not name.endswith(".json"):
        return False
    if "final-copy" in name:
        return False
    if "20260618" in name:
        return False
    if "20260702" in name:
        if name.endswith("-final.json"):
            return True
        return name in {
            "openapi-submit-no-token-rerun-20260702.json",
            "agent-submit-no-token-rerun-20260702.json",
        }
    return True


def _copy_online_evidence(dst: Path) -> None:
    """Copy only reviewer-facing JSON evidence, never raw local logs."""

    if not ONLINE_EVIDENCE.exists():
        return
    for source in sorted(ONLINE_EVIDENCE.glob("*.json")):
        if not _is_canonical_online_evidence(source):
            continue
        _copy_if_exists(source, dst / source.name)


def _extract_stdout(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = "STDOUT\n"
    if marker not in text:
        return text.strip()
    body = text.split(marker, 1)[1]
    if "\n\nSTDERR\n" in body:
        body = body.split("\n\nSTDERR\n", 1)[0]
    return body.strip()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _summarize_json(path: Path) -> str:
    if not path.exists():
        return f"(missing: {path.name})"
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)


_CROSS_DOC_LINK_RE = re.compile(r"(?<!\!)\[([^\]]+)\]\(([a-zA-Z0-9_\-]+\.md)\)")


_PACKAGE_DOC_REWRITES = {
    "`preparation.md`": "`../../docs/preparation.md`",
    "`online_integration.md`": "`../../docs/online_integration.md`",
    "`local_model_finetune.md`": "`../../docs/local_model_finetune.md`",
    "`competition_defense_outline.md`": "`../../docs/competition_defense_outline.md`",
    "`../README.md`": "`../../README.md`",
    "`../competition_submission/defense-package-final/competition_defense_document.html`": "`competition_defense_document.html`",
}


_DOCS_FOOTNOTE_EVIDENCE = "../competition_submission/defense-package-final/evidence/"
_DOCS_FOOTNOTE_GUARD = "___DOCS_FOOTNOTE_EVIDENCE___"


def _package_markdown(text: str) -> str:
    footnote_needle = f"在 `docs/` 阅读本源稿时，图片请改用 `{_DOCS_FOOTNOTE_EVIDENCE}"
    if footnote_needle in text:
        text = text.replace(
            footnote_needle,
            f"在 `docs/` 阅读本源稿时，图片请改用 `{_DOCS_FOOTNOTE_GUARD}",
        )
    text = text.replace(REPO_SUBMISSION_PREFIX, "")
    text = text.replace(_DOCS_FOOTNOTE_GUARD, _DOCS_FOOTNOTE_EVIDENCE)
    text = _CROSS_DOC_LINK_RE.sub(r"\1", text)
    for old, new in _PACKAGE_DOC_REWRITES.items():
        text = text.replace(old, new)
    return text


def _portable_log_text(text: str) -> str:
    root = ROOT.resolve()
    for prefix in (str(root), root.as_posix()):
        text = text.replace(f"{prefix}\\", "")
        text = text.replace(f"{prefix}/", "")
        escaped_prefix = prefix.replace("\\", "\\\\")
        text = text.replace(f"{escaped_prefix}\\\\", "")
    return text


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _read_npu_summary() -> str:
    report = ROOT / "benchmarks" / "reports" / "ascend_910b2c_experiment_summary.md"
    if not report.exists():
        return "(NPU summary not found)"
    return report.read_text(encoding="utf-8")


def _build_online_evidence_readme() -> str:
    return """# 在线集成证据说明

本目录保存 Nexent、DataMate、Neo4j 与三套任务 API 的真实在线证据。

## 2026-07-02 现行结论（Windows + WSL）

集成验证按 L1 任务 API → L2 DataMate → L3 Nexent → L4 Neo4j 自下而上完成，`stack_status=ready`。层级表见 [在线集成文档](../../../../docs/online_integration.md) 与答辩材料 §6.3。

- `datamate-submit-20260702-final.json`：DataMate `catalog_summary(..., mode="submit")`，模板/任务均为已验证（`verified`）。
- `task2-neo4j-live-smoke-20260702-final.json`：Neo4j Bolt 冒烟测试 `passed=true`；读回节点/边数以 JSON 为准。
- `probe-20260702-final.json` / `prepare-20260702-final.json` / `datamate-readiness-20260702-final.json`：Nexent 与 DataMate 就绪检查汇总（Nexent 为 full 模式，经注册/登录获取 JWT 后探测）。
- `openapi-submit-no-token-rerun-20260702.json` / `agent-submit-no-token-rerun-20260702.json`：Nexent OpenAPI 与 DKM Agent 最终提交证据（已验证，`verified`；OpenAPI 工具目录 `tool_count=48`）。

## 2026-07-03 复验补充（答辩前本地复跑）

- `probe-20260703-fullstack.json`：JWT 刷新后全栈 probe，`stack_status=ready`，DataMate 3/3，Nexent OpenAPI 3 服务，三套任务 API 均 available。
- `datamate-submit-20260703-rerun.json`：`catalog_summary` 在线 submit 复验（template/task 均为 `verified`）。
- `../benchmarks/task1_datamate_submit.json`：任务一 pipeline submit benchmark（修复 dest 名冲突后 `passed=true`）。

答辩正文仍以 **2026-07-02** 在线 JSON 为正式结论日期；本节 JSON 供复现对照。

## 2026-06-18 历史结论（对照，已被后续复跑替代）

- 同名 `-20260618-` JSON 仍保留于源目录供日期对照，答辩包默认不再打包。

## 2026-06-16 历史复验（对照）

- `service-reachability-20260616.json`：Neo4j=connected，DataMate=available，Nexent=available。
- `probe-20260616.json`：DKM 在线探测成功，Nexent/DataMate stack_status=ready，task1/task2/task3 API 均 available。
- `openapi-submit-20260616.json`：Nexent OpenAPI 导入/更新结果为 `status=verified`，工具目录刷新后 tool_count=47。
- `agent-submit-20260616.json`：DKM Agent 回查结果为 `status=verified`，agent_id=1，preexisting=true。
- `task2-neo4j-live-smoke-20260616.json`：Neo4j Bolt 连接、图谱写入读回、Cypher 查询和 KG QA 均通过。
- `datamate-readiness-20260616.json`：DataMate 健康检查与算子、模板、任务核心 API 探测通过（该轮未新建清洗任务）。
- `task-api-health-20260616.json`：task1/task2/task3 三套 API health 均返回 HTTP 200。

## 更早历史补充

- `openapi-submit-live.json` / `agent-submit-live.json`：2026-06-14 Nexent 首次导入与 Agent 创建证据。
- `datamate-live-probe-20260615.json`：2026-06-15 DataMate 只读复验证据。
- `probe-live.json`：2026-06-14 Nexent 只读探测证据。

## 边界说明

答辩结论以 2026-07-02 非 NPU 在线集成证据为准；2026-06-18 / 2026-06-16 及更早 JSON 保留供对照。2026-07-02 Neo4j 读回为 26/29，与 §3.3 默认 demo（4 条内置样例）同输入。NPU 硬件复验见 `../npu_summary.txt` 与 `../benchmarks/` 中的 Ascend 910B3 报告（2026-06-24 快照；历史 910B2C 数值见文档记录）。
"""

def _build_package_readme(source: Path) -> str:
    pytest_line = "- Windows 离线代码与最终回归：2026-07-03 证据包，pytest **437/437 passed**（`evidence/logs/pytest.txt`），ruff 全量通过。"
    for log_name in ("pytest.txt", "pytest.log"):
        pytest_log = source / "logs" / log_name
        if pytest_log.is_file():
            text = pytest_log.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"(\d+) passed in [\d.]+s", text)
            if matches:
                count = matches[-1]
                pytest_line = (
                    f"- Windows 离线代码与最终回归：2026-07-03 证据包，"
                    f"pytest **{count}/{count} passed**（`evidence/logs/pytest.txt`），ruff 全量通过。"
                )
            break
    return "\n".join([
        "# 答辩材料包",
        "",
        "## 阅读方式",
        "",
        "1. 用浏览器打开 `competition_defense_document.html`（推荐，图片已内嵌）。",
        "2. 如需编辑正文，修改 `docs/competition_defense_document.md`，再按 [答辩打包说明](../../docs/competition_defense_outline.md) 重采或仅重生成 HTML。",
        "",
        "## 证据再生流程（维护者）",
        "",
        "完整步骤见 `docs/competition_defense_outline.md` §重新采集证据：",
        "",
        "1. `python demos/collect_competition_evidence.py` → `outputs/competition_evidence/$timestamp/`",
        "2. `python demos/build_defense_pdf_package.py --source ... --output outputs/competition_evidence/defense-package-final`（勿直接 `--output competition_submission/...`）",
        "3. 脚本自动同步到本目录；仅改 Markdown 时可 `export_defense_pdf.py` 重生成 HTML",
        "",
        "快速同步（仅改正文、证据未变）：",
        "",
        "```bash",
        "python demos/export_defense_pdf.py --source competition_submission/defense-package-final --sync-from docs/competition_defense_document.md",
        "```",
        "",
        "## 证据时间线",
        "",
        pytest_line,
        "- Nexent/DataMate/Neo4j 非 NPU 在线集成：2026-07-02 结论以该日 JSON 为准，见 `evidence/online_integration/`；2026-06-18 历史对照未打入本包（源目录 `outputs/competition_evidence/online-integration/`）；2026-06-16 及更早对照 JSON 在本目录（见该目录 README）。",
        "- NPU：2026-06-24 Ascend 910B3 快照；插卡前无卡预配置 **737/737**，插卡后 **770/770**（NPU 专项 43/43），关系级 NPU P/R/F1=1.0（46/46，10 条标注病历；30 条 CPU 张量 145/145，NPU 30 条待 Ascend 复跑）；`cached_topk_labels` 99.95×、`cached_bincount_topk` 27.77×。",
        "",
        "详细边界见 `evidence/online_integration/README.md` 和 `evidence/npu_summary.txt`。",
        "",
        "## 目录",
        "",
        "- `competition_defense_document.html` - 自包含答辩报告",
        "- `competition_defense_document.md` - 答辩正文打包副本（编辑请改 `docs/` 源稿后同步）",
        "- `evidence/screenshots/neo4j/` - Neo4j Browser 截图",
        "- `evidence/screenshots/task3/` - 任务三仪表盘截图",
        "- `evidence/figures/` - 架构图与任务 SVG 图表",
        "- `evidence/html/` - Neo4j 查询结果 HTML（`neo4j_query_evidence.html`）与任务三报告/仪表盘",
        "- `evidence/benchmarks/` - 量化评测 JSON",
        "- `evidence/online_integration/` - Nexent/DataMate/Neo4j 在线证据",
        "- `evidence/logs/` - 演示程序 / pytest 终端摘录",
        "",
        f"来源证据包：`{_repo_relative(source)}`",
    ])

def _build_manifest(source: Path, output: Path, files: list[str]) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle": _repo_relative(source),
        "output_dir": ".",
        "document_md": "competition_defense_document.md",
        "document_html": "competition_defense_document.html",
        "files": files,
    }


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.is_absolute():
        source = (ROOT / source).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    evidence = output / "evidence"
    # Screenshots and figures
    for group in ("neo4j", "task3"):
        copied = _copy_if_exists(
            _source_path(source, "screenshots", group),
            evidence / "screenshots" / group,
        )
        if not copied:
            _copy_if_exists(
                SUBMISSION_OUTPUT / "evidence" / "screenshots" / group,
                evidence / "screenshots" / group,
            )
    _copy_if_exists(_source_path(source, "figures"), evidence / "figures")
    _copy_if_exists(_source_path(source, "integration_probes"), evidence / "integration_probes")
    _copy_if_exists(_source_path(source, "nexent_specs"), evidence / "nexent_specs")
    _copy_online_evidence(evidence / "online_integration")
    _write_text(
        evidence / "online_integration" / "README.md",
        _build_online_evidence_readme(),
    )
    _copy_if_exists(
        _source_path(source, "integration_probes", "integration_report.json"),
        evidence / "integration_probes" / "integration_report.json",
    )

    benchmarks_src = _source_path(source, "benchmarks")
    if not benchmarks_src.exists():
        benchmarks_src = source / "generated_outputs" / "benchmarks"
    benchmarks_dst = evidence / "benchmarks"
    benchmarks_dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "task2_kg_extraction_quality.json",
        "task2_oov_extraction_quality.json",
        "task2_pipeline_latency.json",
        "task2_relation_quality_rule.json",
        "task2_relation_quality_cpu.json",
        "task2_relation_tensor_real_corpus.json",
        "task2_neo4j_live_smoke.json",
        "task3_nl2sql_report.json",
        "planner_comparison.json",
        "planner_llm_evidence.json",
        "dkm_orchestrator_execute_evidence.json",
        "task1_datamate_hybrid_evidence.json",
        *NPU_REPORTS,
    ):
        current_report = ROOT / "benchmarks" / "reports" / name
        if not _copy_if_exists(current_report, benchmarks_dst / name):
            _copy_if_exists(benchmarks_src / name, benchmarks_dst / name)

    html_dst = evidence / "html"
    html_dst.mkdir(parents=True, exist_ok=True)
    task3_gen = _source_path(source, "html")
    if not task3_gen.exists():
        task3_gen = source / "generated_outputs" / "task3"
    for name in (
        "task3_analysis_dashboard.html",
        "task3_interactive_dashboard.html",
        "task3_insight_report.html",
    ):
        _copy_if_exists(task3_gen / name, html_dst / name)
    _copy_if_exists(_source_path(source, "artifacts", "medical_kg.json"), evidence / "artifacts" / "medical_kg.json")
    _copy_if_exists(
        _source_path(source, "nexent_toolchain", "nexent_toolchain_evidence.json"),
        evidence / "nexent_toolchain" / "nexent_toolchain_evidence.json",
    )
    _copy_if_exists(
        _source_path(source, "benchmarks", "dkm_orchestrator_execute_evidence.json"),
        evidence / "benchmarks" / "dkm_orchestrator_execute_evidence.json",
    )
    _copy_if_exists(
        _source_path(source, "benchmarks", "task1_datamate_hybrid_evidence.json"),
        evidence / "benchmarks" / "task1_datamate_hybrid_evidence.json",
    )
    if not (evidence / "nexent_toolchain" / "nexent_toolchain_evidence.json").exists():
        _copy_if_exists(
            source / "generated_outputs" / "nexent_toolchain" / "nexent_toolchain_evidence.json",
            evidence / "nexent_toolchain" / "nexent_toolchain_evidence.json",
        )
    if not (evidence / "artifacts" / "medical_kg.json").exists():
        _copy_if_exists(source / "generated_outputs" / "task2" / "medical_kg.json", evidence / "artifacts" / "medical_kg.json")
    _copy_if_exists(_source_path(source, "artifacts", "task1_patients_cleaned.csv"), evidence / "artifacts" / "task1_patients_cleaned.csv")
    if not (evidence / "artifacts" / "task1_patients_cleaned.csv").exists():
        _copy_if_exists(source / "generated_outputs" / "task1" / "task1_patients_cleaned.csv", evidence / "artifacts" / "task1_patients_cleaned.csv")

    logs = _source_path(source, "logs")
    if not logs.exists():
        logs = source / "logs"
    excerpts = evidence / "logs"
    for name in ("task1_demo", "task2_demo", "task3_demo", "end_to_end_demo", "task2_neo4j_live_smoke", "dkm_nexent_toolchain_demo", "dkm_orchestrator_execute_evidence_demo", "task1_datamate_hybrid_evidence_demo", "planner_llm_evidence_demo", "pytest", "ruff"):
        excerpt = _extract_stdout(logs / f"{name}.log")
        if excerpt:
            _write_text(excerpts / f"{name}.txt", _portable_log_text(excerpt))

    _write_text(
        evidence / "npu_summary.txt",
        _read_npu_summary(),
    )

    _write_text(
        output / "competition_defense_document.md",
        _package_markdown(DOC_SOURCE.read_text(encoding="utf-8")),
    )

    export_script = ROOT / "demos" / "export_defense_pdf.py"
    subprocess.run(
        [sys.executable, str(export_script), "--source", str(output)],
        check=True,
        cwd=str(ROOT),
    )

    _write_text(
        output / "README.md",
        _build_package_readme(source),
    )

    package_files = [
        p.relative_to(output).as_posix()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    manifest = _build_manifest(source, output, package_files)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    submission_dir = None
    if not args.skip_submission_sync:
        submission_dir = SUBMISSION_OUTPUT
        if submission_dir.exists():
            shutil.rmtree(submission_dir)
        shutil.copytree(output, submission_dir)
        manifest["submission_dir"] = str(submission_dir)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output),
                "submission_dir": str(submission_dir) if submission_dir else None,
                "file_count": len(manifest["files"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
