from demos.build_defense_pdf_package import (
    ROOT,
    _build_manifest,
    _build_online_evidence_readme,
    _build_package_readme,
    _copy_online_evidence,
    _is_canonical_online_evidence,
    _package_markdown,
    _portable_log_text,
    _read_npu_summary,
)
from demos.export_defense_pdf import _extract_material_date, _markdown_to_html_body


def test_markdown_table_alignment_row_is_not_rendered(tmp_path):
    body = _markdown_to_html_body(
        "| 指标 | 数值 |\n| --- | ---: |\n| F1 | 1.000 |\n",
        tmp_path,
    )

    assert "<th>指标</th>" in body
    assert "<td>F1</td>" in body
    assert "<td>---</td>" not in body
    assert "<td>---:</td>" not in body


def test_package_markdown_rewrites_repo_image_paths():
    source = "![图](../competition_submission/defense-package-final/evidence/figures/a.svg)"

    assert _package_markdown(source) == "![图](evidence/figures/a.svg)"


def test_package_markdown_demotes_cross_doc_links_to_plain_text():
    source = "见 [任务一：数据处理智能体](task1_data_agent.md)。"

    assert _package_markdown(source) == "见 任务一：数据处理智能体。"


def test_material_date_is_read_from_markdown_metadata():
    source = "# 技术答辩材料\n\n**材料日期：** 2026-06-14\n"

    assert _extract_material_date(source) == "2026-06-14"


def test_package_readme_uses_submission_path_and_explains_online_timeline(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pytest.txt").write_text("437 passed in 50.78s\n", encoding="utf-8")

    readme = _build_package_readme(tmp_path)

    assert (
        "python demos/export_defense_pdf.py "
        "--source competition_submission/defense-package-final"
    ) in readme
    assert "--sync-from docs/competition_defense_document.md" in readme
    assert "2026-07-03" in readme
    assert "437/437" in readme
    assert "2026-06-24" in readme
    assert "NPU：2026-06-24 Ascend 910B3 快照" in readme
    assert "770/770" in readme
    assert "evidence/online_integration/README.md" in readme


def test_export_sync_from_rewrites_package_markdown(tmp_path):
    import sys
    from unittest.mock import patch

    from demos.export_defense_pdf import main

    source = tmp_path / "package"
    source.mkdir()
    sync_source = tmp_path / "docs" / "competition_defense_document.md"
    sync_source.parent.mkdir(parents=True)
    sync_source.write_text(
        "![图](../competition_submission/defense-package-final/evidence/figures/a.svg)\n",
        encoding="utf-8",
    )

    with patch.object(sys, "argv", [
        "export_defense_pdf.py",
        "--source",
        str(source),
        "--sync-from",
        str(sync_source),
        "--output",
        str(source / "competition_defense_document.html"),
    ]):
        assert main() == 0

    packaged = (source / "competition_defense_document.md").read_text(encoding="utf-8")
    assert packaged == "![图](evidence/figures/a.svg)\n"
    assert (source / "competition_defense_document.html").is_file()


def test_manifest_uses_portable_repository_paths():
    source = ROOT / "outputs" / "competition_evidence" / "final-review"
    output = ROOT / "outputs" / "competition_evidence" / "defense-package-review"

    manifest = _build_manifest(source, output, ["README.md"])

    assert manifest["source_bundle"] == "outputs/competition_evidence/final-review"
    assert manifest["output_dir"] == "."
    assert manifest["files"] == ["README.md"]
    assert "Administrator" not in str(manifest)
    assert "\\" not in manifest["source_bundle"]


def test_npu_summary_is_complete_and_marks_snapshot_boundary():
    summary = _read_npu_summary()

    assert len(summary) > 3_000
    assert "## 复现命令" in summary
    assert "2026-06-24" in summary
    assert "770/770" in summary
    assert "910B3" in summary
    assert "P/R/F1" in summary


def test_online_evidence_readme_explains_chronology_and_scope():
    readme = _build_online_evidence_readme()

    assert "2026-07-02 现行结论" in readme
    assert "datamate-submit-20260702-final.json" in readme
    assert "2026-06-16 历史复验" in readme
    assert "service-reachability-20260616.json" in readme
    assert "openapi-submit-20260616.json" in readme
    assert "该轮未新建清洗任务" in readme
    assert "NPU 硬件复验见" in readme


def test_is_canonical_online_evidence_filters_20260618_intermediates(tmp_path):
    superseded = tmp_path / "probe-20260618-final.json"
    intermediate = tmp_path / "probe-20260618.json"
    rerun = tmp_path / "openapi-submit-no-token-rerun-20260618.json"
    duplicate = tmp_path / "datamate-submit-20260618-final-copy.json"
    historical = tmp_path / "probe-20260616.json"
    current_final = tmp_path / "probe-20260702-final.json"
    current_rerun = tmp_path / "openapi-submit-no-token-rerun-20260702.json"

    for path in (
        superseded,
        intermediate,
        rerun,
        duplicate,
        historical,
        current_final,
        current_rerun,
    ):
        path.write_text("{}\n", encoding="utf-8")

    assert _is_canonical_online_evidence(superseded) is False
    assert _is_canonical_online_evidence(rerun) is False
    assert _is_canonical_online_evidence(historical) is True
    assert _is_canonical_online_evidence(intermediate) is False
    assert _is_canonical_online_evidence(duplicate) is False
    assert _is_canonical_online_evidence(current_final) is True
    assert _is_canonical_online_evidence(current_rerun) is True


def test_online_evidence_copy_excludes_raw_logs_and_intermediates(monkeypatch, tmp_path):
    source = tmp_path / "online-source"
    source.mkdir()
    (source / "probe.json").write_text('{"status":"available"}\n', encoding="utf-8")
    (source / "probe-20260702-final.json").write_text('{"status":"final"}\n', encoding="utf-8")
    (source / "probe-20260618-final.json").write_text('{"status":"superseded"}\n', encoding="utf-8")
    (source / "probe-20260618.json").write_text('{"status":"intermediate"}\n', encoding="utf-8")
    (source / "probe.log").write_text("local path and diagnostics\n", encoding="utf-8")
    target = tmp_path / "package" / "online_integration"
    monkeypatch.setattr(
        "demos.build_defense_pdf_package.ONLINE_EVIDENCE",
        source,
    )

    _copy_online_evidence(target)

    assert (target / "probe.json").is_file()
    assert (target / "probe-20260702-final.json").is_file()
    assert not (target / "probe-20260618-final.json").exists()
    assert not (target / "probe-20260618.json").exists()
    assert not (target / "probe.log").exists()


def test_packaged_log_text_rewrites_repository_absolute_paths():
    log = f"input: {ROOT}\\data\\samples\\task1_patients.csv"
    json_log = (
        '{"input": "'
        + str(ROOT).replace("\\", "\\\\")
        + '\\\\data\\\\samples\\\\task1_patients.csv"}'
    )

    portable = _portable_log_text(log)
    portable_json = _portable_log_text(json_log)

    assert "Administrator" not in portable
    assert "data\\samples\\task1_patients.csv" in portable
    assert "Administrator" not in portable_json
    assert "data\\\\samples\\\\task1_patients.csv" in portable_json
