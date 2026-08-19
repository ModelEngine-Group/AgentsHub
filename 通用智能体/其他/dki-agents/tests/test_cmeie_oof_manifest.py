import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oof", ROOT / "scripts" / "build_cmeie_oof_manifest.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_oof_partitions_cover_fit_once_without_family_leakage(tmp_path):
    rows = [{"text": f"title{i}@body", "spo_list": [{"predicate": "P"}]} for i in range(20)]
    source = tmp_path / "train.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    base = {"source_train_sha256": MODULE.sha(source), "splits": {"fit": {"indices": list(range(20)), "row_sha256": "x"}}}
    base_path = tmp_path / "base.json"; base_path.write_text(json.dumps(base), encoding="utf-8")
    manifest = MODULE.make_manifest(rows, source, base, base_path, "20260725")
    seen = [index for fold in manifest["partitions"] for index in fold["validation_indices"]]
    assert sorted(seen) == list(range(20))
    assert len(seen) == len(set(seen))
    assert all(not fold["family_intersection"] for fold in manifest["partitions"])
    assert [fold["n_validation"] for fold in manifest["partitions"]] == [4] * 5
