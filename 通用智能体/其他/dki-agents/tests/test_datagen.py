"""数据生成器测试（小参数 + tmp_path，验证确定性与金标准一致性）。"""

import json

from dki.datagen.generate import generate_corpus


def test_generate_small_corpus(tmp_path):
    out = tmp_path / "raw"
    gold = tmp_path / "gold"
    stats = generate_corpus(n_docs=8, seed=7, out_dir=out, gold_dir=gold)
    assert stats["n_docs"] == 8
    assert len(list(out.glob("emr_*.txt"))) == 8
    assert len(list((gold / "triples").glob("*.json"))) == 8
    assert (out / "structured" / "patients.csv").exists()
    # visits 行数 = 文档数 + 表头
    lines = (out / "structured" / "visits.csv").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 9


def test_gold_consistency_with_text(tmp_path):
    """金标准实体的 mention 必须出现在（生成的干净语义中即噪声前）文本里。

    噪声不会删除实体词（只加标签/全角/重复等），故 mention 应在噪声文本中可找到
    （全角化只影响数字，实体词为汉字不受影响）。
    """
    out = tmp_path / "raw"
    gold = tmp_path / "gold"
    generate_corpus(n_docs=6, seed=11, out_dir=out, gold_dir=gold)
    checked = 0
    for ent_file in sorted((gold / "entities").glob("*.json")):
        text = (out / f"{ent_file.stem}.txt").read_text(encoding="utf-8")
        for ent in json.loads(ent_file.read_text(encoding="utf-8")):
            assert ent["mention"] in text, f"{ent_file.stem}: {ent['mention']} 不在文本中"
            checked += 1
    assert checked > 20


def test_determinism(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_corpus(n_docs=4, seed=42, out_dir=a / "raw", gold_dir=a / "gold")
    generate_corpus(n_docs=4, seed=42, out_dir=b / "raw", gold_dir=b / "gold")
    ta = (a / "raw" / "emr_0001.txt").read_text(encoding="utf-8")
    tb = (b / "raw" / "emr_0001.txt").read_text(encoding="utf-8")
    assert ta == tb
