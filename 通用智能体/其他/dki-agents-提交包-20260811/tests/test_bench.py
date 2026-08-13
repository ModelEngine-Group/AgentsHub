"""基准模块测试：无数据也能跑通；bert 后端缺 torch 时给出安装指引。"""

import pytest

from dki.bench.operator_bench import run_bench


def test_operator_bench_smoke(tmp_path):
    md = run_bench(input_dir=tmp_path / "nodata", repeat=1, limit=10, out_dir=tmp_path / "bench")
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    assert "吞吐(docs/s)" in content and "MedicalEntityExtractor" in content
    assert (tmp_path / "bench" / "operator_bench.csv").exists()


def test_bert_backend_hint_without_torch():
    try:
        import torch  # noqa: F401

        pytest.skip("本环境已安装 torch，跳过缺失提示测试")
    except ImportError:
        pass
    from dki.bench.npu.bert_ner_infer import BertNerBackend

    with pytest.raises(RuntimeError) as ei:
        BertNerBackend(device="cpu")
    assert "pip install torch transformers" in str(ei.value)
