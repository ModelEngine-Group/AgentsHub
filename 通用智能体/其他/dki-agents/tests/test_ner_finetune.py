"""微调后端接回测试：类型解析向后兼容（torch 缺失时 skip 需 torch 的部分）。"""

import pytest


def test_resolve_entity_type_backward_compat():
    """真实医疗类型直接采用；占位通用 NER 标签走映射兜底。"""
    from dki.bench.npu.bert_ner_infer import _resolve_entity_type

    # 微调模型的真实类型
    assert _resolve_entity_type("Disease") == "Disease"
    assert _resolve_entity_type("Symptom") == "Symptom"
    assert _resolve_entity_type("Drug") == "Drug"
    assert _resolve_entity_type("Department") == "Department"
    # 未微调占位模型的通用 NER 标签（向后兼容）
    assert _resolve_entity_type("ORG") == "Department"
    assert _resolve_entity_type("PER") == "Disease"
    # 未知标签兜底
    assert _resolve_entity_type("SOMETHING") == "Disease"


def test_finetuned_dir_constant():
    from dki.bench.npu.bert_ner_infer import DEFAULT_FINETUNED_DIR

    # 默认微调模型目录应指向 models/medical_ner_bert
    assert DEFAULT_FINETUNED_DIR.name == "medical_ner_bert"
    assert DEFAULT_FINETUNED_DIR.parent.name == "models"


def test_backend_needs_torch_or_loads():
    """torch 缺失时给安装指引；存在时能构造（自动探测微调模型/回退基线）。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        from dki.bench.npu.bert_ner_infer import BertNerBackend

        with pytest.raises(RuntimeError) as ei:
            BertNerBackend(device="cpu")
        assert "pip install torch transformers" in str(ei.value)
