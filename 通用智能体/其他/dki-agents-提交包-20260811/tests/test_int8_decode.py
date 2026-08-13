"""带类型 BIO 解码测试：decode_typed 手构 logits 校验（纯 numpy，不依赖 acl/torch）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ 不是包，直接加入 sys.path 后导入
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_ID2LABEL = {
    0: "O", 1: "B-Disease", 2: "I-Disease", 3: "B-Symptom", 4: "I-Symptom",
    5: "B-Drug", 6: "I-Drug", 7: "B-Exam", 8: "I-Exam",
    9: "B-Treatment", 10: "I-Treatment", 11: "B-Department", 12: "I-Department",
}


def _decoder():
    try:
        from acl_om_pipeline_bench import VectorizedBioDecoder
    except Exception as e:  # numpy 缺失等
        pytest.skip(f"无法导入 VectorizedBioDecoder: {e}")
    return VectorizedBioDecoder(_ID2LABEL)


def test_decode_typed_two_entities():
    dec = _decoder()
    # seq8: [CLS] B-Disease I-Disease O B-Drug O [SEP] pad
    pred = np.array([[0, 1, 2, 0, 5, 0, 0, 0]])
    offsets = np.array([[[0, 0], [1, 3], [3, 5], [5, 6], [6, 9], [9, 10], [0, 0], [0, 0]]])
    n, _spans = dec.decode(pred, offsets)
    typed = dec.decode_typed(pred, offsets)
    assert n == 2
    types = sorted(e["type"] for e in typed[0])
    assert types == ["Disease", "Drug"]
    # 字符跨度还原
    disease = [e for e in typed[0] if e["type"] == "Disease"][0]
    assert disease["char_start"] == 1 and disease["char_end"] == 5


def test_decode_typed_empty():
    dec = _decoder()
    pred = np.array([[0, 0, 0, 0]])
    offsets = np.array([[[0, 0], [1, 2], [2, 3], [0, 0]]])
    typed = dec.decode_typed(pred, offsets)
    assert typed == [[]]


def test_malformed_i_transitions_follow_canonical_state_machine():
    dec = _decoder()
    # Isolated I starts an entity; changing I type closes and starts a new one.
    pred = np.array([[0, 2, 2, 6, 0, 1, 4, 0]])
    offsets = np.array([[[0, 0], [0, 1], [1, 2], [2, 3], [3, 4],
                         [4, 5], [5, 6], [0, 0]]])
    typed = dec.decode_typed(pred, offsets)[0]
    assert [(item["type"], item["char_start"], item["char_end"]) for item in typed] == [
        ("Disease", 0, 2),
        ("Drug", 2, 3),
        ("Disease", 4, 5),
        ("Symptom", 5, 6),
    ]


def test_decode_validates_shapes():
    dec = _decoder()
    with pytest.raises(ValueError, match="shape mismatch"):
        dec.decode(np.zeros((1, 2), dtype=int), np.zeros((1, 2), dtype=int))


def test_decode_typed_special_tokens_skipped():
    dec = _decoder()
    # B- 出现在 special 位置(offset 0,0) 应被 valid 掩码过滤
    pred = np.array([[1, 1, 2, 0]])
    offsets = np.array([[[0, 0], [0, 2], [2, 4], [0, 0]]])
    n, _ = dec.decode(pred, offsets)
    typed = dec.decode_typed(pred, offsets)
    # 只有第 1 个 token(offset 0,2) 是有效 B-Disease
    assert n == 1 and len(typed[0]) == 1 and typed[0][0]["type"] == "Disease"
