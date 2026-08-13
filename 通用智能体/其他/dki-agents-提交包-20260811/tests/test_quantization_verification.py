import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quantize_ner_int8.py"
_spec = importlib.util.spec_from_file_location("quantize_ner_int8", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
verify_selective_quantization = _module.verify_selective_quantization


def _save(path, *, disabled=True, quantized_disabled=False, dead_quant=False):
    initializers = [
        helper.make_tensor("disabled_weight", TensorProto.FLOAT, [1, 1], [1.0]),
        helper.make_tensor("other_weight", TensorProto.FLOAT, [1, 1], [1.0]),
        helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1]),
        helper.make_tensor("zero", TensorProto.UINT8, [1], [0]),
    ]
    nodes = []
    if disabled:
        protected_weight = "disabled_weight"
        if quantized_disabled:
            nodes.extend([
                helper.make_node("QuantizeLinear", ["disabled_weight", "scale", "zero"], ["disabled_q"], name="disabled_quant"),
                helper.make_node("DequantizeLinear", ["disabled_q", "scale", "zero"], ["disabled_dq"], name="disabled_dequant"),
            ])
            protected_weight = "disabled_dq"
        nodes.append(helper.make_node("MatMul", ["x", protected_weight], ["protected_out"], name="disabled_node"))

    nodes.extend([
        helper.make_node("QuantizeLinear", ["other_weight", "scale", "zero"], ["other_q"], name="other_quant"),
        helper.make_node("DequantizeLinear", ["other_q", "scale", "zero"], ["other_dq"], name="other_dequant"),
        helper.make_node("MatMul", ["x", "other_dq"], ["other_out"], name="other_matmul"),
    ])
    output = "other_out"
    if disabled:
        nodes.append(helper.make_node("Add", ["protected_out", "other_out"], ["combined"], name="combine"))
        output = "combined"
    if dead_quant:
        nodes.append(helper.make_node("QuantizeLinear", ["disabled_weight", "scale", "zero"], ["dead"], name="dead_quant"))
    graph = helper.make_graph(
        nodes,
        "graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, 1])],
        initializer=initializers,
    )
    onnx.save(helper.make_model(graph), str(path))


def test_selective_quantization_requires_requested_component_after_export(tmp_path):
    before = tmp_path / "before.onnx"
    after = tmp_path / "after.onnx"
    _save(before)
    _save(after, disabled=False)
    with pytest.raises(RuntimeError, match="cannot verify disabled"):
        verify_selective_quantization(before, after, ["disabled_node"])


def test_selective_quantization_rejects_unknown_requested_name(tmp_path):
    before = tmp_path / "before.onnx"
    after = tmp_path / "after.onnx"
    _save(before)
    _save(after)
    with pytest.raises(RuntimeError, match="not present"):
        verify_selective_quantization(before, after, ["typo_node"])


def test_selective_quantization_rejects_requested_operator_on_quantized_path(tmp_path):
    before = tmp_path / "before.onnx"
    after = tmp_path / "after.onnx"
    _save(before)
    _save(after, quantized_disabled=True)
    with pytest.raises(RuntimeError, match="still quantized"):
        verify_selective_quantization(before, after, ["disabled_node"])


def test_selective_quantization_accepts_live_graph_proof(tmp_path):
    before = tmp_path / "before.onnx"
    after = tmp_path / "after.onnx"
    _save(before)
    _save(after)
    proof = verify_selective_quantization(before, after, ["disabled_node"])
    assert proof["verified"] is True
    assert proof["requested_quantizable_compute"] == ["disabled_node"]
    assert proof["recognized_quantization_nodes"] == 2


def test_dead_quantization_is_not_evidence(tmp_path):
    before = tmp_path / "before.onnx"
    after = tmp_path / "after.onnx"
    _save(before, disabled=True)
    # Build a graph whose only quantization node is disconnected from output.
    initializers = [
        helper.make_tensor("weight", TensorProto.FLOAT, [1, 1], [1.0]),
        helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1]),
        helper.make_tensor("zero", TensorProto.UINT8, [1], [0]),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "weight"], ["out"], name="disabled_node"),
            helper.make_node("QuantizeLinear", ["weight", "scale", "zero"], ["dead"], name="dead_quant"),
        ],
        "dead",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])],
        initializer=initializers,
    )
    onnx.save(helper.make_model(graph), str(after))
    with pytest.raises(RuntimeError, match="no recognized quantization"):
        verify_selective_quantization(before, after, ["disabled_node"])


def test_full_int8_is_not_mixed_precision_verified(tmp_path):
    before = tmp_path / "before.onnx"
    after = tmp_path / "after.onnx"
    _save(before)
    _save(after)
    proof = verify_selective_quantization(before, after, [])
    assert proof["verified"] is False
