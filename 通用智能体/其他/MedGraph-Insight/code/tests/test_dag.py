"""任务一 DAG 编排、数据质量指标与血缘追踪测试。"""

from __future__ import annotations

import pytest

from mgi.datamate.dag import OperatorDAG, data_quality

SRC = [
    "<p>患者手机号13800138000，繁體字檢驗</p>",
    "<p>患者手机号13800138000，繁體字檢驗</p>",
    "短",
    "正常一条医疗记录文本内容足够长度测试",
]


def _etl_dag() -> OperatorDAG:
    return (OperatorDAG()
            .add("clean_html", "HtmlTagCleaner")
            .add("t2s", "TraditionalChineseCleaner", inputs=["clean_html"])
            .add("anon_phone", "PhoneAnonymizer", inputs=["t2s"])
            .add("ws", "WhitespaceNormalizer", inputs=["anon_phone"])
            .add("dedup", "DedupFilter", inputs=["ws"])
            .add("minlen", "MinLengthFilter", inputs=["dedup"], min_len=10))


def test_dag_topological_order() -> None:
    order = _etl_dag().topo_order()
    assert order.index("clean_html") < order.index("t2s") < order.index("dedup")


def test_dag_cycle_detection() -> None:
    dag = OperatorDAG()
    dag.add("a", "HtmlTagCleaner", inputs=["b"])
    dag.add("b", "DedupFilter", inputs=["a"])
    with pytest.raises(ValueError, match="环"):
        dag.topo_order()


def test_dag_unknown_upstream() -> None:
    dag = OperatorDAG().add("a", "HtmlTagCleaner", inputs=["missing"])
    with pytest.raises(ValueError, match="不存在的上游"):
        dag.topo_order()


def test_dag_run_end_to_end() -> None:
    run = _etl_dag().run(SRC)
    final = run["final"]
    assert len(final) == 2  # 去重 + 去短文本
    assert all("<PHONE>" in r or "13800138000" not in r for r in final)
    assert all("<" not in r or "<PHONE>" in r for r in final)


def test_dag_quality_metrics_improve() -> None:
    run = _etl_dag().run(SRC)
    by_id = {r.id: r for r in run["results"]}
    assert by_id["anon_phone"].dq_in["pii_leak_rate"] > 0
    assert by_id["anon_phone"].dq_out["pii_leak_rate"] == 0
    assert by_id["dedup"].dq_in["dup_rate"] > 0
    assert by_id["dedup"].dq_out["dup_rate"] == 0
    assert by_id["clean_html"].dq_in["html_rate"] > 0
    assert by_id["clean_html"].dq_out["html_rate"] == 0


def test_dag_lineage() -> None:
    lineage = _etl_dag().lineage()
    assert lineage["clean_html"] == []
    assert lineage["minlen"] == ["clean_html", "t2s", "anon_phone", "ws", "dedup"]


def test_dag_branch_and_merge() -> None:
    dag = (OperatorDAG()
           .add("html", "HtmlTagCleaner")
           .add("phone", "PhoneAnonymizer", inputs=["html"])
           .add("idcard", "IDCardAnonymizer", inputs=["html"])
           .add("merge", "DedupFilter", inputs=["phone", "idcard"]))
    run = dag.run(["<b>x13800138000y</b>"])
    assert run["sinks"] == ["merge"]
    assert len(run["outputs"]["merge"]) == 2  # 两条支路结果不同，合并后保留两条


def test_data_quality_empty() -> None:
    dq = data_quality([])
    assert dq["count"] == 0 and dq["dup_rate"] == 0.0
