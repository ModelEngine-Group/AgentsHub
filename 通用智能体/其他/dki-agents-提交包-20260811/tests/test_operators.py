"""医疗算子测试：清洗 / 分段 / 实体 / 关系 / 三元组 + 全链 e2e。"""

import json

import pytest

from dki.pipeline import LocalPipelineRunner, make_sample

NOISY = (
    "<p>基本信息：李四，男，62岁，来自北京。</p>\n"
    "主　诉：反复胸痛　３天。&nbsp;<br/>\n"
    "现病史：患者3天前出现胸痛，伴胸闷、气短。患者3天前出现胸痛，伴胸闷、气短。\n"
    "既往史：胃溃疡病史5年。\n"
    "辅助检查：行心电图检查，结果提示异常。完善冠脉造影，明确诊断。\n"
    "诊断：冠心病。病程中并发心律失常。\n"
    "治疗方案：给予阿托伐他汀口服治疗。因患者既往有胃溃疡病史，阿司匹林禁用。\n"
    "医嘱：于心血管内科门诊就诊。联系电话：13912345678。\n"
)


@pytest.fixture(scope="module")
def runner():
    return LocalPipelineRunner()


@pytest.fixture(scope="module")
def kg_chain_result(runner):
    """全链产物（module 级缓存，多个测试复用）。"""
    process = [
        {"MedicalTextCleaner": {"maskPii": True}},
        {"MedicalSectionSplitter": {}},
        {"MedicalEntityExtractor": {"backend": "dict"}},
        {"MedicalRelationExtractor": {}},
        {"TripleBuilderValidator": {"minConfidence": 0.55}},
    ]
    return runner.run(process, [make_sample(text=NOISY, file_id="t1")], name="test-chain")


def test_cleaner_removes_noise(runner):
    res = runner.run([{"MedicalTextCleaner": {"maskPii": True}}],
                     [make_sample(text=NOISY, file_id="c1")])
    text = res.ok_samples[0]["text"]
    assert "<p>" not in text and "<br/>" not in text and "&nbsp;" not in text
    assert "３" not in text and "主　诉" not in text
    assert "13912345678" not in text and "<tel>" in text
    # 重复句去重
    assert text.count("患者3天前出现胸痛") == 1


def test_section_splitter(runner):
    res = runner.run(
        [{"MedicalTextCleaner": {}}, {"MedicalSectionSplitter": {}}],
        [make_sample(text=NOISY, file_id="s1")],
    )
    payload = json.loads(res.ok_samples[0]["text"])
    secs = payload["sections"]
    assert "主诉" in secs and "诊断" in secs and "治疗方案" in secs
    assert "冠心病" in secs["诊断"]


def test_entity_extractor_alias_and_span(runner):
    res = runner.run(
        [{"MedicalTextCleaner": {}}, {"MedicalSectionSplitter": {}},
         {"MedicalEntityExtractor": {"backend": "dict"}}],
        [make_sample(text=NOISY, file_id="e1")],
    )
    ents = json.loads(res.ok_samples[0]["text"])["entities"]
    by_name = {(e["name"], e["type"]) for e in ents}
    assert ("冠心病", "Disease") in by_name
    assert ("胸痛", "Symptom") in by_name
    assert ("阿托伐他汀", "Drug") in by_name
    assert ("心血管内科", "Department") in by_name
    # 别名归一：原文"冠脉造影" -> 规范名"冠状动脉造影"
    assert ("冠状动脉造影", "Exam") in by_name
    exam = next(e for e in ents if e["name"] == "冠状动脉造影")
    assert exam["mention"] == "冠脉造影" and exam["span"] is not None


def test_relation_extractor_rules(kg_chain_result):
    # 用全链前一步的关系检查：从 ext_params 读
    sample = kg_chain_result.samples[0]
    triples = sample["ext_params"]["triples"]
    keys = {(t["head"], t["relation"], t["tail"]) for t in triples}
    assert ("冠心病", "HAS_SYMPTOM", "胸痛") in keys
    assert ("冠心病", "TREATED_BY_DRUG", "阿托伐他汀") in keys
    assert ("冠心病", "DIAGNOSED_BY", "心电图") in keys
    assert ("冠心病", "HAS_COMPLICATION", "心律失常") in keys
    assert ("冠心病", "BELONGS_TO_DEPT", "心血管内科") in keys
    assert ("阿司匹林", "CONTRAINDICATED_FOR", "胃溃疡") in keys
    # 禁忌语境的药物不得成为治疗药物
    assert ("冠心病", "TREATED_BY_DRUG", "阿司匹林") not in keys
    # 既往史疾病不作为关系头
    assert not any(t["head"] == "胃溃疡" and t["relation"] != "CONTRAINDICATED_FOR" for t in triples)


def test_triple_validator_stats(kg_chain_result):
    sample = kg_chain_result.samples[0]
    v = sample["ext_params"]["validation"]
    assert v["n_valid"] > 0 and v["n_candidates"] >= v["n_valid"]
    out = json.loads(sample["text"])
    assert all(t["confidence"] >= 0.55 for t in out["triples"])


def test_validator_rejects_bad_types():
    from dki.pipeline.runner import resolve_operator

    cls = resolve_operator("TripleBuilderValidator")
    op = cls(op_name="TripleBuilderValidator")
    bad = {
        "relations": [
            {"head": "高血压", "head_type": "Disease", "relation": "TREATED_BY_DRUG",
             "tail": "头痛", "tail_type": "Symptom", "confidence": 0.9},   # 值域违规
            {"head": "高血压", "head_type": "Disease", "relation": "HAS_SYMPTOM",
             "tail": "头痛", "tail_type": "Symptom", "confidence": 0.3},   # 低置信
            {"head": "高血压", "head_type": "Disease", "relation": "HAS_SYMPTOM",
             "tail": "头晕", "tail_type": "Symptom", "confidence": 0.9},   # 合法
        ]
    }
    sample = make_sample(text=json.dumps(bad, ensure_ascii=False), file_id="v1")
    out = json.loads(op.execute(sample)["text"])
    assert len(out["triples"]) == 1
    assert out["validation"]["n_rejected"] == 2
    assert out["validation"]["reject_reasons"]


def test_full_chain_e2e(kg_chain_result):
    assert kg_chain_result.report.status.value == "COMPLETED"
    out = json.loads(kg_chain_result.ok_samples[0]["text"])
    assert out["triples"] and out["validation"]["n_valid"] == len(out["triples"])
