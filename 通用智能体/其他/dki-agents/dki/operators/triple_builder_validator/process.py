"""TripleBuilderValidator —— 三元组生成与校验算子。

对上游关系抽取结果执行四道工序：
1. 生成：关系 -> 三元组（补 source=文档ID）；
2. Schema 校验：实体/关系类型合法性、定义域/值域约束、自环检查
   （优先复用 dki.kg.schema.validate_triple；算子市场独立部署时使用
   文件内置的等价约束表）；
3. 去重合并：同 (head, relation, tail) 取最高置信度并合并证据；
4. 置信度过滤：minConfidence 以下丢弃。

输出附带 validation 统计（候选数/通过数/淘汰原因分布），保证可解释性。
"""

import json
import time
from typing import Any, Dict, List, Tuple

from loguru import logger

from dki.dmcompat import Mapper, OPERATORS

try:
    from dki.kg.schema import validate_triple as _validate
except ImportError:
    # 算子市场独立部署回退：与 dki/kg/schema.py 等价的最小约束表
    _ENTITY_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}
    _REL_CONSTRAINTS = {
        "HAS_SYMPTOM": ("Disease", "Symptom"),
        "TREATED_BY_DRUG": ("Disease", "Drug"),
        "TREATED_BY": ("Disease", "Treatment"),
        "DIAGNOSED_BY": ("Disease", "Exam"),
        "HAS_COMPLICATION": ("Disease", "Disease"),
        "CONTRAINDICATED_FOR": ("Drug", "Disease"),
        "BELONGS_TO_DEPT": ("Disease", "Department"),
    }

    def _validate(head, head_type, relation, tail, tail_type) -> Tuple[bool, str]:
        if not head or not tail:
            return False, "头/尾实体为空"
        if head_type not in _ENTITY_TYPES:
            return False, f"未知头实体类型: {head_type}"
        if tail_type not in _ENTITY_TYPES:
            return False, f"未知尾实体类型: {tail_type}"
        c = _REL_CONSTRAINTS.get(relation)
        if c is None:
            return False, f"未知关系类型: {relation}"
        if c[0] != head_type:
            return False, f"关系 {relation} 要求头实体为 {c[0]}，实际 {head_type}"
        if c[1] != tail_type:
            return False, f"关系 {relation} 要求尾实体为 {c[1]}，实际 {tail_type}"
        if head == tail:
            return False, "头尾实体相同（自环）"
        return True, "ok"


class TripleBuilderValidator(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_confidence = float(kwargs.get("minConfidence", 0.55))

    def execute(self, sample):
        start = time.time()
        try:
            payload = json.loads(sample[self.text_key])
        except json.JSONDecodeError as e:
            raise ValueError(f"上游输出不是 JSON（需先经 MedicalRelationExtractor）: {e}")
        relations: List[Dict[str, Any]] = payload.get("relations", [])
        doc_id = str(sample.get(self.fileid_key, ""))

        n_candidates = len(relations)
        reject_reasons: Dict[str, int] = {}
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for r in relations:
            ok, reason = _validate(
                r.get("head", ""), r.get("head_type", ""), r.get("relation", ""),
                r.get("tail", ""), r.get("tail_type", ""),
            )
            if not ok:
                key = reason.split("，")[0][:30]
                reject_reasons[key] = reject_reasons.get(key, 0) + 1
                continue
            conf = float(r.get("confidence", 1.0))
            if conf < self.min_confidence:
                reject_reasons["低于置信度阈值"] = reject_reasons.get("低于置信度阈值", 0) + 1
                continue
            k = (r["head"], r["relation"], r["tail"])
            if k in merged:
                t = merged[k]
                if conf > t["confidence"]:
                    t["confidence"] = conf
                    t["rule"] = r.get("rule", t.get("rule", ""))
                if r.get("evidence") and r["evidence"] not in t["evidence"]:
                    if len(t["evidence"]) < 300:
                        t["evidence"] = (t["evidence"] + " / " + r["evidence"])[:300]
            else:
                merged[k] = {
                    "head": r["head"], "head_type": r["head_type"],
                    "relation": r["relation"],
                    "tail": r["tail"], "tail_type": r["tail_type"],
                    "confidence": round(conf, 3),
                    "evidence": r.get("evidence", ""),
                    "rule": r.get("rule", ""),
                    "source": doc_id,
                }

        triples = list(merged.values())
        n_rejected = sum(reject_reasons.values())
        validation = {
            "n_candidates": n_candidates,
            "n_valid": len(triples),
            "n_rejected": n_rejected,
            "n_merged_dup": n_candidates - n_rejected - len(triples),
            "reject_reasons": reject_reasons,
        }

        out = {"triples": triples, "validation": validation}
        sample[self.text_key] = json.dumps(out, ensure_ascii=False)
        ext = sample.get(self.ext_params_key) or {}
        ext["triples"] = triples
        ext["validation"] = validation
        sample[self.ext_params_key] = ext
        logger.info(
            "fileName: {}, method: TripleBuilderValidator costs {:.6f} s, "
            "candidates={} valid={} rejected={}",
            sample.get(self.filename_key), time.time() - start,
            n_candidates, len(triples), validation["n_rejected"],
        )
        return sample
