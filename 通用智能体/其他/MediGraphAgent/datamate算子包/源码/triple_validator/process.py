# -*- coding: utf-8 -*-
"""DataMate operator: triple validation & governance (rule-based, self-contained).

Reads sample['text'] expected to be JSON {"triples":[...]} (e.g. MedicalRE output)
and applies: schema legality, dedup/normalization, confidence filtering and
conflict detection. Writes {"valid":[...], "rejected":[...]} with reasons.
"""
import json
from typing import Any, Dict

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

from datamate.core.base_op import Mapper

_LESION = {"Disease", "Tumor"}
RELATION_CONSTRAINTS = {
    "has_symptom": (_LESION, {"Symptom"}),
    "recommend_drug": (_LESION, {"Drug"}),
    "need_examination": (_LESION, {"Examination"}),
    "complication": (_LESION, _LESION),
    "contraindication": ({"Drug"}, _LESION | {"Symptom"}),
    "treated_in_department": (_LESION, {"Department"}),
    "positive_marker": (_LESION, {"Biomarker"}),
    "negative_marker": (_LESION, {"Biomarker"}),
    "associated_gene": (_LESION, {"Gene"}),
    "has_morphology": (_LESION, {"Morphology"}),
    "located_in": (_LESION | {"Symptom"}, {"Body"}),
    "subtype_of": (_LESION, _LESION),
    "treated_by_procedure": (_LESION, {"Procedure"}),
    "adverse_reaction": ({"Drug"}, {"Symptom", "Disease"}),
}
_CONFLICTS = {("positive_marker", "negative_marker")}


class TripleValidator(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_confidence = float(kwargs.get("minConfidence", 0.5))

    @staticmethod
    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self.read_file_first(sample)
        text = (sample.get(self.text_key, "") or "").strip()
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {}
        triples = payload.get("triples", []) if isinstance(payload, dict) else []

        valid, rejected = [], []
        seen, pair_rel = set(), {}
        for t in triples:
            if not isinstance(t, dict):
                continue
            head, tail = str(t.get("head", "")).strip(), str(t.get("tail", "")).strip()
            rel = str(t.get("relation", "")).strip()
            ht, tt = str(t.get("head_type", "")).strip(), str(t.get("tail_type", "")).strip()
            conf = self._as_float(t.get("confidence", 0.0))

            if not head or not tail or not rel:
                rejected.append({"triple": t, "reason": "missing head/tail/relation"}); continue
            if rel not in RELATION_CONSTRAINTS:
                rejected.append({"triple": t, "reason": f"unknown relation '{rel}'"}); continue
            head_ok, tail_ok = RELATION_CONSTRAINTS[rel]
            if ht not in head_ok or tt not in tail_ok:
                rejected.append({"triple": t, "reason": f"schema violation {ht}-{rel}->{tt}"}); continue
            if conf < self.min_confidence:
                rejected.append({"triple": t, "reason": f"low confidence {conf:.2f}"}); continue
            key = f"{ht}:{head.lower()}|{rel}|{tt}:{tail.lower()}"
            if key in seen:
                continue
            pair = (head.lower(), tail.lower())
            rels = pair_rel.setdefault(pair, set())
            conflict = next((r for r in rels if (r, rel) in _CONFLICTS or (rel, r) in _CONFLICTS), None)
            if conflict:
                rejected.append({"triple": t, "reason": f"conflicts with '{conflict}'"}); continue
            rels.add(rel)
            seen.add(key)
            valid.append({
                "head": head, "head_type": ht, "relation": rel,
                "tail": tail, "tail_type": tt, "confidence": round(conf, 3),
            })

        logger.info(f"TripleValidator: {len(valid)} valid / {len(rejected)} rejected")
        sample[self.text_key] = json.dumps({"valid": valid, "rejected": rejected}, ensure_ascii=False, indent=2)
        return sample
