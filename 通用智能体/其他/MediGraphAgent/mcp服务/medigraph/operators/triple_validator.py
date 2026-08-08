"""Triple generation & validation operator.

Validates candidate relation triples against the ontology and quality rules:
  1. Schema legality: relation's (head_type, tail_type) must match the ontology.
  2. Dedup + normalization: collapse case/whitespace duplicates.
  3. Confidence threshold filtering.
  4. Conflict detection: mutually exclusive relations on the same entity pair.
Each rejected triple carries an explicit reason (explainability / result quality).
An optional LLM adjudication path can rescue borderline-confidence triples.
"""
from __future__ import annotations

from typing import Any

from medigraph.operators.base import BaseOperator, OperatorMeta
from medigraph.schema.ontology import check_relation_constraint, is_valid_relation
from medigraph.schema.normalize import canonical_key, is_valid_entity_name

# Relations that cannot both hold for the same (head, tail) pair.
_CONFLICTS = {("positive_marker", "negative_marker")}
_REVERSED_CONFLICTS = {("recommend_drug", "contraindication")}


class TripleValidatorOperator(BaseOperator):
    def __init__(self, llm: Any | None = None, min_confidence: float = 0.5,
                 adjudicate_band: float = 0.2):
        self.llm = llm
        self.min_confidence = min_confidence
        # Borderline triples in [min_conf - band, min_conf) are sent to the LLM
        # for a keep/drop decision (with reason) instead of being auto-rejected.
        self.adjudicate_band = adjudicate_band
        self.meta = OperatorMeta(
            name="triple_validator",
            version="2.0.0",
            description=(
                "对候选三元组做校验与治理：schema 合法性、去重归一、置信度过滤、冲突检测，"
                "输出通过的三元组与每条拒因。输入 {triples, min_confidence?}, 输出 {valid:[...], rejected:[...]}。"
            ),
            input_schema={
                "type": "object",
                "properties": {"triples": {"type": "array"}, "min_confidence": {"type": "number"}},
                "required": ["triples"],
            },
            output_schema={
                "type": "object",
                "properties": {"valid": {"type": "array"}, "rejected": {"type": "array"}},
            },
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        triples = inputs.get("triples", []) or []
        min_conf = float(inputs.get("min_confidence") or self.min_confidence)

        valid: list[dict] = []
        rejected: list[dict] = []
        seen: dict[str, dict] = {}
        pair_relations: dict[tuple, set] = {}

        # Higher-confidence claims win deterministic conflict resolution.
        ordered = sorted(
            (item for item in triples if isinstance(item, dict)),
            key=lambda item: -self._as_float(item.get("confidence", 0.0)),
        )
        for t in ordered:
            if not isinstance(t, dict):
                continue
            head = str(t.get("head", "")).strip()
            tail = str(t.get("tail", "")).strip()
            rel = str(t.get("relation", "")).strip()
            ht = str(t.get("head_type", "")).strip()
            tt = str(t.get("tail_type", "")).strip()
            conf = self._as_float(t.get("confidence", 0.0))

            if not head or not tail or not rel:
                self._reject(rejected, t, "missing head/tail/relation")
                continue
            if not is_valid_entity_name(head) or not is_valid_entity_name(tail):
                self._reject(rejected, t, "invalid or placeholder entity name")
                continue
            if not is_valid_relation(rel):
                self._reject(rejected, t, f"unknown relation '{rel}'")
                continue
            if not check_relation_constraint(rel, ht, tt):
                self._reject(rejected, t, f"schema violation: {ht}-{rel}->{tt}")
                continue
            if conf < min_conf:
                # LLM adjudication for borderline triples (plan 4.4 rule 4).
                if (
                    self.llm is not None
                    and conf >= min_conf - self.adjudicate_band
                    and self._adjudicate(t, inputs.get("text", ""))
                ):
                    conf = min_conf  # rescued; promote to threshold
                    t["adjudicated"] = True
                else:
                    self._reject(rejected, t, f"low confidence {conf:.2f} < {min_conf:.2f}")
                    continue

            key = f"{ht}:{canonical_key(head)}|{rel}|{tt}:{canonical_key(tail)}"
            if key in seen:
                # keep the higher-confidence duplicate
                if conf > seen[key]["confidence"]:
                    seen[key]["confidence"] = round(conf, 3)
                continue

            # conflict detection on the (head, tail) pair
            pair = (canonical_key(head), canonical_key(tail))
            rels = pair_relations.setdefault(pair, set())
            conflict = next(
                (r for r in rels if (r, rel) in _CONFLICTS or (rel, r) in _CONFLICTS),
                None,
            )
            if conflict:
                self._reject(rejected, t, f"conflicts with existing relation '{conflict}'")
                continue
            reverse_rels = pair_relations.get((pair[1], pair[0]), set())
            reverse_conflict = next(
                (
                    other
                    for other in reverse_rels
                    if (rel, other) in _REVERSED_CONFLICTS or (other, rel) in _REVERSED_CONFLICTS
                ),
                None,
            )
            if reverse_conflict:
                self._reject(rejected, t, f"conflicts with reverse relation '{reverse_conflict}'")
                continue
            rels.add(rel)

            norm = {
                "head": head,
                "head_type": ht,
                "relation": rel,
                "tail": tail,
                "tail_type": tt,
                "confidence": round(conf, 3),
            }
            # Retain extraction evidence and routing provenance instead of
            # discarding it at the validation boundary.
            for field in (
                "extractor",
                "model_version",
                "evidence_method",
                "evidence",
                "cascade_sources",
                "cmeie_relation",
                "cmeie_predicate",
                "adjudicated",
                "head_canonical_id",
                "tail_canonical_id",
            ):
                if field in t:
                    norm[field] = t[field]
            seen[key] = norm
            valid.append(norm)

        return {
            "valid": valid,
            "rejected": rejected,
            "stats": {
                "candidates": len(ordered),
                "valid": len(valid),
                "rejected": len(rejected),
                "pass_rate": round(len(valid) / len(ordered), 4) if ordered else 0.0,
            },
        }

    def _adjudicate(self, triple: dict, context: str = "") -> bool:
        """Ask the LLM whether a borderline triple should be kept. Returns bool."""
        ctx = f"\n参考文本片段：\n{context[:800]}\n" if context else ""
        prompt = (
            "判断下面这条医学知识三元组是否成立、是否应保留进知识图谱。"
            f"{ctx}\n三元组：{triple.get('head')} -[{triple.get('relation')}]-> {triple.get('tail')}\n"
            '只输出 JSON：{"keep": true/false, "reason": "简要理由"}'
        )
        try:
            res = self.llm.chat_json(prompt, default={"keep": False})
            return bool(isinstance(res, dict) and res.get("keep"))
        except Exception:  # noqa: BLE001 - adjudication is best-effort
            return False

    @staticmethod
    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _reject(bucket: list, triple: dict, reason: str) -> None:
        bucket.append({"triple": triple, "reason": reason})
