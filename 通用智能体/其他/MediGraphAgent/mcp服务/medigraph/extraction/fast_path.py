"""Millisecond CPU fast path for span and relation extraction.

This is an honest deterministic baseline trained from public gold annotations
and a structured medical KB.  It is not presented as a neural model: its job is
to provide a zero-download fast path, a reproducible lower bound, and the L1
interface into which a GLiNER/OneRel checkpoint can later be plugged.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable

from medigraph.extraction.calibration import TemperatureCalibrator
from medigraph.schema.ontology import check_relation_constraint
from medigraph.schema.normalize import canonical_key, canonical_name, is_valid_entity_name

_END = "\0"

# Conservative lexical triggers used only for unseen pairs.  Exact supervised/KB
# facts do not need a trigger.
_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "has_symptom": ("症状", "表现", "伴有", "出现", "可见"),
    "recommend_drug": ("药物", "用药", "治疗", "首选", "给予", "可用"),
    "need_examination": ("检查", "检测", "检验", "筛查", "诊断"),
    "complication": ("并发", "并发症", "可引起", "导致"),
    "treated_in_department": ("科室", "就诊", "诊治"),
    "located_in": ("位于", "部位", "好发于", "发生于"),
    "treated_by_procedure": ("手术", "切除", "化疗", "放疗"),
    "adverse_reaction": ("不良反应", "副作用", "可引起"),
    "contraindication": ("禁忌", "禁用于", "慎用"),
    "positive_marker": ("阳性", "升高", "表达"),
    "negative_marker": ("阴性", "不表达"),
    "associated_gene": ("基因", "突变", "扩增", "相关"),
    "has_morphology": ("镜下", "形态", "可见"),
    "subtype_of": ("亚型", "分型", "属于"),
}


class FastSpanRelationExtractor:
    """Trie-based span NER plus schema-constrained relation extraction."""

    def __init__(
        self,
        artifact: dict,
        calibrator: TemperatureCalibrator | None = None,
        min_entity_confidence: float = 0.5,
        min_relation_confidence: float = 0.55,
    ):
        self.artifact = artifact
        self.version = str(artifact.get("version", "unknown"))
        self.min_entity_confidence = min_entity_confidence
        self.min_relation_confidence = min_relation_confidence
        self.calibrator = calibrator or TemperatureCalibrator()
        self._trie: dict = {}
        self._entries: dict[tuple[str, str], dict] = {}
        for entry in artifact.get("entities", []):
            name = canonical_name(str(entry.get("name", "")))
            if not is_valid_entity_name(name):
                continue
            key = name.casefold()
            normalized = {
                "name": name,
                "type": str(entry.get("type", "")),
                "count": max(1, int(entry.get("count", 1))),
                "source": str(entry.get("source", "public_train")),
                "canonical_id": str(entry.get("canonical_id", "")),
            }
            typed_key = (key, normalized["type"])
            current = self._entries.get(typed_key)
            if current is None or normalized["count"] > current["count"]:
                self._entries[typed_key] = normalized
        for (key, _), entry in self._entries.items():
            node = self._trie
            for char in key:
                node = node.setdefault(char, {})
            node.setdefault(_END, []).append(entry)

        self._facts_by_head: dict[str, list[dict]] = {}
        for fact in artifact.get("relations", []):
            head_key = canonical_key(str(fact.get("head", "")))
            tail_key = canonical_key(str(fact.get("tail", "")))
            relation = str(fact.get("relation", ""))
            if not head_key or not tail_key or not relation:
                continue
            normalized = {
                "head_key": head_key,
                "tail_key": tail_key,
                "relation": relation,
                "count": max(1, int(fact.get("count", 1))),
                "source": str(fact.get("source", "public_train")),
            }
            self._facts_by_head.setdefault(head_key, []).append(normalized)
        self._cmeie_facts_by_head: dict[str, list[dict]] = {}
        for fact in artifact.get("cmeie_benchmark_relations", []):
            head_key = canonical_key(str(fact.get("head", "")))
            tail_key = canonical_key(str(fact.get("tail", "")))
            if not head_key or not tail_key:
                continue
            self._cmeie_facts_by_head.setdefault(head_key, []).append(
                {
                    "head_key": head_key,
                    "tail_key": tail_key,
                    "head_type": str(fact.get("head_type", "")),
                    "tail_type": str(fact.get("tail_type", "")),
                    "relation": str(fact.get("relation", "")),
                    "predicate": str(fact.get("predicate", "")),
                    "count": max(1, int(fact.get("count", 1))),
                }
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        calibration_path: str | Path | None = None,
        **kwargs,
    ) -> "FastSpanRelationExtractor":
        """Load the dictionary artifact, transparently accepting a gzipped copy.

        The uncompressed artifact is 23 MiB, over the 10 MiB per-file ceiling this
        repository enforces, so it ships as `fast_extractor.json.gz` (2.3 MiB).
        Falling back to the `.gz` beside the requested path keeps the offline
        dictionary route working with no unpacking step for the user, and callers
        that already have the plain file keep working unchanged.
        """
        path = Path(path)
        if path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
        else:
            packed = path.with_suffix(path.suffix + ".gz")
            if not packed.exists():
                raise FileNotFoundError(
                    f"neither {path.name} nor {packed.name} found in {path.parent}"
                )
            with gzip.open(packed, "rt", encoding="utf-8") as handle:
                artifact = json.load(handle)
        calibrator = (
            TemperatureCalibrator.load(calibration_path)
            if calibration_path and Path(calibration_path).exists()
            else None
        )
        return cls(artifact, calibrator=calibrator, **kwargs)

    @staticmethod
    def _frequency_confidence(count: int, floor: float = 0.62) -> float:
        # Smooth, deliberately conservative confidence prior.
        return min(0.97, floor + 0.075 * math.log1p(max(1, count)))

    def extract_entities(
        self,
        text: str,
        allowed_types: set[str] | None = None,
        include_type_alternatives: bool = False,
        overlap_policy: str = "nested",
    ) -> list[dict]:
        raw = str(text or "")
        folded = raw.casefold()
        candidates: list[dict] = []
        for start in range(len(folded)):
            node = self._trie
            cursor = start
            while cursor < len(folded) and folded[cursor] in node:
                node = node[folded[cursor]]
                cursor += 1
                for entry in node.get(_END, []):
                    if allowed_types and entry["type"] not in allowed_types:
                        continue
                    confidence = self.calibrator.transform_one(
                        self._frequency_confidence(entry["count"])
                    )
                    if confidence < self.min_entity_confidence:
                        continue
                    candidates.append(
                        {
                            "name": raw[start:cursor],
                            "type": entry["type"],
                            "start": start,
                            "end": cursor,
                            "confidence": round(confidence, 3),
                            "extractor": "fast_span_lexicon",
                            "model_version": self.version,
                            "canonical_id": entry.get("canonical_id", ""),
                            "training_source": entry["source"],
                        }
                    )
        # Retain nested spans, but remove exact duplicates and prefer the
        # higher-confidence type if the same boundary has conflicting labels.
        best: dict[tuple[int, int, str], dict] = {}
        for entity in candidates:
            key = (entity["start"], entity["end"], entity["type"])
            if key not in best or entity["confidence"] > best[key]["confidence"]:
                best[key] = entity
        if not include_type_alternatives:
            by_span: dict[tuple[int, int], dict] = {}
            for entity in best.values():
                span = (entity["start"], entity["end"])
                current = by_span.get(span)
                if current is None or (
                    entity["confidence"],
                    entity["type"],
                ) > (
                    current["confidence"],
                    current["type"],
                ):
                    by_span[span] = entity
            best = {
                (entity["start"], entity["end"], entity["type"]): entity
                for entity in by_span.values()
            }
        values = list(best.values())
        if overlap_policy == "maximal":
            selected: list[dict] = []
            for entity in sorted(
                values,
                key=lambda item: (
                    -float(item["confidence"]),
                    -(int(item["end"]) - int(item["start"])),
                    int(item["start"]),
                ),
            ):
                if any(self._spans_overlap(entity, accepted) for accepted in selected):
                    continue
                selected.append(entity)
            values = selected
        elif overlap_policy != "nested":
            raise ValueError("overlap_policy must be 'nested' or 'maximal'")
        return sorted(values, key=lambda item: (item["start"], -(item["end"] - item["start"]), item["type"]))

    def extract_relations(self, text: str, entities: Iterable[dict] | None = None) -> list[dict]:
        entity_list = list(entities) if entities is not None else self.extract_entities(text)
        by_key: dict[str, list[dict]] = {}
        for entity in entity_list:
            by_key.setdefault(canonical_key(str(entity.get("name", ""))), []).append(entity)

        triples: dict[tuple[str, str, str], dict] = {}
        # High-precision path: memorized relation facts from public train/KB.
        for head_key, heads in by_key.items():
            for fact in self._facts_by_head.get(head_key, []):
                tails = by_key.get(fact["tail_key"], [])
                for head in heads:
                    for tail in tails:
                        if self._spans_overlap(head, tail):
                            continue
                        relation = fact["relation"]
                        if not check_relation_constraint(relation, str(head.get("type", "")), str(tail.get("type", ""))):
                            continue
                        confidence = self.calibrator.transform_one(
                            self._frequency_confidence(fact["count"], floor=0.68)
                        )
                        self._keep_triple(
                            triples,
                            head,
                            relation,
                            tail,
                            confidence,
                            "memorized_fact",
                            fact["source"],
                        )

        # Schema/cue path for unseen pairs.  Only nearby entities and explicit
        # lexical evidence are allowed, preventing a combinatorial hallucination.
        positioned = [
            entity
            for entity in entity_list
            if "start" in entity
            and "end" in entity
            and not self._shadowed_same_type(entity, entity_list)
        ]
        for head in positioned:
            for tail in positioned:
                if head is tail:
                    continue
                distance = max(0, max(head["start"], tail["start"]) - min(head["end"], tail["end"]))
                if distance > 96:
                    continue
                left = max(0, min(head["start"], tail["start"]) - 12)
                right = min(len(text), max(head["end"], tail["end"]) + 12)
                context = text[left:right]
                for relation, cues in _RELATION_CUES.items():
                    if not check_relation_constraint(relation, str(head.get("type", "")), str(tail.get("type", ""))):
                        continue
                    matched = next((cue for cue in cues if cue in context), "")
                    if not matched:
                        continue
                    confidence = self.calibrator.transform_one(
                        min(float(head.get("confidence", 0.5)), float(tail.get("confidence", 0.5))) * 0.86
                    )
                    if confidence >= self.min_relation_confidence:
                        self._keep_triple(
                            triples,
                            head,
                            relation,
                            tail,
                            confidence,
                            "schema_cue",
                            f"cue:{matched}",
                        )
        return sorted(
            triples.values(),
            key=lambda item: (-item["confidence"], item["head"], item["relation"], item["tail"]),
        )

    def extract_cmeie_relations(
        self,
        text: str,
        entities: Iterable[dict] | None = None,
    ) -> list[dict]:
        """Predict in the complete official CMeIE predicate label space.

        The deterministic baseline emits only supervised entity-pair facts; it
        does not guess unseen relations.  This makes the full-schema metric
        honest (often recall-limited) and provides a reproducible comparison
        target for the later OneRel checkpoint.
        """
        entity_list = (
            list(entities)
            if entities is not None
            else self.extract_entities(text, include_type_alternatives=True)
        )
        by_key: dict[str, list[dict]] = {}
        for entity in entity_list:
            by_key.setdefault(canonical_key(str(entity.get("name", ""))), []).append(entity)
        predictions: dict[tuple[str, str, str], dict] = {}
        for head_key, heads in by_key.items():
            for fact in self._cmeie_facts_by_head.get(head_key, []):
                for head in heads:
                    if str(head.get("type", "")) != fact["head_type"]:
                        continue
                    for tail in by_key.get(fact["tail_key"], []):
                        if str(tail.get("type", "")) != fact["tail_type"] or self._spans_overlap(head, tail):
                            continue
                        confidence = self.calibrator.transform_one(
                            self._frequency_confidence(fact["count"], floor=0.68)
                        )
                        key = (
                            canonical_key(str(head.get("name", ""))),
                            fact["predicate"],
                            canonical_key(str(tail.get("name", ""))),
                        )
                        predictions[key] = {
                            "head": str(head.get("name", "")),
                            "head_type": fact["head_type"],
                            "relation": fact["relation"],
                            "predicate": fact["predicate"],
                            "tail": str(tail.get("name", "")),
                            "tail_type": fact["tail_type"],
                            "confidence": round(confidence, 3),
                            "extractor": "fast_cmeie_fact",
                            "model_version": self.version,
                        }
        return sorted(
            predictions.values(),
            key=lambda item: (-item["confidence"], item["head"], item["predicate"], item["tail"]),
        )

    @staticmethod
    def _spans_overlap(left: dict, right: dict) -> bool:
        if not all(key in left and key in right for key in ("start", "end")):
            return False
        return max(int(left["start"]), int(right["start"])) < min(int(left["end"]), int(right["end"]))

    @staticmethod
    def _shadowed_same_type(entity: dict, entities: Iterable[dict]) -> bool:
        """Suppress shorter same-type spans only for relation inference.

        Span NER still returns nested mentions for honest nested-entity metrics.
        """
        start, end = int(entity["start"]), int(entity["end"])
        entity_type = str(entity.get("type", ""))
        for other in entities:
            if other is entity or str(other.get("type", "")) != entity_type:
                continue
            if "start" not in other or "end" not in other:
                continue
            o_start, o_end = int(other["start"]), int(other["end"])
            if o_start <= start and end <= o_end and (o_end - o_start) > (end - start):
                return True
        return False

    def _keep_triple(
        self,
        bucket: dict,
        head: dict,
        relation: str,
        tail: dict,
        confidence: float,
        method: str,
        evidence: str,
    ) -> None:
        key = (
            canonical_key(str(head.get("name", ""))),
            relation,
            canonical_key(str(tail.get("name", ""))),
        )
        value = {
            "head": str(head.get("name", "")),
            "head_type": str(head.get("type", "")),
            "relation": relation,
            "tail": str(tail.get("name", "")),
            "tail_type": str(tail.get("type", "")),
            "confidence": round(float(confidence), 3),
            "extractor": "fast_joint_relation",
            "model_version": self.version,
            "evidence_method": method,
            "evidence": evidence,
        }
        if key not in bucket or value["confidence"] > bucket[key]["confidence"]:
            bucket[key] = value

    def uncertainty(self, items: Iterable[dict], threshold: float = 0.72) -> dict:
        values = [float(item.get("confidence", 0.0)) for item in items]
        if not values:
            return {"uncertain": True, "reason": "no_prediction", "mean_confidence": 0.0, "minimum_confidence": 0.0}
        mean = sum(values) / len(values)
        minimum = min(values)
        return {
            "uncertain": mean < threshold or minimum < self.min_entity_confidence,
            "reason": "below_route_threshold" if mean < threshold else "accepted",
            "mean_confidence": round(mean, 4),
            "minimum_confidence": round(minimum, 4),
        }
