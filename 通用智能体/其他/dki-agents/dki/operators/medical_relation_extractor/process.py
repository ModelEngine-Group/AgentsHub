"""MedicalRelationExtractor —— 医疗关系抽取算子。

两级策略（置信度分层，供下游校验算子过滤）：
1. 句内模式规则（高置信 0.85-0.9）：同句实体对 + 触发词
   （禁忌：禁用/慎用/避免使用；并发：并发/合并/继发/警惕…）；
2. 章节先验共现（中置信 0.6，命中触发词提升）：临床文书的章节语义先验——
   诊断章节的主疾病 × 主诉/现病史的症状 → HAS_SYMPTOM；
   × 治疗方案的药物/治疗 → TREATED_BY_DRUG / TREATED_BY；
   × 辅助检查的检查 → DIAGNOSED_BY；× 科室 → BELONGS_TO_DEPT。

关键精度设计：
- 头实体只取"诊断章节"的疾病（首个为主疾病），既往史疾病是干扰项，
  仅允许作为 CONTRAINDICATED_FOR 的尾实体；
- 出现在禁忌语境（禁用/停用句）中的药物不再参与 TREATED_BY_DRUG。
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from dki.dmcompat import Mapper, OPERATORS
from dki.kg.schema import RELATION_TYPES, validate_triple
from dki.nlp.cmeie_pair import (
    allowed_label_ids,
    load_relation_contract,
    mark_pair,
    signature_text,
)

_SENT_ENDS = set("。！？；;\n")


def _split_sentences(text: str) -> List[Tuple[int, int, str]]:
    """按中文句读切句，返回 (start, end, sentence)（保留原文位置）。"""
    out: List[Tuple[int, int, str]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _SENT_ENDS:
            seg = text[start:i + 1]
            if seg.strip():
                out.append((start, i + 1, seg))
            start = i + 1
    if start < len(text):
        seg = text[start:]
        if seg.strip():
            out.append((start, len(text), seg))
    return out

# 触发词表（关系 -> 触发词）
TRIGGERS = {
    "HAS_SYMPTOM": ["主诉", "出现", "自觉", "伴有", "伴", "表现为", "明显", "反复"],
    "TREATED_BY_DRUG": ["给予", "予", "服用", "口服", "处方", "继续", "规律"],
    "DIAGNOSED_BY": ["行", "完善", "查", "示", "结果", "支持", "明确"],
    "TREATED_BY": ["行", "建议", "接受", "择期", "术"],
    "HAS_COMPLICATION": ["并发", "合并", "继发", "警惕"],
    "CONTRAINDICATED_FOR": ["禁用", "慎用", "避免使用", "停用", "避免"],
}

# 章节先验：尾实体类型 -> (允许的章节, 关系)
SECTION_PRIOR = {
    "Symptom": (["主诉", "现病史", "全文", "全文头"], "HAS_SYMPTOM"),
    "Drug": (["治疗方案", "医嘱", "全文"], "TREATED_BY_DRUG"),
    "Exam": (["辅助检查", "全文"], "DIAGNOSED_BY"),
    "Treatment": (["治疗方案", "医嘱", "全文"], "TREATED_BY"),
    "Department": (None, "BELONGS_TO_DEPT"),  # None = 任意章节
}

_CMEIE_DEFAULT_MAP = {
    "临床表现": "HAS_SYMPTOM",
    "药品治疗": "TREATED_BY_DRUG",
    "药物治疗": "TREATED_BY_DRUG",
    "治疗方法": "TREATED_BY",
    "治疗方式": "TREATED_BY",
    "辅助检查": "DIAGNOSED_BY",
    "并发症": "HAS_COMPLICATION",
    "就诊科室": "BELONGS_TO_DEPT",
    "禁忌用药": "CONTRAINDICATED_FOR",
}


def _sha256_tree(path: Path) -> str:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"CMeIE checkpoint 必须是实体目录: {path}")
    digest = hashlib.sha256()
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"CMeIE checkpoint 不允许符号链接: {item}")
        if item.is_file():
            files.append(item)
    if not files:
        raise ValueError(f"CMeIE checkpoint 目录为空: {path}")
    for item in files:
        file_digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_threshold_map(
    path: Path,
    model_path: Path,
    predicates: List[str],
) -> Dict[str, Optional[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("re_model_sha256") != _sha256_tree(model_path):
        raise ValueError("CMeIE 逐谓词阈值文件与 checkpoint 哈希不匹配")
    thresholds = payload.get("selected_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(predicates):
        raise ValueError("CMeIE 逐谓词阈值文件的谓词集合与 checkpoint 不匹配")
    normalized: Dict[str, Optional[float]] = {}
    for predicate, value in thresholds.items():
        if value is None:
            normalized[predicate] = None
            continue
        threshold = float(value)
        if not 0 <= threshold <= 1:
            raise ValueError(f"CMeIE 谓词阈值越界: {predicate}={threshold}")
        normalized[predicate] = threshold
    return normalized


class MedicalRelationExtractor(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = str(kwargs.get("backend", "rule")).strip().lower()
        if self.backend not in {"rule", "cmeie", "hybrid"}:
            raise ValueError("backend 仅支持 rule/cmeie/hybrid")
        use_llm = kwargs.get("useLlm", False)
        self.use_llm = str(use_llm).lower() in ("true", "1", "yes")
        self.low_conf = float(kwargs.get("lowConfidence", 0.6))
        self.high_conf = float(kwargs.get("highConfidence", 0.9))
        self.cmeie_model_path = kwargs.get("cmeieModelPath", "")
        self.cmeie_threshold_map_path = kwargs.get("cmeieThresholdMapPath", "")
        raw_threshold = kwargs.get("cmeieThreshold")
        self.cmeie_threshold = (
            float(raw_threshold)
            if raw_threshold is not None and str(raw_threshold).strip()
            else None
        )
        self.cmeie_batch_size = max(1, int(kwargs.get("cmeieBatchSize", 16)))
        raw_map = kwargs.get("predicateMap", {})
        if isinstance(raw_map, str) and raw_map.strip():
            raw_map = json.loads(raw_map)
        self.predicate_map = {**_CMEIE_DEFAULT_MAP, **(raw_map or {})}
        self._cmeie = None

    # ------------------------------------------------------ 主流程
    def execute(self, sample):
        start = time.time()
        text = sample[self.text_key]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"上游输出不是 JSON（需先经 MedicalEntityExtractor）: {e}")
        entities: List[Dict[str, Any]] = payload.get("entities", [])
        sections: Dict[str, str] = payload.get("sections", {}) or {}
        if not sections:
            sections = {"全文": ""}

        rule_relations = self._extract_relations(entities, sections)
        if self.backend == "rule":
            relations = rule_relations
        else:
            try:
                cmeie_relations = self._extract_cmeie_relations(entities, sections)
            except Exception as exc:
                if self.backend == "cmeie":
                    raise
                logger.warning("CMeIE 后端不可用，hybrid 保留规则结果: {}", exc)
                cmeie_relations = []
            relations = cmeie_relations if self.backend == "cmeie" else self._merge_relations(
                rule_relations, cmeie_relations
            )

        if self.use_llm:
            relations = self._llm_verify(relations)

        payload["relations"] = relations
        sample[self.text_key] = json.dumps(payload, ensure_ascii=False)
        ext = sample.get(self.ext_params_key) or {}
        ext["relations"] = relations
        sample[self.ext_params_key] = ext
        logger.info("fileName: {}, method: MedicalRelationExtractor costs {:.6f} s, relations={}",
                    sample.get(self.filename_key), time.time() - start, len(relations))
        return sample

    # ------------------------------------------------------ 关系抽取
    def _extract_relations(self, entities, sections) -> List[Dict[str, Any]]:
        # 每个实体定位到句子
        sec_sentences = {name: _split_sentences(txt) for name, txt in sections.items()}

        def sentence_of(ent) -> str:
            sec = ent.get("section", "")
            sents = sec_sentences.get(sec, [])
            span = ent.get("span")
            if span and sents:
                for s, e, t in sents:
                    if s <= span[0] < e:
                        return t
            for s, e, t in sents:  # 无 span 时退化为包含判断
                if ent.get("mention", ent["name"]) in t:
                    return t
            return ""

        # 诊断章节疾病（按 span 排序），首个为主疾病
        diag_secs = [s for s in ("诊断", "全文") if s in sections]
        diag_diseases = [
            e for e in entities
            if e["type"] == "Disease" and e.get("section") in diag_secs
        ]
        diag_diseases.sort(key=lambda e: (e.get("span") or [1 << 30])[0])
        past_diseases = {
            e["name"] for e in entities
            if e["type"] == "Disease" and e.get("section") == "既往史"
        }
        main_disease = diag_diseases[0]["name"] if diag_diseases else None
        if main_disease is None:  # 无诊断章节：退化取全文首个疾病
            all_d = [e for e in entities if e["type"] == "Disease"]
            all_d.sort(key=lambda e: (e.get("span") or [1 << 30])[0])
            main_disease = all_d[0]["name"] if all_d else None

        relations: List[Dict[str, Any]] = []

        def add(head, relation, tail, tail_type, conf, evidence, rule):
            relations.append({
                "head": head, "head_type": "Disease" if relation != "CONTRAINDICATED_FOR" else "Drug",
                "relation": relation,
                "tail": tail, "tail_type": tail_type,
                "confidence": round(conf, 3),
                "evidence": evidence.strip()[:160],
                "rule": rule,
            })

        # ---- 规则1：句内禁忌（Drug + Disease + 触发词）
        contra_drugs = set()
        for ent in entities:
            if ent["type"] != "Drug":
                continue
            sent = sentence_of(ent)
            if not sent or not any(t in sent for t in TRIGGERS["CONTRAINDICATED_FOR"]):
                continue
            contra_drugs.add(ent["name"])
            # 同句疾病（含既往史疾病提及在同句的情况）
            for other in entities:
                if other["type"] == "Disease" and other.get("section") == ent.get("section"):
                    if sentence_of(other) == sent:
                        add(ent["name"], "CONTRAINDICATED_FOR", other["name"], "Disease",
                            self.high_conf, sent, "pattern:禁忌")

        # ---- 规则2：并发症（诊断章节，非主疾病 + 触发词）
        for ent in diag_diseases[1:] if main_disease else []:
            sent = sentence_of(ent)
            if any(t in sent for t in TRIGGERS["HAS_COMPLICATION"]):
                add(main_disease, "HAS_COMPLICATION", ent["name"], "Disease",
                    0.85, sent, "pattern:并发")

        # ---- 规则3：章节先验（头=主疾病）
        if main_disease:
            seen_pairs = set()
            for ent in entities:
                etype = ent["type"]
                if etype not in SECTION_PRIOR:
                    continue
                allowed_secs, relation = SECTION_PRIOR[etype]
                if allowed_secs is not None and ent.get("section") not in allowed_secs:
                    continue
                if etype == "Drug" and ent["name"] in contra_drugs:
                    continue  # 禁忌语境药物不作治疗药物
                if ent["name"] == main_disease:
                    continue
                key = (relation, ent["name"])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                sent = sentence_of(ent)
                has_trigger = any(t in sent for t in TRIGGERS.get(relation, []))
                conf = self.high_conf if has_trigger else self.low_conf
                if relation == "HAS_SYMPTOM":
                    conf = 0.85 if has_trigger else 0.6
                if relation == "BELONGS_TO_DEPT":
                    conf = 0.8
                rule = f"section:{ent.get('section','')}" + ("+trigger" if has_trigger else "")
                add(main_disease, relation, ent["name"], etype, conf, sent, rule)

        return relations

    @staticmethod
    def _merge_relations(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep one provenance-rich triple per semantic key, preferring confidence."""
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for group in groups:
            for relation in group:
                key = (relation["head"], relation["relation"], relation["tail"])
                current = merged.get(key)
                if current is None or relation["confidence"] > current["confidence"]:
                    merged[key] = relation
        return list(merged.values())

    # ------------------------------------------------------ CMeIE pair classifier
    def _candidate_pairs(
        self, entities: List[Dict[str, Any]], sections: Dict[str, str]
    ) -> List[
        Tuple[
            Dict[str, Any],
            Dict[str, Any],
            str,
            Tuple[int, int],
            Tuple[int, int],
        ]
    ]:
        """Generate only sentence-local, schema-compatible directional pairs.

        The old evaluation path cross-producted every document entity and placed
        markers outside the original text.  Production inference instead uses
        real sentence context and the project relation domain/range contract.
        """
        sent_map = {section: _split_sentences(text) for section, text in sections.items()}

        def locate(
            entity: Dict[str, Any],
        ) -> Tuple[str, int, str, Optional[Tuple[int, int]]]:
            section = entity.get("section", "")
            mention = entity.get("mention") or entity.get("name", "")
            span = entity.get("span") or []
            for index, (start, end, sentence) in enumerate(sent_map.get(section, [])):
                if span and start <= span[0] < end:
                    local = (int(span[0]) - start, int(span[1]) - start)
                    return section, index, sentence, local
                if not span and mention and mention in sentence:
                    local_start = sentence.find(mention)
                    return (
                        section,
                        index,
                        sentence,
                        (local_start, local_start + len(mention)),
                    )
            return section, -1, "", None

        located = [(entity, *locate(entity)) for entity in entities]
        pairs = []
        seen = set()
        for head, section, sentence_index, sentence, head_span in located:
            if sentence_index < 0 or head_span is None:
                continue
            for tail, other_section, other_index, _, tail_span in located:
                if (
                    head is tail
                    or head.get("name") == tail.get("name")
                    or section != other_section
                    or sentence_index != other_index
                ):
                    continue
                if tail_span is None:
                    continue
                if not any(
                    relation.head_type == head.get("type") and relation.tail_type == tail.get("type")
                    for relation in RELATION_TYPES.values()
                ):
                    continue
                key = (
                    section,
                    sentence_index,
                    tuple(head_span),
                    head.get("type"),
                    tuple(tail_span),
                    tail.get("type"),
                )
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((head, tail, sentence, head_span, tail_span))
        return pairs

    @staticmethod
    def _mark_pair(
        sentence: str,
        head: Dict[str, Any],
        tail: Dict[str, Any],
        head_span: Optional[Tuple[int, int]] = None,
        tail_span: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        """Insert directional markers at actual non-overlapping mentions."""
        head_text = str(head.get("mention") or head.get("name") or "")
        tail_text = str(tail.get("mention") or tail.get("name") or "")
        return mark_pair(
            sentence,
            head_text,
            tail_text,
            subject_span=head_span,
            object_span=tail_span,
        )

    def _load_cmeie(self):
        if self._cmeie is not None:
            return self._cmeie
        if not self.cmeie_model_path:
            raise RuntimeError("backend=cmeie 需要 cmeieModelPath 指向已冻结的 CMeIE checkpoint")
        model_path = Path(self.cmeie_model_path)
        predicates_path = model_path / "predicates.json"
        if not predicates_path.is_file():
            raise FileNotFoundError(f"CMeIE checkpoint 缺少 predicates.json: {model_path}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("backend=cmeie 需要 transformers 与 torch 可选依赖") from exc
        device = "npu" if self.use_npu() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
        predicates = json.loads(predicates_path.read_text(encoding="utf-8"))
        contract = load_relation_contract(model_path)
        threshold = (
            self.cmeie_threshold
            if self.cmeie_threshold is not None
            else float(contract.get("threshold", contract.get("selected_threshold", 0.5)))
        )
        threshold_map: Dict[str, Optional[float]] = {}
        if self.cmeie_threshold_map_path:
            threshold_map = _load_threshold_map(
                Path(self.cmeie_threshold_map_path),
                model_path,
                predicates,
            )
        max_len = int(contract.get("max_len", 256))
        self._cmeie = (
            torch, tokenizer, model, predicates, contract, threshold,
            threshold_map, max_len, device,
        )
        return self._cmeie

    def _extract_cmeie_relations(
        self, entities: List[Dict[str, Any]], sections: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        (
            torch, tokenizer, model, predicates, contract, threshold,
            threshold_map, max_len, device,
        ) = self._load_cmeie()
        relations: List[Dict[str, Any]] = []
        encoded_pairs = []
        for head, tail, sentence, head_span, tail_span in self._candidate_pairs(
            entities, sections
        ):
            marked = self._mark_pair(sentence, head, tail, head_span, tail_span)
            if not marked:
                continue
            encoded_pairs.append((
                head,
                tail,
                sentence,
                marked,
                signature_text(head["type"], tail["type"]),
            ))

        for start in range(0, len(encoded_pairs), self.cmeie_batch_size):
            batch = encoded_pairs[start:start + self.cmeie_batch_size]
            inputs = tokenizer(
                [item[3] for item in batch],
                [item[4] for item in batch],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                probability_rows = torch.sigmoid(model(**inputs).logits.float()).cpu().tolist()
            for item, probabilities in zip(batch, probability_rows):
                head, tail, sentence = item[:3]
                allowed = allowed_label_ids(
                    contract, head["type"], tail["type"], len(probabilities)
                )
                for label_id, probability in enumerate(probabilities):
                    if isinstance(predicates, list):
                        raw_relation = predicates[label_id] if label_id < len(predicates) else ""
                    else:
                        raw_relation = predicates.get(str(label_id), predicates.get(label_id, ""))
                    if not raw_relation:
                        continue
                    cutoff = threshold_map.get(raw_relation, threshold)
                    if (
                        label_id not in allowed
                        or cutoff is None
                        or probability < cutoff
                    ):
                        continue
                    relation = self.predicate_map.get(raw_relation, raw_relation)
                    valid, _ = validate_triple(
                        head["name"], head["type"], relation, tail["name"], tail["type"]
                    )
                    if not valid:
                        continue
                    relations.append({
                        "head": head["name"], "head_type": head["type"],
                        "relation": relation, "tail": tail["name"], "tail_type": tail["type"],
                        "confidence": round(float(probability), 3),
                        "evidence": sentence.strip()[:160],
                        "rule": f"cmeie:{raw_relation}",
                    })
        return relations

    # ------------------------------------------------------ LLM 复核（可选）
    def _llm_verify(self, relations):
        try:
            from dki.llm import chat, llm_available
        except ImportError:
            return relations
        if not llm_available():
            return relations
        verified = []
        for r in relations:
            if r["confidence"] >= 0.7:
                verified.append(r)
                continue
            try:
                q = (f"根据句子：「{r['evidence']}」，判断三元组（{r['head']}，"
                     f"{r['relation']}，{r['tail']}）是否成立？只回答 是 或 否。")
                ans = chat(q, temperature=0.0, max_tokens=8)
                if "是" in ans:
                    r["confidence"] = 0.8
                    r["rule"] += "+llm_verified"
                    verified.append(r)
                # 答"否"则丢弃
            except Exception as e:
                logger.warning("LLM 复核失败，保留原关系: {}", e)
                verified.append(r)
        return verified
