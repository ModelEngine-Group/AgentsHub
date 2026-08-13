"""MedicalEntityExtractor —— 医疗实体识别算子（疾病/症状/药物/检查/治疗/科室）。

三种后端（settings.backend）：
- dict（默认）：词典多模式匹配。**多模式扫描复用 DataMate 的 AC 自动机**
  （datamate/common/utils/aho_corasick.py，其 search 仅返回命中词表——面向敏感词
  场景），在其结果上补充 span 定位与最长匹配消歧；DataMate 不可用时回退为
  逐词查找（词典量级 ~400，性能足够）。别名统一归一到规范名。
- llm：调用 OpenAI 兼容 LLM 抽取（增强路径，需配置 LLM_*环境变量），与词典
  结果按 (name,type) 去重合并。
- bert：Transformer NER 推理（对接 NPU 优化，见 dki/bench/npu/bert_ner_infer.py）。
- cmeee/cmeee_gp：PyTorch CMeEE GlobalPointer。
- cmeee_om：schema-v3 绑定的 CMeEE GlobalPointer OM；缺任一证据工件即失败关闭。

输出（写回 sample["text"] 的 JSON，并镜像 ext_params["entities"]）：
{"entities":[{"name","type","mention","span":[s,e],"span_base","section",
              "source","confidence"}], "sections":{...}}
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from dki.dmcompat import Mapper, OPERATORS

try:
    from datamate.common.utils.aho_corasick import AhoCorasic
    _HAS_DM_AC = True
except ImportError:
    AhoCorasic = None
    _HAS_DM_AC = False

# 词典类别 -> 实体类型
_CATEGORY_TYPE = {
    "diseases": "Disease",
    "symptoms": "Symptom",
    "drugs": "Drug",
    "exams": "Exam",
    "treatments": "Treatment",
}


def _default_dict_paths() -> List[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "resources" / "medical_dict.json",              # 算子市场自含包
        here.parent.parent / "kg" / "dictionaries" / "medical_dict.json",  # dki 仓库布局
    ]


def load_vocab(dict_path: Optional[str] = None) -> Dict[str, Tuple[str, str]]:
    """加载词典 -> {表面形式(含规范名与别名): (规范名, 实体类型)}。

    冲突处置：同一表面形式命中多个类别时，按 Disease>Drug>Symptom>Exam>Treatment
    的优先级保留（临床文本中歧义倾向）。
    """
    paths = [Path(dict_path)] if dict_path else _default_dict_paths()
    data = None
    for p in paths:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            break
    if data is None:
        raise FileNotFoundError(
            f"医疗词典不存在，尝试过: {[str(p) for p in paths]}；"
            f"可先运行 python -m dki.datagen.build_dict 生成，或以 dictPath 参数指定。"
        )
    priority = {"Disease": 0, "Drug": 1, "Symptom": 2, "Exam": 3, "Treatment": 4, "Department": 5}
    vocab: Dict[str, Tuple[str, str]] = {}

    def put(surface: str, canonical: str, etype: str):
        if not surface:
            return
        old = vocab.get(surface)
        if old is None or priority[etype] < priority[old[1]]:
            vocab[surface] = (canonical, etype)

    for cat, etype in _CATEGORY_TYPE.items():
        for name, info in data.get(cat, {}).items():
            put(name, name, etype)
            for alias in info.get("aliases", []):
                put(alias, name, etype)
    for dept in data.get("departments", []):
        put(dept, dept, "Department")
    return vocab


def _resolve_spans(text: str, hits: List[str], vocab) -> List[Dict[str, Any]]:
    """对命中的表面形式做全位置定位 + 最长匹配消歧（重叠时保留更长者）。"""
    candidates = []
    for surface in hits:
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            canonical, etype = vocab[surface]
            candidates.append((idx, idx + len(surface), surface, canonical, etype))
            start = idx + 1
    # 最长优先的区间选择（同长按出现顺序）
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    chosen: List[Tuple[int, int, str, str, str]] = []
    occupied: List[Tuple[int, int]] = []
    for c in candidates:
        s, e = c[0], c[1]
        if any(not (e <= os_ or s >= oe) for os_, oe in occupied):
            continue
        chosen.append(c)
        occupied.append((s, e))
    chosen.sort(key=lambda c: c[0])
    return [
        {"span": [s, e], "mention": surf, "name": canon, "type": etype}
        for s, e, surf, canon, etype in chosen
    ]


class MedicalEntityExtractor(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = kwargs.get("backend", "dict")
        self.dict_path = kwargs.get("dictPath", None)
        # bert 后端可选：显式指定微调模型目录；None 时后端自动探测 models/medical_ner_bert
        self.model_path = kwargs.get("modelPath", None)
        self.om_path = kwargs.get("omPath", None)
        self.manifest_path = kwargs.get("manifestPath", None)
        self.export_receipt_path = kwargs.get("exportReceiptPath", None)
        self.atc_log_path = kwargs.get("atcLogPath", None)
        self.om_device = int(kwargs.get("device", 0))
        self.vocab = load_vocab(self.dict_path)
        self._ac = None
        if _HAS_DM_AC:
            # 复用 DataMate AC 自动机做多模式扫描（返回命中词表）
            self._ac = AhoCorasic(list(self.vocab.keys()))
        else:
            logger.info("DataMate AC 自动机不可用，实体扫描使用逐词查找回退。")
        self._bert = None  # 惰性初始化
        self._cmeee = None  # cmeee 后端惰性初始化

    # ------------------------------------------------------ dict 后端
    def _extract_dict(self, text: str) -> List[Dict[str, Any]]:
        if self._ac is not None:
            hits = [h for h in self._ac.search(text, set()) if h in self.vocab]
        else:
            hits = [w for w in self.vocab if w in text]
        ents = _resolve_spans(text, hits, self.vocab)
        for e in ents:
            e["source"] = "dict"
            e["confidence"] = 1.0
        return ents

    # ------------------------------------------------------ llm 后端
    def _extract_llm(self, text: str) -> List[Dict[str, Any]]:
        try:
            from dki.llm import chat
        except ImportError as e:
            raise RuntimeError(f"llm 后端需要 dki.llm 模块: {e}")
        prompt = (
            "从下面的中文病历文本中抽取医疗实体，类型限定：Disease(疾病)、Symptom(症状)、"
            "Drug(药物)、Exam(检查)、Treatment(治疗方式)、Department(科室)。"
            '只输出 JSON 数组，元素形如 {"name":"规范名","type":"Disease","mention":"原文提及"}。\n\n'
            f"文本：\n{text[:3000]}"
        )
        raw = chat(prompt, temperature=0.0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        items = json.loads(raw)
        out = []
        for it in items:
            if it.get("type") in _CATEGORY_TYPE.values() or it.get("type") == "Department":
                out.append({
                    "name": it.get("name", ""), "type": it["type"],
                    "mention": it.get("mention", it.get("name", "")),
                    "span": None, "source": "llm", "confidence": 0.85,
                })
        return out

    # ------------------------------------------------------ bert 后端
    def _extract_bert(self, text: str) -> List[Dict[str, Any]]:
        if self._bert is None:
            from dki.bench.npu.bert_ner_infer import BertNerBackend
            device = "npu" if self.use_npu() else "cpu"
            self._bert = BertNerBackend(model_name=self.model_path, device=device)
        ents = self._bert.extract([text])[0]
        for e in ents:
            e.setdefault("source", "bert")
            e.setdefault("confidence", 0.8)
        return ents

    # ------------------------------------------------------ 主流程
    def execute(self, sample):
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        sections: Dict[str, str] = {}
        if text.lstrip().startswith("{"):
            try:
                payload = json.loads(text)
                sections = payload.get("sections", {}) or {}
            except json.JSONDecodeError:
                pass

        entities: List[Dict[str, Any]] = []
        if sections:
            for sec_name, sec_text in sections.items():
                ents = self._extract_backend(sec_text)
                for e in ents:
                    e["section"] = sec_name
                    e["span_base"] = "section"
                entities.extend(ents)
        else:
            entities = self._extract_backend(text)
            for e in entities:
                e["section"] = ""
                e["span_base"] = "text"

        # llm 增强：与 dict 结果按 (name,type) 合并
        if self.backend == "llm" and sections:
            pass  # llm 后端已在 _extract_backend 中处理

        out = {"entities": entities, "sections": sections}
        sample[self.text_key] = json.dumps(out, ensure_ascii=False)
        ext = sample.get(self.ext_params_key) or {}
        ext["entities"] = entities
        ext["sections"] = sections
        sample[self.ext_params_key] = ext
        logger.info("fileName: {}, method: MedicalEntityExtractor({}) costs {:.6f} s, entities={}",
                    sample.get(self.filename_key), self.backend, time.time() - start, len(entities))
        return sample

    def _extract_backend(self, text: str) -> List[Dict[str, Any]]:
        if self.backend == "dict":
            return self._extract_dict(text)
        if self.backend == "llm":
            dict_ents = self._extract_dict(text)
            try:
                llm_ents = self._extract_llm(text)
            except Exception as e:
                logger.warning("LLM 实体抽取失败，退回词典结果: {}", e)
                return dict_ents
            seen = {(x["name"], x["type"]) for x in dict_ents}
            merged = dict_ents + [x for x in llm_ents if (x["name"], x["type"]) not in seen]
            return merged
        if self.backend == "bert":
            return self._extract_bert(text)
        if self.backend in {"cmeee", "cmeee_gp", "cmeee_om"}:
            return self._extract_cmeee(text)
        raise ValueError(
            f"未知 backend: {self.backend}"
            "（可选 dict/llm/bert/cmeee/cmeee_gp/cmeee_om）"
        )

    # ------------------------------------------------------ cmeee 后端（真实基准模型）
    def _extract_cmeee(self, text: str) -> List[Dict[str, Any]]:
        """CBLUE CMeEE-V2 训练的 RoBERTa+GlobalPointer 后端（真实医疗泛化）。

        9 类 CMeEE 标签映射到本项目 6 类 schema（dis→Disease/sym→Symptom/dru→Drug/
        ite→Exam/pro→Treatment/dep→Department；bod/equ/mic 不在 6 类内，丢弃）。
        span 归一到词典规范名（与 dict/gold 口径一致，便于图谱聚合）。
        """
        if self._cmeee is None:
            if self.backend == "cmeee_om":
                required = {
                    "modelPath": self.model_path,
                    "omPath": self.om_path,
                    "manifestPath": self.manifest_path,
                    "exportReceiptPath": self.export_receipt_path,
                    "atcLogPath": self.atc_log_path,
                }
                missing = [name for name, value in required.items() if not value]
                if missing:
                    raise ValueError(f"cmeee_om 缺少发布工件参数: {missing}")
                from dki.bench.npu.cmeee_ner_infer import (
                    CmeeeGlobalPointerOmBackend,
                )

                self._cmeee = CmeeeGlobalPointerOmBackend(
                    model_dir=self.model_path,
                    om_path=self.om_path,
                    manifest_path=self.manifest_path,
                    export_receipt_path=self.export_receipt_path,
                    atc_log_path=self.atc_log_path,
                    device=self.om_device,
                )
            else:
                from dki.bench.npu.cmeee_ner_infer import (
                    CmeeeGlobalPointerBackend,
                )

                device = "npu" if self.use_npu() else "cpu"
                self._cmeee = CmeeeGlobalPointerBackend(
                    model_dir=self.model_path,
                    device=device,
                )
        ents = self._cmeee.extract([text])[0]
        out = []
        for e in ents:
            surface = e["mention"]
            canon = self.vocab.get(surface, (surface, e["type"]))[0]
            out.append({"span": e.get("span"), "mention": surface, "name": canon,
                        "type": e["type"], "source": e.get("source", "cmeee"),
                        "confidence": e.get("confidence", 0.85)})
        return out
