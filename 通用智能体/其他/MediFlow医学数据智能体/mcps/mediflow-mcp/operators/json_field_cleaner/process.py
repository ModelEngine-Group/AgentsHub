# -*- coding: utf-8 -*-
"""
JSON/JSONL 字段级清洗算子。

问题：JSON 是结构化数据，DataMate 默认把它当纯文本整串清洗 → 可能误删引号/逗号破坏结构、
破坏缩进、误改 key 名、无差别清洗标识字段(姓名/age)。

方案（与 TableColumnCleaner 同构）：解析 JSON → 递归遍历 → 只清洗"医疗文本字段"的字符串 value
→ 保留 JSON 结构、key、标识字段、非文本值 → 复用 term_kb(术语标准化) + noise_kb(规则去噪)。

支持三种输入：JSON 对象、JSON 数组、JSONL(每行一个 JSON)。
字段分类（按 key 名）：
  - 医疗文本字段（note/text/content/主诉/诊断/现病史…）→ 清洗
  - 标识字段（id/name/姓名/age/date…）→ 原样保留
  - 其他字符串字段 → 保守标准化（仅术语，不去噪）

设计取舍：字段级用规则（不调 LLM）——JSON 常含大量记录，逐字段调 LLM 不现实，
也契合"减少 LLM 调用"。语义级噪声留给非结构化文本路径的 LLMNoiseFilter。
"""
import re
import time
import json
import sqlite3
from typing import Dict, Any

from loguru import logger
from datamate.core.base_op import Mapper

try:
    from ops.user.llm_noise_filter.noise_rule_engine import NoiseRuleEngine
    from ops.user.llm_noise_filter.structured_clean_utils import deterministic_field_clean
except ImportError:
    from noise_rule_engine import NoiseRuleEngine
    from structured_clean_utils import deterministic_field_clean

_TERM_KB = "/opt/runtime/datamate/ops/user/medical_term_normalizer/term_kb.db"
_NOISE_KB = "/opt/runtime/datamate/ops/user/llm_noise_filter/noise_kb.db"

# 医疗文本字段 key 关键词（命中→清洗+去噪）
_MEDICAL_KEY = ["note", "text", "content", "ask", "answer", "question", "title",
                "desc", "remark", "comment", "complaint", "diagnos", "symptom",
                "history", "present", "illness", "finding", "impression", "advice",
                "noisy_text", "cleaned_text", "record_text", "self_report",
                "主诉", "现病史", "既往史", "诊断", "症状", "描述", "病情", "建议",
                "医嘱", "检查", "所见", "印象", "结论", "病历", "记录"]
# 标识字段 key 关键词（命中→保留）
_ID_KEY = ["id", "name", "姓名", "age", "年龄", "sex", "gender", "性别", "date",
           "time", "时间", "日期", "phone", "电话", "dept", "department", "科室",
           "编号", "卡号", "床号", "工号"]
_DROP_KEYS = {"clean_reference", "noise_labels", "noise", "output_format_hint"}

_STRUCTURED_NOISE_RE = re.compile(
    r"(?:由.{0,12}(?:HIS|EMR|his|emr|电子病历|医院信息|信息)?系统.{0,8}"
    r"(?:自动)?(?:导出|生成|产生))"
    r"|(?:(?:HIS|EMR|his|emr|电子病历|医院信息)系统?.{0,6}(?:自动)?(?:导出|生成|产生))"
    r"|(?:系统自动(?:导出|生成))"
    r"|(?:填表时间[:：]?\s*\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)"
    r"|(?:填表人[:：]?\s*[\u4e00-\u9fffA-Za-z0-9_]{1,16})"
    r"|(?:@[A-Za-z0-9_\-\u4e00-\u9fff]{1,30})"
    r"|(?:加(?:我)?微信[:：]?\s*[A-Za-z0-9_\-]{3,30})"
    r"|(?:图片链接(?:已)?失效|资料链接(?:已)?失效|(?:图片|资料|链接)(?:已)?失效)"
    r"|(?:仅供(?:内部)?测试|测试数据请忽略)"
    r"|(?:controlled_noise_\d{8}\??)"
    r"|(?:HIS\s+system\s+auto\s+export)",
    re.IGNORECASE,
)


class JsonFieldCleaner(Mapper):

    def __init__(self, *args, **kwargs):
        super(JsonFieldCleaner, self).__init__(*args, **kwargs)
        self.term_kb_path = kwargs.get("termKbPath", _TERM_KB)
        self.noise_kb_path = kwargs.get("noiseKbPath", _NOISE_KB)
        self._rule_engine = NoiseRuleEngine(self.noise_kb_path)
        self._load_kbs()

    def _load_kbs(self):
        # 术语映射
        self.term_dict = {}
        try:
            conn = sqlite3.connect(self.term_kb_path)
            self.term_dict = {a: f for a, f in conn.execute(
                "SELECT abbr, full FROM term_mappings WHERE status='active'").fetchall()
                if a and f}
            conn.close()
        except Exception as e:
            logger.warning(f"JsonFieldCleaner: term_kb 加载失败 {e}")
        self.term_sorted = sorted(self.term_dict.keys(), key=len, reverse=True)

        # 噪声规则（exact 转义、regex 原样，单条容错）
        self.noise_re = None
        try:
            conn = sqlite3.connect(self.noise_kb_path)
            cols = [c[1] for c in conn.execute("PRAGMA table_info(noise_rules)").fetchall()]
            has_mt = "match_type" in cols
            sel = ("SELECT pattern, match_type FROM noise_rules "
                   "WHERE status='active' AND medical_safe=1") if has_mt else \
                  ("SELECT pattern, 'regex' FROM noise_rules "
                   "WHERE status='active' AND medical_safe=1")
            rows = conn.execute(sel).fetchall()
            conn.close()
            valid = []
            for pat, mt in rows:
                if not pat:
                    continue
                frag = re.escape(pat) if mt == "exact" else pat
                try:
                    re.compile(frag)
                    valid.append(f"(?:{frag})")
                except re.error:
                    pass
            if valid:
                self.noise_re = re.compile("|".join(valid), re.IGNORECASE)
            self._noise_count = len(valid)
        except Exception as e:
            logger.debug(f"JsonFieldCleaner: noise_kb 加载 {e}")
            self._noise_count = 0
        logger.info(f"JsonFieldCleaner: {len(self.term_dict)} terms, "
                    f"{self._noise_count} noise rules")

    _DEDUP_RE = re.compile(r'([\u4e00-\u9fffA-Za-z0-9·\-]{2,20})[（(]\1[）)]')

    def _classify_key(self, key: str) -> str:
        k = str(key).lower()
        if any(kw.lower() in k for kw in _ID_KEY):
            return "identifier"
        if any(kw.lower() in k for kw in _MEDICAL_KEY):
            return "medical_text"
        return "other"

    def _normalize_terms(self, text: str) -> str:
        for abbr in self.term_sorted:
            if abbr not in text:
                continue
            full = self.term_dict[abbr]
            skip_re = re.compile(re.escape(full) + r'[（(]' + re.escape(abbr) + r'[）)]')
            if skip_re.search(text):
                continue
            text = re.sub(r'(?<![A-Za-z0-9])' + re.escape(abbr) + r'(?![A-Za-z0-9])',
                          full, text)
        return self._DEDUP_RE.sub(r'\1', text)

    def _clean_value(self, value: str, do_noise: bool) -> str:
        text = deterministic_field_clean(value, rule_engine=self._rule_engine, remove_noise=do_noise)
    # 先完成全角与繁简处理，再匹配医学缩写。
        text = self._normalize_terms(text)
        if do_noise:
            if self.noise_re is not None:
                text = self.noise_re.sub("", text)
            text = _STRUCTURED_NOISE_RE.sub("", text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
            text = re.sub(r'^[，。；;、\s]+|[，。；;、\s]+$', '', text)
        return text

    def _clean_obj(self, obj, parent_key="", stats=None):
        """递归清洗：dict 按 key 分类，list 逐元素，str 按父 key 类型处理"""
        if isinstance(obj, dict):
            return {
                k: self._clean_obj(v, k, stats)
                for k, v in obj.items()
                if str(k).strip() not in _DROP_KEYS
            }
        if isinstance(obj, list):
            return [self._clean_obj(v, parent_key, stats) for v in obj]
        if isinstance(obj, str):
            ctype = self._classify_key(parent_key)
            if ctype == "identifier" or not obj.strip():
                return obj  # 标识字段/空串原样
            if stats is not None:
                stats[ctype] = stats.get(ctype, 0) + 1
            return self._clean_value(obj, do_noise=(ctype == "medical_text"))
        return obj  # 数字/布尔/None 原样

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        filetype = str(sample.get(self.filetype_key, "")).lower()
        filename = str(sample.get(self.filename_key, "")).lower()
        if filename.endswith(".jsonl"):
            filetype = "jsonl"
        elif filename.endswith(".json"):
            filetype = "json"
        text = sample[self.text_key]
        stripped = text.lstrip() if isinstance(text, str) else ""
        if filetype not in ("json", "jsonl") and stripped.startswith(("{", "[")):
            first_lines = [line.strip() for line in text.splitlines() if line.strip()][:2]
            if len(first_lines) > 1 and all(line.startswith("{") for line in first_lines):
                filetype = "jsonl"
            else:
                filetype = "json"

        if filetype not in ("json", "jsonl") or not text.strip():
            logger.info(f"JsonFieldCleaner: {sample.get(self.filename_key)} "
                        f"非JSON({filetype})，跳过")
            return sample

        stats = {}
        try:
            if filetype == "jsonl":
                # 逐行解析
                out_lines = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    out_lines.append(json.dumps(self._clean_obj(obj, stats=stats),
                                                ensure_ascii=False))
                sample[self.text_key] = "\n".join(out_lines)
            else:
                obj = json.loads(text)
                cleaned = self._clean_obj(obj, stats=stats)
                sample[self.text_key] = json.dumps(cleaned, ensure_ascii=False, indent=2)
            sample["target_type"] = filetype
            sample[self.filetype_key] = filetype
        except Exception as e:
            logger.warning(f"JsonFieldCleaner: JSON 解析失败 {e}，保留原文")
            return sample

        logger.info(
            f"fileName: {sample[self.filename_key]}, "
            f"JsonFieldCleaner costs {time.time()-start:.3f}s, cleaned fields: {stats}")
        return sample
