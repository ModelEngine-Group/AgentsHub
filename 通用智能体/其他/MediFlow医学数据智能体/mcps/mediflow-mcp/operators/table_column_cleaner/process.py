# -*- coding: utf-8 -*-
"""
结构化表格列级清洗算子。

核心：DataMate 默认 read_file 对 CSV 用 unstructured.partition 拍平成纯文本会丢列结构，
因此本算子在 execute 里【绕过默认读取】，直接用 pandas 读 CSV/Excel 保留列，
按列名自动分类并差异化清洗，复用已建的两个外置知识库：
  - term_kb.db  → 医疗术语标准化（缩写→全称）
  - noise_kb.db → 规则去噪（exact 类固定噪声删除）

列分类策略：
  - 医疗文本列（主诉/现病史/诊断/症状/描述/ask/answer/content…）→ 标准化 + 去噪
  - 标识列（姓名/ID/编号/日期/性别/年龄/科室…）→ 原样保留
  - 其他列 → 保守清洗（仅术语标准化，不删内容）

设计取舍：列级清洗只用规则（不调 LLM）——表格常上千行，逐格调 LLM 不现实，
也契合"减少 LLM 调用"。语义级噪声留给非结构化文本路径的 LLMNoiseFilter。
"""
import re
import time
import sqlite3
import io
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

# 列名分类关键词（中英文都覆盖）
_MEDICAL_COL_KW = ["主诉", "现病史", "既往史", "诊断", "症状", "描述", "病情", "建议",
                   "处理", "医嘱", "检查", "所见", "印象", "结论",
                   "ask", "answer", "content", "title", "question", "text", "desc", "note",
                   "complaint", "chief", "symptom", "diagnos", "present", "history",
                   "illness", "finding", "impression", "condition", "suggest", "advice",
                   "remark", "comment", "noisy_text", "cleaned_text", "record_text",
                   "self_report"]
_ID_COL_KW = ["姓名", "name", "id", "编号", "卡号", "床号", "日期", "date", "time",
              "性别", "sex", "gender", "年龄", "age", "科室", "department", "dept",
              "电话", "phone", "地址", "address", "民族", "职业", "工号"]
_DROP_COLS = {"clean_reference", "noise_labels", "noise_injected", "output_format_hint"}

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


class TableColumnCleaner(Mapper):

    def __init__(self, *args, **kwargs):
        super(TableColumnCleaner, self).__init__(*args, **kwargs)
        self.term_kb_path = kwargs.get("termKbPath", _TERM_KB)
        self.noise_kb_path = kwargs.get("noiseKbPath", _NOISE_KB)
        self._rule_engine = NoiseRuleEngine(self.noise_kb_path)
        self._load_kbs()

    def _load_kbs(self):
        """加载术语映射 + 噪声规则（与算子代码分离的外置 SQLite）"""
        # 术语库：abbr -> full，按长度降序避免 LDL-C 被拆
        self.term_dict = {}
        try:
            conn = sqlite3.connect(self.term_kb_path)
            self.term_dict = {a: f for a, f in conn.execute(
                "SELECT abbr, full FROM term_mappings WHERE status='active'").fetchall()
                if a and f}
            conn.close()
        except Exception as e:
            logger.warning(f"TableColumnCleaner: term_kb 加载失败 {e}")
        self.term_sorted = sorted(self.term_dict.keys(), key=len, reverse=True)

        # 噪声库：表格列级不调 LLM，所有 active 规则都用 re.sub 删除（都是真实数据
        # 提取的固定噪声短语，删除安全；过宽规则已在 noise_kb 中剔除）。
        # exact 类做 re.escape，regex 类原样，单条 try-compile 容错。
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
            logger.debug(f"TableColumnCleaner: noise_kb 加载 {e}")
            self._noise_count = 0
        logger.info(f"TableColumnCleaner: {len(self.term_dict)} terms, "
                    f"{self._noise_count} noise rules")

    _DEDUP_RE = re.compile(r'([\u4e00-\u9fffA-Za-z0-9·\-]{2,20})[（(]\1[）)]')

    def _classify_column(self, col_name: str) -> str:
        name = str(col_name).lower()
        if any(kw.lower() in name for kw in _ID_COL_KW):
            return "identifier"
        if any(kw.lower() in name for kw in _MEDICAL_COL_KW):
            return "medical_text"
        return "other"

    def _normalize_terms(self, text: str) -> str:
        """术语标准化（复用 term_kb 逻辑：跳过已有中文括号、去重）"""
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

    def _clean_cell(self, value, do_noise: bool) -> str:
        """清洗单个单元格：术语标准化 +（可选）去固定噪声"""
        if value is None:
            return value
        text = str(value)
        if not text.strip():
            return text
        text = deterministic_field_clean(text, rule_engine=self._rule_engine, remove_noise=do_noise)
    # 全角归一化必须先于缩写匹配执行。
        text = self._normalize_terms(text)
        if do_noise:
            if self.noise_re is not None:
                text = self.noise_re.sub("", text)
            text = _STRUCTURED_NOISE_RE.sub("", text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
            text = re.sub(r'^[，。；;、\s]+|[，。；;、\s]+$', '', text)
        return text

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        filepath = sample[self.filepath_key]
        filetype = str(sample.get(self.filetype_key, "")).lower()

        # 非表格文件：交给默认文本读取，本算子不处理（原样透传）
        if filetype not in ("csv", "xlsx", "xls"):
            self.read_file_first(sample)
            logger.info(f"TableColumnCleaner: {sample.get(self.filename_key)} "
                        f"非表格({filetype})，跳过")
            return sample

        import pandas as pd
        try:
            if filetype == "csv":
                df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
            else:
                df = pd.read_excel(filepath, dtype=str, keep_default_na=False)
        except Exception as e:
            logger.warning(f"TableColumnCleaner: 读表失败 {e}，回退文本处理")
            self.read_file_first(sample)
            return sample

        drop_cols = [col for col in df.columns if str(col).strip() in _DROP_COLS]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        col_report = {"__dropped__": drop_cols} if drop_cols else {}
        for col in df.columns:
            ctype = self._classify_column(col)
            col_report[str(col)] = ctype
            if ctype == "identifier":
                continue  # 标识列原样保留
            do_noise = (ctype == "medical_text")  # 医疗文本列才去噪，其他列仅标准化
            df[col] = df[col].apply(lambda v: self._clean_cell(v, do_noise))

        # 写回为 CSV 文本，交给 FileExporter 落盘为 csv
        # FileExporter.get_textfile_handler 依据 sample["target_type"] 决定输出后缀，
        # 不设则默认存 .txt → 这里显式设 csv 保持结构化格式
        sample[self.text_key] = df.to_csv(index=False)
        sample[self.data_key] = b""
        sample[self.filetype_key] = "csv"
        sample["target_type"] = "csv"

        logger.info(
            f"fileName: {sample[self.filename_key]}, "
            f"TableColumnCleaner costs {time.time()-start:.3f}s, "
            f"rows={len(df)}, cols={col_report}")
        return sample
