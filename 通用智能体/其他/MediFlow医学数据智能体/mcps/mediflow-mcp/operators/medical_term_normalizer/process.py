# -*- coding: utf-8 -*-
"""
医疗术语标准化算子 v3.0：规则字典优先 + LLM 兜底
- 快速路径：基于 medical_abbrev.ABBR_DICT 做正则替换（0 API 调用，<1ms）
- LLM 路径：仅当文本中仍存在未识别英文缩写时才调用（处理生僻缩写）
- 来源：RuijinDiabetes 临床标注数据（493篇）+ 通用中文医学缩写库
"""
import os
import re
import time
from pathlib import Path
from typing import Dict, Any

import requests
from loguru import logger

from datamate.core.base_op import Mapper

# 回退缩写库（仅当外部 SQLite term_kb.db 不可用时使用）
try:
    from ops.user.medical_term_normalizer.medical_abbrev import ABBR_DICT as _FALLBACK_DICT
except ImportError:
    try:
        from medical_abbrev import ABBR_DICT as _FALLBACK_DICT
    except ImportError:
        _FALLBACK_DICT = {}

# 检测未识别英文缩写：全大写2-6字母（排除单字母和 DNA/RNA 等已知非医学缩写）
_UNKNOWN_ABBR_RE = re.compile(r'\b[A-Z]{2,6}(?:-\d)?\b')
_KNOWN_NON_MEDICAL = {"DNA", "RNA", "ATP", "CPU", "AI", "ID", "OK", "WHO", "UN"}

PROMPT_TEMPLATE = """你是医学文本标准化专家。
以下文本中存在未识别的医学缩写（已用【】标出），请将其替换为规范中文术语，其他内容保持不变，直接输出处理后的完整文本，不要解释。

文本：
{text}"""


class MedicalTermNormalizer(Mapper):

    def __init__(self, *args, **kwargs):
        super(MedicalTermNormalizer, self).__init__(*args, **kwargs)
        self.ollama_url = kwargs.get("ollamaUrl", "https://api.deepseek.com/v1/chat/completions")
        self.model = kwargs.get("modelName", "deepseek-chat")
        self.timeout = int(kwargs.get("timeoutSeconds", 60))
        self.api_key = kwargs.get("apiKey") or os.environ.get("CCF_LLM_API_KEY", "")
        if not self.api_key:
            key_file = os.environ.get("CCF_LLM_API_KEY_FILE", "/run/secrets/ccf_llm_api_key")
            try:
                self.api_key = Path(key_file).read_text(encoding="utf-8").strip()
            except OSError:
                self.api_key = ""
        # 从外部 SQLite 术语知识库加载映射（与算子代码物理分离，支持增量学习）
        kb_path = kwargs.get("termKbPath",
            "/opt/runtime/datamate/ops/user/medical_term_normalizer/term_kb.db")
        self._load_term_kb(kb_path)

    def _load_term_kb(self, kb_path):
        """从 term_kb.db 加载 abbr→full 映射；失败时回退到内置字典"""
        import sqlite3 as _sql
        self.abbr_dict = {}
        self.abbr_negative_patterns = {}
        try:
            conn = _sql.connect(kb_path)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(term_mappings)")}
            if "negative_patterns" in columns:
                rows = conn.execute(
                    """
                    SELECT abbr, full, COALESCE(negative_patterns, '')
                      FROM term_mappings WHERE status='active'
                    """
                ).fetchall()
            else:
                rows = [
                    (abbr, full, "")
                    for abbr, full in conn.execute(
                        "SELECT abbr, full FROM term_mappings WHERE status='active'"
                    )
                ]
            conn.close()
            self.abbr_dict = {a: f for a, f, _ in rows if a and f}
            self.abbr_negative_patterns = {
                a: [item for item in negative.split("||") if item]
                for a, _full, negative in rows if a and negative
            }
            logger.info(f"MedicalTermNormalizer: loaded {len(self.abbr_dict)} terms from KB")
        except Exception as e:
            logger.warning(f"MedicalTermNormalizer: KB unavailable ({e}), using fallback dict")
        if not self.abbr_dict:
            self.abbr_dict = dict(_FALLBACK_DICT)
        # 按长度降序排序，避免 LDL-C 被 LDL 先匹配拆分
        self.abbr_sorted = sorted(self.abbr_dict.keys(), key=len, reverse=True)

    # 匹配"中文术语（缩写）"模式，替换后用于去重清理
    _DEDUP_RE = re.compile(r'([\u4e00-\u9fffA-Za-z0-9·\-]{2,20})[（(]\1[）)]')

    def _dict_normalize(self, text: str) -> str:
        """规则快速路径：按字典从长到短替换，避免 LDL-C 被拆分。
        同时跳过缩写已有对应中文术语在括号外紧邻的情况，防止重复展开。"""
        for abbr in self.abbr_sorted:
            if abbr not in text:
                continue
            full = self.abbr_dict[abbr]
            # 若 full 已出现在 abbr 的紧邻括号之外（如 "冠心病（CHD）"），跳过替换
            # 模式：full 紧跟（abbr）→ 不替换
            skip_re = re.compile(re.escape(full) + r'[（(]' + re.escape(abbr) + r'[）)]')
            if skip_re.search(text):
                continue
            match_re = re.compile(
                r'(?<![A-Za-z0-9])' + re.escape(abbr) + r'(?![A-Za-z0-9])'
            )
            negative_patterns = self.abbr_negative_patterns.get(abbr, [])

            def replace(match):
                left = text.rfind("\n", 0, match.start()) + 1
                right = text.find("\n", match.end())
                right = len(text) if right < 0 else right
                window = text[left:right]
                relative_start = match.start() - left
                relative_end = match.end() - left
                for pattern in negative_patterns:
                    for protected in re.finditer(pattern, window, re.IGNORECASE):
                        if (
                            protected.start() < relative_end
                            and protected.end() > relative_start
                        ):
                            return match.group(0)
                return full

            text = match_re.sub(replace, text)
        # 清理双重展开残留：X（X）→ X
        text = self._DEDUP_RE.sub(r'\1', text)
        return text

    def _has_unknown_abbrs(self, text: str) -> bool:
        """判断字典替换后是否仍有未识别的英文缩写。
        排除：(1)已知非医学缩写 (2)已在字典中但被skip_re保留的（如"冠心病（CHD）"里的CHD）"""
        matches = _UNKNOWN_ABBR_RE.findall(text)
        return any(m for m in matches
                   if m not in _KNOWN_NON_MEDICAL and m not in self.abbr_dict)

    def _call_llm(self, text: str) -> str:
        prompt = PROMPT_TEMPLATE.format(text=text)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(self.ollama_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        if not text.strip():
            return sample

        # 1. 规则快速路径（覆盖95%常见缩写，几乎无耗时）
        text = self._dict_normalize(text)
        used_llm = False

        # 2. LLM 兜底（仅当字典处理后仍有未识别缩写）
        if self._has_unknown_abbrs(text):
            try:
                text = self._call_llm(text)
                used_llm = True
            except Exception as e:
                logger.warning(f"MedicalTermNormalizer LLM fallback failed: {e}")

        sample[self.text_key] = text
        elapsed = time.time() - start
        logger.info(
            f"fileName: {sample[self.filename_key]}, "
            f"MedicalTermNormalizer costs {elapsed:.6f}s "
            f"({'dict+LLM' if used_llm else 'dict-only'})"
        )
        return sample
