# -*- coding: utf-8 -*-
"""
空白规范化算子：处理空行、多余空格、噪声标记行、广告话术。
在 HtmlTagCleaner 之后、MedicalTermNormalizer 之前执行。
"""
import re
import time
from typing import Dict, Any

from loguru import logger
from datamate.core.base_op import Mapper

# 噪声标记行：整行只含 # / - / = / * / @ 及空白
_NOISE_LINE_RE = re.compile(r'^[\s#\-=\*@~_|]{1,30}$')
# 广告/模板话术行：@@...@@ 或 【广告】 形式
_AD_LINE_RE = re.compile(r'@@[^@]{1,100}@@|【广告[^】]{0,30}】')
# 行内连续空白（不含换行）：多个空格/制表符 → 单空格
_MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')
# 中文字符之间的行内空格通常来自 OCR/导出拆字，如“检 查”“患 者”。
_CJK_INLINE_SPACE_RE = re.compile(r'(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])')
_SPACE_BEFORE_CJK_PUNCT_RE = re.compile(r'(?<=[\u4e00-\u9fff])[ \t]+(?=[,.;:!?，。；：！？、）】》])')
_SPACE_AFTER_OPEN_CJK_PUNCT_RE = re.compile(r'(?<=[（【《])[ \t]+')
# 行首尾空白
_LINE_STRIP_RE = re.compile(r'^[ \t]+|[ \t]+$', re.MULTILINE)
# 连续3个以上空行 → 单空行
_MULTI_BLANK_RE = re.compile(r'\n{3,}')
# 疑似乱码 token 行：整行纯小写字母、长度>=8（如 augagagaafaf）
_GIBBERISH_RE = re.compile(r'^[a-z]{8,}$')
# 任务一真实问诊数据中高频出现的拆字噪声，保守重连。
_SPLIT_TOKEN_MAP = {
    "患 者": "患者",
    "医 生": "医生",
    "检 查": "检查",
    "治 疗": "治疗",
    "传 染": "传染",
    "诊 断": "诊断",
    "症 状": "症状",
}
# 轻量繁体转简体：只覆盖本项目真实样本中出现的医学常见字，不替代完整 OpenCC。
_TRADITIONAL = str.maketrans({
    "醫": "医", "藥": "药", "診": "诊", "療": "疗", "壓": "压",
    "囑": "嘱", "複": "复", "體": "体", "檢": "检", "驗": "验",
    "術": "术", "門": "门", "腫": "肿", "裡": "里", "裏": "里",
    "還": "还", "來": "来", "陰": "阴", "陽": "阳", "婦": "妇",
    "兒": "儿", "產": "产", "內": "内", "癥": "症",
})
    # 基准控制字段和重复尾部块不应进入正式清洗结果。
_NOISE_LABEL_LINE_RE = re.compile(r'^\s*noise_labels\s*[:：].*$', re.IGNORECASE)
_DUP_FRAGMENT_RE = re.compile(r'重复片段[:：].*?(?=^\s*record_id\s*[:：]|\Z)', re.DOTALL | re.MULTILINE)
_PAGE_BREAK_RE = re.compile(r'[-=]{3,}\s*PAGE_BREAK\s*[-=]{3,}', re.IGNORECASE)


def _is_gibberish(line: str) -> bool:
    """整行纯小写字母且字符重复度高 → 判为乱码 token，删除。
    用 不同字符数/长度 比例区分真实英文单词与乱码：
      metformin → 8/9≈0.89（保留）；augagagaafaf → 4/12≈0.33（删除）。"""
    if not _GIBBERISH_RE.match(line):
        return False
    return len(set(line)) / len(line) < 0.5


class WhitespaceNormalizer(Mapper):

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        if not text.strip():
            return sample

        if r"\n" in text:
            text = text.replace(r"\n", "\n")
        text = _DUP_FRAGMENT_RE.sub("", text)
        text = _PAGE_BREAK_RE.sub("", text)
        text = text.translate(_TRADITIONAL)
        for bad, good in _SPLIT_TOKEN_MAP.items():
            text = text.replace(bad, good)

        lines = text.splitlines()
        kept = []
        for line in lines:
            # 全角空格统一成普通空格，再去除行首尾空白。
            line = line.replace("\u3000", " ").strip()
            if _NOISE_LABEL_LINE_RE.match(line):
                continue
            # 丢弃纯噪声标记行（整行只有 ### 或 --- 等）
            if _NOISE_LINE_RE.match(line):
                continue
            # 丢弃广告话术行
            if _AD_LINE_RE.search(line):
                continue
            # 丢弃疑似乱码 token 行（如 augagagaafaf）
            if _is_gibberish(line):
                continue
            # 去除行末尾的噪声标记（如 "内容  ###" → "内容"）
            line = re.sub(r'[\s#\-=\*@~_|]+$', '', line).strip()
            if not line:
                continue
            # 压缩行内多余空格
            line = _MULTI_SPACE_RE.sub(' ', line)
            # 只处理行内空格/制表符，不跨换行合并，避免破坏段落边界。
            line = _CJK_INLINE_SPACE_RE.sub('', line)
            line = _SPACE_BEFORE_CJK_PUNCT_RE.sub('', line)
            line = _SPACE_AFTER_OPEN_CJK_PUNCT_RE.sub('', line)
            kept.append(line)

        # 重组，并将 3+ 个连续空行压缩为最多 1 个
        text = '\n'.join(kept)
        text = _MULTI_BLANK_RE.sub('\n\n', text).strip()

        sample[self.text_key] = text
        logger.info(
            f"fileName: {sample[self.filename_key]}, "
            f"WhitespaceNormalizer costs {time.time()-start:.6f}s, "
            f"lines: {len(lines)}→{len(kept)}"
        )
        return sample
