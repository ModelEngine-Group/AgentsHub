# -*- coding: utf-8 -*-
"""
医疗文本结构性预处理（纯规则，无需 LLM）。
在质量评分和术语标准化之前运行，处理脏数据。

被复用于：
  - mcp_server/ (preprocess_medical_text 工具)
  - operators/medical_term_normalizer/ 的前置步骤

处理内容：
  1. HTML 实体解码     (&amp; → &, &lt; → <, &#x8840; → 血)
  2. HTML 标签剥离     (<p>文本</p> → 文本)
  3. Emoji 去除        (😢😭 → 删除)
  4. 零宽/控制字符清除 (​, ﻿ 等)
  5. 全角转半角        （Ａ → A，１２３ → 123，，→ ,）
  6. 多余空白压缩      (多个空格/换行 → 单个)
  7. 重复字符压缩      (啊啊啊啊啊 → 啊啊)
"""
import re
import html
from dataclasses import dataclass
from typing import List


@dataclass
class PreprocessResult:
    cleaned: str
    changes: List[str]   # 记录做了哪些处理，供 Agent 展示给用户


# ---- 全角→半角映射表（数字、字母、常用标点） ----
_FULLWIDTH = str.maketrans(
    "　！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
    " !\"#$%&'()*+,-./"
    "0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
)

# ---- Emoji Unicode 范围 ----
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # 表情符号
    "\U0001F300-\U0001F5FF"   # 各类符号
    "\U0001F680-\U0001F6FF"   # 交通/地图
    "\U0001F1E0-\U0001F1FF"   # 国旗
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"   # 补充符号
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"   # 杂项符号
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)

# ---- 零宽/不可见字符 ----
_INVISIBLE_RE = re.compile(
    "[­​‌‍‎‏‪-‮⁠-⁯﻿￹-￻]"
)

# ---- 控制字符（保留 \t\n\r） ----
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ---- 重复字符（5次以上压缩到2次） ----
_REPEAT_RE = re.compile(r"(.)\1{4,}")


def preprocess(text: str) -> PreprocessResult:
    """对医疗文本做全面结构性清洗，返回清洗后文本和变更记录。"""
    if not text:
        return PreprocessResult(cleaned="", changes=[])

    original = text
    changes = []

    # 1. HTML 实体解码
    decoded = html.unescape(text)
    if decoded != text:
        changes.append("HTML实体解码")
        text = decoded

    # 2. HTML 标签剥离（保留标签内文本）
    stripped = re.sub(r"<[^>]{1,500}>", " ", text)
    if stripped != text:
        changes.append("HTML标签剥离")
        text = stripped

    # 3. Emoji 去除
    no_emoji = _EMOJI_RE.sub("", text)
    if no_emoji != text:
        changes.append("Emoji去除")
        text = no_emoji

    # 4. 零宽/不可见字符
    no_invis = _INVISIBLE_RE.sub("", text)
    if no_invis != text:
        changes.append("零宽字符去除")
        text = no_invis

    # 5. 控制字符
    no_ctrl = _CTRL_RE.sub("", text)
    if no_ctrl != text:
        changes.append("控制字符去除")
        text = no_ctrl

    # 6. 全角转半角（数字、字母、常用标点）
    half = text.translate(_FULLWIDTH)
    if half != text:
        changes.append("全角→半角")
        text = half

    # 7. 重复字符压缩（连续5+次 → 2次）
    compressed = _REPEAT_RE.sub(r"\1\1", text)
    if compressed != text:
        changes.append("重复字符压缩")
        text = compressed

    # 8. 多余空白压缩（保留段落换行）
    # 把连续3+行空行压缩为1行，行内多空格压缩为1格
    text = re.sub(r"[^\S\n]{2,}", " ", text)      # 行内多空格
    text = re.sub(r"\n{3,}", "\n\n", text)         # 多余空行
    text = text.strip()

    if text != original and "空白压缩" not in changes:
        if not changes or original.replace(" ", "").replace("\n", "") != text.replace(" ", "").replace("\n", ""):
            pass  # 空白压缩是隐式做的，不单独记录，除非引发实质性变化

    return PreprocessResult(cleaned=text, changes=changes)
