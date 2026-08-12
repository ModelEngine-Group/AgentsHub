# -*- coding: utf-8 -*-
"""
结构化字段清理工具。

该模块为 CSV、JSON 和 JSONL 字段提供确定性清理函数。
"""

from __future__ import annotations

import html
import re


_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FFFF"
    "\U00002702-\U000027B0"
    "\U0000200D"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
_URL_RE = re.compile(
    r"(?:https?|ftp|file)://\S+|"
    r"\b(?:www|w{2,3})\s*[.\u3002\uff0e]\s*[^\s,;\u3001\u3002]{1,80}?\s*[.\u3002\uff0e]\s*"
    r"(?:com|cn|net|org|top|xyz|wang)\b",
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)
_OCR_BARE_URL_RE = re.compile(
    r"\b(?:www|w{2,3})\s*(?:[.\u3002\uff0e]\s*)+"
    r"(?:[A-Za-z0-9_\-]{2,40}[瑚坶]"
    r"|[A-Za-z0-9_\-]{2,40}(?:[.\u3002\uff0e]\s*)+"
    r"[A-Za-z](?:\s*[.\u3002\uff0e]?\s*[A-Za-z]){0,4}"
    r"|[A-Za-z0-9_\-]{3,40})",
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]*)?/?>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\ufffd]")
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_GARBLED_TOKEN_RE = re.compile(r"(?:锟斤拷|�)+")
_QUESTION_MOJIBAKE_RE = re.compile(r"(?<![\u4e00-\u9fffA-Za-z0-9])\?{2,}(?![\u4e00-\u9fffA-Za-z0-9])")
_CONTROLLED_NOISE_RE = re.compile(
    r"controlled_noise_\d{8}\??|"
    r"HIS\s+system\s+auto\s+export|"
    r"EMR\s+system\s+auto\s+export|"
    r"\?{1,}\d{4}-\d{1,2}-\d{1,2}\s*\?*",
    re.IGNORECASE,
)
_NOISE_MARKER_RUN_RE = re.compile(r"[#@=*~_|]{3,}")
_NOISE_DASH_LINE_RE = re.compile(r"(?m)^\s*[-—–]{3,}\s*$")
_PAGE_BREAK_RE = re.compile(r"[-=]{3,}\s*PAGE_BREAK\s*[-=]{3,}", re.IGNORECASE)
_DUP_FRAGMENT_RE = re.compile(r"重复片段[:：].*?(?=(?:\n|$))")
_VIEW_MORE_RE = re.compile(r"查看更多关于[^\n]{0,180}(?:\.\.\.|…)", re.IGNORECASE)
_OCR_CONFIDENCE_RE = re.compile(r"@{0,3}#{0,3}\s*OCR_CONFIDENCE\s*=\s*[0-9.]+\s*#{0,3}@{0,3}", re.IGNORECASE)
_HIS_EXPORT_RE = re.compile(
    r"(?:HIS|EMR)?系统自动(?:导出|生成)|"
    r"HIS系统自动导出\s*填表时间[:：][^\n]{0,80}|"
    r"填表时间[:：][^\n]{0,80}填表人[:：][^\n]{0,80}"
)
_SYSTEM_BOILERPLATE_RE = re.compile(r"【系统提示】[^\n]{0,160}(?:自动生成|内部流转)[^\n]*")
_DEBUG_METADATA_RE = re.compile(
    r"患者ID[:：]\s*TMP-[^\n]{0,120}|"
    r"设备[:：]\s*EMR-DEBUG[^\n]{0,120}|"
    r"操作员[:：]\s*admin_test[^\n]{0,120}"
)
_LOST_LINK_RE = re.compile(r"(?:图片|资料|链接)?链接?已失效[:：]?\s*", re.IGNORECASE)
_AD_RE = re.compile(
    r"(?:关注|注)?公众号领取健康资料|"
    r"广告合作请联系\s*\S*|"
    r"免责声明[:：][^\n]{0,200}|"
    r"您可以提供微信号[^\n，。；;]{0,80}|"
    r"微信号\s*[A-Za-z0-9_\-]{3,40}"
)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_CJK_INLINE_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")
_SPACE_BEFORE_CJK_PUNCT_RE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[,.;:!?，。；：！？、）】》])")
_SPACE_AFTER_OPEN_CJK_PUNCT_RE = re.compile(r"(?<=[（【《])[ \t]+")


try:
    from opencc import OpenCC  # type: ignore

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None

try:
    from zhconv import convert as _zhconv_convert  # type: ignore
except Exception:
    _zhconv_convert = None


def normalize_fullwidth(text: str) -> str:
    chars = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            chars.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            chars.append(chr(code - 0xFEE0))
        else:
            chars.append(char)
    return "".join(chars)


def restore_cjk_punctuation(text: str) -> str:
    """在上游全角清理后保持结构化医学字段可读。"""
    cjk_end = r"(?=[\u4e00-\u9fff\s\"'，,。；;：:\)）\]】]|$)"
    text = re.sub(r"(?<=[\u4e00-\u9fff])\?" + cjk_end, "？", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])!" + cjk_end, "！", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]),(?=[\u4e00-\u9fff])", "，", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]);(?=[\u4e00-\u9fff])", "；", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]):(?=[\u4e00-\u9fff])", "：", text)
    return text


def deterministic_field_clean(text: str, *, rule_engine=None, remove_noise: bool = True) -> str:
    if not isinstance(text, str) or not text:
        return text
    text = html.unescape(text)
    if r"\n" in text:
        text = text.replace(r"\r\n", "\n").replace(r"\n", "\n")
    text = normalize_fullwidth(text)
    if _OPENCC is not None:
        try:
            text = _OPENCC.convert(text)
        except Exception:
            pass
    elif _zhconv_convert is not None:
        try:
            text = _zhconv_convert(text, "zh-hans")
        except Exception:
            pass
    text = _HTML_COMMENT_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _OCR_BARE_URL_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = _GARBLED_TOKEN_RE.sub("", text)
    text = _CONTROLLED_NOISE_RE.sub(" ", text)
    text = _QUESTION_MOJIBAKE_RE.sub(" ", text)
    text = _NOISE_MARKER_RUN_RE.sub("", text)
    text = _NOISE_DASH_LINE_RE.sub("", text)
    text = _PAGE_BREAK_RE.sub(" ", text)
    text = _DUP_FRAGMENT_RE.sub(" ", text)
    text = _VIEW_MORE_RE.sub(" ", text)
    text = _OCR_CONFIDENCE_RE.sub(" ", text)
    text = _HIS_EXPORT_RE.sub(" ", text)
    text = _SYSTEM_BOILERPLATE_RE.sub(" ", text)
    text = _DEBUG_METADATA_RE.sub(" ", text)
    text = _LOST_LINK_RE.sub(" ", text)
    text = _AD_RE.sub(" ", text)
    if remove_noise and rule_engine is not None:
        try:
            text, _removed = rule_engine.clean(text)
        except Exception:
            pass
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _CJK_INLINE_SPACE_RE.sub("", text)
    text = _SPACE_BEFORE_CJK_PUNCT_RE.sub("", text)
    text = _SPACE_AFTER_OPEN_CJK_PUNCT_RE.sub("", text)
    text = restore_cjk_punctuation(text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()
