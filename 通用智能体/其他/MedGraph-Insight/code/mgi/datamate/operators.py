"""本地文本清洗算子（与 DataMate 内置文本算子对齐的轻量实现）。

命名对应 DataMate 算子市场中的同名/同义算子：
HtmlTagCleaner / TraditionalChineseCleaner / 去重过滤 / 长度过滤 / 脱敏等。
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable

_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t\u3000]+")
_PHONE = re.compile(r"1[3-9]\d{9}")
_IDCARD = re.compile(r"\d{17}[\dXx]|\d{15}")

# 常用繁->简映射（示例子集，可按需扩充）
_T2S = str.maketrans("醫藥檢驗臟腎糖尿病體溫壓細胞氣血管療診斷復發燒", "医药检验脏肾糖尿病体温压细胞气血管疗诊断复发烧")


def HtmlTagCleaner(records: list[str]) -> list[str]:
    """去除 HTML 标签。"""
    return [_HTML_TAG.sub("", r) for r in records]


def WhitespaceNormalizer(records: list[str]) -> list[str]:
    """空白归一化：合并连续空白、去首尾空白。"""
    return [_SPACES.sub(" ", r).strip() for r in records]


def TraditionalChineseCleaner(records: list[str]) -> list[str]:
    """繁体转简体（常用医疗字符子集）。"""
    return [r.translate(_T2S) for r in records]


def DedupFilter(records: list[str]) -> list[str]:
    """基于内容哈希的精确去重。"""
    seen: set[str] = set()
    out: list[str] = []
    for r in records:
        h = hashlib.md5(r.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out


def MinLengthFilter(records: list[str], min_len: int = 10) -> list[str]:
    """过滤过短文本。"""
    return [r for r in records if len(r) >= min_len]


def PhoneAnonymizer(records: list[str]) -> list[str]:
    """手机号脱敏。"""
    return [_PHONE.sub("<PHONE>", r) for r in records]


def IDCardAnonymizer(records: list[str]) -> list[str]:
    """身份证号脱敏。"""
    return [_IDCARD.sub("<ID_CARD>", r) for r in records]


def build_text_pipeline() -> dict[str, Callable[..., list[str]]]:
    """返回 算子名 -> 可调用 的注册表。"""
    return {
        "HtmlTagCleaner": HtmlTagCleaner,
        "WhitespaceNormalizer": WhitespaceNormalizer,
        "TraditionalChineseCleaner": TraditionalChineseCleaner,
        "DedupFilter": DedupFilter,
        "MinLengthFilter": MinLengthFilter,
        "PhoneAnonymizer": PhoneAnonymizer,
        "IDCardAnonymizer": IDCardAnonymizer,
    }
