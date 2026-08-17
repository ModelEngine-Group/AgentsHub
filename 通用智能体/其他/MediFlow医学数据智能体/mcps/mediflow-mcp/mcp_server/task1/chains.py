"""
任务一清洗链定义模块。
"""

from __future__ import annotations

def operator_config(operator_id: str, name: str) -> dict:
    return {"id": operator_id, "name": name, "inputs": "text", "outputs": "text", "overrides": {}}


DEFAULT_OPERATORS = [
    operator_config("EmojiCleaner", "颜文字/Emoji去除"),
    operator_config("UrlRemover", "URL/HTML注释/实体清理"),
    operator_config("GrableCharactersCleaner", "文档乱码去除"),
    operator_config("InvisibleCharactersCleaner", "不可见字符去除"),
    operator_config("FullWidthCharacterCleaner", "全角转半角"),
    operator_config("TraditionalChineseCleaner", "繁体转简体"),
    operator_config("HtmlTagCleaner", "HTML标签去除"),
    operator_config("WhitespaceNormalizer", "空行/空格/噪声标记/广告行清理"),
    operator_config("FileWithShortOrLongLengthFilter", "文档字数检查"),
    operator_config("FileWithHighRepeatPhraseRateFilter", "文档词重复率检查"),
    operator_config("FileWithHighSpecialCharRateFilter", "文档特殊字符率检查"),
    operator_config("DuplicateFilesFilter", "相似文档去除"),
    operator_config("MedicalTermNormalizer", "医疗术语标准化"),
]


def task1_mixed_chain_map() -> dict[str, list[str | dict]]:
    base = [
        "EmojiCleaner",
        "UrlRemover",
        "GrableCharactersCleaner",
        "InvisibleCharactersCleaner",
        "FullWidthCharacterCleaner",
        "TraditionalChineseCleaner",
        "HtmlTagCleaner",
        "WhitespaceNormalizer",
    ]
    text_filters = [
        "FileWithShortOrLongLengthFilter",
        "FileWithHighRepeatPhraseRateFilter",
        "FileWithHighSpecialCharRateFilter",
        "DuplicateFilesFilter",
    ]
    return {
        "text": base + text_filters + ["MedicalTermNormalizer", "LLMNoiseFilter"],
        "csv": base + ["TableColumnCleaner"],
        "json": base + ["JsonFieldCleaner"],
        "jsonl": base + ["JsonFieldCleaner"],
        "pdf": base + text_filters + ["MedicalTermNormalizer", "LLMNoiseFilter"],
    }


def task1_mixed_chain_descriptions() -> dict[str, str]:
    return {
        "text": "文本链：基础噪声清理、字符规范化、HTML/URL/HIS/OCR/debug/广告等清理、医学术语标准化、LLM语义噪声过滤。",
        "csv": "表格链：TableColumnCleaner，按单元格/字段清理内容并保持 CSV 输出。",
        "json": "JSON链：JsonFieldCleaner，递归清理文本字段并保持 JSON 输出。",
        "jsonl": "JSONL链：JsonFieldCleaner，逐行清理文本字段并保持 JSONL 输出。",
        "pdf": "PDF链：MCP 前置调用远程 MinerU 提取 TXT，再由 DataMate 执行文本噪声清理、字符规范化和医学术语标准化。",
    }
