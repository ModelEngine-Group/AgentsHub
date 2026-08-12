"""
任务一清洗证据整理模块。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


TERM_REPLACEMENTS = [
    ("T2DM", "2型糖尿病"),
    ("DM", "糖尿病"),
    ("BP", "血压"),
    ("HbA1c", "糖化血红蛋白"),
    ("FPG", "空腹血糖"),
    ("Hp", "幽门螺杆菌"),
    ("qd", "每日一次"),
    ("bid", "每日两次"),
    ("tid", "每日三次"),
    ("po", "口服"),
    ("iv", "静脉注射"),
]

NOISE_REMOVAL_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("url_removed", "URL 链接", re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)),
    ("html_tag_removed", "HTML 标签", re.compile(r"<[^>]{1,500}>")),
    ("emoji_removed", "Emoji", re.compile(r"[\U0001F300-\U0001FFFF\U00002700-\U000027BF]+")),
    ("invisible_removed", "不可见字符", re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")),
    ("fullwidth_to_halfwidth", "全角字符", re.compile(r"[\u3000\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]")),
    ("traditional_to_simplified", "繁体医疗词", re.compile(r"[醫藥診療壓囑複體檢驗術門腫]")),
    ("mojibake_removed", "乱码占位符", re.compile(r"(?:锟斤拷|�)+")),
    ("test_banner_removed", "测试/内部标记", re.compile(r"仅供(?:内部)?测试|测试数据请忽略")),
    ("controlled_noise_removed", "受控噪声标记", re.compile(r"controlled_noise_\d{8}\??", re.IGNORECASE)),
    ("question_mojibake_removed", "问号乱码串", re.compile(r"(?<![\u4e00-\u9fffA-Za-z0-9])\?{2,}(?![\u4e00-\u9fffA-Za-z0-9])")),
    (
        "his_header_removed",
        "HIS/系统导出头",
        re.compile(r"(?:HIS|EMR)?系统自动(?:导出|生成)|填表时间|填表人", re.IGNORECASE),
    ),
    (
        "social_or_ad_removed",
        "非病历社交/广告文本",
        re.compile(r"扫码进群|关注公众号|领取.{0,12}偏方|@[A-Za-z0-9_\-\u4e00-\u9fff]{1,30}"),
    ),
]


def _read_text(path: Path, limit: int = 2_000_000) -> str:
    if not path.exists() or path.stat().st_size > limit:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _match_outputs(source_paths: Iterable[Path], output_paths: Iterable[Path]) -> list[tuple[Path, Path]]:
    outputs = list(output_paths)
    by_name = {path.name: path for path in outputs}
    by_stem = {path.stem.replace(".cleaned", ""): path for path in outputs}
    pairs: list[tuple[Path, Path]] = []
    for source in source_paths:
        output = by_name.get(source.name) or by_stem.get(source.stem)
        if output:
            pairs.append((source, output))
    if not pairs and len(list(source_paths)) == 1 and len(outputs) == 1:
        source = list(source_paths)[0]
        pairs.append((source, outputs[0]))
    return pairs


def summarize_cleaning_evidence(
    source_paths: Iterable[Path],
    output_paths: Iterable[Path],
) -> dict:
    source_list = list(source_paths)
    output_list = list(output_paths)
    pairs = _match_outputs(source_list, output_list)

    replacements: list[dict] = []
    noise_removals: list[dict] = []
    changed_files = 0
    compared_files = 0
    total_before = 0
    total_after = 0

    for source, output in pairs:
        before = _read_text(source)
        after = _read_text(output)
        if not before or not after:
            continue
        compared_files += 1
        total_before += len(before)
        total_after += len(after)
        if before != after:
            changed_files += 1
        for raw, normalized in TERM_REPLACEMENTS:
            before_raw = before.count(raw)
            after_raw = after.count(raw)
            before_norm = before.count(normalized)
            after_norm = after.count(normalized)
            if before_raw > after_raw and after_norm > before_norm:
                replacements.append(
                    {
                        "file": source.name,
                        "from": raw,
                        "to": normalized,
                        "before_count": before_raw,
                        "after_count": after_raw,
                    }
                )
        for key, label, pattern in NOISE_REMOVAL_PATTERNS:
            before_count = len(pattern.findall(before))
            if not before_count:
                continue
            after_count = len(pattern.findall(after))
            if after_count < before_count:
                noise_removals.append(
                    {
                        "file": source.name,
                        "type": key,
                        "label": label,
                        "before_count": before_count,
                        "after_count": after_count,
                        "removed_count": before_count - after_count,
                    }
                )

    return {
        "compared_files": compared_files,
        "changed_files": changed_files,
        "char_delta": total_after - total_before,
        "observed_term_replacements": replacements,
        "observed_noise_removals": noise_removals,
        "semantic_noise_filter": {
            "reported": False,
            "note": "per_file_semantic_noise_evidence_unavailable",
        },
        "reporting_rule": (
            "Only observed_term_replacements may be reported as concrete replacements. "
            "Only observed_noise_removals may be reported as concrete noise removals. "
            "If semantic_noise_filter.reported is false, omit detailed semantic-noise claims."
        ),
    }
