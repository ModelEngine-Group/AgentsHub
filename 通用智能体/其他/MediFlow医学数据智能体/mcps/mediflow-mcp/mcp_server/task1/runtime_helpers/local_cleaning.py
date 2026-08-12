# -*- coding: utf-8 -*-
"""任务一结构化字段清理辅助函数。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .quality_eval import evaluate_file, summarize


TEXT_KEYS = {
    "noisy_text",
    "cleaned_text",
    "text",
    "content",
    "question",
    "answer",
    "ask",
    "self_report",
    "record_text",
    "title",
}

TERM_MAP = {
    "T2DM": "2型糖尿病",
    "DM": "糖尿病",
    "HTN": "高血压",
    "BP": "血压",
    "CHD": "冠心病",
    "HbA1c": "糖化血红蛋白",
    "LDL-C": "低密度脂蛋白胆固醇",
}

FULLWIDTH = str.maketrans(
    "　！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
    " !\"#$%&'()*+,-./"
    "0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
)
TRADITIONAL = str.maketrans({
    "醫": "医",
    "藥": "药",
    "診": "诊",
    "療": "疗",
    "壓": "压",
    "囑": "嘱",
    "複": "复",
    "體": "体",
    "檢": "检",
    "驗": "验",
    "術": "术",
    "門": "门",
    "腫": "肿",
    "裡": "里",
    "裏": "里",
    "還": "还",
    "來": "来",
    "陰": "阴",
    "陽": "阳",
    "婦": "妇",
    "兒": "儿",
    "產": "产",
    "內": "内",
    "癥": "症",
})

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FFFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
HTML_TAG_RE = re.compile(r"<[^>]{1,500}>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b|"
    r"@[A-Za-z0-9_\-\u4e00-\u9fff]{1,30}"
)
INVISIBLE_RE = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PAGE_BREAK_RE = re.compile(r"[-=]{3,}\s*PAGE_BREAK\s*[-=]{3,}", re.IGNORECASE)
OCR_RE = re.compile(r"@{0,3}#{0,3}\s*OCR_CONFIDENCE\s*=\s*[0-9.]+\s*#{0,3}@{0,3}", re.IGNORECASE)
MARKUP_BLOB_RE = re.compile(r"@{2,}#{2,}|#{3,}|@@[^@]{0,100}@@")
MOJIBAKE_RE = re.compile(r"(?:锟斤拷|�)+")
NOISE_LABEL_LINE_RE = re.compile(r"(?m)^\s*noise_labels\s*[:：].*$")
DUP_FRAGMENT_RE = re.compile(r"(?s)重复片段[:：].*?(?=\nrecord_id\s*[:：]|\Z)")
VIEW_MORE_RE = re.compile(
    r"查看更多关于[^\n]{0,180}(?:\.\.\.|…)",
    re.IGNORECASE,
)
HIS_RE = re.compile(
    r"(?:HIS|EMR)?系统自动(?:导出|生成)|"
    r"HIS系统自动导出\s*填表时间[:：][^\n]{0,80}|"
    r"填表时间[:：][^\n]{0,80}填表人[:：][^\n]{0,80}"
)
SYSTEM_RE = re.compile(r"【系统提示】[^\n]{0,120}(?:自动生成|内部流转)[^\n]*")
TEST_BANNER_RE = re.compile(r"仅供(?:内部)?测试|测试数据请忽略")
DEBUG_META_RE = re.compile(r"患者ID[:：]\s*TMP-[^\n]{0,80}|设备[:：]\s*EMR-DEBUG[^\n]{0,80}|操作员[:：]\s*admin_test")
LOST_LINK_RE = re.compile(r"(?:图片|资料|链接)?链接?已失效[:：]?\s*", re.IGNORECASE)
AD_RE = re.compile(
    r"(?:关注|注)?公众号领取健康资料|"
    r"广告合作请联系\s*\S*|"
    r"免责声明[:：][^\n]{0,200}|"
    r"您可以提供微信号[^\n，。；;]{0,80}|"
    r"微信号\s*[A-Za-z0-9_\-]{3,40}"
)
MARKUP_TAIL_RE = re.compile(r"[\s#=\-*@~_|]{3,}$")
SPLIT_TOKEN_MAP = {
    "患 者": "患者",
    "医 生": "医生",
    "检 查": "检查",
    "治 疗": "治疗",
    "传 染": "传染",
    "诊 断": "诊断",
    "症 状": "症状",
}


@dataclass
class CleanStats:
    text_fields: int = 0
    changed_fields: int = 0
    changes: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str) -> None:
        self.changes[name] = self.changes.get(name, 0) + 1


def normalize_terms(text: str, stats: CleanStats) -> str:
    for abbr in sorted(TERM_MAP, key=len, reverse=True):
        if abbr not in text:
            continue
        full = TERM_MAP[abbr]
        skip_re = re.compile(re.escape(full) + r"[（(]" + re.escape(abbr) + r"[）)]")
        if skip_re.search(text):
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(abbr) + r"(?![A-Za-z0-9])")
        text, count = pattern.subn(full, text)
        if count:
            stats.add("medical_term_normalized")
    text = re.sub(r"(.{2,10})（\1）", r"\1", text)
    return text


def clean_text(text: str, stats: CleanStats) -> str:
    original = text
    stats.text_fields += 1

    replacements: List[Tuple[str, re.Pattern[str], str]] = [
        ("html_tag_removed", HTML_TAG_RE, " "),
        ("url_removed", URL_RE, " "),
        ("email_removed", EMAIL_RE, " "),
        ("page_break_removed", PAGE_BREAK_RE, " "),
        ("noise_label_line_removed", NOISE_LABEL_LINE_RE, " "),
        ("duplicated_fragment_removed", DUP_FRAGMENT_RE, " "),
        ("view_more_removed", VIEW_MORE_RE, " "),
        ("ocr_tail_removed", OCR_RE, " "),
        ("markup_blob_removed", MARKUP_BLOB_RE, " "),
        ("mojibake_removed", MOJIBAKE_RE, " "),
        ("his_header_removed", HIS_RE, " "),
        ("system_boilerplate_removed", SYSTEM_RE, " "),
        ("test_banner_removed", TEST_BANNER_RE, " "),
        ("debug_metadata_removed", DEBUG_META_RE, " "),
        ("lost_link_removed", LOST_LINK_RE, " "),
        ("ad_disclaimer_removed", AD_RE, " "),
        ("invisible_removed", INVISIBLE_RE, ""),
        ("control_char_removed", CTRL_RE, ""),
    ]

    text = html.unescape(text)
    if text != original:
        stats.add("html_entity_unescaped")

    if r"\n" in text:
        text = text.replace(r"\n", "\n")
        stats.add("escaped_newline_decoded")

    before = text
    text = text.translate(FULLWIDTH)
    if text != before:
        stats.add("fullwidth_to_halfwidth")

    before = text
    text = text.translate(TRADITIONAL)
    if text != before:
        stats.add("traditional_to_simplified")

    for name, pattern, repl in replacements:
        text, count = pattern.subn(repl, text)
        if count:
            stats.add(name)

    before = text
    text = EMOJI_RE.sub("", text)
    if text != before:
        stats.add("emoji_removed")

    text = normalize_terms(text, stats)

    for bad, good in SPLIT_TOKEN_MAP.items():
        if bad in text:
            text = text.replace(bad, good)
            stats.add("split_token_rejoined")

    lines = []
    for line in text.splitlines():
        line = MARKUP_TAIL_RE.sub("", line.strip())
        if not line:
            continue
        # 删除前序清理后只剩分隔符的行。
        if re.fullmatch(r"[-=*_#\s]{3,}", line):
            continue
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if text != original:
        stats.changed_fields += 1
    return text


def repair_json_text_for_parse(text: str, stats: CleanStats | None = None) -> str:
    """在字段级清理前修复导出器造成的 JSON 控制字符问题。

    部分文本导出路径会把换行或制表符直接写入 JSON 字符串，导致解析失败。
    本函数只在扫描器位于字符串内部时转义控制字符，避免改变 JSON 结构。
    """
    repaired: List[str] = []
    in_string = False
    escaped = False
    changed = False

    for char in text:
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            repaired.append(char)
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            repaired.append(char)
            continue
        if in_string and char in {"\n", "\r", "\t"}:
            repaired.append({"\n": r"\n", "\r": r"\r", "\t": r"\t"}[char])
            changed = True
            continue
        repaired.append(char)

    if changed and stats is not None:
        stats.add("json_control_char_escaped")
    return "".join(repaired)


def clean_obj(value: Any, stats: CleanStats, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: clean_obj(v, stats, k) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_obj(item, stats, key) for item in value]
    if isinstance(value, str):
        return clean_text(value, stats)
    return value


def strip_eval_only_fields(value: Any) -> Any:
    """移除不应进入任务一最终数据集的评测辅助字段。"""
    if isinstance(value, dict):
        return {
            k: strip_eval_only_fields(v)
            for k, v in value.items()
            if k not in {"clean_reference", "noise_labels", "output_format_hint"}
        }
    if isinstance(value, list):
        return [strip_eval_only_fields(item) for item in value]
    return value


def iter_files(input_dir: Path, max_files_per_format: int) -> Iterable[Path]:
    for subdir, suffix in (("txt", "*.txt"), ("csv", "*.csv"), ("jsonl", "*.jsonl"), ("json", "*.json")):
        paths = sorted((input_dir / subdir).glob(suffix))
        yield from paths[:max_files_per_format]


def clean_txt(path: Path, out: Path, stats: CleanStats) -> None:
    out.write_text(clean_text(path.read_text(encoding="utf-8", errors="replace"), stats) + "\n", encoding="utf-8")


def clean_csv(path: Path, out: Path, stats: CleanStats) -> None:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = list(csv.DictReader(raw.splitlines()))
    fieldnames = list(rows[0].keys()) if rows else []
    for row in rows:
        row.pop("clean_reference", None)
        row.pop("noise_labels", None)
        row.pop("output_format_hint", None)
        for key, value in list(row.items()):
            if key in TEXT_KEYS and isinstance(value, str):
                row[key] = clean_text(value, stats)
    fieldnames = [
        name for name in fieldnames
        if name not in {"clean_reference", "noise_labels", "output_format_hint"}
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_jsonl(path: Path, out: Path, stats: CleanStats) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(repair_json_text_for_parse(line, stats))
            obj = strip_eval_only_fields(obj)
            dst.write(json.dumps(clean_obj(obj, stats), ensure_ascii=False) + "\n")


def clean_json(path: Path, out: Path, stats: CleanStats) -> None:
    obj = json.loads(repair_json_text_for_parse(path.read_text(encoding="utf-8"), stats))
    obj = strip_eval_only_fields(obj)
    out.write_text(json.dumps(clean_obj(obj, stats), ensure_ascii=False, indent=2), encoding="utf-8")


def clean_file(path: Path, input_dir: Path, output_dir: Path, stats: CleanStats) -> Path:
    rel = path.relative_to(input_dir)
    out = output_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        clean_txt(path, out, stats)
    elif suffix == ".csv":
        clean_csv(path, out, stats)
    elif suffix == ".jsonl":
        clean_jsonl(path, out, stats)
    elif suffix == ".json":
        clean_json(path, out, stats)
    else:
        raise ValueError(f"Unsupported file suffix: {path}")
    return out


def apply_medical_false_positive_exemptions(quality: Dict[str, Any]) -> Dict[str, Any]:
    """避免宽泛残留噪声规则误判合法医学文本。"""
    ignored = []
    for item in quality.get("files", []):
        kept = []
        for issue in item.get("residual_noise", []):
            if (
                issue.get("issue") == "his_export_banner"
                and "免疫系统" in str(issue.get("match", ""))
            ):
                ignored.append({**issue, "file": item.get("path")})
                continue
            kept.append(issue)
        item["residual_noise"] = kept

    totals = quality["totals"]
    totals["residual_noise"] = sum(len(item.get("residual_noise", [])) for item in quality.get("files", []))
    quality["ignored_medical_false_positives"] = ignored
    quality["pass"] = (
        totals["files"] > 0
        and totals["records"] >= 1
        and totals["parse_errors"] == 0
        and totals["empty_text"] == 0
        and totals["residual_noise"] == 0
        and totals.get("duplicate_content", 0) == 0
    )
    return quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--max-files-per-format", type=int, default=6)
    parser.add_argument("--min-records", type=int, default=1000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = CleanStats()
    cleaned_files = [
        clean_file(path, input_dir, output_dir, stats)
        for path in iter_files(input_dir, args.max_files_per_format)
    ]
    quality = summarize([evaluate_file(path) for path in cleaned_files], min_records=args.min_records)
    quality = apply_medical_false_positive_exemptions(quality)
    if quality["totals"]["records"] < args.min_records:
        quality["pass"] = False
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "cleaned_file_count": len(cleaned_files),
        "text_fields": stats.text_fields,
        "changed_fields": stats.changed_fields,
        "change_counts": dict(sorted(stats.changes.items())),
        "quality": quality,
    }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "cleaned_file_count": report["cleaned_file_count"],
        "text_fields": report["text_fields"],
        "changed_fields": report["changed_fields"],
        "quality_pass": quality["pass"],
        "quality_totals": quality["totals"],
    }, ensure_ascii=False, indent=2))
    return 0 if quality["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
