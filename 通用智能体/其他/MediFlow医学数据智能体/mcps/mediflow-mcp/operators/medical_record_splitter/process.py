# -*- coding: utf-8 -*-
"""
多段病历分段算子。

DataMate 默认以文件为处理粒度；该算子支持处理一个 txt 内包含多份
病历/多位患者的情况。本算子通常放在标准清洗链末尾，把已清洗文本切成 JSONL：
  {source_file, record_id, cleaned_text, noise_score, data_type}
"""
import json
import re
import time
from typing import Dict, Any, List

from loguru import logger
from datamate.core.base_op import Mapper


_SEP_RE = re.compile(r'^\s*(?:={3,}|-{3,}|#{3,}|\*{3,})\s*$', re.MULTILINE)
_CASE_MARKER_RE = re.compile(r'^\s*病例\s*\d+', re.MULTILINE)
_PATIENT_MARKER_RE = re.compile(
    r'^\s*(?:患者\s*[:：]|姓名\s*[:：]|病历号\s*[:：]?|住院号\s*[:：]?)',
    re.MULTILINE
)
_BLANK_BLOCK_RE = re.compile(r'\n\s*\n\s*\n+')
_NOISE_HINT_RE = re.compile(r'https?://|<[^>]+>|@[^\s@]+|微信|自动导出|HIS系统|###|---')


class MedicalRecordSplitter(Mapper):

    def _split_by_separator(self, text: str) -> List[str]:
        text = _SEP_RE.sub("\n<<<DM_RECORD_SEP>>>\n", text)
        text = _BLANK_BLOCK_RE.sub("\n<<<DM_RECORD_SEP>>>\n", text)
        parts = [p.strip() for p in text.split("<<<DM_RECORD_SEP>>>") if p.strip()]
        return parts

    def _split_by_matches(self, text: str, matches) -> List[str]:
        if len(matches) < 2:
            return [text.strip()] if text.strip() else []

        spans = [m.start() for m in matches] + [len(text)]
        parts = []
        prefix = text[:spans[0]].strip()
        for i in range(len(spans) - 1):
            chunk = text[spans[i]:spans[i + 1]].strip()
            if i == 0 and prefix:
                chunk = prefix + "\n" + chunk
            if chunk:
                parts.append(chunk)
        return parts

    def _split_by_markers(self, text: str) -> List[str]:
        case_matches = list(_CASE_MARKER_RE.finditer(text))
        if len(case_matches) >= 2:
            return self._split_by_matches(text, case_matches)
        patient_matches = list(_PATIENT_MARKER_RE.finditer(text))
        return self._split_by_matches(text, patient_matches)

    def _split_records(self, text: str) -> List[str]:
        parts = self._split_by_separator(text)
        if len(parts) > 1:
            return parts
        return self._split_by_markers(text)

    def _noise_score(self, text: str, sample: Dict[str, Any]) -> float:
        meta = sample.get("_noise") or {}
        if isinstance(meta, dict) and meta.get("noise_score") is not None:
            try:
                return float(meta["noise_score"])
            except Exception:
                pass
        hints = len(_NOISE_HINT_RE.findall(text))
        return round(min(1.0, hints / 6.0), 3)

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        if not text.strip():
            return sample

        records = self._split_records(text)
        source_file = sample.get(self.filename_key, "")
        base = re.sub(r'\.[^.]+$', '', source_file) or "record"
        noise_score = self._noise_score(text, sample)

        rows = []
        for idx, record_text in enumerate(records, 1):
            rows.append({
                "source_file": source_file,
                "record_id": f"{base}_{idx:03d}",
                "cleaned_text": record_text,
                "noise_score": noise_score,
                "data_type": "medical_record_text",
            })

        sample[self.text_key] = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        sample[self.data_key] = b""
        # 保持 fileType 为 txt，让 FileExporter 进入文本保存分支；target_type 控制后缀。
        sample[self.filetype_key] = "txt"
        sample["target_type"] = "jsonl"

        logger.info(
            f"fileName: {source_file}, MedicalRecordSplitter costs "
            f"{time.time() - start:.3f}s, records={len(rows)}"
        )
        return sample
