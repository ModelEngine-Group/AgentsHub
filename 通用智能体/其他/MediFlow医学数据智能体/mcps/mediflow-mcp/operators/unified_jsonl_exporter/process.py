# -*- coding: utf-8 -*-
"""
JSONL 统一输出算子。

放在文本链/表格链/JSON链末尾，将不同输入统一为下游任务二易消费的 JSONL：
  {source_file, record_id, cleaned_text, noise_score, data_type}
"""
import csv
import io
import json
import re
import time
from typing import Dict, Any, Iterable, List

from loguru import logger
from datamate.core.base_op import Mapper


_TEXT_KEYS = ("cleaned_text", "text", "content", "note", "description", "主诉",
              "现病史", "诊断", "症状", "检查", "结论", "answer", "question")
_NOISE_HINT_RE = re.compile(r'https?://|<[^>]+>|@[^\s@]+|微信|自动导出|HIS系统|###|---')


class UnifiedJsonlExporter(Mapper):

    def _noise_score(self, text: str, sample: Dict[str, Any]) -> float:
        meta = sample.get("_noise") or {}
        if isinstance(meta, dict) and meta.get("noise_score") is not None:
            try:
                return float(meta["noise_score"])
            except Exception:
                pass
        return round(min(1.0, len(_NOISE_HINT_RE.findall(text or "")) / 6.0), 3)

    def _row_text(self, row: Dict[str, Any]) -> str:
        parts = []
        lowered = {str(k).lower(): v for k, v in row.items()}
        for key in _TEXT_KEYS:
            if key in row and str(row[key]).strip():
                parts.append(str(row[key]).strip())
            lk = key.lower()
            if lk in lowered and str(lowered[lk]).strip():
                parts.append(str(lowered[lk]).strip())
        if parts:
            return "\n".join(dict.fromkeys(parts))
        return "；".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())

    def _load_json_records(self, text: str) -> List[Any]:
        stripped = text.strip()
        if not stripped:
            return []
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, list) else [obj]
        except Exception:
            records = []
            for line in stripped.splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records

    def _from_jsonish(self, text: str, source_file: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for idx, obj in enumerate(self._load_json_records(text), 1):
            if isinstance(obj, dict):
                cleaned_text = str(obj.get("cleaned_text") or self._row_text(obj))
                data_type = str(obj.get("data_type") or "json_record")
                record = {
                    "source_file": obj.get("source_file", source_file),
                    "record_id": obj.get("record_id", f"{source_file}_{idx:03d}"),
                    "cleaned_text": cleaned_text,
                    "noise_score": obj.get("noise_score", self._noise_score(cleaned_text, sample)),
                    "data_type": data_type,
                }
                extra = {k: v for k, v in obj.items() if k not in record}
                if extra:
                    record["structured_record"] = extra
                out.append(record)
            else:
                cleaned_text = json.dumps(obj, ensure_ascii=False)
                out.append({
                    "source_file": source_file,
                    "record_id": f"{source_file}_{idx:03d}",
                    "cleaned_text": cleaned_text,
                    "noise_score": self._noise_score(cleaned_text, sample),
                    "data_type": "json_record",
                })
        return out

    def _from_csv(self, text: str, source_file: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        reader = csv.DictReader(io.StringIO(text))
        out = []
        for idx, row in enumerate(reader, 1):
            cleaned_text = self._row_text(row)
            out.append({
                "source_file": source_file,
                "record_id": f"{source_file}_row_{idx:05d}",
                "cleaned_text": cleaned_text,
                "noise_score": self._noise_score(cleaned_text, sample),
                "data_type": "table_row",
                "structured_record": row,
            })
        return out

    def _from_text(self, text: str, source_file: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = re.sub(r'\.[^.]+$', '', source_file) or 'record'
        return [{
            "source_file": source_file,
            "record_id": f"{base}_001",
            "cleaned_text": text.strip(),
            "noise_score": self._noise_score(text, sample),
            "data_type": "medical_text",
        }]

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        source_file = sample.get(self.filename_key, "")
        filetype = str(sample.get(self.filetype_key, "")).lower()

        if not text.strip():
            return sample

        records = []
        try:
            if filetype in ("json", "jsonl") or text.lstrip().startswith(("{", "[")):
                records = self._from_jsonish(text, source_file, sample)
            elif filetype in ("csv", "xlsx", "xls") or ("," in text.splitlines()[0] and "\n" in text):
                records = self._from_csv(text, source_file, sample)
            else:
                records = self._from_text(text, source_file, sample)
        except Exception as exc:
            logger.warning(f"UnifiedJsonlExporter parse fallback: {exc}")
            records = self._from_text(text, source_file, sample)

        sample[self.text_key] = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        sample[self.data_key] = b""
        sample[self.filetype_key] = "txt"
        sample["target_type"] = "jsonl"

        logger.info(
            f"fileName: {source_file}, UnifiedJsonlExporter costs "
            f"{time.time() - start:.3f}s, records={len(records)}"
        )
        return sample
