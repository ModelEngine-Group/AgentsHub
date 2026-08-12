"""
混合医学数据解析工具。

该模块把 TXT、CSV、JSON、JSONL 文件转换为统一记录流。
"""

import re, json, csv, io


def _record_lineage(source_file: str, index: int, row: object = None) -> dict:
    source_record_id = ""
    if isinstance(row, dict):
        source_record_id = str(row.get("record_id") or row.get("id") or "").strip()
    stable_part = source_record_id or str(index)
    return {
        "record_id": f"{source_file}:{stable_part}",
        "source_record_id": source_record_id,
    }

def infer_format(file_name: str, file_type: str = '') -> str:
    name = file_name.lower()
    if name.endswith('.csv'): return 'csv'
    if name.endswith('.jsonl'): return 'jsonl'
    if name.endswith('.json'): return 'json'
    return 'txt'

def split_text(text: str, source_file: str, source_format: str, max_chars: int = 1200) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    records = []
    for i, para in enumerate(paragraphs):
        if len(para) <= max_chars:
            records.append({'text': para, 'source_file': source_file,
                          'record_id': f'{source_file}:{i}', 'source_format': source_format})
        else:
            sentences = re.split(r'(?<=[。！？])\s*', para)
            buf, sid = '', 0
            for s in sentences:
                if len(buf) + len(s) > max_chars and buf:
                    records.append({'text': buf.strip(), 'source_file': source_file,
                                  'record_id': f'{source_file}:{i}.{sid}', 'source_format': source_format})
                    buf, sid = s, sid + 1
                else:
                    buf += s
            if buf.strip():
                records.append({'text': buf.strip(), 'source_file': source_file,
                              'record_id': f'{source_file}:{i}.{sid}', 'source_format': source_format})
    return records

def parse_csv(text: str, source_file: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    return [{'text': json.dumps(r, ensure_ascii=False), 'source_file': source_file,
             **_record_lineage(source_file, i, r), 'source_format': 'csv', 'row_data': r}
            for i, r in enumerate(rows) if any(str(v).strip() for v in r.values())]

def parse_json(text: str, source_file: str, source_format: str) -> list[dict]:
    data = json.loads(text)
    if isinstance(data, list):
        return [{'text': json.dumps(item, ensure_ascii=False), 'source_file': source_file,
                 **_record_lineage(source_file, i, item), 'source_format': source_format,
                 'row_data': item if isinstance(item, dict) else {}}
                for i, item in enumerate(data)]
    return [{'text': text, 'source_file': source_file,
             **_record_lineage(source_file, 0, data), 'source_format': source_format,
             'row_data': data if isinstance(data, dict) else {}}]

def parse_jsonl(text: str, source_file: str) -> list[dict]:
    records = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        data = json.loads(line)
        records.append({'text': json.dumps(data, ensure_ascii=False), 'source_file': source_file,
                        **_record_lineage(source_file, i, data), 'source_format': 'jsonl',
                        'row_data': data if isinstance(data, dict) else {}})
    return records

def parse_files(files: list[dict]) -> tuple[list[dict], dict]:
    records = []
    stats = {}
    for f in files:
        fmt = infer_format(f.get('file_name', ''), f.get('file_type', ''))
        stats[fmt] = stats.get(fmt, 0) + 1
        text = f.get('content') or f.get('text') or ''
        if not text: continue
        if fmt == 'csv': records.extend(parse_csv(text, f.get('file_name', '')))
        elif fmt == 'jsonl': records.extend(parse_jsonl(text, f.get('file_name', '')))
        elif fmt == 'json': records.extend(parse_json(text, f.get('file_name', ''), fmt))
        else: records.extend(split_text(text, f.get('file_name', ''), 'txt'))
    return records, stats
