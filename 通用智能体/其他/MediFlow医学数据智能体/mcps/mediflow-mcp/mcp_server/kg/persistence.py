"""
任务二知识图谱持久化模块。

该模块负责三元组入库、来源登记、重复过滤和写入统计。
"""

import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any
from mcp_server.shared.sqlite_utils import connect_kg
from mcp_server.kg.schema import _task2_ensure_kg_schema
from mcp_server.kg.normalization import (
    normalize_kg_entity_type,
    normalize_kg_relation_code,
    relation_display_name,
)

def ensure_kg_schema(conn=None):
    c = conn or connect_kg()
    _task2_ensure_kg_schema(c)
    if conn is None:
        c.close()

def ensure_source(conn, dataset, record_count):
    c = conn.cursor()
    source_name = dataset.get('name') or dataset.get('id') or 'unknown'
    now = time.strftime('%Y-%m-%dT%H:%M:%S')
    c.execute(
        '''INSERT OR IGNORE INTO kg_sources(source_name, source_path, source_type, record_count, created_at)
           VALUES(?,?,?,?,?)''',
        (source_name, dataset.get('id', ''), 'datamate', record_count, now),
    )
    c.execute(
        '''UPDATE kg_sources SET source_path=?, source_type=?, record_count=?
           WHERE source_name=?''',
        (dataset.get('id', ''), 'datamate', record_count, source_name),
    )
    c.execute('SELECT source_id FROM kg_sources WHERE source_name=?', (source_name,))
    row = c.fetchone()
    return row[0] if row else None

def ensure_entity(conn, name, etype=''):
    if not (name or '').strip():
        return None
    c = conn.cursor()
    now = time.strftime('%Y-%m-%dT%H:%M:%S')
    normalized_type = normalize_kg_entity_type(etype)
    c.execute(
        '''INSERT OR IGNORE INTO kg_entities(canonical_name, entity_type, confidence, created_at)
           VALUES(?,?,?,?)''',
        (name, normalized_type, 1.0, now),
    )
    c.execute('SELECT entity_id FROM kg_entities WHERE canonical_name=? AND entity_type=?', (name, normalized_type))
    row = c.fetchone()
    return row[0] if row else None

def ensure_relation(conn, code, display=''):
    if not (code or '').strip():
        return None
    c = conn.cursor()
    normalized_code = normalize_kg_relation_code(code)
    c.execute(
        'INSERT OR IGNORE INTO kg_relations(relation_code, display_name) VALUES(?,?)',
        (normalized_code, display or relation_display_name(normalized_code)),
    )
    return normalized_code

def _triple_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"unsupported triple payload type: {type(value).__name__}")


def persist_triples(
    conn,
    record_triples,
    source_file='',
    source_id=None,
    source_record_id='',
    return_details=False,
    include_quality_metrics=False,
):
    count = 0
    candidate_count = 0
    rejected_count = 0
    accepted_high_count = 0
    deduplicated_count = 0
    invalid_count = 0
    c = conn.cursor()
    for raw_triple in record_triples:
        t = _triple_to_dict(raw_triple)
        reliability = str(t.get('reliability_level') or '').strip().lower()
        evidence = t.get('evidence') or source_file
        method = str(t.get('extraction_method') or t.get('method') or 'unknown').strip().lower()
        if reliability not in {'high', 'medium', 'low'}:
            reliability = 'medium' if method == 'llm' else 'low'
        if reliability in {'medium', 'low'}:
            c.execute(
                '''INSERT INTO kg_quality_issues(source_id, field_name, value, issue_type,
                   evidence, created_at) VALUES(?,?,?,?,?,?)''',
                (
                    source_id,
                    'triple',
                    json.dumps(
                        {
                            'subject': t.get('subject', ''),
                            'predicate': t.get('predicate', ''),
                            'object': t.get('object', ''),
                            'extraction_method': method,
                            'reliability_level': reliability,
                        },
                        ensure_ascii=False,
                    ),
                    'candidate_fact' if reliability == 'medium' else 'rejected_fact',
                    evidence,
                    time.strftime('%Y-%m-%dT%H:%M:%S'),
                ),
            )
            if reliability == 'medium':
                candidate_count += 1
            else:
                rejected_count += 1
            continue
        accepted_high_count += 1
        sid = ensure_entity(conn, t.get('subject',''), t.get('subject_type', ''))
        oid = ensure_entity(conn, t.get('object',''), t.get('object_type', ''))
        rel = ensure_relation(conn, t.get('predicate',''), '')
        if not sid or not oid or not rel:
            invalid_count += 1
            continue
        c.execute(
            '''INSERT OR IGNORE INTO kg_triples(subject_id, relation_code, object_id, source_id,
               source_file, source_record_id, evidence, confidence, extractor,
               reliability_level, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
            (
                sid,
                rel,
                oid,
                source_id,
                source_file,
                source_record_id,
                evidence,
                t.get('confidence', 0.7),
                method,
                reliability or None,
                time.strftime('%Y-%m-%dT%H:%M:%S'),
            ),
        )
        inserted = max(c.rowcount, 0)
        count += inserted
        if inserted == 0:
            deduplicated_count += 1
    conn.commit()
    if return_details:
        details = {
            'inserted': count,
            'candidate': candidate_count,
            'rejected': rejected_count,
        }
        if include_quality_metrics:
            details.update(
                {
                    'accepted_high': accepted_high_count,
                    'deduplicated': deduplicated_count,
                    'invalid': invalid_count,
                }
            )
        return details
    return count
