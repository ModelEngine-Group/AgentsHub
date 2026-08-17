# -*- coding: utf-8 -*-
"""
医学抽取示例检索模块。

该模块延迟读取可信训练样例，为需要提示词增强的抽取流程提供少量示例。
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any, Callable


def _char_bigrams(text: str) -> set[str]:
    compact = "".join(text.split())
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


class _Retriever:
    def __init__(
        self,
        records: list[dict[str, Any]],
        formatter: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.records = records
        self.formatter = formatter
        self.features = [_char_bigrams(str(record.get("text") or "")) for record in records]
        self.index: dict[str, list[int]] = collections.defaultdict(list)
        for record_index, features in enumerate(self.features):
            for feature in features:
                self.index[feature].append(record_index)

    def retrieve(self, text: str, limit: int = 2) -> list[dict[str, Any]]:
        query = _char_bigrams(text)
        candidate_counts: collections.Counter[int] = collections.Counter()
        for feature in query:
            for record_index in self.index.get(feature, ()):
                candidate_counts[record_index] += 1
        scored = []
        for record_index, overlap in candidate_counts.most_common(200):
            denominator = len(query) + len(self.features[record_index])
            score = (2 * overlap / denominator) if denominator else 0.0
            scored.append((score, record_index))
        scored.sort(reverse=True)
        return [
            self.formatter(self.records[record_index])
            for _, record_index in scored[:limit]
        ]


def _object_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("@value", value.get("value", "")) or "").strip()
    return str(value or "").strip()


def _format_cmeee(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(record.get("text") or ""),
        "entities": [
            {
                "text": str(entity.get("entity") or ""),
                "type": str(entity.get("type") or ""),
                "start_idx": entity.get("start_idx"),
                "end_idx": entity.get("end_idx"),
                "confidence": 1.0,
            }
            for entity in record.get("entities") or []
        ],
    }


def _format_cmeie(record: dict[str, Any]) -> dict[str, Any]:
    relations = []
    for spo in record.get("spo_list") or []:
        subject = str(spo.get("subject") or "").strip()
        predicate = str(spo.get("predicate") or "").strip()
        obj = _object_value(spo.get("object"))
        if subject and predicate and obj:
            relations.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "subject_type": spo.get("subject_type", ""),
                    "object_type": _object_value(spo.get("object_type")),
                    "confidence": 1.0,
                }
            )
    return {"text": str(record.get("text") or ""), "relations": relations}


_CACHE: dict[tuple[str, str], _Retriever] = {}


def _candidate_paths(task: str) -> list[Path]:
    env_name = "CCF_CMEEE_TRAIN" if task == "cmeee" else "CCF_CMEIE_TRAIN"
    file_name = "CMeEE-V2_train.json" if task == "cmeee" else "CMeIE_train.jsonl"
    dataset_dir = "CMeEE-V2" if task == "cmeee" else "CMeIE"
    candidates = []
    if os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]))
    candidates.extend(
        [
            Path("data/eval_sources/CBLUE_data") / dataset_dir / file_name,
            Path("/opt/runtime/datamate/task2_fewshot") / dataset_dir / file_name,
        ]
    )
    return candidates


def _load(task: str) -> _Retriever | None:
    formatter = _format_cmeee if task == "cmeee" else _format_cmeie
    for path in _candidate_paths(task):
        if not path.exists():
            continue
        key = (task, str(path.resolve()))
        if key in _CACHE:
            return _CACHE[key]
        if task == "cmeee":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        else:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                data = [json.loads(line) for line in handle if line.strip()]
        retriever = _Retriever(data, formatter)
        _CACHE[key] = retriever
        return retriever
    return None


def retrieve_cmeee_examples(text: str, limit: int = 2) -> list[dict[str, Any]]:
    retriever = _load("cmeee")
    return retriever.retrieve(text, limit) if retriever else []


def retrieve_cmeie_examples(text: str, limit: int = 2) -> list[dict[str, Any]]:
    retriever = _load("cmeie")
    return retriever.retrieve(text, limit) if retriever else []
