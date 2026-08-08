"""Lightweight local vector store for GraphRAG hybrid retrieval.

Stores text chunks with their embeddings and does cosine top-k search in numpy.
No external service required (Milvus is the documented production upgrade). Used
by the QA agent to fuse vector-retrieved passages with graph subgraph evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LocalVectorStore:
    def __init__(self):
        self.chunks: list[dict] = []          # [{text, source}]
        self._matrix: np.ndarray | None = None  # (N, dim), L2-normalized

    def add(self, texts: list[str], sources: list[str], vectors: list[list[float]]) -> None:
        rows = []
        for text, source, vec in zip(texts, sources, vectors):
            if not vec:  # embedding failed for this item
                continue
            self.chunks.append({"text": text, "source": source})
            rows.append(vec)
        if rows:
            arr = np.array(rows, dtype=np.float32)
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            self._matrix = arr if self._matrix is None else np.vstack([self._matrix, arr])

    def search(self, query_vec: list[float], k: int = 3) -> list[dict]:
        if not query_vec or self._matrix is None or len(self.chunks) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = self._matrix @ q
        idx = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in idx]

    @property
    def size(self) -> int:
        return len(self.chunks)

    # -- persistence --------------------------------------------------- #
    def save_json(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "matrix": self._matrix.tolist() if self._matrix is not None else [],
        }
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(p)

    @classmethod
    def load_json(cls, path: str | Path) -> "LocalVectorStore":
        store = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store.chunks = data.get("chunks", [])
        matrix = data.get("matrix", [])
        store._matrix = np.array(matrix, dtype=np.float32) if matrix else None
        return store
