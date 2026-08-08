"""pgvector-backed ANN store: the indexed upgrade of `LocalVectorStore`.

`LocalVectorStore` is a numpy matrix and a dot product -- exact, zero-dependency,
and O(N) per query with no index. That is the right default for a corpus of a few
thousand chunks and for offline runs. This store is the production-shaped path:
vectors live in PostgreSQL under an HNSW index, so search cost stops scaling
linearly with corpus size, at the price of approximate results.

The interface mirrors `LocalVectorStore` (`add` / `search` / `size`) so the QA
agent can take either. Two extra knobs matter and are exposed rather than hidden:

* build-time: ``m`` (graph degree) and ``ef_construction`` (build beam width) --
  larger values improve the graph's recall ceiling at the cost of build time;
* query-time: ``ef_search`` (search beam width) -- the recall/latency dial. The
  benchmark (`scripts/bench_vector_search.py`) publishes the measured curve
  instead of a single cherry-picked point.

Cosine everywhere: embeddings are L2-normalised on insert (matching
`LocalVectorStore`), and the index uses ``vector_cosine_ops`` with the ``<=>``
operator, so scores are directly comparable between the two stores.
"""
from __future__ import annotations

import threading

import numpy as np

from config.settings import AnalyticsConfig

TABLE = "vector_chunks"


class PgVectorStore:
    def __init__(
        self,
        dim: int,
        config: AnalyticsConfig | None = None,
        m: int = 16,
        ef_construction: int = 64,
        table: str = TABLE,
    ):
        from config.settings import get_analytics_config
        from medigraph.analysis.pg_relational import get_pool

        self.dim = int(dim)
        self.m = m
        self.ef_construction = ef_construction
        self.table = table
        self._config = config or get_analytics_config()
        self._pool = get_pool(self._config)
        self._lock = threading.Lock()
        self._size: int | None = None  # lazy COUNT cache, invalidated on add
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    def _ensure_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "  id BIGSERIAL PRIMARY KEY,"
                "  text TEXT NOT NULL,"
                "  source TEXT NOT NULL DEFAULT '',"
                f"  embedding vector({self.dim}) NOT NULL"
                ")"
            )
            conn.commit()

    def create_index(self, maintenance_work_mem: str = "1GB", parallel_workers: int = 4) -> None:
        """Build (or rebuild) the HNSW index.

        Called once after bulk load rather than before: inserting through an
        existing HNSW index is much slower than indexing a loaded table.

        The two session settings matter enormously at scale: with the default
        64 MB ``maintenance_work_mem`` a 100k x 1024-dim build does not fit the
        graph in memory and pgvector falls back to a much slower on-disk path
        (measured: >8 min vs ~1 min). Session-scoped, so nothing leaks to other
        pool users.
        """
        with self._pool.connection() as conn:
            conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
            conn.execute(f"SET max_parallel_maintenance_workers = {parallel_workers}")
            conn.execute(f"DROP INDEX IF EXISTS {self.table}_hnsw")
            conn.execute(
                f"CREATE INDEX {self.table}_hnsw ON {self.table} "
                f"USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {self.m}, ef_construction = {self.ef_construction})"
            )
            conn.commit()

    def clear(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(f"TRUNCATE {self.table} RESTART IDENTITY")
            conn.commit()
        self._size = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)

    @staticmethod
    def _vec_literal(vector: np.ndarray) -> str:
        # pgvector's text input format; float32 precision is plenty for cosine.
        return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"

    def add(self, texts: list[str], sources: list[str], vectors: list[list[float]]) -> int:
        """Bulk-insert via COPY. Returns the number of rows written."""
        rows = [
            (text, source, vector)
            for text, source, vector in zip(texts, sources, vectors)
            if vector
        ]
        if not rows:
            return 0
        matrix = self._normalize(np.asarray([vector for _, _, vector in rows], dtype=np.float32))
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                with cur.copy(f"COPY {self.table} (text, source, embedding) FROM STDIN") as copy:
                    for (text, source, _), embedding in zip(rows, matrix):
                        copy.write_row((text, source, self._vec_literal(embedding)))
            conn.commit()
        self._size = None
        return len(rows)

    def search(self, query_vec: list[float], k: int = 3, ef_search: int | None = None) -> list[dict]:
        """Approximate top-k by cosine similarity.

        `ef_search` must be >= k (pgvector requirement); when unset, the server
        default (40) applies. Returns the same shape as `LocalVectorStore.search`.
        """
        if not query_vec:
            return []
        query = np.asarray(query_vec, dtype=np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)
        literal = self._vec_literal(query)
        with self._pool.connection() as conn:
            if ef_search is not None:
                conn.execute(f"SET hnsw.ef_search = {max(int(ef_search), k)}")
            cur = conn.execute(
                f"SELECT text, source, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (literal, literal, k),
            )
            hits = cur.fetchall()
            conn.rollback()
        return [
            {"text": text, "source": source, "score": round(float(score), 3)}
            for text, source, score in hits
        ]

    @property
    def size(self) -> int:
        if self._size is None:
            with self._lock:
                if self._size is None:
                    with self._pool.connection() as conn:
                        self._size = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
        return self._size
