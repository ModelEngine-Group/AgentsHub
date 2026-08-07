from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from medgraph_agent.core.models import Entity, GraphSnapshot, PipelineRun, Relation, to_dict


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_graph_json(path: str | Path) -> GraphSnapshot:
    payload = read_json(path)
    return GraphSnapshot(
        entities=[Entity(**item) for item in payload.get("entities", [])],
        relations=[Relation(**item) for item in payload.get("relations", [])],
        generated_at=payload["generated_at"],
        source_record_count=payload["source_record_count"],
    )


class GraphStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_record_ids TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    subject_name TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    predicate_label TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_name TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    confidence REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    payload TEXT NOT NULL
                );
                """
            )

    def save_graph(self, graph: GraphSnapshot) -> None:
        self.initialize()
        with self.connection() as conn:
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM entities")
            conn.executemany(
                """
                INSERT INTO entities(id, name, type, label, confidence, source_record_ids)
                VALUES(:id, :name, :type, :label, :confidence, :source_record_ids)
                """,
                [
                    {
                        **entity.__dict__,
                        "source_record_ids": json.dumps(entity.source_record_ids, ensure_ascii=False),
                    }
                    for entity in graph.entities
                ],
            )
            conn.executemany(
                """
                INSERT INTO relations(
                    id, subject_id, subject_name, predicate, predicate_label, object_id,
                    object_name, evidence, source_record_id, confidence
                )
                VALUES(
                    :id, :subject_id, :subject_name, :predicate, :predicate_label, :object_id,
                    :object_name, :evidence, :source_record_id, :confidence
                )
                """,
                [relation.__dict__ for relation in graph.relations],
            )

    def save_run(self, run: PipelineRun) -> None:
        self.initialize()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(id, task, source, status, started_at, finished_at, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.task,
                    run.source,
                    run.status,
                    run.started_at,
                    run.finished_at,
                    json.dumps(to_dict(run), ensure_ascii=False),
                ),
            )

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
