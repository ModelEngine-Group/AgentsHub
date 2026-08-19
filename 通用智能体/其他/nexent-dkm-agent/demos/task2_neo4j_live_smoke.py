"""Neo4j live integration smoke test and evidence collector.

Runs connection check, optional pipeline persist, Cypher spot-checks, and
round-trip read/query against a running Neo4j instance (see
``docker-compose.neo4j.yml``). Writes a JSON report for答辩 handoff.

Usage:
    # Configure NEO4J_AUTH in an ignored .env before starting the container.
    docker compose -f docker-compose.neo4j.yml up -d
    python demos/task2_neo4j_live_smoke.py \
        --password-file .local/neo4j.password
    python demos/task2_neo4j_live_smoke.py \
        --password-file .local/neo4j.password \
        --report benchmarks/reports/task2_neo4j_live_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.kg_ops.neo4j_query import neo4j_answer_question, neo4j_find_entities
from src.operators.kg_ops.neo4j_store import (
    check_neo4j_connection,
    neo4j_to_graph,
)
from src.pipelines.task2_kg_pipeline import run_task2_pipeline

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_INPUT = ROOT / "data" / "samples" / "task2_medical_notes.txt"
DEFAULT_REPORT = ROOT / "benchmarks" / "reports" / "task2_neo4j_live_smoke.json"


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neo4j live smoke test for task 2.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--user", default=DEFAULT_USER)
    password_group = parser.add_mutually_exclusive_group()
    password_group.add_argument(
        "--password",
        default=None,
        help="Neo4j password. Prefer --password-stdin in automated runs.",
    )
    password_group.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the Neo4j password from standard input.",
    )
    password_group.add_argument(
        "--password-file",
        default=None,
        help="Read the Neo4j password from an ignored local file.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--question", default="高血压有哪些症状和用药？")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Only probe connection and query existing graph (skip rebuild).",
    )
    return parser.parse_args()


def _run_cypher(
    uri: str,
    user: str,
    password: str,
    cypher: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from src.operators.kg_ops.neo4j_store import _get_driver

    driver = _get_driver(uri, user, password)
    if driver is None:
        raise RuntimeError("neo4j driver unavailable")
    try:
        with driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [dict(record) for record in result]
    finally:
        driver.close()


def main() -> int:
    args = parse_args()
    password = _resolve_password(args)
    report: dict[str, Any] = {
        "uri": args.uri,
        "user": args.user,
        "input": _portable_path(args.input),
        "question": args.question,
    }

    connection = check_neo4j_connection(args.uri, args.user, password)
    report["connection"] = connection
    if connection.get("status") != "connected":
        _write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    pipeline_artifact: dict[str, Any] | None = None
    if not args.skip_pipeline:
        result = run_task2_pipeline(
            input_path=args.input,
            question=args.question,
            neo4j_config={
                "uri": args.uri,
                "user": args.user,
                "password": password,
            },
        )
        pipeline_artifact = result.artifacts.get("neo4j", {})
        report["pipeline_status"] = result.status
        report["neo4j_persist"] = pipeline_artifact

    graph = neo4j_to_graph(args.uri, args.user, password)
    report["readback"] = {
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
    }

    report["cypher_spot_checks"] = {
        "node_labels": _run_cypher(
            args.uri,
            args.user,
            password,
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt "
            "ORDER BY cnt DESC",
        ),
        "hypertension_edges": _run_cypher(
            args.uri,
            args.user,
            password,
            "MATCH (d:Disease {name: $name})-[r]->(t) "
            "RETURN type(r) AS rel, t.name AS target ORDER BY rel, target LIMIT 20",
            {"name": "高血压"},
        ),
    }

    find_result = neo4j_find_entities("高血压", args.uri, args.user, password)
    qa_result = neo4j_answer_question(args.question, args.uri, args.user, password)
    report["find_entities"] = {
        "status": find_result.get("status"),
        "matches": [m.get("name") for m in find_result.get("matches", [])[:5]],
    }
    report["neo4j_qa"] = {
        "status": qa_result.get("status"),
        "answer": qa_result.get("answer"),
    }

    report["passed"] = (
        connection.get("status") == "connected"
        and report["readback"]["node_count"] > 0
        and report["readback"]["edge_count"] > 0
        and find_result.get("status") == "matched"
        and qa_result.get("status") == "answered"
        and (args.skip_pipeline or pipeline_artifact.get("status") == "completed")
    )
    report["browser_url"] = "http://localhost:7474"
    report["screenshot_queries"] = [
        "MATCH (n) RETURN labels(n), n.name, n.type LIMIT 25",
        "MATCH (d:Disease {name:'高血压'})-[r]->(t) RETURN d.name, type(r), t.name LIMIT 20",
    ]

    _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_password(args: argparse.Namespace) -> str:
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
        if not password:
            raise ValueError("Neo4j password from stdin must not be empty.")
        return password
    if args.password_file:
        password = Path(args.password_file).read_text(encoding="utf-8").strip()
        if not password:
            raise ValueError("Neo4j password file must not be empty.")
        return password
    if args.password:
        return args.password
    raise ValueError(
        "Neo4j password is required; use --password-file, --password-stdin, "
        "or --password."
    )


if __name__ == "__main__":
    raise SystemExit(main())
