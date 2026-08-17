"""DataMate 数据血缘持久化适配器。

本模块只负责把任务一的源数据集、最终数据集及清洗关系写入 DataMate
血缘表。业务编排层无需了解表结构，重复登记同一条血缘也不会产生重复记录。
"""

from __future__ import annotations

import json
import subprocess
import uuid
from typing import Any


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _run_psql(sql: str, *, capture: bool = False) -> str:
    command = [
        "docker",
        "exec",
        "-i",
        "datamate-database",
        "psql",
        "-U",
        "postgres",
        "-d",
        "datamate",
        "-v",
        "ON_ERROR_STOP=1",
    ]
    if capture:
        command.extend(["-At", "-F", "\t"])
    completed = subprocess.run(
        command,
        input=sql,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def _dataset_info(dataset_id: str) -> dict[str, str]:
    output = _run_psql(
        "select id, name, coalesce(description, '') "
        "from t_dm_datasets "
        f"where id = {_sql_literal(dataset_id)};",
        capture=True,
    )
    if not output:
        raise RuntimeError(f"DataMate 数据集不存在: {dataset_id}")
    row = output.splitlines()[0].split("\t")
    return {"id": row[0], "name": row[1], "description": row[2] if len(row) > 2 else ""}


def _deterministic_edge_id(source_id: str, delivery_id: str) -> str:
    key = f"mediflow:task1-lineage:{source_id}:{delivery_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def persist_task1_lineage(
    source_dataset_id: str,
    delivery_dataset_id: str,
    report: dict[str, Any],
) -> dict[str, str]:
    """幂等登记任务一源数据集到最终数据集的清洗血缘。"""

    source = _dataset_info(source_dataset_id)
    delivery = _dataset_info(delivery_dataset_id)
    graph_id = source["id"]
    edge_id = _deterministic_edge_id(source["id"], delivery["id"])

    common_metadata = {
        "task": "task1",
        "quality_pass": bool(report.get("pass")),
        "source_files": len(report.get("source_mixed_dataset", {}).get("files", [])),
        "delivery_files": int(report.get("delivery_report", {}).get("files", 0) or 0),
        "delivery_mode": "format_preserved",
    }
    source_metadata = json.dumps(
        {"role": "mixed_source", **common_metadata},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    delivery_metadata = json.dumps(
        {"role": "final_delivery", **common_metadata},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    edge_metadata = json.dumps(common_metadata, ensure_ascii=False, separators=(",", ":"))

    sql = f"""
    insert into t_lineage_node
      (id, graph_id, node_type, name, description, node_metadata)
    values
      ({_sql_literal(source['id'])}, {_sql_literal(graph_id)}, 'DATASET',
       {_sql_literal(source['name'])}, {_sql_literal(source['description'])},
       {_sql_literal(source_metadata)})
    on conflict (id) do update set
      graph_id = excluded.graph_id,
      node_type = excluded.node_type,
      name = excluded.name,
      description = excluded.description,
      node_metadata = excluded.node_metadata;

    insert into t_lineage_node
      (id, graph_id, node_type, name, description, node_metadata)
    values
      ({_sql_literal(delivery['id'])}, {_sql_literal(graph_id)}, 'DATASET',
       {_sql_literal(delivery['name'])}, {_sql_literal(delivery['description'])},
       {_sql_literal(delivery_metadata)})
    on conflict (id) do update set
      graph_id = excluded.graph_id,
      node_type = excluded.node_type,
      name = excluded.name,
      description = excluded.description,
      node_metadata = excluded.node_metadata;

    insert into t_lineage_edge
      (id, process_id, graph_id, edge_type, name, description, edge_metadata,
       from_node_id, to_node_id)
    values
      ({_sql_literal(edge_id)}, 'task1-mixed-cleaning', {_sql_literal(graph_id)},
       'DATA_CLEANING', '任务一混合格式清洗',
       '按文件类型编排清洗算子并归集为保留源格式的最终数据集',
       {_sql_literal(edge_metadata)}, {_sql_literal(source['id'])},
       {_sql_literal(delivery['id'])})
    on conflict (id) do update set
      process_id = excluded.process_id,
      graph_id = excluded.graph_id,
      edge_type = excluded.edge_type,
      name = excluded.name,
      description = excluded.description,
      edge_metadata = excluded.edge_metadata,
      from_node_id = excluded.from_node_id,
      to_node_id = excluded.to_node_id;
    """
    _run_psql(sql)
    return {
        "status": "persisted",
        "graph_id": graph_id,
        "source_node_id": source["id"],
        "delivery_node_id": delivery["id"],
        "edge_id": edge_id,
    }
