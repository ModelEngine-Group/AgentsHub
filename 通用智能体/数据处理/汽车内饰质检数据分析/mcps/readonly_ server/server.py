# -*- coding: utf-8 -*-
"""Self-contained read-only MCP server for the quality data analyst demo.

Includes 18 read/compute tools, a demo SQLite database and demo standards.
No write tool is registered. For a real deployment replace data/demo.db and
knowledge/*.json with your own data.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "demo.db"
DEFAULT_KNOWLEDGE = HERE / "knowledge"


def _db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _rows(con, sql, params=()):
    return [dict(row) for row in con.execute(sql, params).fetchall()]


def _date_clause(column: str, start: str, end: str):
    clauses, params = [], []
    if start:
        clauses.append(f"{column} >= ?")
        params.append(start)
    if end:
        clauses.append(f"{column} <= ?")
        params.append(end + " 23:59:59" if len(end) == 10 else end)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _standard(vehicle: str, task: str, knowledge_dir: Path) -> dict:
    for name in (vehicle or "通用", "通用"):
        path = knowledge_dir / f"{name}.json"
        if path.is_file():
            try:
                return (json.loads(path.read_text(encoding="utf-8")).get("standards") or {}).get(task, {})
            except (OSError, ValueError):
                pass
    return {}


def _grade(value: float, target: float, tol: float, mild: float, moderate: float) -> tuple[str, str]:
    dev = abs(value - target)
    if dev <= tol:
        return "OK", "合格"
    if dev <= mild:
        return "NG", "轻微"
    if dev <= moderate:
        return "NG", "中度"
    return "NG", "严重"


def build_server(db_path: str, knowledge_dir: str, host: str, port: int):
    db_path = str(Path(db_path).resolve())
    knowledge_dir = Path(knowledge_dir).resolve()
    mcp = FastMCP("autodetect-readonly-mcp", host=host, port=port)

    @mcp.tool()
    def query_records(
        batch_number: str = "", vehicle: str = "", content: str = "",
        workstation: str = "", task_type: str = "", class_id: str = "",
        start_date: str = "", end_date: str = "", search_text: str = "",
        limit: int = 100, offset: int = 0,
    ) -> str:
        """查询检测记录明细（只读）。class_id: 0=不合格, 1=合格, 留空=全部。"""
        where = ["s.is_deleted = 0"]
        params: list = []
        for col, value in (("s.batch_number", batch_number), ("s.vehicle", vehicle),
                           ("s.content", content), ("s.workstation", workstation),
                           ("s.task_type", task_type)):
            if value:
                where.append(f"{col} = ?")
                params.append(value)
        if class_id not in ("", None):
            where.append("s.class_id = ?")
            params.append(int(class_id))
        date_sql, date_params = _date_clause("s.created_at", start_date, end_date)
        if date_sql:
            where.append(date_sql.removeprefix(" AND "))
            params.extend(date_params)
        if search_text:
            where.append("(s.image_path LIKE ? OR s.label_text LIKE ? OR s.remark LIKE ?)")
            like = f"%{search_text}%"
            params.extend([like, like, like])
        clause = " AND ".join(where)
        con = _db(db_path)
        try:
            total = con.execute(f"SELECT COUNT(*) FROM detection_records s WHERE {clause}", params).fetchone()[0]
            rows = _rows(
                con,
                f"""SELECT s.*, u.display_name AS operator_name
                      FROM detection_records s
                      LEFT JOIN users u ON u.id = s.operator_id
                     WHERE {clause}
                     ORDER BY s.id DESC LIMIT ? OFFSET ?""",
                params + [max(1, min(int(limit), 500)), max(0, int(offset))],
            )
        finally:
            con.close()
        return json.dumps({"total": total, "items": rows}, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_record(record_id: int) -> str:
        """按 ID 查询单条检测记录。"""
        con = _db(db_path)
        try:
            rows = _rows(
                con,
                """SELECT s.*, u.display_name AS operator_name
                     FROM detection_records s
                     LEFT JOIN users u ON u.id = s.operator_id
                    WHERE s.id = ? AND s.is_deleted = 0""",
                (int(record_id),),
            )
        finally:
            con.close()
        return json.dumps(rows[0] if rows else None, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_stats(start_date: str = "", end_date: str = "") -> str:
        """汇总双针/单针/孔径三张回归表的 OK/NG 和合格率。"""
        con = _db(db_path)
        try:
            breakdown, total, ok, ng = [], 0, 0, 0
            for name, table in (("双针装饰线", "stitch_inspection"),
                                ("单针偏移", "single_stitch_inspection"),
                                ("孔径分析", "hole_inspection")):
                clause, params = _date_clause("created_at", start_date, end_date)
                row = con.execute(
                    f"""SELECT COUNT(*) total,
                               SUM(CASE WHEN verdict='OK' THEN 1 ELSE 0 END) ok,
                               SUM(CASE WHEN verdict='NG' THEN 1 ELSE 0 END) ng
                          FROM {table} WHERE 1=1{clause}""",
                    params,
                ).fetchone()
                t, o, n = int(row["total"] or 0), int(row["ok"] or 0), int(row["ng"] or 0)
                total += t; ok += o; ng += n
                breakdown.append({"task_type": name, "total": t, "pass": o, "fail": n,
                                  "pass_rate_pct": round(o / t * 100, 2) if t else 0})
        finally:
            con.close()
        return json.dumps({"total": total, "pass": ok, "fail": ng,
                           "pass_rate_pct": round(ok / total * 100, 2) if total else 0,
                           "by_task_type": breakdown}, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_daily_trend(start_date: str = "", end_date: str = "") -> str:
        """按日期查看三张回归表的合格率趋势。"""
        con = _db(db_path)
        try:
            union = " UNION ALL ".join(
                f"SELECT substr(created_at,1,10) d, verdict FROM {t}" for t in
                ("stitch_inspection", "single_stitch_inspection", "hole_inspection"))
            clause, params = _date_clause("d", start_date, end_date)
            rows = _rows(
                con,
                f"""SELECT d AS date, COUNT(*) total,
                           SUM(CASE WHEN verdict='OK' THEN 1 ELSE 0 END) pass,
                           SUM(CASE WHEN verdict='NG' THEN 1 ELSE 0 END) fail
                      FROM ({union}) WHERE 1=1{clause}
                     GROUP BY d ORDER BY d""",
                params,
            )
        finally:
            con.close()
        for row in rows:
            row["pass_rate_pct"] = round(row["pass"] / row["total"] * 100, 2) if row["total"] else 0
        return json.dumps({"items": rows}, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_vehicle_stats(start_date: str = "", end_date: str = "") -> str:
        """按车型和任务类型统计实时检测记录。class_id 1=合格, 0=不合格。"""
        con = _db(db_path)
        try:
            clause, params = _date_clause("created_at", start_date, end_date)
            rows = _rows(
                con,
                f"""SELECT vehicle, task_type, COUNT(*) total,
                           SUM(CASE WHEN class_id=1 THEN 1 ELSE 0 END) pass,
                           SUM(CASE WHEN class_id=0 THEN 1 ELSE 0 END) fail
                      FROM detection_records
                     WHERE is_deleted=0{clause}
                     GROUP BY vehicle, task_type ORDER BY vehicle, task_type""",
                params,
            )
        finally:
            con.close()
        for row in rows:
            row["pass_rate_pct"] = round(row["pass"] / row["total"] * 100, 2) if row["total"] else 0
        return json.dumps({"items": rows}, ensure_ascii=False, indent=2)

    @mcp.tool()
    def list_vehicles() -> str:
        """列出检测记录中出现过的车型和工位。"""
        con = _db(db_path)
        try:
            vehicles = [r[0] for r in con.execute(
                "SELECT DISTINCT vehicle FROM detection_records WHERE is_deleted=0 ORDER BY vehicle")]
            stations = [r[0] for r in con.execute(
                "SELECT DISTINCT workstation FROM detection_records WHERE is_deleted=0 ORDER BY workstation")]
        finally:
            con.close()
        return json.dumps({"vehicles": vehicles, "workstations": stations}, ensure_ascii=False, indent=2)

    @mcp.tool()
    def query_standards(vehicle: str = "通用") -> str:
        """查询双针装饰线质量标准。"""
        std = _standard(vehicle, "双针装饰线", knowledge_dir)
        target = float(std.get("standard_value", 4.0)); tol = float(std.get("tolerance", 1.0))
        return json.dumps({"vehicle": vehicle, "task_type": "双针装饰线",
                           "spec": {"standard_distance": f"{target} mm", "tolerance": f"±{tol} mm"},
                           "safety_redlines": [f"双针线距 <{std.get('redline', 1.0)}mm → 立即停机"]},
                          ensure_ascii=False, indent=2)

    def _query_simple(table: str, batch_number: str, workstation: str, verdict: str,
                      start_date: str, end_date: str, limit: int, offset: int) -> str:
        where, params = ["1=1"], []
        for col, value in (("batch_number", batch_number), ("workstation", workstation),
                           ("verdict", verdict)):
            if value:
                where.append(f"{col} = ?"); params.append(value)
        clause, dp = _date_clause("created_at", start_date, end_date)
        if clause:
            where.append(clause.removeprefix(" AND ")); params.extend(dp)
        sql = " AND ".join(where)
        con = _db(db_path)
        try:
            total = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {sql}", params).fetchone()[0]
            rows = _rows(con, f"SELECT * FROM {table} WHERE {sql} ORDER BY id DESC LIMIT ? OFFSET ?",
                         params + [max(1, min(int(limit), 500)), max(0, int(offset))])
        finally:
            con.close()
        return json.dumps({"total": total, "items": rows}, ensure_ascii=False, indent=2)

    @mcp.tool()
    def query_stitch_records(batch_number: str = "", workstation: str = "", verdict: str = "",
                             start_date: str = "", end_date: str = "",
                             limit: int = 100, offset: int = 0) -> str:
        """查询双针检测记录。"""
        return _query_simple("stitch_inspection", batch_number, workstation, verdict,
                            start_date, end_date, limit, offset)

    def _stats_simple(table: str, batch_number: str, start_date: str, end_date: str) -> str:
        where, params = ["1=1"], []
        if batch_number:
            where.append("batch_number = ?"); params.append(batch_number)
        clause, dp = _date_clause("created_at", start_date, end_date)
        if clause:
            where.append(clause.removeprefix(" AND ")); params.extend(dp)
        sql = " AND ".join(where)
        con = _db(db_path)
        try:
            row = con.execute(
                f"""SELECT COUNT(*) total,
                           SUM(CASE WHEN verdict='OK' THEN 1 ELSE 0 END) pass,
                           SUM(CASE WHEN verdict='NG' THEN 1 ELSE 0 END) fail
                      FROM {table} WHERE {sql}""", params).fetchone()
        finally:
            con.close()
        total, ok = int(row["total"] or 0), int(row["pass"] or 0)
        return json.dumps({"total": total, "pass": ok, "fail": int(row["fail"] or 0),
                           "pass_rate_pct": round(ok / total * 100, 2) if total else 0},
                          ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_stitch_stats(batch_number: str = "", start_date: str = "", end_date: str = "") -> str:
        """双针检测统计。"""
        return _stats_simple("stitch_inspection", batch_number, start_date, end_date)

    @mcp.tool()
    def query_single_stitch_records(batch_number: str = "", workstation: str = "", verdict: str = "",
                                    start_date: str = "", end_date: str = "",
                                    limit: int = 100, offset: int = 0) -> str:
        """查询单针检测记录。"""
        return _query_simple("single_stitch_inspection", batch_number, workstation, verdict,
                            start_date, end_date, limit, offset)

    @mcp.tool()
    def get_single_stitch_stats(batch_number: str = "", start_date: str = "", end_date: str = "") -> str:
        """单针检测统计。"""
        return _stats_simple("single_stitch_inspection", batch_number, start_date, end_date)

    @mcp.tool()
    def query_single_stitch_standards(vehicle: str = "通用") -> str:
        """查询单针偏移质量标准。"""
        std = _standard(vehicle, "单针偏移", knowledge_dir)
        return json.dumps({"vehicle": vehicle, "task_type": "单针偏移",
                           "spec": {"standard_distance": f"{std.get('standard_value', 4.0)} mm",
                                    "tolerance": f"±{std.get('tolerance', 1.0)} mm"}},
                          ensure_ascii=False, indent=2)

    @mcp.tool()
    def query_hole_records(batch_number: str = "", workstation: str = "", verdict: str = "",
                           start_date: str = "", end_date: str = "",
                           limit: int = 100, offset: int = 0) -> str:
        """查询孔径检测记录。"""
        return _query_simple("hole_inspection", batch_number, workstation, verdict,
                            start_date, end_date, limit, offset)

    @mcp.tool()
    def get_hole_stats(batch_number: str = "", start_date: str = "", end_date: str = "") -> str:
        """孔径检测统计。"""
        return _stats_simple("hole_inspection", batch_number, start_date, end_date)

    @mcp.tool()
    def query_hole_standards(vehicle: str = "通用") -> str:
        """查询孔径质量标准。"""
        std = _standard(vehicle, "孔径分析", knowledge_dir)
        return json.dumps({"vehicle": vehicle, "task_type": "孔径分析",
                           "spec": {"standard_diameter": f"{std.get('standard_value', 10.0)} mm",
                                    "tolerance": f"±{std.get('tolerance', 0.8)} mm",
                                    "ovality_limit": f"椭圆度 ≤ {std.get('ovality_limit', 0.3)}mm"}},
                          ensure_ascii=False, indent=2)

    @mcp.tool()
    def analyze_stitch(d1_mean: float, d2_mean: float, workstation: str = "",
                       angle_deg: float = 0, vehicle: str = "通用") -> str:
        """单次双针计算（不保存）。"""
        std = _standard(vehicle, "双针装饰线", knowledge_dir)
        target = float(std.get("standard_value", 4.0)); tol = float(std.get("tolerance", 1.0))
        dev = max(abs(d1_mean - target), abs(d2_mean - target))
        status, level = _grade(dev + target, target, tol, float(std.get("severity_mild", 1.0)),
                               float(std.get("severity_moderate", 1.5)))
        return json.dumps({"task_type": "双针装饰线", "d1_mean_mm": d1_mean, "d2_mean_mm": d2_mean,
                           "standard_mm": target, "tolerance_mm": tol,
                           "max_deviation_mm": round(dev, 3), "verdict": status,
                           "severity": level, "workstation": workstation, "angle_deg": angle_deg},
                          ensure_ascii=False, indent=2)

    @mcp.tool()
    def analyze_single_stitch(d_mean: float, workstation: str = "", angle_deg: float = 0,
                              vehicle: str = "通用") -> str:
        """单次单针计算（不保存）。"""
        std = _standard(vehicle, "单针偏移", knowledge_dir)
        target = float(std.get("standard_value", 4.0)); tol = float(std.get("tolerance", 1.0))
        dev = abs(d_mean - target)
        status, level = _grade(d_mean, target, tol, float(std.get("severity_mild", 1.0)),
                               float(std.get("severity_moderate", 1.5)))
        return json.dumps({"task_type": "单针偏移", "d_mean_mm": d_mean, "standard_mm": target,
                           "tolerance_mm": tol, "deviation_mm": round(dev, 3),
                           "verdict": status, "severity": level,
                           "workstation": workstation, "angle_deg": angle_deg},
                          ensure_ascii=False, indent=2)

    @mcp.tool()
    def analyze_hole(d_mean: float, d_std: float = 0, ovality_mm: float = 0,
                     workstation: str = "", vehicle: str = "通用") -> str:
        """单次孔径计算（不保存）。"""
        std = _standard(vehicle, "孔径分析", knowledge_dir)
        target = float(std.get("standard_value", 10.0)); tol = float(std.get("tolerance", 0.8))
        oval_limit = float(std.get("ovality_limit", 0.3))
        dev = abs(d_mean - target)
        diameter_ok = dev <= tol
        oval_ok = ovality_mm <= oval_limit
        status = "OK" if diameter_ok and oval_ok else "NG"
        return json.dumps({"task_type": "孔径分析", "d_mean_mm": d_mean, "d_std_mm": d_std,
                           "ovality_mm": ovality_mm, "standard_mm": target,
                           "tolerance_mm": tol, "ovality_limit_mm": oval_limit,
                           "deviation_mm": round(dev, 3), "verdict": status,
                           "workstation": workstation}, ensure_ascii=False, indent=2)

    return mcp


def main() -> int:
    p = argparse.ArgumentParser(description="Read-only quality-data MCP server")
    p.add_argument("--db-path", default=str(DEFAULT_DB))
    p.add_argument("--knowledge-dir", default=str(DEFAULT_KNOWLEDGE))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5071)
    args = p.parse_args()
    if not Path(args.db_path).is_file():
        raise SystemExit(f"database not found: {args.db_path}; run data/seed_demo.py first")
    server = build_server(args.db_path, args.knowledge_dir, args.host, args.port)
    server.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
