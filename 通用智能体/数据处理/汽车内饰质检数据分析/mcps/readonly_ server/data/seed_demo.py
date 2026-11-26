# -*- coding: utf-8 -*-
"""Create a deterministic demo database for the reproduction package."""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "demo.db"
SCHEMA = HERE / "schema.sql"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_DB))
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(str(out))
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    rng = random.Random(20261126)
    base = datetime(2026, 11, 20, 8, 0, 0)
    vehicles = ["F520M", "X5M", "E260"]
    workstations = ["WS-01", "WS-02", "WS-03"]
    tasks = ["双针装饰线", "单针偏移", "孔径分析"]
    users = [(1, "admin", "管理员"), (2, "operator1", "操作员1")]
    con.executemany(
        "INSERT INTO users (id, username, display_name, role) VALUES (?, ?, ?, 'admin')",
        [(i, u, n) for i, u, n in users],
    )
    # 180 realtime detection records, spread over 7 days
    for i in range(180):
        ts = base + timedelta(hours=i * 0.9)
        task = tasks[i % 3]
        vehicle = vehicles[i % len(vehicles)]
        ws = workstations[(i // 3) % len(workstations)]
        bad = (i % 7 == 0) or (i % 17 == 0)
        class_id = 0 if bad else 1
        con.execute(
            """INSERT INTO detection_records
               (vehicle, content, workstation, image_path, class_id, class_name,
                label_text, confidence, inference_time_ms, batch_number, operator_id,
                task_type, is_deleted, created_at)
               VALUES (?, '前靠背面套', ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 0, ?)""",
            (vehicle, ws, f"demo/{i:04d}.jpg", class_id,
             "unqualified" if bad else "qualified",
             "不合格" if bad else "合格",
             round(0.7 + rng.random() * 0.29, 4),
             round(20 + rng.random() * 40, 2),
             ts.strftime("%Y-%m-%d"), task, ts.strftime("%Y-%m-%d %H:%M:%S")),
        )
    # 60 double-needle inspections
    for i in range(60):
        ts = base + timedelta(hours=i * 2.2)
        bad = i % 6 == 0
        verdict = "NG" if bad else "OK"
        d1 = round(4.0 + (rng.uniform(-1.8, 1.8) if bad else rng.uniform(-0.8, 0.8)), 3)
        d2 = round(4.0 + (rng.uniform(-1.8, 1.8) if bad else rng.uniform(-0.8, 0.8)), 3)
        con.execute(
            """INSERT INTO stitch_inspection
               (image_id, batch_number, workstation, verdict, severity, defect_type,
                max_deviation_mm, d1_mean_mm, d1_std_mm, d1_min_mm, d1_max_mm,
                d1_bad_points, d1_total_points, d2_mean_mm, d2_std_mm, d2_min_mm,
                d2_max_mm, d2_bad_points, d2_total_points, angle_deg,
                causes, fixes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?)""",
            (f"stitch/{i:04d}", ts.strftime("%Y-%m-%d"), workstations[i % 3],
             verdict, "严重" if bad else "正常", "线距偏移" if bad else "",
             round(abs(d1 - 4.0), 3), d1, 0.12, d1 - 0.3, d1 + 0.3, 0 if not bad else 3, 20,
             d2, 0.12, d2 - 0.3, d2 + 0.3, 0 if not bad else 2, 20,
             round(rng.uniform(-2, 2), 2),
             "线距偏移" if bad else "", "调整张力/针距" if bad else "",
             ts.strftime("%Y-%m-%d %H:%M:%S")),
        )
    # 45 single-needle inspections
    for i in range(45):
        ts = base + timedelta(hours=i * 2.7)
        bad = i % 5 == 0
        d = round(4.0 + (rng.uniform(-2.2, 2.2) if bad else rng.uniform(-0.8, 0.8)), 3)
        con.execute(
            """INSERT INTO single_stitch_inspection
               (image_id, batch_number, workstation, verdict, severity, defect_type,
                max_deviation_mm, d_mean_mm, d_std_mm, d_min_mm, d_max_mm,
                bad_points, total_points, angle_deg, causes, fixes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"single/{i:04d}", ts.strftime("%Y-%m-%d"), workstations[(i + 1) % 3],
             "NG" if bad else "OK", "严重" if bad else "正常",
             "单针偏移" if bad else "", round(abs(d - 4.0), 3), d, 0.15,
             d - 0.4, d + 0.4, 0 if not bad else 4, 20,
             round(rng.uniform(-2, 2), 2), "单针偏移" if bad else "",
             "调整机针/送布" if bad else "", ts.strftime("%Y-%m-%d %H:%M:%S")),
        )
    # 45 hole inspections
    for i in range(45):
        ts = base + timedelta(hours=i * 2.9)
        bad = i % 5 == 1
        d = round(10.0 + (rng.uniform(-1.6, 1.6) if bad else rng.uniform(-0.5, 0.5)), 3)
        oval = round(0.3 + (rng.uniform(0.2, 1.2) if bad else rng.uniform(-0.15, 0.2)), 3)
        con.execute(
            """INSERT INTO hole_inspection
               (image_id, batch_number, workstation, verdict, severity, defect_type,
                max_deviation_mm, d_mean_mm, d_std_mm, d_min_mm, d_max_mm,
                ovality_mm, sample_angles, position_x_mm, position_y_mm,
                causes, fixes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"hole/{i:04d}", ts.strftime("%Y-%m-%d"), workstations[(i + 2) % 3],
             "NG" if bad else "OK", "严重" if bad else "正常",
             "孔径超差/椭圆度" if bad else "", round(abs(d - 10.0), 3), d, 0.08,
             d - 0.2, d + 0.2, oval, 8, 0.0, 0.0,
             "模具磨损" if bad else "", "换模/调整冲头" if bad else "",
             ts.strftime("%Y-%m-%d %H:%M:%S")),
        )
    con.commit()
    con.close()
    print(f"demo database: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
