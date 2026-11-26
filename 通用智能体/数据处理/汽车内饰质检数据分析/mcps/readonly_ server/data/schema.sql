-- Demo schema for the quality-data-analyst reproduction package.
-- The real deployment can point --db-path at its own database.

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password_hash TEXT,
    role TEXT,
    display_name TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS detection_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle TEXT,
    content TEXT,
    workstation TEXT,
    image_path TEXT,
    class_id INTEGER,
    class_name TEXT,
    label_text TEXT,
    confidence REAL,
    inference_time_ms REAL,
    batch_number TEXT,
    operator_id INTEGER,
    remark TEXT,
    task_type TEXT,
    reviewed_by INTEGER,
    reviewed_at TEXT,
    is_deleted INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS stitch_inspection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT,
    batch_number TEXT,
    workstation TEXT,
    verdict TEXT,
    severity TEXT,
    defect_type TEXT,
    max_deviation_mm REAL,
    d1_mean_mm REAL, d1_std_mm REAL, d1_min_mm REAL, d1_max_mm REAL,
    d1_bad_points INTEGER, d1_total_points INTEGER,
    d2_mean_mm REAL, d2_std_mm REAL, d2_min_mm REAL, d2_max_mm REAL,
    d2_bad_points INTEGER, d2_total_points INTEGER,
    angle_deg REAL,
    causes TEXT,
    fixes TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS single_stitch_inspection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT,
    batch_number TEXT,
    workstation TEXT,
    verdict TEXT,
    severity TEXT,
    defect_type TEXT,
    max_deviation_mm REAL,
    d_mean_mm REAL, d_std_mm REAL, d_min_mm REAL, d_max_mm REAL,
    bad_points INTEGER, total_points INTEGER,
    angle_deg REAL,
    causes TEXT,
    fixes TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS hole_inspection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT,
    batch_number TEXT,
    workstation TEXT,
    verdict TEXT,
    severity TEXT,
    defect_type TEXT,
    max_deviation_mm REAL,
    d_mean_mm REAL, d_std_mm REAL, d_min_mm REAL, d_max_mm REAL,
    ovality_mm REAL,
    sample_angles INTEGER,
    position_x_mm REAL,
    position_y_mm REAL,
    causes TEXT,
    fixes TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_detection_created ON detection_records(created_at);
CREATE INDEX IF NOT EXISTS idx_stitch_created ON stitch_inspection(created_at);
CREATE INDEX IF NOT EXISTS idx_single_created ON single_stitch_inspection(created_at);
CREATE INDEX IF NOT EXISTS idx_hole_created ON hole_inspection(created_at);
