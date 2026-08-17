# -*- coding: utf-8 -*-
"""
噪声日志模块：记录 LLMNoiseFilter 每次运行产生的噪声信号和清洗结果，
供后续模式挖掘和规则自学习使用。
"""
import json
import os
import re
import sqlite3
import time
from typing import Dict, List, Optional

_DB_PATH = os.environ.get("NOISE_DB_PATH",
                          "/opt/runtime/datamate/ops/user/llm_noise_filter/noise_log.db")


def _get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    _init_tables(conn)
    return conn


def _init_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS noise_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT,
            file_name   TEXT,
            char_count  INTEGER,
            status      TEXT,           -- skipped / llm_cleaned / llm_rejected / llm_error
            noise_score REAL DEFAULT 0, -- 0=clean, 1=very noisy
            chars_removed INTEGER DEFAULT 0,
            noise_signals TEXT,         -- JSON dict: {signal_name: matched_text}
            categories  TEXT,           -- JSON list of matched categories
            llm_label   TEXT,           -- LLM自己判断的噪声标签（如果有）
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS noise_diffs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER REFERENCES noise_runs(id),
            removed_text TEXT,           -- 被LLM删除的文本片段
            char_count  INTEGER,
            category    TEXT,            -- 噪声分类
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_noise_runs_score ON noise_runs(noise_score);
        CREATE INDEX IF NOT EXISTS idx_noise_runs_status ON noise_runs(status);
        CREATE INDEX IF NOT EXISTS idx_noise_diffs_category ON noise_diffs(category);
    """)


# 噪声信号定义：正则 → 信号名 → 分类
NOISE_SIGNALS = [
# 三元组含义：匹配模式、信号名称、分类。
    (r'哎[呀ya]|呜呜|卧槽|嗷嗷|好吧好吧|OMG|WTF',
     '口语感叹词', 'colloquial'),
    (r'太[^\s]{0,4}了',
     '口语强调("太X了")', 'colloquial'),
    (r'@[^\s@]{1,20}',
     '@mention通知', 'mention'),
    (r'由.{1,10}系统.{0,6}(导出|生成|产生)',
     '系统导出废话', 'system_export'),
    (r'图片链接已失效|该网页无法显示',
     '垃圾系统提示', 'system_export'),
    (r'填表[时间：:].{0,30}[0-9]',
     '填表时间戳', 'form_metadata'),
    (r'记得把这个录入|交接班记录',
     '工作指令废文', 'work_instruction'),
    (r'跟.{1,8}聊天',
     '闲聊(与人聊天)', 'chitchat'),
    (r'邻居|煮饭|买菜|散步.{0,3}(遇到|碰见|看见)',
     '日常琐事描述', 'chitchat'),
    (r'听.{1,6}说[^.。]{0,20}(可以|能|管用|有效)',
     '道听途说', 'chitchat'),
    (r'去[大中心]{0,2}医院看看',
     '随意建议', 'chitchat'),
    (r'试.{0,3}(试|一下|看看)(?!表|剂|药|方|案)',
     '口语试探', 'chitchat'),
    (r'不知道吃[什么啥](?!药)',
     '不确定语', 'chitchat'),
]


def _scan_signals(text: str) -> dict:
    """扫描文本，返回所有匹配到的噪声信号"""
    result = {}
    for pat, name, cat in NOISE_SIGNALS:
        m = re.search(pat, text)
        if m:
            result[name] = {
                "matched": m.group(0)[:60],
                "category": cat,
            }
    return result


def _compute_noise_score(signals: dict, chars_removed: int, total_chars: int) -> float:
    """计算噪声分数 0.0(完全干净) ~ 1.0(噪声极重)"""
    if not total_chars:
        return 0.0
    # 信号密度 + 删除比例 加权
    categories = set(v["category"] for v in signals.values())
    signal_density = len(categories) / max(len({c for _, _, c in NOISE_SIGNALS}), 1)
    removal_ratio = min(1.0, chars_removed / max(total_chars, 1))
    return round(0.4 * signal_density + 0.6 * removal_ratio, 3)


def log_run(task_id: str, file_name: str, char_count: int,
            status: str, noise_signals: dict, chars_removed: int,
            removed_segments: Optional[List[str]] = None) -> int:
    """记录一次运行结果，返回 run_id 供写入 diffs"""
    score = _compute_noise_score(noise_signals, chars_removed, char_count)
    categories = list(set(v["category"] for v in noise_signals.values()))
    signals_json = json.dumps({k: v["matched"] for k, v in noise_signals.items()},
                              ensure_ascii=False)

    conn = _get_db()
    cur = conn.execute(
        "INSERT INTO noise_runs (task_id, file_name, char_count, status, "
        "noise_score, chars_removed, noise_signals, categories) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, file_name, char_count, status, score, chars_removed,
         signals_json, json.dumps(categories)))
    run_id = cur.lastrowid

    if removed_segments:
        for seg in removed_segments:
            # 猜测分类
            cat = _guess_category(seg, noise_signals)
            conn.execute(
                "INSERT INTO noise_diffs (run_id, removed_text, char_count, category) "
                "VALUES (?, ?, ?, ?)",
                (run_id, seg, len(seg), cat))
    conn.commit()
    conn.close()
    return run_id


def _guess_category(text: str, signals: dict) -> str:
    """根据已检测信号猜测被删文本的噪声类别"""
    # 收集所有已触发的类别
    active_cats = set(v["category"] for v in signals.values())
    for cat in ["system_export", "form_metadata", "work_instruction",
                 "colloquial", "chitchat", "mention"]:
        if cat in active_cats:
            return cat
    return "unknown"


def get_recent_stats(limit: int = 50) -> dict:
    """获取最近的噪声统计摘要，用于监控和学习"""
    conn = _get_db()
    # 总览
    total = conn.execute("SELECT count(*) FROM noise_runs").fetchone()[0]
    skipped = conn.execute("SELECT count(*) FROM noise_runs WHERE status='skipped'").fetchone()[0]
    cleaned = conn.execute("SELECT count(*) FROM noise_runs WHERE status LIKE 'llm_cleaned%'").fetchone()[0]
    avg_score = conn.execute("SELECT avg(noise_score) FROM noise_runs WHERE noise_score > 0").fetchone()[0] or 0
    # 按分类统计
    cat_rows = conn.execute(
        "SELECT category, count(*) as cnt, avg(char_count) as avg_len "
        "FROM noise_diffs GROUP BY category ORDER BY cnt DESC").fetchall()
    conn.close()
    return {
        "total_runs": total,
        "skipped": skipped,
        "llm_cleaned": cleaned,
        "llm_rate": f"{cleaned/max(total,1):.1%}",
        "avg_noise_score": round(float(avg_score), 3),
        "top_noise_categories": [(r[0], r[1], round(r[2], 0)) for r in cat_rows[:10]],
    }
