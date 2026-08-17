"""
SQLite 连接工具模块。
"""

import sqlite3
from pathlib import Path
from mcp_server.config import KG_DB, ANALYTICS_DB

def connect_kg():
    c = sqlite3.connect(str(KG_DB))
    c.row_factory = sqlite3.Row
    return c

def connect_analytics():
    c = sqlite3.connect(str(ANALYTICS_DB))
    c.row_factory = sqlite3.Row
    return c

def row_dicts(cursor):
    return [dict(r) for r in cursor.fetchall()]
