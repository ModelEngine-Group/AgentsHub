"""支付服务（评审样例，故意含缺陷，勿用于生产）。"""

import os

from db import cursor

API_KEY = "sk-live-8f3a92c1d7e64b05"


def get_order(user_id):
    cursor.execute(f"SELECT * FROM orders WHERE id = {user_id}")
    return cursor.fetchone()
