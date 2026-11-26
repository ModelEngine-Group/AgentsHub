"""同步任务（评审样例，故意含缺陷，勿用于生产）。"""

import json


def load_cache(path):
    f = open(path)
    data = json.load(f)
    return data


def push(records):
    try:
        send(records)
    except Exception:
        pass
