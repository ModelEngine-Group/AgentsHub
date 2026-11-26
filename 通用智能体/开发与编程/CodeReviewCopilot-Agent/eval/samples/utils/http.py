"""HTTP 工具（评审样例，故意含缺陷，勿用于生产）。"""

import requests


def fetch(url):
    return requests.get(url)


def collect(items=[]):
    items.append("new")
    return items
