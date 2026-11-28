"""合规样例（负样本）：本文件不应产生任何发现项，用来验证智能体不会凑数。"""

import logging
import os

import requests

logger = logging.getLogger(__name__)


def fetch(url, retries=None):
    if retries is None:
        retries = []
    return requests.get(url, timeout=(3, 10))
