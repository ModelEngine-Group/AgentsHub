"""Spawn-safe fake adapters for process runtime acceptance tests."""
from __future__ import annotations

import os
import time


class ProcessTestAdapter:
    def __init__(self, mode="normal", marker=""):
        self.mode = mode
        self.marker = marker

    def open(self):
        return None

    def execute(self, value):
        if self.mode == "hang":
            time.sleep(30)
        if self.mode == "short_hang":
            time.sleep(0.35)
        if self.mode == "crash":
            os._exit(97)
        if self.mode == "error":
            raise RuntimeError("adapter error")
        return {"value": value * 2, "pid": os.getpid()}

    def execute_batch(self, values):
        if self.mode == "bad_batch_cardinality":
            return []
        if self.mode == "error":
            raise RuntimeError("adapter error")
        return [{"value": value * 2, "pid": os.getpid()} for value in values]

    def close(self):
        return None


def process_test_adapter(**config):
    return ProcessTestAdapter(**config)
