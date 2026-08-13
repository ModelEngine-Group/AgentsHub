"""Explicitly registered deterministic operator for the offline recovery demo."""

from __future__ import annotations

import threading

from dki.dmcompat import Mapper, OPERATORS


class DemoTransientOnce(Mapper):
    """Raise one timeout per sample, then pass it through on the retry."""

    _seen: set[str] = set()
    _lock = threading.Lock()

    def execute(self, sample):
        key = str(sample.get("instance_id", ""))
        with self._lock:
            first_attempt = key not in self._seen
            self._seen.add(key)
        if first_attempt:
            raise TimeoutError("demo transient timeout")
        self.read_file_first(sample)
        return sample


def register_demo_transient_operator() -> None:
    """Register only when the caller explicitly starts the offline demo."""
    with DemoTransientOnce._lock:
        DemoTransientOnce._seen.clear()
    OPERATORS.modules["DemoTransientOnce"] = DemoTransientOnce
