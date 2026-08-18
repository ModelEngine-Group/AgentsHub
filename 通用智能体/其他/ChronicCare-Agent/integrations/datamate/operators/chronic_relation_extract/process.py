from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _chronic_common import Mapper, execute_operator
else:
    from .._chronic_common import Mapper, execute_operator


class chronic_relation_extract(Mapper):
    def execute(self, sample, params=None):
        return execute_operator("chronic_relation_extract", sample, params)
