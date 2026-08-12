from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="medinsight-agent-test-"))
os.environ["MEDINSIGHT_RUNTIME_ROOT"] = str(TEST_RUNTIME)
sys.path.insert(0, str(ROOT / "mcps"))
sys.path.insert(0, str(ROOT / "mcps" / "src"))

from medinsight_tools import process_medical_data  # noqa: E402


class MedInsightPackageSmokeTest(unittest.TestCase):
    def test_task1_accepts_portable_json(self) -> None:
        payload = json.dumps(
            {
                "records": [
                    {
                        "doc_id": "smoke-001",
                        "title": "结构契约测试",
                        "text": "这是一条用于验证数据处理契约的中性测试文本。",
                    }
                ]
            },
            ensure_ascii=False,
        )
        result = process_medical_data(payload, run_id="portable-json")
        self.assertEqual(result["status"], "ok")
        self.assertIn("quality", result)


def tearDownModule() -> None:  # noqa: N802
    shutil.rmtree(TEST_RUNTIME, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
