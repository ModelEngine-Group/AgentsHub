import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcps"))
from analysis_core import analyze_python

class AnalysisTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(analyze_python("x = 1")["issues"], [])
    def test_syntax(self):
        self.assertFalse(analyze_python("def f(")["ok"])
    def test_rules(self):
        result = analyze_python("def f(x=[]):\n try: eval('1')\n except: pass")
        self.assertEqual({x["rule"] for x in result["issues"]}, {"mutable-default", "bare-except", "dynamic-execution"})
        self.assertFalse(result["executed"])
    def test_limit(self):
        self.assertFalse(analyze_python(" " * 100001)["ok"])
    def test_no_execution(self):
        self.assertTrue(analyze_python("raise RuntimeError('must not run')")["ok"])
