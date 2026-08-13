import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "eval_cmeee_deployment_backends.py"
)
SPEC = importlib.util.spec_from_file_location("eval_cmeee_deployment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gold_uses_exclusive_character_end():
    row = {
        "entities": [
            {"type": "dis", "start": 2, "end": 5, "entity": "示例病"},
        ],
    }
    assert MODULE._gold(row) == {("dis", 2, 5)}


def test_evaluate_reports_exact_span_type_counts():
    class Backend:
        @staticmethod
        def execute_batch(texts):
            return [
                [{
                    "cmeee_type": "dis",
                    "type": "dis",
                    "span": [0, 2],
                }],
                [],
            ]

    rows = [
        {
            "text": "甲乙",
            "entities": [
                {"type": "dis", "start": 0, "end": 2, "entity": "甲乙"},
            ],
        },
        {
            "text": "丙丁",
            "entities": [
                {"type": "sym", "start": 0, "end": 2, "entity": "丙丁"},
            ],
        },
    ]
    result = MODULE._evaluate(Backend(), rows, 2, ["dis", "sym"])
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 1
