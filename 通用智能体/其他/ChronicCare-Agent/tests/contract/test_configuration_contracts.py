import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_current_metrics_has_core_counts() -> None:
    metrics = json.loads((PROJECT_ROOT / "configs/current_metrics.json").read_text(encoding="utf-8"))
    for key in ("patient_count", "node_count", "edge_count"):
        assert isinstance(metrics[key], int)
        assert metrics[key] > 0


def test_nexent_mcp_example_uses_streamable_http() -> None:
    config = json.loads(
        (PROJECT_ROOT / "integrations/nexent/chroniccare_mcp_config.example.json").read_text(encoding="utf-8")
    )
    assert config["transport"] == "streamable-http"
    assert config["mcp_endpoint"].endswith("/mcp")


def test_manifest_key_paths_exist() -> None:
    manifest = json.loads((PROJECT_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    missing = [path for path in manifest["key_paths"] if not (PROJECT_ROOT / path).exists()]
    assert missing == []
