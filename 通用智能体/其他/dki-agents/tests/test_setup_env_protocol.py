from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_environment_installers_include_sqlglot():
    for relative in ("scripts/setup_env.ps1", "scripts/setup_env.sh"):
        content = (ROOT / relative).read_text(encoding="utf-8").casefold()
        assert "sqlglot" in content, relative
