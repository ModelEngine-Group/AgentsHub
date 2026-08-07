from __future__ import annotations

import os
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MEDGRAPH_SOURCE", str(AGENT_ROOT / "knowledge" / "medical_cases.jsonl"))
os.environ.setdefault("MEDGRAPH_OUTPUT_DIR", str(AGENT_ROOT / ".runtime" / "latest"))

from medgraph_agent.integrations.fastmcp_server import main  # noqa: E402


if __name__ == "__main__":
    main()
