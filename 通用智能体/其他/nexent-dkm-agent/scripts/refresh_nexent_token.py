"""Refresh Nexent JWT from /api/user/signin and write ignored token files."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Nexent access token.")
    parser.add_argument("--base-url", default="http://localhost:3000")
    parser.add_argument("--email", default="dkm_evidence@example.com")
    parser.add_argument("--password", default="DkmEvidence2026!")
    parser.add_argument(
        "--token-file",
        action="append",
        default=[
            str(ROOT / ".local" / "nexent.token"),
            str(ROOT.parent / ".local" / "nexent.token"),
        ],
    )
    return parser.parse_args()


def signin(base_url: str, email: str, password: str) -> str:
    payload = json.dumps({"email": email, "password": password}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/user/signin",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            cookie = response.headers.get("Set-Cookie", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"signin failed HTTP {exc.code}: {detail}") from exc

    match = re.search(r"nexent_access_token=([^;]+)", cookie)
    if match:
        return match.group(1)

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("signin response missing nexent_access_token cookie") from exc

    for key in ("access_token", "token"):
        if isinstance(data, dict) and data.get(key):
            return str(data[key])
        nested = data.get("data") if isinstance(data, dict) else None
        if isinstance(nested, dict) and nested.get(key):
            return str(nested[key])

    raise RuntimeError("signin succeeded but no access token found in cookie or JSON")


def main() -> int:
    args = parse_args()
    token = signin(args.base_url, args.email, args.password)
    for path_str in args.token_file:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "token_length": len(token)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
