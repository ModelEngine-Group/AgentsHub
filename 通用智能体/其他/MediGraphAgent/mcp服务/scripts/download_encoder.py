"""Fetch a HuggingFace encoder to a local dir via hf-mirror (robust to SSL
interception in restricted networks).  Used to obtain the Chinese encoder for
the neural extractor without relying on the hub client.

    python scripts/download_encoder.py --repo hfl/chinese-roberta-wwm-ext
"""
from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

FILES = [
    "config.json", "vocab.txt", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "added_tokens.json",
    "pytorch_model.bin",
]
REQUIRED = {"config.json", "vocab.txt", "pytorch_model.bin"}


def fetch(url: str, dest: Path) -> bool:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  OK {dest.name} ({len(data)/1e6:.1f} MB)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {dest.name}: {type(exc).__name__} {str(exc)[:80]}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "data" / "models" / "encoder"
        / args.repo.split("/")[-1])
    out.mkdir(parents=True, exist_ok=True)
    got = set()
    for name in FILES:
        url = f"{args.endpoint}/{args.repo}/resolve/main/{name}"
        if fetch(url, out / name):
            got.add(name)
        elif (out / name).exists() and (out / name).stat().st_size == 0:
            (out / name).unlink()
    missing = REQUIRED - got
    if missing:
        print(f"FAILED: missing required files {missing}")
        sys.exit(1)
    print(f"encoder ready at {out}")


if __name__ == "__main__":
    main()
