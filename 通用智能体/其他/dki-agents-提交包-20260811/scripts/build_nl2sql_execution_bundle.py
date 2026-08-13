"""Build a deterministic, source-only NL2SQL execution bundle.

This command only snapshots the local Python closure and explicitly named
runtime assets.  It does not invoke an evaluator or inspect a database,
dataset, candidate seal, or recovery artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _require_isolated_startup() -> None:
    if (
        not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.dont_write_bytecode
    ):
        raise SystemExit(
            "execution bundle builder requires Python startup flags -I -S -B"
        )
    preloaded = {"site", "sitecustomize", "usercustomize"}.intersection(sys.modules)
    if preloaded:
        raise SystemExit(
            "execution bundle builder rejects preloaded Python startup modules: "
            + ", ".join(sorted(preloaded))
        )


_require_isolated_startup()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_execution_bundle import build_execution_bundle


def _runtime_asset(value: str) -> tuple[str, Path]:
    destination, separator, source = value.partition("=")
    if not separator or not destination or not source:
        raise argparse.ArgumentTypeError(
            "runtime assets must use DESTINATION=SOURCE"
        )
    return destination, Path(source)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic source-only NL2SQL execution bundle; "
            "this command does not run evaluation."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--entry-module", required=True)
    parser.add_argument(
        "--runner-environment-manifest-sha256",
        required=True,
        help=(
            "SHA-256 of an independently verified runner/dependency-environment "
            "manifest; package versions alone are insufficient"
        ),
    )
    parser.add_argument(
        "--runtime-asset",
        type=_runtime_asset,
        action="append",
        default=[],
        metavar="DESTINATION=SOURCE",
        help="explicit non-code runtime asset to include (repeatable)",
    )
    parser.add_argument(
        "--extra-module",
        action="append",
        default=[],
        help=(
            "additional reviewed local module root to snapshot (repeatable); "
            "this does not authorize unresolved dynamic loading"
        ),
    )
    args = parser.parse_args()
    try:
        bundle = build_execution_bundle(
            args.output,
            source_root=args.source_root,
            entry_module=args.entry_module,
            runner_environment_manifest_sha256=(
                args.runner_environment_manifest_sha256
            ),
            runtime_assets=args.runtime_asset,
            extra_modules=args.extra_module,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    manifest = bundle.manifest
    print(
        json.dumps(
            {
                "archive": str(bundle.archive_path),
                "archive_sha256": bundle.archive_sha256,
                "entry_module": manifest["entry_module"],
                "kind": manifest["kind"],
                "manifest_sha256": bundle.manifest_sha256,
                "n_files": len(manifest["files"]),
                "n_modules": len(manifest["module_registry"]),
                "n_runtime_assets": len(manifest["runtime_assets"]),
                "runner_environment_manifest_sha256": manifest[
                    "runner_environment_manifest_sha256"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
