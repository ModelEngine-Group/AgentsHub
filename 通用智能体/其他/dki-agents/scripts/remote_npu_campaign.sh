#!/usr/bin/env bash
# Retired unsafe remote campaign launcher.
#
# It formerly mixed host probing, diagnostic smoke training, full candidate
# training, and OM validation in one mutable flow.  That ordering can consume
# an NPU before the archive hashes, frozen checkpoint, and OM manifest have
# been validated; it also runs the second CMeIE seed before seed-42 evidence is
# reviewed.  Keep this executable as a fail-closed migration marker rather than
# a second remote-control path.
set -u

printf '%s\n' \
  "Refusing obsolete automatic remote campaign. Build the local credential-free bundle with scripts/prepare_remote_npu_bundle.py, inspect dist/remote_npu/REMOTE_NPU_RUNBOOK.json, then execute its verified target-host steps manually after preflight passes." \
  >&2
exit 2
