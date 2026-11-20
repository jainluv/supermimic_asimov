#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$PWD}"
shift || true
exec python3 "$BUNDLE_DIR/apply_audit_fixes.py" "$ROOT" "$@"
