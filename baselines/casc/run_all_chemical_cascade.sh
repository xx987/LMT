#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 baselines/casc/run_cascade_chemical.py --repo-root "$ROOT" --preset all
