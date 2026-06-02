#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 baselines/vocabulary/role_distill_vocabulary_k16_500.py "$@"
