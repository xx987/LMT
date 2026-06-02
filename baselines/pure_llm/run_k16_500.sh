#!/usr/bin/env bash
# k16, 500 events: OpenAI -> 16×16 cluster A_matrix.csv + metrics vs GT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec python3 baselines/pure_llm/pure_llm_k16_500_cluster_graph.py "$@"
