#!/usr/bin/env bash
# k8, 500 events: OpenAI -> 8×8 cluster A_matrix.csv + metrics vs GT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec python3 baselines/pure_llm/pure_llm_k8_500_cluster_graph.py "$@"
