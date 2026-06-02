#!/usr/bin/env bash
# k5, 500 events: OpenAI -> cluster A_matrix.csv + metrics vs GT.
# Requires: pip install openai  and  OPENAI_API_KEY
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 baselines/pure_llm/pure_llm_k5_500_cluster_graph.py "$@"
