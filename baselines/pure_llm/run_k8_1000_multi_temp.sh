#!/usr/bin/env bash
# k8-1000: same as run_k5_1000_multi_temp.sh but --k 8.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ -z "${OPENAI_API_KEY:-}" && "${*:-}" != *--mock* ]]; then
  echo "OPENAI_API_KEY is not set. Export it once, e.g.: export OPENAI_API_KEY=sk-..." >&2
  echo "(Or pass --mock for a smoke test.)" >&2
  exit 1
fi
exec python3 baselines/pure_llm/pure_llm_cluster_graph.py \
  --k 8 \
  --n-events 1000 \
  --temperatures 0.0,0.2,0.4,0.6,0.8 \
  --result-tau 0.2\
  --model "${OPENAI_MODEL:-gpt-4o-mini}" \
  "$@"
