#!/usr/bin/env bash
# k5-1000: one OpenAI call per listed LLM temperature; metrics_by_tau.csv + run_*/; one-line Result summary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ -z "${OPENAI_API_KEY:-}" && "${*:-}" != *--mock* ]]; then
  echo "OPENAI_API_KEY is not set. Export it once, e.g.: export OPENAI_API_KEY=sk-..." >&2
  echo "(Or pass --mock for a smoke test.)" >&2
  exit 1
fi
exec python3 baselines/pure_llm/pure_llm_cluster_graph.py \
  --k 5 \
  --n-events 1000 \
  --temperatures 0.0,0.4,0.6,0.8,1.0 \
  --result-tau 0.25 \
  --model "${OPENAI_MODEL:-gpt-4o-mini}" \
  "$@"
