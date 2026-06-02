#!/usr/bin/env bash
# k5-500: one OpenAI call per listed LLM temperature; writes metrics_by_tau.csv + run_*/
# and prints only the Result line (default --result-tau 0.4).
#
# API key: set OPENAI_API_KEY once in your environment (e.g. ~/.zshrc, direnv, or
# `export OPENAI_API_KEY=...` in the same terminal) — no prompt in this script.
# Optional: OPENAI_MODEL (default gpt-4o-mini).
#
# Usage (from anywhere):
#   bash baselines/pure_llm/run_k5_500_multi_temp.sh
#   bash baselines/pure_llm/run_k5_500_multi_temp.sh --mock
#   bash baselines/pure_llm/run_k5_500_multi_temp.sh --result-tau 0.5 --out-dir baselines/pure_llm/output/k5_500_sweep
#
# Extra args are appended after defaults; later flags override earlier ones for the same option.
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
  --n-events 500 \
  --temperatures 0.0,0.2,0.4,0.6,0.8 \
  --model "${OPENAI_MODEL:-gpt-4o-mini}" \
  "$@"
