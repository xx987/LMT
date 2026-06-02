#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Vocabulary baseline — k5, 500 events: 5 training seeds + eval vs ground truth.
#
# From repository root:
#   bash baselines/vocabulary/run_k5_500_voc.sh
#
# Outputs:
#   baselines/vocabulary/output/vocab_teacher_scores_k5_500.csv   (shared teacher)
#   baselines/vocabulary/output/run_seed{100,200,300,400,500}/  (.pt + A_matrix)
#   baselines/vocabulary/output/vocab_k5_500_metrics.csv  (default τ=0.5, one row)
#   Full τ grid CSV only: add  --all-thresholds  (writes vocab_k5_500_metrics_by_threshold.csv)
#   Same + print table:  --all-thresholds --print-grid
#
# Requires: Python 3 with torch, sentence-transformers, pandas, numpy (see repo).
# Data are under baselines/vocabulary/data/ (bundled for GitHub repro).
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

TEACHER="baselines/vocabulary/output/vocab_teacher_scores_k5_500.csv"
SEEDS=(100 200 300 400 500)

for s in "${SEEDS[@]}"; do
  outd="baselines/vocabulary/output/run_seed${s}"
  mkdir -p "$outd"
  python3 baselines/vocabulary/role_distill_vocabulary_k5_500.py \
    --seed "$s" \
    --teacher-cache "$TEACHER" \
    --out-pt "${outd}/vocab_role_distill_k5_500.pt" \
    --out-a-matrix-csv "${outd}/vocab_k5_500_A_matrix.csv" \
    --out-a-matrix-npy "${outd}/vocab_k5_500_A_matrix.npy"
done

python3 baselines/vocabulary/eval_vocab_k5_500_runs.py \
  --runs-root baselines/vocabulary/output \
  --seeds "${SEEDS[@]}"

echo "Done. See baselines/vocabulary/output/vocab_k5_500_metrics.csv"
