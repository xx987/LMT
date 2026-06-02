#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Vocabulary baseline — k8, 500 events: 5 training seeds + eval vs ground truth.
#
# From repository root:
#   bash baselines/vocabulary/run_k8_500_voc.sh
#
# Outputs (separate from k5 runs under output/k8_500/):
#   baselines/vocabulary/output/k8_500/vocab_teacher_scores_k8_500.csv
#   baselines/vocabulary/output/k8_500/run_seed{100,200,300,400,500}/
#   baselines/vocabulary/output/k8_500/vocab_k8_500_metrics.csv
#
# Data: baselines/vocabulary/data/ (k8 500 sim + oracle clusters + GT; same as CASCADE bundle).
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_K8="baselines/vocabulary/output/k8_500"
TEACHER="${OUT_K8}/vocab_teacher_scores_k8_500.csv"
SEEDS=(100 200 300 400 500)

for s in "${SEEDS[@]}"; do
  outd="${OUT_K8}/run_seed${s}"
  mkdir -p "$outd"
  python3 baselines/vocabulary/role_distill_vocabulary_k8_500.py \
    --seed "$s" \
    --teacher-cache "$TEACHER" \
    --out-pt "${outd}/vocab_role_distill_k8_500.pt" \
    --out-a-matrix-csv "${outd}/vocab_k8_500_A_matrix.csv" \
    --out-a-matrix-npy "${outd}/vocab_k8_500_A_matrix.npy"
done

python3 baselines/vocabulary/eval_vocab_k8_500_runs.py \
  --runs-root "$OUT_K8" \
  --seeds "${SEEDS[@]}"

echo "Done. See ${OUT_K8}/vocab_k8_500_metrics.csv"
