#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Vocabulary baseline — k16, 500 events: 5 training seeds + eval vs ground truth.
#
# From repository root:
#   bash baselines/vocabulary/run_k16_500_voc.sh
#
# Outputs under baselines/vocabulary/output/k16_500/ (separate from k5 / k8).
# Data: baselines/vocabulary/data/ (same as baselines/casc/chemical_500_data/output/).
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_K16="baselines/vocabulary/output/k16_500"
TEACHER="${OUT_K16}/vocab_teacher_scores_k16_500.csv"
SEEDS=(100 200 300 400 500)

for s in "${SEEDS[@]}"; do
  outd="${OUT_K16}/run_seed${s}"
  mkdir -p "$outd"
  python3 baselines/vocabulary/role_distill_vocabulary_k16_500.py \
    --seed "$s" \
    --teacher-cache "$TEACHER" \
    --out-pt "${outd}/vocab_role_distill_k16_500.pt" \
    --out-a-matrix-csv "${outd}/vocab_k16_500_A_matrix.csv" \
    --out-a-matrix-npy "${outd}/vocab_k16_500_A_matrix.npy"
done

python3 baselines/vocabulary/eval_vocab_k16_500_runs.py \
  --runs-root "$OUT_K16" \
  --seeds "${SEEDS[@]}"

echo "Done. See ${OUT_K16}/vocab_k16_500_metrics.csv"
