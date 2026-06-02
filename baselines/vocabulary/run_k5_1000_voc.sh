#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Vocabulary baseline — k5, 1000 events: 5 training seeds + eval vs ground truth.
#
# From repository root:
#   bash baselines/vocabulary/run_k5_1000_voc.sh
#
# Outputs under baselines/vocabulary/output/k5_1000/:
#   vocab_teacher_scores_k5_1000.csv                    (shared teacher)
#   run_seed{100,200,300,400,500}/vocab_k5_1000_A_matrix.csv
#   vocab_k5_1000_metrics.csv                          (τ=0.5, mean±sd)
#   (optional) vocab_k5_1000_metrics_by_threshold.csv  (use eval --all-thresholds)
#
# Data are bundled for GitHub repro under baselines/vocabulary/data/.
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_K5="baselines/vocabulary/output/k5_1000"
TEACHER="${OUT_K5}/vocab_teacher_scores_k5_1000.csv"
SEEDS=(100 200 300 400 500)

INPUT="baselines/vocabulary/data/sim_chemical_1000_k5_uniform_passset.csv"
CLUSTER="baselines/vocabulary/data/sim_chemical1000_k5_uniform_passset_oracle_clusters.csv"

for s in "${SEEDS[@]}"; do
  outd="${OUT_K5}/run_seed${s}"
  mkdir -p "$outd"
  python3 baselines/vocabulary/role_distill_vocabulary_k5_500.py \
    --seed "$s" \
    --teacher-cache "$TEACHER" \
    --input "$INPUT" \
    --cluster-csv "$CLUSTER" \
    --max-events 1000 \
    --out-pt "${outd}/vocab_role_distill_k5_1000.pt" \
    --out-a-matrix-csv "${outd}/vocab_k5_1000_A_matrix.csv" \
    --out-a-matrix-npy "${outd}/vocab_k5_1000_A_matrix.npy"
done

python3 baselines/vocabulary/eval_vocab_k5_1000_runs.py \
  --runs-root "$OUT_K5" \
  --seeds "${SEEDS[@]}"

echo "Done. See ${OUT_K5}/vocab_k5_1000_metrics.csv"
