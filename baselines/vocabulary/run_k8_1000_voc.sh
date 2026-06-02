#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Vocabulary baseline — k8, 1000 events: 5 training seeds + eval vs ground truth.
#
# From repository root:
#   bash baselines/vocabulary/run_k8_1000_voc.sh
#
# Outputs under baselines/vocabulary/output/k8_1000/:
#   vocab_teacher_scores_k8_1000.csv
#   run_seed{100,200,300,400,500}/vocab_k8_1000_A_matrix.csv
#   vocab_k8_1000_metrics.csv
#
# Data are bundled for GitHub repro under baselines/vocabulary/data/.
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_K8="baselines/vocabulary/output/k8_1000"
TEACHER="${OUT_K8}/vocab_teacher_scores_k8_1000.csv"
SEEDS=(100 200 300 400 500)

INPUT="baselines/vocabulary/data/sim_chemical_1000_k8_uniform_passset.csv"
CLUSTER="baselines/vocabulary/data/sim_chemical1000_k8_uniform_passset_oracle_clusters.csv"

for s in "${SEEDS[@]}"; do
  outd="${OUT_K8}/run_seed${s}"
  mkdir -p "$outd"
  python3 baselines/vocabulary/role_distill_vocabulary_k8_500.py \
    --seed "$s" \
    --teacher-cache "$TEACHER" \
    --input "$INPUT" \
    --cluster-csv "$CLUSTER" \
    --max-events 1000 \
    --out-pt "${outd}/vocab_role_distill_k8_1000.pt" \
    --out-a-matrix-csv "${outd}/vocab_k8_1000_A_matrix.csv" \
    --out-a-matrix-npy "${outd}/vocab_k8_1000_A_matrix.npy"
done

python3 baselines/vocabulary/eval_vocab_k8_1000_runs.py \
  --runs-root "$OUT_K8" \
  --seeds "${SEEDS[@]}"

echo "Done. See ${OUT_K8}/vocab_k8_1000_metrics.csv"
