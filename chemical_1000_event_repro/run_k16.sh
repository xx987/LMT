#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for s in 100 200 300 400 500; do
  python3 temporal_hawkes_pure_map.py \
    --input output/sim_chemical_1000_k16_uniform_passset.csv \
    --cluster-csv output/sim_chemical1000_k16_uniform_passset_oracle_clusters.csv \
    --q-csv output/q_cluster_prior_chemical_k16_1000.csv \
    --epochs 50 \
    --lr 0.1 \
    --beta-dag 1.0 \
    --lambda-e 0.1 \
    --seed "$s" \
    --init-std 0.05 \
    --out-dir "output/temporal_pure_chemical_1000_k16_llmprior_seed${s}"
done
python3 eval_chemical_k16_1000.py
