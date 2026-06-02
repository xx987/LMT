#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for s in 100 200 300 400 500; do
  python3 temporal_hawkes_pure_map.py \
    --input output/sim_chemical_1000_k8_uniform_passset.csv \
    --cluster-csv output/sim_chemical1000_k8_uniform_passset_oracle_clusters.csv \
    --q-csv output/q_cluster_prior_chemical_k8_1000.csv \
    --epochs 200 \
    --lr 0.01 \
    --beta-dag 1.0 \
    --lambda-e 0.1 \
    --seed "$s" \
    --init-std 0.12 \
    --out-dir "output/temporal_pure_chemical_1000_k8_llmprior_seed${s}"
done
python3 eval_chemical_k8_1000.py
