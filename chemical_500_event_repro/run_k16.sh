#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for s in 100 200 300 400 500 600 700 800 900 1000; do
  python3 temporal_hawkes_pure_map.py \
    --input output/sim_chemical_500_k16_cascadefriendly.csv \
    --cluster-csv output/sim_chemical500_k16_cascadefriendly_oracle_clusters.csv \
    --q-csv output/q_cluster_prior_chemical_k16_v1.csv \
    --epochs 200 --lr 0.01 --beta-dag 1.0 --lambda-e 0.1 \
    --seed "$s" --init-std 0.02 \
    --out-dir "output/temporal_pure_chemical_k16_map_llmprior_seed${s}"
done
python3 eval_chemical_k16_500.py
