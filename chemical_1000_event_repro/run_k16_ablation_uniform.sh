#!/usr/bin/env bash
# k16 uniform-prior ablation: same training setup as run_k16.sh, no --q-csv (uses --q-default).
set -euo pipefail
cd "$(dirname "$0")"
for s in 100 200 300 400 500; do
  python3 temporal_hawkes_pure_map.py \
    --input output/sim_chemical_1000_k16_uniform_passset.csv \
    --cluster-csv output/sim_chemical1000_k16_uniform_passset_oracle_clusters.csv \
    --epochs 50 \
    --lr 0.08 \
    --beta-dag 1.0 \
    --lambda-e 0.1 \
    --seed "$s" \
    --init-std 0.05 \
    --q-default 0.5 \
    --out-dir "output/temporal_pure_chemical_1000_k16_uniformprior_ablation_seed${s}"
done
python3 eval_chemical_k16_ablation_uniform_1000.py
