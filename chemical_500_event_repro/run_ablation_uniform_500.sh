
set -euo pipefail
cd "$(dirname "$0")"

for s in 100 200 300 400 500; do
  python3 temporal_hawkes_pure_map.py \
    --input output/sim_chemical_500_k16_cascadefriendly.csv \
    --cluster-csv output/sim_chemical500_k16_cascadefriendly_oracle_clusters.csv \
    --epochs 50 --lr 0.07 --beta-dag 1.0 --lambda-e 0.1 \
    --seed "$s" --init-std 0.2 \
    --q-default 0.5 \
    --out-dir "output/temporal_pure_chemical_k16_uniformprior_ablation_seed${s}"
done1

echo "Done k16 uniform-prior ablation. Eval: python3 eval_chemical_k16_ablation_uniform_500.py"
