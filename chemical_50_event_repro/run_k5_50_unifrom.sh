#!/usr/bin/env bash
set -euo pipefail

# Run 5 seeds with uniform prior for k=5 chemical 50-event setup.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for s in 100 200 300 400 500; do
  python3 temporal_hawkes_pure_map.py \
    --input output/sim_chemical_50_k5_uniform_passset.csv \
    --cluster-csv output/sim_chemical50_k5_uniform_passset_oracle_clusters.csv \
    --q-default 0.5 \
    --seed "$s" \
    --init-std 0.65 \
    --epochs 50 \
    --lr 0.03 \
    --beta-dag 1.0 \
    --lambda-prior 0.2 \
    --lambda-e 0.1 \
    --out-dir "output/temporal_pure_chemical_k5_n50_uniform_prior_s${s}"
done

python3 - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path

seeds = [100, 200, 300, 400, 500]
out_dirs = [
    Path(f"output/temporal_pure_chemical_k5_n50_uniform_prior_s{s}")
    for s in seeds
]
gt_path = Path("output/sim_chemical_true_graph_k5_uniform_passset.csv")
tau = 0.200

G = pd.read_csv(gt_path, index_col=0).to_numpy(dtype=int)
K = G.shape[0]
mask = ~np.eye(K, dtype=bool)
G_off = G.copy()
np.fill_diagonal(G_off, 0)
g = G_off[mask]

def metrics_for_A(A: np.ndarray, tau: float):
    pred = (A > tau).astype(int)
    np.fill_diagonal(pred, 0)
    p = pred[mask]
    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    shd = float(fp + fn)
    return shd, fdr, tpr

def load_A(d: Path) -> np.ndarray:
    p = d / "A_matrix.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p, index_col=0).to_numpy(dtype=float)

As = [load_A(d) for d in out_dirs]
for A in As:
    if A.shape != (K, K):
        raise SystemExit(f"shape mismatch G {G.shape} vs A {A.shape}")

rows = np.array([metrics_for_A(A, float(tau)) for A in As], dtype=float)
m = rows.mean(axis=0)
s = rows.std(axis=0, ddof=1)
print(f"threshold={tau:.3f}")
print(f"SHD={m[0]:.1f} ± {s[0]:.1f}  |  FDR={m[1]:.2f} ± {s[1]:.2f}  |  TPR={m[2]:.2f} ± {s[2]:.2f}")
PY
