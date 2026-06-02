#!/usr/bin/env bash
set -euo pipefail
#
# Generate chemical-style sim bundles with N=100 events (same pipeline as chemical_50_event_repro/gen_50_data.sh).
# Run from anywhere: bash chemical_100_event_repro/gen_100_data.sh
#
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="$REPO/chemical_100_event_repro/output"
N=100
SEED="${SEED:-42}"

mkdir -p "$OUT"

echo "=== k=5 (paired episodes), N=$N ==="
python3 generate_chemical_k5_uniform_passset.py \
  --output-csv "$OUT/sim_chemical_100_k5_uniform_passset.csv" \
  --output-graph "$OUT/sim_chemical_true_graph_k5_uniform_passset.csv" \
  --output-links "$OUT/sim_chemical_links_k5_uniform_passset.csv" \
  --output-oracle-clusters "$OUT/sim_chemical100_k5_uniform_passset_oracle_clusters.csv" \
  --n-events "$N" \
  --seed "$SEED"

echo "=== k=8 (episode + extra roots), N=$N ==="
python3 generate_chemical_uniform_passset.py \
  --output-csv "$OUT/sim_chemical_100_k8_uniform_passset.csv" \
  --output-graph "$OUT/sim_chemical_true_graph_k8_uniform_passset.csv" \
  --output-links "$OUT/sim_chemical_links_k8_uniform_passset.csv" \
  --output-oracle-clusters "$OUT/sim_chemical100_k8_uniform_passset_oracle_clusters.csv" \
  --n-events "$N" \
  --seed "$SEED"

echo "=== k=8 CASCADE compat CSV (event_id + Alarm Text + time only) ==="
python3 <<'PY'
import pandas as pd
from pathlib import Path

out = Path("chemical_100_event_repro/output")
full = pd.read_csv(out / "sim_chemical_100_k8_uniform_passset.csv")
compact = pd.DataFrame(
    {
        "event_id": [f"e{i}" for i in range(len(full))],
        "Alarm Text": [f"alarm_{int(z)}" for z in full["cluster_id_true"]],
        "time": full["time"].astype(float),
    }
)
compact.to_csv(out / "sim_chemical_100_k8_uniform_passset_compat.csv", index=False)
print("wrote", out / "sim_chemical_100_k8_uniform_passset_compat.csv", f"({len(compact)} rows)")
PY

echo "=== k=16 (chemical_plant_16, cascade-friendly), N=$N ==="
python3 simulate_engineering_events.py \
  --output-csv "$OUT/sim_chemical_100_k16_cascadefriendly.csv" \
  --output-graph "$OUT/sim_chemical_true_graph_k16_cascadefriendly.csv" \
  --output-links "$OUT/sim_chemical_links_k16_cascadefriendly.csv" \
  --n-events "$N" \
  --seed "$SEED" \
  --scenario chemical_plant_16 \
  --cascade-friendly

python3 <<PY
import pandas as pd
from pathlib import Path
out = Path("$OUT")
df = pd.read_csv(out / "sim_chemical_100_k16_cascadefriendly.csv")
df[["cluster_id_true"]].rename(columns={"cluster_id_true": "cluster_id"}).to_csv(
    out / "sim_chemical100_k16_cascadefriendly_oracle_clusters.csv", index=False
)
print("wrote oracle clusters:", out / "sim_chemical100_k16_cascadefriendly_oracle_clusters.csv")
PY

echo "=== copy cluster priors (K fixed; reuse 500-era matrices) ==="
for f in q_cluster_prior_k5.csv q_cluster_prior_chemical_k8_v1.csv q_cluster_prior_chemical_k16_v1.csv; do
  SRC="$REPO/baselines/casc/chemical_500_data/output/$f"
  if [[ -f "$SRC" ]]; then
    cp "$SRC" "$OUT/$f"
    echo "  copied $f"
  else
    echo "  SKIP missing $SRC"
  fi
done

echo "Done. Output under $OUT"
