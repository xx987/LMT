# LMT: A Bayesian Framework for Causal Discovery from Textual Alarm Records in Manufacturing Systems



**Working directory:** run all commands from `LMT/` unless a script `cd`s into its own folder.

---

## Examples of Reproduction

### \(K=5\), \(N=50\)

```bash
bash chemical_50_event_repro/run_k5_50.sh
```

### \(K=5\), \(N=500\)

```bash
bash chemical_500_event_repro/run_k5.sh
```

**Outputs:** `chemical_500_event_repro/output/k5_500_event_gt.png`, `k5_500_event_ours.png`, plus metrics from `eval_chemical_k5_500.py`.

---

### \(K=5\), \(N=1000\)

```bash
bash chemical_1000_event_repro/run_k5.sh
```

**Outputs:** metrics + figures via `eval_chemical_k5_1000.py` under `chemical_1000_event_repro/output/`.

---

## From scratch: regenerate data + your own LLM prior

Use this if you do **not** want bundled `output/` files and will call **OpenAI** yourself. All steps below are for **\(K=5\), \(N=50\)**.

**Prerequisites**

```bash
cd LMT
pip install torch pandas numpy matplotlib scikit-learn sentence-transformers openai
export OPENAI_API_KEY="sk-..."          # required for real LLM steps
export OPENAI_MODEL="gpt-4o-mini"      # optional; default in code
```

### Step 1 — Generate synthetic events + ground-truth graph

```bash
bash chemical_50_event_repro/gen_50_data.sh
```

### Step 2 — Build semantic prior `q` with OpenAI (role distillation → cluster matrix)

```bash
cd chemical_50_event_repro


python3 llm_role_distillation_demo.py --use-openai --max-events 50 \
  --input output/sim_chemical_50_k5_uniform_passset.csv \
  --cluster-csv output/sim_chemical50_k5_uniform_passset_oracle_clusters.csv \
  --llm-cache output/llm_role_scores_cache_n50.csv


python3 ../chemical_100_event_repro/build_cluster_prior_from_role_distill.py \
  --input output/sim_chemical_50_k5_uniform_passset.csv \
  --cluster-csv output/sim_chemical50_k5_uniform_passset_oracle_clusters.csv \
  --checkpoint output/llm_role_distill_demo.pt \
  --out-csv output/q_cluster_prior_k5_llm_n50.csv \
  --out-npy output/q_cluster_prior_k5_llm_n50.npy
```




