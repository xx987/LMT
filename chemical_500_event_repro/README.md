# Reviewer Reproduction Guide (Chemical k=5, k=8, k=16)

This directory reproduces the chemical experiments for three settings:
- k=5
- k=8
- k=16

Each run script performs:
1. 10-seed training with `temporal_hawkes_pure_map.py` (seeds: 100, 200, ..., 1000)
2. Evaluation with the matching `eval_chemical_*_500.py` script

## Directory Contents

- `temporal_hawkes_pure_map.py`: training script (pure temporal Hawkes MAP)
- `eval_chemical_k5_500.py`: evaluation script for k=5
- `eval_chemical_k8_500.py`: evaluation script for k=8
- `eval_chemical_k16_500.py`: evaluation script for k=16
- `run_k5.sh`, `run_k8.sh`, `run_k16.sh`: one-command reproduction scripts
- `output/`: input event CSVs, oracle cluster CSVs, prior CSVs, and ground-truth graph CSVs
- `text_causal/`: local package dependency used by the training script

## Environment

Recommended:
- Python 3.10+
- `numpy`, `pandas`, `torch`

Install example:

```bash
pip install numpy pandas torch
```

## How To Run

From this directory, run one of:

```bash
bash run_k5.sh
bash run_k8.sh
bash run_k16.sh
```

## What You Will Get




