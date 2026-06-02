"""Bootstrap uncertainty for temporal MAP evals (no retraining)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_sd_across_seeds(
    shd_run: np.ndarray,
    fdr_run: np.ndarray,
    tpr_run: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    """Resample runs with replacement; SD of bootstrap replicate means (one scalar per metric)."""
    n = int(shd_run.shape[0])
    if n < 1 or n_boot < 2:
        return 0.0, 0.0, 0.0
    shd_bs: list[float] = []
    fdr_bs: list[float] = []
    tpr_bs: list[float] = []
    for _ in range(n_boot):
        j = rng.integers(0, n, size=n)
        shd_bs.append(float(np.mean(shd_run[j])))
        fdr_bs.append(float(np.mean(fdr_run[j])))
        tpr_bs.append(float(np.mean(tpr_run[j])))
    return (
        float(np.std(shd_bs, ddof=1)),
        float(np.std(fdr_bs, ddof=1)),
        float(np.std(tpr_bs, ddof=1)),
    )


def bootstrap_sd_run_edge(
    preds: np.ndarray,
    gt_flat: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    metric_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Same construction as chemical_500 eval: resample runs and edges with replacement."""
    run_count, edge_count = int(preds.shape[0]), int(preds.shape[1])
    if run_count < 1 or edge_count < 1 or n_boot < 2:
        return 0.0, 0.0, 0.0
    shd_bs: list[float] = []
    fdr_bs: list[float] = []
    tpr_bs: list[float] = []
    for _ in range(n_boot):
        ridx = rng.integers(0, run_count, size=run_count)
        eidx = rng.integers(0, edge_count, size=edge_count)
        pred_boot = preds[ridx][:, eidx]
        gt_boot = gt_flat[eidx]
        shd_tmp: list[float] = []
        fdr_tmp: list[float] = []
        tpr_tmp: list[float] = []
        for r in range(run_count):
            shd, fdr, tpr = metric_fn(pred_boot[r], gt_boot)
            shd_tmp.append(shd)
            fdr_tmp.append(fdr)
            tpr_tmp.append(tpr)
        shd_bs.append(float(np.mean(shd_tmp)))
        fdr_bs.append(float(np.mean(fdr_tmp)))
        tpr_bs.append(float(np.mean(tpr_tmp)))
    return (
        float(np.std(shd_bs, ddof=1)),
        float(np.std(fdr_bs, ddof=1)),
        float(np.std(tpr_bs, ddof=1)),
    )
