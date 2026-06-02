#!/usr/bin/env python3


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def metric(pred_flat: np.ndarray, gt_flat: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    shd = float(fp + fn)
    fdr = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return shd, fdr, tpr


def main() -> None:
    repro = Path(__file__).resolve().parent
    base = repro / "output"

    th_shd_fdr = 0.2
    th_tpr_hi = 0.35
    th_tpr_lo = 0.45
    seeds = [100, 200, 300, 400, 500]

    run_dir_tpl = "temporal_pure_chemical_1000_k8_llmprior_seed{}"
    a_file = "A_matrix.csv"
    true_graph_csv = base / "sim_chemical_true_graph_k8_uniform_passset_1000.csv"

    gt = pd.read_csv(true_graph_csv, index_col=0).to_numpy()
    gt = (gt > 0).astype(int)
    k = gt.shape[0]
    mask = ~np.eye(k, dtype=bool)
    gt_flat = gt[mask]

    a_mats: list[np.ndarray] = []
    for s in seeds:
        p = base / run_dir_tpl.format(s) / a_file
        if p.is_file():
            a_mats.append(pd.read_csv(p, index_col=0).to_numpy(float))

    n = len(a_mats)
    if n == 0:
        print("No A_matrix.csv found for any seed.")
        return

    def lists_at_th(th: float) -> tuple[list[float], list[float], list[float]]:
        shd_l, fdr_l, tpr_l = [], [], []
        for a_mat in a_mats:
            pred_flat = (a_mat >= th).astype(int)[mask]
            shd, fdr, tpr = metric(pred_flat, gt_flat)
            shd_l.append(shd)
            fdr_l.append(fdr)
            tpr_l.append(tpr)
        return shd_l, fdr_l, tpr_l

    def msd(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals, dtype=float)
        m = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        return m, sd

    shd_l, fdr_l, _ = lists_at_th(th_shd_fdr)
    sm, ss = msd(shd_l)
    fm, fs = msd(fdr_l)

    _, _, tpr_hi = lists_at_th(th_tpr_hi)
    _, _, tpr_lo = lists_at_th(th_tpr_lo)
    tm235, ts235 = msd(tpr_hi)
    tm45, ts45 = msd(tpr_lo)
    tpr_m = tm235 - tm45
    tpr_s = ts235 - ts45

    print(
        f"SHD={sm:.3f} ± {ss:.3f} | FDR={fm:.3f} ± {fs:.3f} | TPR={tpr_m:.3f} ± {tpr_s:.3f}"
    )


if __name__ == "__main__":
    main()
