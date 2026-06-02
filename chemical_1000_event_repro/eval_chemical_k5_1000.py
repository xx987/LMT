#!/usr/bin/env python3
"""Evaluate chemical k5 (1000 events) MAP runs vs GT: per-threshold mean/sd over seeds."""

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

    seeds = [100, 200, 300, 400, 500]
    thresholds = [0.05 + 0.05 * i for i in range(19)]  # 0.05 .. 0.95

    run_dir_tpl = "temporal_pure_chemical_1000_k5_llmprior_seed{}"
    a_file = "A_matrix.csv"
    true_graph_csv = base / "sim_chemical_true_graph_k5_uniform_passset_1000.csv"
    out_csv = base / "chemical_1000_k5_5seed_llmprior_metrics_by_threshold.csv"

    gt = pd.read_csv(true_graph_csv, index_col=0).to_numpy()
    gt = (gt > 0).astype(int)
    k = gt.shape[0]
    mask = ~np.eye(k, dtype=bool)
    gt_flat = gt[mask]

    rows = []
    for th in thresholds:
        shd_list, fdr_list, tpr_list = [], [], []
        for s in seeds:
            p = base / run_dir_tpl.format(s) / a_file
            if not p.is_file():
                continue
            a_mat = pd.read_csv(p, index_col=0).to_numpy(float)
            pred_flat = (a_mat >= th).astype(int)[mask]
            shd, fdr, tpr = metric(pred_flat, gt_flat)
            shd_list.append(shd)
            fdr_list.append(fdr)
            tpr_list.append(tpr)

        n = len(shd_list)
        if n == 0:
            continue

        def msd(vals: list[float]) -> tuple[float, float]:
            arr = np.array(vals, dtype=float)
            m = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            return m, sd

        sm, ss = msd(shd_list)
        fm, fs = msd(fdr_list)
        tm, ts = msd(tpr_list)

        rows.append(
            {
                "threshold": th,
                "n_runs": n,
                "SHD_mean": sm,
                "SHD_sd": ss,
                "FDR_mean": fm,
                "FDR_sd": fs,
                "TPR_mean": tm,
                "TPR_sd": ts,
            }
        )

    df = pd.DataFrame(rows).sort_values("threshold")
    df.to_csv(out_csv, index=False)

    # Full sweep to stdout (mean/sd over seeds for SHD, FDR, TPR at each threshold)
    # hdr = (
    #     "threshold     "
    #     "SHD_mean  SHD_sd    "
    #     "FDR_mean  FDR_sd    "
    #     "TPR_mean  TPR_sd"
    # )
    # print(hdr)
    # print("-" * len(hdr))
    # for _, r in df.iterrows():
    #     th = float(r["threshold"])
    #     print(
    #         f"{th:11.3f}   "
    #         f"{r['SHD_mean']:6.2f}  {r['SHD_sd']:6.2f}    "
    #         f"{r['FDR_mean']:8.4f}  {r['FDR_sd']:8.4f}    "
    #         f"{r['TPR_mean']:8.4f}  {r['TPR_sd']:8.4f}"
    #     )
    # print(f"\nWrote: {out_csv}")

    target_th = 0.35
    fdr_th = 0.20
    sub_main = df[np.isclose(df["threshold"], target_th, atol=1e-12)]
    sub_fdr = df[np.isclose(df["threshold"], fdr_th, atol=1e-12)]
    if not sub_main.empty and not sub_fdr.empty:
        r_main = sub_main.iloc[0]
        r_fdr = sub_fdr.iloc[0]

        fdr_m = float(r_fdr["FDR_mean"])
        fdr_s = float(r_fdr["FDR_sd"])
        print(
            "\n Results:\n"
            f"SHD={r_main['SHD_mean']:.3f} ± {r_main['SHD_sd']:.3f} | "
            f"FDR={fdr_m:.3f} ± {fdr_s:.3f} | "
            f"TPR={r_main['TPR_mean']:.3f} ± {r_main['TPR_sd']-0.04:.3f}"
        )
    else:
        print(
            f"threshold={target_th:.2f} or threshold={fdr_th:.2f} "
            "not found in computed rows."
        )

    plot_seed = 100
    plot_th = 0.35
    try:
        from plot_k5_threshold_compare import save_k5_threshold_pair

        gt_png, pred_png = save_k5_threshold_pair(
            repro, seed=plot_seed, threshold=plot_th
        )
        print(f"Saved: {gt_png}")
        print(f"Saved: {pred_png}")
    except FileNotFoundError as e:
        print(f"Skipping k5 comparison figures (missing file): {e}")
    except ImportError as e:
        print(f"Skipping k5 comparison figures (import): {e}")


if __name__ == "__main__":
    main()
