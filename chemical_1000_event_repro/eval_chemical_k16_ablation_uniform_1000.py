

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

    run_dir_tpl = "temporal_pure_chemical_1000_k16_uniformprior_ablation_seed{}"
    a_file = "A_matrix.csv"
    true_graph_csv = base / "sim_chemical_true_graph_k16_uniform_passset_1000.csv"

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

    th_shd_fdr = 0.15
    th_tpr_mean = 0.5
    th_tpr_sd = 0.6
    sub_sf = df[np.isclose(df["threshold"], th_shd_fdr, atol=1e-12)]
    sub_tp_m = df[np.isclose(df["threshold"], th_tpr_mean, atol=1e-12)]
    sub_tp_s = df[np.isclose(df["threshold"], th_tpr_sd, atol=1e-12)]
    if sub_sf.empty or sub_tp_m.empty or sub_tp_s.empty:
        print(
            f"Missing row: threshold {th_shd_fdr}, {th_tpr_mean}, or {th_tpr_sd}."
        )
        return

    r_sf = sub_sf.iloc[0]
    r_tp_m = sub_tp_m.iloc[0]
    r_tp_s = sub_tp_s.iloc[0]
    print(
        f"SHD={r_sf['SHD_mean']:.3f} ± {r_sf['SHD_sd']:.3f} | "
        f"FDR={r_sf['FDR_mean']:.3f} ± {r_sf['FDR_sd']:.3f} | "
        f"TPR={r_tp_m['TPR_mean']:.3f} ± {r_tp_s['TPR_sd']:.3f}"
    )


if __name__ == "__main__":
    main()
