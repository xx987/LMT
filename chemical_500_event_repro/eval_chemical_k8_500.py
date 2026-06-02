#!/usr/bin/env python3
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
    seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    thresholds = [i / 100 for i in range(5, 100, 5)]
    target_threshold = 0.25

    run_dir_tpl = "output/temporal_pure_chemical_k8_llmprior_seed{}"
    a_file = "A_matrix.csv"
    true_graph_csv = "output/sim_chemical_true_graph_uniform_easy.csv"
    out_csv = "output/chemical_k8_10seed_llmprior_metrics_compact.csv"
    bootstrap_rounds = 1000

    gt = pd.read_csv(true_graph_csv, index_col=0).to_numpy()
    gt = (gt > 0).astype(int)
    k = gt.shape[0]
    mask = ~np.eye(k, dtype=bool)
    gt_flat = gt[mask]
    edge_count = gt_flat.size
    rng = np.random.default_rng(2026)

    rows = []
    for th in thresholds:
        preds = []
        shd_run, fdr_run, tpr_run = [], [], []

        for s in seeds:
            p = Path(run_dir_tpl.format(s)) / a_file
            if not p.exists():
                continue
            a_mat = pd.read_csv(p, index_col=0).to_numpy()
            pred_flat = (a_mat >= th).astype(int)[mask]
            preds.append(pred_flat)
            shd, fdr, tpr = metric(pred_flat, gt_flat)
            shd_run.append(shd)
            fdr_run.append(fdr)
            tpr_run.append(tpr)

        run_count = len(preds)
        if run_count == 0:
            continue
        preds = np.stack(preds, axis=0)  # [R, E]

        shd_bs, fdr_bs, tpr_bs = [], [], []
        for _ in range(bootstrap_rounds):
            ridx = rng.integers(0, run_count, size=run_count)
            eidx = rng.integers(0, edge_count, size=edge_count)
            pred_boot = preds[ridx][:, eidx]
            gt_boot = gt_flat[eidx]

            shd_tmp, fdr_tmp, tpr_tmp = [], [], []
            for r in range(run_count):
                shd, fdr, tpr = metric(pred_boot[r], gt_boot)
                shd_tmp.append(shd)
                fdr_tmp.append(fdr)
                tpr_tmp.append(tpr)
            shd_bs.append(np.mean(shd_tmp))
            fdr_bs.append(np.mean(fdr_tmp))
            tpr_bs.append(np.mean(tpr_tmp))

        rows.append(
            {
                "threshold": th,
                "n_runs": run_count,
                "SHD_mean": float(np.mean(shd_run)),
                "SHD_sd": float(np.std(shd_bs, ddof=1)),
                "FDR_mean": float(np.mean(fdr_run)),
                "FDR_sd": float(np.std(fdr_bs, ddof=1)),
                "TPR_mean": float(np.mean(tpr_run)),
                "TPR_sd": float(np.std(tpr_bs, ddof=1)),
            }
        )

    df = pd.DataFrame(rows).sort_values("threshold")
    df.to_csv(out_csv, index=False)

    hit = df[np.isclose(df["threshold"], target_threshold, atol=1e-12)]
    if hit.empty:
        print(f"threshold={target_threshold:.2f} not found; saved full table to {out_csv}")
        return

    r = hit.iloc[0]
    print(
        f"SHD={r['SHD_mean']:.4f}±{r['SHD_sd']:.4f} | "
        f"FDR={r['FDR_mean']:.4f}±{r['FDR_sd']:.4f} | "
        f"TPR={r['TPR_mean']:.4f}±{r['TPR_sd']:.4f}"
    )


if __name__ == "__main__":
    main()
