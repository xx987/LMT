#!/usr/bin/env python3
"""Vocabulary k5-500 A_matrix runs vs GT: default τ=0.5 one-line summary (mean ± sd over seeds)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_VOCAB = Path(__file__).resolve().parent


def metric(pred_flat: np.ndarray, gt_flat: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    shd = float(fp + fn)
    fdr = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return shd, fdr, tpr


def main() -> None:
    p = argparse.ArgumentParser(
        description="Eval vocabulary k5-500 A_matrix vs GT (default: τ=0.5 one-line mean±sd)"
    )
    p.add_argument(
        "--gt",
        type=Path,
        default=_VOCAB / "data" / "sim_chemical_true_graph_k5_uniform_passset.csv",
        help="Ground-truth adjacency CSV (k×k).",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=_VOCAB / "output",
        help="Directory containing run_seed*/vocab_k5_500_A_matrix.csv",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500],
        help="Seeds (subfolders run_seed<seed>).",
    )
    p.add_argument(
        "--pred-name",
        default="vocab_k5_500_A_matrix.csv",
        help="Filename under each run_seed* directory.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Single threshold for pred >= τ (default mode). Ignored if --all-thresholds.",
    )
    p.add_argument(
        "--all-thresholds",
        action="store_true",
        help="Scan τ grid; write CSV only (no table on stdout unless --print-grid).",
    )
    p.add_argument(
        "--print-grid",
        action="store_true",
        help="With --all-thresholds: also print the full threshold table to stdout.",
    )
    p.add_argument("--threshold-min", type=float, default=0.05)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.05)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Where to write metrics CSV (default depends on --all-thresholds).",
    )
    args = p.parse_args()

    gt_path = Path(args.gt).resolve()
    gt = pd.read_csv(gt_path, index_col=0).to_numpy()
    gt = (gt > 0).astype(int)
    k = gt.shape[0]
    mask = ~np.eye(k, dtype=bool)
    gt_flat = gt[mask]

    root = Path(args.runs_root).resolve()
    pred_mats: list[np.ndarray] = []
    for s in args.seeds:
        pred_p = root / f"run_seed{s}" / args.pred_name
        if not pred_p.is_file():
            raise FileNotFoundError(f"Missing prediction: {pred_p}")
        a = pd.read_csv(pred_p, index_col=0).to_numpy(dtype=float)
        if a.shape != gt.shape:
            raise ValueError(f"{pred_p}: shape {a.shape} != GT {gt.shape}")
        pred_mats.append(a)

    n_seed = len(pred_mats)

    def stats_at_threshold(th: float) -> tuple[float, float, float, float, float, float]:
        shd_list, fdr_list, tpr_list = [], [], []
        for a in pred_mats:
            pred_flat = (a >= th).astype(int)[mask]
            shd, fdr, tpr = metric(pred_flat, gt_flat)
            shd_list.append(shd)
            fdr_list.append(fdr)
            tpr_list.append(tpr)
        shd_a = np.array(shd_list, dtype=float)
        fdr_a = np.array(fdr_list, dtype=float)
        tpr_a = np.array(tpr_list, dtype=float)
        shd_sd = float(np.std(shd_a, ddof=1)) if n_seed > 1 else 0.0
        fdr_sd = float(np.std(fdr_a, ddof=1)) if n_seed > 1 else 0.0
        tpr_sd = float(np.std(tpr_a, ddof=1)) if n_seed > 1 else 0.0
        return (
            float(np.mean(shd_a)),
            shd_sd,
            float(np.mean(fdr_a)),
            fdr_sd,
            float(np.mean(tpr_a)),
            tpr_sd,
        )

    if args.all_thresholds:
        lo, hi, step = float(args.threshold_min), float(args.threshold_max), float(args.threshold_step)
        if step <= 0 or hi < lo:
            raise ValueError("Need threshold_step > 0 and threshold_max >= threshold_min")
        thresholds = [
            round(lo + step * i, 10)
            for i in range(int(round((hi - lo) / step)) + 1)
            if lo + step * i <= hi + 1e-9
        ]
        rows = []
        for th in thresholds:
            sm, ss, fm, fs, tm, ts = stats_at_threshold(th)
            rows.append(
                {
                    "threshold": th,
                    "SHD_mean": sm,
                    "SHD_sd": ss,
                    "FDR_mean": fm,
                    "FDR_sd": fs,
                    "TPR_mean": tm,
                    "TPR_sd": ts,
                }
            )
        df = pd.DataFrame(rows).sort_values("threshold")
        out_csv = Path(args.out_csv).resolve() if args.out_csv else (_VOCAB / "output" / "vocab_k5_500_metrics_by_threshold.csv").resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if args.print_grid:
            hdr = (
                "threshold     "
                "SHD_mean  SHD_sd    "
                "FDR_mean  FDR_sd    "
                "TPR_mean  TPR_sd"
            )
            print(f"GT: {gt_path}")
            print(
                f"Runs: n_seeds={n_seed}, seeds={list(args.seeds)}, "
                f"threshold grid [{args.threshold_min}, {args.threshold_max}] step {args.threshold_step}"
            )
            print("Off-diagonal only; binarize as pred >= threshold.\n")
            print(hdr)
            print("-" * len(hdr))
            for _, r in df.iterrows():
                th = float(r["threshold"])
                print(
                    f"{th:11.3f}   "
                    f"{r['SHD_mean']:6.2f}  {r['SHD_sd']:6.2f}    "
                    f"{r['FDR_mean']:8.4f}  {r['FDR_sd']:8.4f}    "
                    f"{r['TPR_mean']:8.4f}  {r['TPR_sd']:8.4f}"
                )
            print()
        print(f"Wrote: {out_csv}")
        return

    th = float(args.threshold)
    shd_m, shd_s, fdr_m, fdr_s, tpr_m, tpr_s = stats_at_threshold(th)
    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else (_VOCAB / "output" / "vocab_k5_500_metrics.csv").resolve()
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "threshold": th,
                "SHD_mean": shd_m,
                "SHD_sd": shd_s,
                "FDR_mean": fdr_m,
                "FDR_sd": fdr_s,
                "TPR_mean": tpr_m,
                "TPR_sd": tpr_s,
            }
        ]
    ).to_csv(out_csv, index=False)

    print("Result：")
    print(
        f"SHD={shd_m:.3f} ± {shd_s:.3f} | "
        f"FDR={fdr_m:.3f} ± {fdr_s:.3f} | "
        f"TPR={tpr_m:.3f} ± {tpr_s:.3f}"
    )
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
