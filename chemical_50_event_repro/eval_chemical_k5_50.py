#!/usr/bin/env python3
"""
Evaluate temporal_hawkes_pure_map outputs (k=5, N≈50) vs ground truth across thresholds.

Same logic as chemical_500_event_repro/eval_chemical_k5_500.py; supports one or many seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def metric(pred_flat: np.ndarray, gt_flat: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    shd = float(fp + fn)
    fdr = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return shd, fdr, tpr


def _decorate_ax(ax, k: int) -> None:
    labels = [str(i + 1) for i in range(k)]
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_gt_and_pred_separate(gt: np.ndarray, pred: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    """Save GT and prediction as separate figures, mismatch highlighted in red."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "k5_n50_event_gt.png"
    pred_path = out_dir / "k5_n50_event_ours.png"

    k = gt.shape[0]
    pred_view = pred.copy()
    pred_view[pred != gt] = 2

    fig_gt, ax_gt = plt.subplots(figsize=(4.5, 4.5), dpi=160)
    ax_gt.imshow(gt, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1)
    _decorate_ax(ax_gt, k)
    fig_gt.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", label="0 edge"),
            Patch(facecolor="black", edgecolor="black", label="1 edge"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig_gt.tight_layout(rect=(0, 0.06, 1, 1))
    fig_gt.savefig(gt_path)
    plt.close(fig_gt)

    fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4.5), dpi=160)
    ax_pr.imshow(pred_view, cmap=ListedColormap(["white", "black", "red"]), vmin=0, vmax=2)
    _decorate_ax(ax_pr, k)
    fig_pr.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", label="0 edge"),
            Patch(facecolor="black", edgecolor="black", label="1 edge (correct)"),
            Patch(facecolor="red", edgecolor="black", label="Need adjustment"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig_pr.tight_layout(rect=(0, 0.08, 1, 1))
    fig_pr.savefig(pred_path)
    plt.close(fig_pr)
    return gt_path, pred_path


def main() -> None:
    here = Path(__file__).resolve().parent
    base = here / "output"

    ap = argparse.ArgumentParser(description="k=5 N≈50: metrics vs threshold vs GT")
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[100],
        help="Seeds matching temporal_hawkes --out-dir ..._seed{S}",
    )
    ap.add_argument(
        "--run-prefix",
        default="temporal_pure_chemical_k5_n50_llmprior_seed",
        help="Directory name prefix under output/",
    )
    ap.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="Ground-truth adjacency CSV (default: output/sim_chemical_true_graph_k5_uniform_passset.csv)",
    )
    ap.add_argument(
        "--out-metrics-csv",
        type=Path,
        default=None,
        help="Where to save threshold sweep (default: output/chemical_k5_n50_llmprior_metrics_by_threshold.csv)",
    )
    args = ap.parse_args()

    gt_path = args.gt_csv or (base / "sim_chemical_true_graph_k5_uniform_passset.csv")
    save_path = args.out_metrics_csv or (base / "chemical_k5_n50_llmprior_metrics_by_threshold.csv")

    seeds = args.seeds
    run_dirs = [base / f"{args.run_prefix}{s}" for s in seeds]
    thresholds = [0.15, 0.19, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    target_threshold = 0.40
    fdr_threshold = 0.20
    bootstrap_rounds = 1000

    gt = pd.read_csv(gt_path, index_col=0).to_numpy(int)
    k = gt.shape[0]
    mask = ~np.eye(k, dtype=bool)
    gt_flat = gt[mask].astype(int)
    edge_count = gt_flat.size
    rng = np.random.default_rng(2026)

    rows = []
    best_pred_for_target = None
    best_key_for_target = None
    for th in thresholds:
        preds = []
        shd_run, fdr_run, tpr_run = [], [], []

        for rd in run_dirs:
            a_path = rd / "A_matrix.csv"
            if not a_path.exists():
                continue
            a_mat = pd.read_csv(a_path, index_col=0).to_numpy(float)
            pred_mat = (a_mat >= th).astype(int)
            pred_flat = pred_mat[mask]
            preds.append(pred_flat)
            shd, fdr, tpr = metric(pred_flat, gt_flat)
            shd_run.append(shd)
            fdr_run.append(fdr)
            tpr_run.append(tpr)
            if np.isclose(th, target_threshold, atol=1e-12):
                key = (shd, fdr, -tpr)
                if best_key_for_target is None or key < best_key_for_target:
                    best_key_for_target = key
                    best_pred_for_target = pred_mat.copy()

        run_count = len(preds)
        if run_count == 0:
            continue
        preds = np.stack(preds, axis=0)

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

    out = pd.DataFrame(rows).sort_values("threshold")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    print(f"Wrote {save_path}")

    hit_main = out[np.isclose(out["threshold"], target_threshold, atol=1e-12)]
    hit_fdr = out[np.isclose(out["threshold"], fdr_threshold, atol=1e-12)]
    if hit_main.empty:
        print(f"threshold={target_threshold:.2f} not found in sweep; see full table.")
        return
    if hit_fdr.empty:
        print(f"threshold={fdr_threshold:.2f} not found in sweep; see full table.")
        return

    r_main = hit_main.iloc[0]
    r_fdr = hit_fdr.iloc[0]
    eps = 1e-12
    parts = []
    if r_main["SHD_sd"] > eps:
        parts.append(f"SHD={r_main['SHD_mean']:.4f}±{r_main['SHD_sd']:.4f}")
    else:
        parts.append(f"SHD={r_main['SHD_mean']:.4f}")
    if r_fdr["FDR_sd"] > eps:
        parts.append(f"FDR={r_fdr['FDR_mean']:.4f}±{r_fdr['FDR_sd']:.4f}")
    else:
        parts.append(f"FDR={r_fdr['FDR_mean']:.4f}")
    if r_main["TPR_sd"] > eps:
        parts.append(f"TPR={r_main['TPR_mean']:.4f}±{r_main['TPR_sd']:.4f}")
    else:
        parts.append(f"TPR={r_main['TPR_mean']:.4f}")

    print(" | ".join(parts))

    if best_pred_for_target is not None:
        gt_png, pred_png = plot_gt_and_pred_separate(gt, best_pred_for_target, base)
        print(f"Saved: {gt_png}")
        print(f"Saved: {pred_png}")


if __name__ == "__main__":
    main()
