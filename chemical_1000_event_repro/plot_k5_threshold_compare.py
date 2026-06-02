#!/usr/bin/env python3
"""Plot GT and thresholded prediction (k5, 1000 events) as two separate figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def load_binary_matrix(csv_path: Path, threshold: float | None = None) -> np.ndarray:
    mat = pd.read_csv(csv_path, index_col=0).to_numpy(dtype=float)
    if threshold is None:
        return (mat > 0).astype(int)
    return (mat >= threshold).astype(int)


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


def save_k5_threshold_pair(
    repro: Path,
    *,
    seed: int,
    threshold: float,
    gt_csv: Path | None = None,
    pred_csv: Path | None = None,
    gt_out: Path | None = None,
    pred_out: Path | None = None,
) -> tuple[Path, Path]:
    """Write ground-truth and prediction (with mismatch in red) as two PNGs."""
    if gt_csv is None:
        gt_csv = repro / "output/sim_chemical_true_graph_k5_uniform_passset_1000.csv"
    if pred_csv is None:
        pred_csv = (
            repro / f"output/temporal_pure_chemical_1000_k5_llmprior_seed{seed}/A_matrix.csv"
        )

    gt_csv = Path(gt_csv).resolve()
    pred_csv = Path(pred_csv).resolve()
    if not gt_csv.is_file():
        raise FileNotFoundError(f"GT csv not found: {gt_csv}")
    if not pred_csv.is_file():
        raise FileNotFoundError(f"Pred csv not found: {pred_csv}")

    gt = load_binary_matrix(gt_csv, threshold=None)
    pred = load_binary_matrix(pred_csv, threshold=threshold)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")

    pred_view = pred.copy()
    mismatch = pred != gt
    pred_view[mismatch] = 2
    k = gt.shape[0]
    cmap_pred = ListedColormap(["white", "black", "red"])

    if gt_out is None:
        gt_out = repro / "output/k5_1000_event_gt.png"
    if pred_out is None:
        pred_out = repro / "output/k5_1000_event_ours.png"
    gt_out = Path(gt_out).resolve()
    pred_out = Path(pred_out).resolve()
    gt_out.parent.mkdir(parents=True, exist_ok=True)

    # Ground truth only
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
    fig_gt.savefig(gt_out)
    plt.close(fig_gt)

    # Prediction with mismatch highlighted
    fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4.5), dpi=160)
    ax_pr.imshow(pred_view, cmap=cmap_pred, vmin=0, vmax=2)
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
    fig_pr.savefig(pred_out)
    plt.close(fig_pr)

    return gt_out, pred_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=500, help="Seed folder to visualize.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Binarization threshold.")
    parser.add_argument(
        "--gt-csv",
        default=None,
        help="Ground-truth adjacency CSV (default: output/...k5...1000.csv).",
    )
    parser.add_argument(
        "--pred-csv",
        default=None,
        help="Predicted A matrix CSV. If omitted, derive from --seed.",
    )
    parser.add_argument("--gt-png", default=None, help="Output path for GT figure.")
    parser.add_argument("--pred-png", default=None, help="Output path for prediction figure.")
    args = parser.parse_args()

    repro = Path(__file__).resolve().parent
    gt_csv = (repro / args.gt_csv).resolve() if args.gt_csv else None
    pred_csv = (repro / args.pred_csv).resolve() if args.pred_csv else None
    gt_png = (repro / args.gt_png).resolve() if args.gt_png else None
    pred_png = (repro / args.pred_png).resolve() if args.pred_png else None

    gt_path, pred_path = save_k5_threshold_pair(
        repro,
        seed=args.seed,
        threshold=args.threshold,
        gt_csv=gt_csv,
        pred_csv=pred_csv,
        gt_out=gt_png,
        pred_out=pred_png,
    )
    print(f"Saved: {gt_path}")
    print(f"Saved: {pred_path}")


if __name__ == "__main__":
    main()
