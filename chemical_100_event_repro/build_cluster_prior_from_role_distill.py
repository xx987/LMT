#!/usr/bin/env python3
"""
Build cluster-level semantic prior from a trained role-distillation checkpoint.

Pipeline:
1) Load events, clusters, and trained Wc/We checkpoint.
2) Compute event-pair score r_ij = sigmoid(<h_i^(c), h_j^(e)>), i != j.
3) Aggregate r_ij by cluster pair (a, b) -> q_ab.
4) Save KxK prior matrix as CSV and NPY for downstream --q-npy usage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class RoleDistillModel(nn.Module):
    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.Wc = nn.Linear(d_in, d_h)
        self.We = nn.Linear(d_in, d_h)
        self.head_c = nn.Linear(d_h, 1)
        self.head_e = nn.Linear(d_h, 1)

    def forward(self, g: torch.Tensor):
        hc = self.Wc(g)
        he = self.We(g)
        c_hat = torch.sigmoid(self.head_c(hc)).squeeze(-1)
        e_hat = torch.sigmoid(self.head_e(he)).squeeze(-1)
        return hc, he, c_hat, e_hat


def aggregate_values(vals: np.ndarray, method: str, trim_ratio: float, top_p: float) -> float:
    if vals.size == 0:
        return np.nan
    if method == "mean":
        return float(np.mean(vals))
    if method == "median":
        return float(np.median(vals))
    if method == "trimmed_mean":
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError("--trim-ratio must be in [0, 0.5)")
        if vals.size < 4 or trim_ratio == 0.0:
            return float(np.mean(vals))
        lo = np.quantile(vals, trim_ratio)
        hi = np.quantile(vals, 1.0 - trim_ratio)
        kept = vals[(vals >= lo) & (vals <= hi)]
        if kept.size == 0:
            return float(np.mean(vals))
        return float(np.mean(kept))
    if method == "top_p_mean":
        if not (0.0 < top_p <= 1.0):
            raise ValueError("--top-p must be in (0, 1]")
        n_keep = max(1, int(np.ceil(vals.size * top_p)))
        sorted_vals = np.sort(vals)
        return float(np.mean(sorted_vals[-n_keep:]))
    raise ValueError(f"Unknown aggregation method: {method}")


def spread_prior_bands(
    q: np.ndarray,
    *,
    diag_value: float,
    clip_eps: float,
    low_max: float,
    high_min: float,
) -> np.ndarray:
    """
    Linearly remap off-diagonal entries away from 0.5:

    - q < 0.5 → [clip_eps, low_max], order preserved within the bucket
    - q > 0.5 → [high_min, 1-clip_eps]
    - q == 0.5 stays 0.5 on off-diagonal (neutral)
    - diagonal set to diag_value
    """
    K = q.shape[0]
    lo_bd, hi_bd = float(clip_eps), float(1.0 - clip_eps)
    if not (lo_bd <= low_max + 1e-12 < 0.5 < high_min - 1e-12 <= hi_bd + 1e-12):
        raise ValueError(
            "Need clip_eps <= low_max < 0.5 < high_min <= 1-clip_eps "
            f"(got low_max={low_max}, high_min={high_min}, clip_eps={clip_eps})"
        )

    out = np.asarray(q, dtype=np.float64).copy()
    tol = 1e-14
    od = np.arange(K)[:, None] != np.arange(K)[None, :]

    def lin_map(vals: np.ndarray, vmin: float, vmax: float, a: float, b: float) -> np.ndarray:
        if vals.size == 0:
            return vals
        span = float(vmax - vmin)
        if span < tol:
            return np.full_like(vals, float(0.5 * (a + b)))
        t = (vals - vmin) / span
        return a + t * (b - a)

    blk = od & (out < 0.5 - tol)
    vb = out[blk]
    if vb.size > 0:
        out[blk] = lin_map(vb, vmin=float(np.min(vb)), vmax=float(np.max(vb)), a=lo_bd, b=low_max)

    blk = od & (out > 0.5 + tol)
    va = out[blk]
    if va.size > 0:
        out[blk] = lin_map(va, vmin=float(np.min(va)), vmax=float(np.max(va)), a=high_min, b=hi_bd)

    np.fill_diagonal(out, diag_value)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate event-pair semantic scores into cluster prior matrix")
    p.add_argument("--input", required=True, help="Event CSV with text/time columns")
    p.add_argument("--cluster-csv", required=True, help="CSV with cluster_id column, same row order as events")
    p.add_argument("--checkpoint", default="output/llm_role_distill_demo.pt", help="Role distillation checkpoint")
    p.add_argument("--text-col", default="Alarm Text")
    p.add_argument("--time-col", default="time")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "median", "trimmed_mean", "top_p_mean"],
        help="How to aggregate event-pair scores within each cluster pair",
    )
    p.add_argument("--trim-ratio", type=float, default=0.1, help="Used only when --aggregation=trimmed_mean")
    p.add_argument("--top-p", type=float, default=0.2, help="Used only when --aggregation=top_p_mean")
    p.add_argument("--respect-time-order", action="store_true", help="Only aggregate pairs with t_i < t_j")
    p.add_argument("--diag-value", type=float, default=0.5, help="Diagonal q_aa value")
    p.add_argument("--clip-eps", type=float, default=1e-3, help="Clip non-diagonal q_ab to [eps, 1-eps]")
    p.add_argument(
        "--prior-spread-low-max",
        type=float,
        default=None,
        help="If set with --prior-spread-high-min: remap off-diag q<0.5 into [clip_eps, this] (order preserved).",
    )
    p.add_argument(
        "--prior-spread-high-min",
        type=float,
        default=None,
        help="If set with --prior-spread-low-max: remap off-diag q>0.5 into [this, 1-clip_eps].",
    )
    p.add_argument("--out-csv", required=True, help="Output prior matrix CSV")
    p.add_argument("--out-npy", required=True, help="Output prior matrix NPY (for --q-npy)")
    p.add_argument("--out-meta", default=None, help="Optional JSON summary path")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    events = pd.read_csv(args.input)
    clusters_df = pd.read_csv(args.cluster_csv)
    if "cluster_id" not in clusters_df.columns:
        raise ValueError("--cluster-csv must contain column 'cluster_id'")
    if len(events) != len(clusters_df):
        raise ValueError(f"events rows {len(events)} != cluster rows {len(clusters_df)}")

    texts = events[args.text_col].astype(str).tolist()
    t = pd.to_numeric(events[args.time_col], errors="coerce").fillna(0.0).to_numpy()
    z = clusters_df["cluster_id"].to_numpy(dtype=int)
    K = int(np.max(z)) + 1
    N = len(texts)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    d_in = int(ckpt["d_in"])
    d_h = int(ckpt["d_h"])
    model = RoleDistillModel(d_in=d_in, d_h=d_h)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    print(f"Encoding {N} events with {args.embed_model} ...")
    st = SentenceTransformer(args.embed_model)
    g_np = st.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    if g_np.shape[1] != d_in:
        raise ValueError(f"Embedding dim {g_np.shape[1]} != checkpoint d_in {d_in}")

    with torch.no_grad():
        g = torch.from_numpy(g_np).to(device)
        hc, he, _, _ = model(g)
        logits = hc @ he.T
        r = torch.sigmoid(logits).cpu().numpy()  # (N, N)

    valid = np.ones((N, N), dtype=bool)
    np.fill_diagonal(valid, False)
    if args.respect_time_order:
        valid &= t[:, None] < t[None, :]

    q = np.full((K, K), np.nan, dtype=np.float64)
    counts = np.zeros((K, K), dtype=int)

    for a in range(K):
        ia = np.where(z == a)[0]
        if ia.size == 0:
            continue
        for b in range(K):
            ib = np.where(z == b)[0]
            if ib.size == 0:
                continue
            sub_scores = r[np.ix_(ia, ib)]
            sub_valid = valid[np.ix_(ia, ib)]
            vals = sub_scores[sub_valid]
            counts[a, b] = int(vals.size)
            q[a, b] = aggregate_values(vals, args.aggregation, args.trim_ratio, args.top_p)

    # Fill empty entries conservatively to neutral prior.
    q = np.where(np.isnan(q), 0.5, q)
    for k in range(K):
        q[k, k] = args.diag_value

    if not (0.0 <= args.clip_eps < 0.5):
        raise ValueError("--clip-eps must be in [0, 0.5)")
    lo, hi = args.clip_eps, 1.0 - args.clip_eps
    for i in range(K):
        for j in range(K):
            if i != j:
                q[i, j] = float(np.clip(q[i, j], lo, hi))

    spread_lo, spread_hi = args.prior_spread_low_max, args.prior_spread_high_min
    if (spread_lo is None) ^ (spread_hi is None):
        raise ValueError("Use both --prior-spread-low-max and --prior-spread-high-min together, or neither.")
    if spread_lo is not None and spread_hi is not None:
        q = spread_prior_bands(
            q,
            diag_value=args.diag_value,
            clip_eps=args.clip_eps,
            low_max=float(spread_lo),
            high_min=float(spread_hi),
        )

    out_csv = Path(args.out_csv)
    out_npy = Path(args.out_npy)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(q, index=range(K), columns=range(K)).to_csv(out_csv, index=True, header=True)
    np.save(out_npy, q)

    print(f"Saved cluster prior CSV: {out_csv}")
    print(f"Saved cluster prior NPY: {out_npy}")
    print(f"K={K}, N={N}, aggregation={args.aggregation}, respect_time_order={args.respect_time_order}")

    if args.out_meta:
        out_meta = Path(args.out_meta)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "input": str(Path(args.input).resolve()),
            "cluster_csv": str(Path(args.cluster_csv).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "embed_model": args.embed_model,
            "aggregation": args.aggregation,
            "trim_ratio": args.trim_ratio,
            "top_p": args.top_p,
            "respect_time_order": bool(args.respect_time_order),
            "diag_value": args.diag_value,
            "clip_eps": args.clip_eps,
            "prior_spread_low_max": spread_lo,
            "prior_spread_high_min": spread_hi,
            "K": K,
            "N": N,
            "pair_counts": counts.tolist(),
        }
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved meta JSON: {out_meta}")


if __name__ == "__main__":
    main()
