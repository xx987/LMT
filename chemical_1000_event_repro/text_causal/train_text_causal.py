#!/usr/bin/env python3
"""
Train embedding-marked Hawkes + graph prior on an event CSV.

Example:
  cd ".../cascade"
  python -m text_causal.train_text_causal \\
    --input new_dm048.csv \\
    --event-col IntID \\
    --text-col "Alarm Text" \\
    --time-col Set \\
    --parse-time \\
    --n-clusters 15 \\
    --out-dir output/text_causal_dm048
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import attribution as attr_mod
from . import graph_prior
from .load_events import events_to_arrays, load_events_csv
from .marked_hawkes import MarkedHawkes
from .preprocess import load_or_compute_clusters

_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / ".cursor" / "debug-ce48b3.log"


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    run_id: str,
) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "ce48b3",
            "runId": run_id,
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def build_q_matrix(K: int, q_default: float, q_npy: str | None, device: torch.device) -> torch.Tensor:
    q = torch.full((K, K), q_default, dtype=torch.float64)
    if q_npy is not None:
        arr = np.load(q_npy)
        if arr.shape != (K, K):
            raise ValueError(f"q npy shape {arr.shape} != ({K},{K})")
        q = torch.from_numpy(arr.astype(np.float64))
    eye = torch.eye(K)
    q = q * (1.0 - eye) + 0.5 * eye
    return q.to(device=device, dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Train text_causal marked Hawkes + graph prior")
    p.add_argument("--input", required=True, help="CSV path (one row per event)")
    p.add_argument("--event-col", default="event_id", help="Event id column")
    p.add_argument("--text-col", default="Alarm Text", help="Text description column")
    p.add_argument("--time-col", default="time", help="Time column (numeric or datetime)")
    p.add_argument("--parse-time", action="store_true", help="Parse time column as datetime")
    p.add_argument("--n-clusters", type=int, default=20, help="KMeans K when no --cluster-csv")
    p.add_argument("--cluster-csv", default=None, help="Optional CSV with cluster_id column, same row order")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--d-latent", type=int, default=64, help="Latent dim after Linear(d_in, d_latent)")
    p.add_argument("--q-npy", default=None, help="Optional KxK semantic prior q_ij (numpy)")
    p.add_argument("--q-default", type=float, default=0.5, help="Default q_ij if no --q-npy")
    p.add_argument("--beta-dag", type=float, default=1.0, help="DAG penalty weight")
    p.add_argument("--lambda-e", type=float, default=0.1, help="Edge sparsity weight")
    p.add_argument(
        "--a-temperature",
        type=float,
        default=1.0,
        help="A = sigmoid(A_raw/tau). tau<1 pushes A toward extremes (easier 0.xx vs >0.5).",
    )
    p.add_argument(
        "--init-a-from-q-npy",
        action="store_true",
        help="If --q-npy set: init A_raw so sigmoid(A_raw/tau)≈q on off-diagonal (needs oracle q).",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--out-dir", required=True, help="Directory for checkpoints and CSV outputs")
    p.add_argument(
        "--multi-parent-min-prob",
        type=float,
        default=None,
        help="If set (or if --multi-parent-top-k set), write event_attribution_multi_edges.csv: "
        "parents with joint softmax prob >= this. Use 0 to only apply top-k.",
    )
    p.add_argument(
        "--multi-parent-top-k",
        type=int,
        default=None,
        help="Optional cap on parents per child after min-prob filter (e.g. 5).",
    )
    args = p.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_events_csv(
        args.input,
        event_col=args.event_col,
        text_col=args.text_col,
        time_col=args.time_col,
        parse_time_as_datetime=args.parse_time,
    )
    t_np, event_ids, texts = events_to_arrays(df)
    emb_np, c_np = load_or_compute_clusters(
        texts, n_clusters=args.n_clusters, cluster_csv=args.cluster_csv, embed_model=args.embed_model
    )
    K = int(c_np.max()) + 1
    if K < 2:
        raise ValueError("Need at least 2 clusters (increase --n-clusters or check data)")

    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    c = torch.tensor(c_np, dtype=torch.long, device=device)
    x_in = torch.tensor(emb_np, dtype=torch.float32, device=device)

    q = build_q_matrix(K, args.q_default, args.q_npy, device)
    model = MarkedHawkes(
        n_types=K,
        d_in=emb_np.shape[1],
        d_latent=args.d_latent,
        a_temperature=args.a_temperature,
    ).to(device)

    if args.init_a_from_q_npy:
        if args.q_npy is None:
            raise ValueError("--init-a-from-q-npy requires --q-npy")
        arr = np.load(args.q_npy)
        if arr.shape != (K, K):
            raise ValueError(f"q npy shape {arr.shape} != ({K},{K})")
        eps = 1e-4
        p = np.clip(arr.astype(np.float64), eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        eye = np.eye(K)
        raw = logit * args.a_temperature * (1.0 - eye)
        with torch.no_grad():
            model.A_raw.copy_(torch.from_numpy(raw).to(device=device, dtype=model.A_raw.dtype))

    # Initialize background rates to a scale consistent with the observed number of events.
    # In a Hawkes process, if triggers are small initially, the expected count is ~ T * sum_k mu_k.
    # Our time stamps are in seconds (t is large), so default mu_init would make the compensator enormous.
    # We set mu_target = N / (K * T) and invert softplus (mu_vec = softplus(log_mu) + 1e-5).
    T_obs = float(t_np[-1])
    N_obs = float(len(df))
    mu_target = N_obs / max(1.0, K * T_obs)
    softplus_target = max(mu_target - 1e-5, 1e-8)
    init_log_mu = float(np.log(np.expm1(softplus_target)))
    with torch.no_grad():
        model.log_mu.data.fill_(init_log_mu)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    meta = {
        "n_events": len(df),
        "K": K,
        "d_in": emb_np.shape[1],
        "d_latent": args.d_latent,
        "beta_dag": args.beta_dag,
        "lambda_e": args.lambda_e,
        "a_temperature": args.a_temperature,
        "q_default": args.q_default,
        "q_npy": args.q_npy,
        "init_a_from_q_npy": args.init_a_from_q_npy,
        "input": str(Path(args.input).resolve()),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    pd.DataFrame(
        {
            "row_idx": np.arange(len(df)),
            "event_id": event_ids,
            "cluster_id": c_np,
            "time": t_np,
            "text": texts,
        }
    ).to_csv(out / "events_with_clusters.csv", index=False)

    for ep in range(args.epochs):
        opt.zero_grad()
        log_lam, comp = model(t, c, x_in)
        loglike = log_lam.sum() - comp
        A = model.A_matrix()
        sem = graph_prior.semantic_log_bernoulli(A, q)
        h = graph_prior.dag_penalty_h(A)
        ssum = graph_prior.sparsity_sum(A)
        J = loglike + sem - args.beta_dag * h - args.lambda_e * ssum
        loss = -J

        # #region agent log
        if ep in (0, 1) or ep == args.epochs - 1:
            eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            A_off = A.detach() * (1.0 - eye)
            num_off = A.shape[0] * (A.shape[0] - 1)
            A_sum_off = float(A_off.sum().cpu().item())
            A_mean_off = float(A_sum_off / max(1, num_off))

            mu = model.mu_vec().detach()
            beta = model.beta_matrix().detach()
            sig_bg = torch.exp(model.log_sig_bg.detach())

            # log-lambda magnitude: if it's extremely negative, then intensity is tiny and NLL explodes.
            t_min = float(t.detach().min().cpu().item())
            t_max = float(t.detach().max().cpu().item())
            T = float(t.detach()[-1].cpu().item())

            _debug_log(
                hypothesis_id="H_time_or_beta",
                location="train_text_causal.py:epoch_stats",
                message="stats for loss decomposition (time/beta/sig/loglam/comp)",
                run_id=f"ep{ep}",
                data={
                    "t_min": t_min,
                    "t_max": t_max,
                    "T": T,
                    "mu_min": float(mu.min().cpu().item()),
                    "mu_max": float(mu.max().cpu().item()),
                    "beta_min": float(beta.min().cpu().item()),
                    "beta_max": float(beta.max().cpu().item()),
                    "sig_bg_min": float(sig_bg.min().cpu().item()),
                    "sig_bg_max": float(sig_bg.max().cpu().item()),
                    "log_lam_sum": float(log_lam.detach().sum().cpu().item()),
                    "log_lam_min": float(log_lam.detach().min().cpu().item()),
                    "log_lam_max": float(log_lam.detach().max().cpu().item()),
                    "comp": float(comp.detach().cpu().item()),
                    "loglike": float(loglike.detach().cpu().item()),
                },
            )

            _debug_log(
                hypothesis_id="H_prior_terms",
                location="train_text_causal.py:prior_terms",
                message="semantic/dag/sparsity terms and loss",
                run_id=f"ep{ep}",
                data={
                    "sem": float(sem.detach().cpu().item()),
                    "h": float(h.detach().cpu().item()),
                    "ssum": float(ssum.detach().cpu().item()),
                    "J": float(J.detach().cpu().item()),
                    "loss": float(loss.detach().cpu().item()),
                    "A_sum_off": A_sum_off,
                    "A_mean_off": A_mean_off,
                },
            )
        # #endregion

        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(
                f"epoch {ep:4d}  loss={loss.item() / 1000.0:.4f}  h={h.item():.4f}"
            )

    torch.save(model.state_dict(), out / "marked_hawkes.pt")
    with torch.no_grad():
        A_final = model.A_matrix().cpu().numpy()
    np.save(out / "A_matrix.npy", A_final)
    pd.DataFrame(A_final).to_csv(out / "A_matrix.csv", index=True, header=True)

    q_bg, q_par = attr_mod.event_responsibilities(model, t, c, x_in)
    rows = []
    for n in range(len(df)):
        row = {
            "row_idx": n,
            "event_id": event_ids[n],
            "q_background": float(q_bg[n].cpu()),
        }
        qp = q_par[n]
        if qp.numel() == 0:
            row["argmax_parent_row"] = ""
            row["argmax_parent_q"] = ""
        else:
            idx = int(torch.argmax(qp).item())
            row["argmax_parent_row"] = idx
            row["argmax_parent_q"] = float(qp[idx].cpu())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "event_attribution_summary.csv", index=False)

    if args.multi_parent_min_prob is not None or args.multi_parent_top_k is not None:
        min_p = 0.0 if args.multi_parent_min_prob is None else float(args.multi_parent_min_prob)
        multi_rows = attr_mod.multi_parent_edges_from_responsibilities(
            q_bg,
            q_par,
            event_ids,
            min_joint_prob=min_p,
            top_k=args.multi_parent_top_k,
            include_argmax_if_empty=True,
        )
        pd.DataFrame(multi_rows).to_csv(out / "event_attribution_multi_edges.csv", index=False)

    print(f"Done. Wrote: {out}")


if __name__ == "__main__":
    main()
