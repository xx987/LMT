#!/usr/bin/env python3
"""
Pure temporal multivariate Hawkes MAP fitter (no text / no marks in the likelihood).

This aligns with the manuscript-style likelihood:

  D_time = {(t_n, z_n)}_{n=1}^N, ordered 0 < t_1 < ... < t_N <= T.

  λ_b(t | H_t, A, Θ)
    = μ_b + Σ_{m: t_m < t} A_{z_m,b} α_{z_m,b} β_{z_m,b} exp(-β_{z_m,b}(t - t_m)).

At an observed event n of type z_n at time t_n:

  λ_{z_n}(t_n | H_{t_n}) = μ_{z_n}
    + Σ_{m<n} A_{z_m,z_n} α_{z_m,z_n} β_{z_m,z_n} exp(-β_{z_m,z_n}(t_n - t_m)).

Log-likelihood (standard point-process form):

  L_time = Σ_n log λ_{z_n}(t_n) - ∫_0^T Σ_b λ_b(t | H_t) dt.

Under exponential kernel the compensator has the closed form used in the text:

  ∫_0^T Σ_b λ_b(t) dt
    = T Σ_b μ_b
    + Σ_b Σ_m A_{z_m,b} α_{z_m,b} (1 - exp(-β_{z_m,b}(T - t_m))).

MAP objective (same structure as text_causal.train_text_causal, but likelihood is pure-time):

  J = L_time + λ_prior log p(A | q) - β_dag h(A) - λ_e Σ_{i≠j} A_ij,
      loss = -J,

where A is relaxed to continuous [0,1] for optimization, and log p(A|q) is the
Bernoulli semantic prior from text_causal.graph_prior.semantic_log_bernoulli.

This is MAP / penalized ML, not full Bayesian posterior sampling.

Example (chemical 100-event k=5, from repo root ``cascade/``)
------------------------------------------------------------
::

  cd chemical_100_event_repro
  python3 temporal_hawkes_pure_map.py \\
    --input output/sim_chemical_100_k5_uniform_passset.csv \\
    --cluster-csv output/sim_chemical100_k5_uniform_passset_oracle_clusters.csv \\
    --q-csv output/q_cluster_prior_k5.csv \\
    --epochs 200 --out-dir output/temporal_pure_chemical_k5_n100_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import ``text_causal`` from repo root ``cascade/`` when cwd is chemical_100_event_repro/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_causal import graph_prior
from text_causal.load_events import load_events_csv, events_to_arrays


def load_q_prior_matrix(K: int, q_default: float, q_npy: str | None, q_csv: str | None, device: torch.device) -> torch.Tensor:
    """KxK prior q_ij in [0,1]; diagonal forced to 0.5 (same as build_q_matrix)."""
    if q_npy is not None and q_csv is not None:
        raise ValueError("Use at most one of --q-npy and --q-csv")
    q = torch.full((K, K), q_default, dtype=torch.float64)
    if q_npy is not None:
        arr = np.load(q_npy)
        if arr.shape != (K, K):
            raise ValueError(f"q npy shape {arr.shape} != ({K},{K})")
        q = torch.from_numpy(arr.astype(np.float64))
    elif q_csv is not None:
        q_path = Path(q_csv)
        if not q_path.is_file():
            raise FileNotFoundError(
                f"q prior CSV not found: {q_path.resolve()}\n"
                f"  Build one with: python3 build_cluster_prior_from_role_distill.py ... --out-csv <path>\n"
                f"  Chemical k=5 demo prior in this repo: chemical_100_event_repro/output/q_cluster_prior_k5.csv"
            )
        mat = pd.read_csv(q_csv, index_col=0).to_numpy(dtype=np.float64)
        if mat.shape != (K, K):
            raise ValueError(f"q csv shape {mat.shape} != ({K},{K})")
        q = torch.from_numpy(mat)
    eye = torch.eye(K)
    q = q * (1.0 - eye) + 0.5 * eye
    return q.to(device=device, dtype=torch.float32)


class PureTemporalMultivariateHawkes(nn.Module):
    """
    Cluster-only Hawkes. Indices are 0..K-1 (paper often uses 1..K).

    A_gate: soft adjacency in (0,1) on off-diagonal (sigmoid).
    alpha: nonnegative trigger strengths (softplus).
    beta: positive decay rates (softplus).
    """

    def __init__(self, n_types: int, a_gate_temperature: float = 1.0):
        super().__init__()
        if n_types < 2:
            raise ValueError("Need at least 2 types")
        if a_gate_temperature <= 0:
            raise ValueError("a_gate_temperature must be > 0")
        self.K = n_types
        self.a_gate_temperature = float(a_gate_temperature)
        self.log_mu = nn.Parameter(torch.zeros(n_types))
        self.A_raw = nn.Parameter(torch.zeros(n_types, n_types))
        self.log_alpha = nn.Parameter(torch.zeros(n_types, n_types))
        self.log_beta = nn.Parameter(torch.zeros(n_types, n_types))

    def A_matrix(self) -> torch.Tensor:
        eye = torch.eye(self.K, device=self.A_raw.device, dtype=self.A_raw.dtype)
        return torch.sigmoid(self.A_raw / self.a_gate_temperature) * (1.0 - eye)

    def alpha_matrix(self) -> torch.Tensor:
        eye = torch.eye(self.K, device=self.log_alpha.device, dtype=self.log_alpha.dtype)
        return F.softplus(self.log_alpha) * (1.0 - eye) + 1e-8

    def beta_matrix(self) -> torch.Tensor:
        eye = torch.eye(self.K, device=self.log_beta.device, dtype=self.log_beta.dtype)
        return F.softplus(self.log_beta) * (1.0 - eye) + 1e-4

    def mu_vec(self) -> torch.Tensor:
        return F.softplus(self.log_mu) + 1e-5

    def forward(self, t: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        t: (N,) float, sorted non-decreasing
        c: (N,) int64 cluster indices in [0, K-1]
        Returns log_lambda (N,), compensator (scalar).
        """
        N = t.shape[0]
        device = t.device
        dtype = t.dtype
        T = t[-1]

        A = self.A_matrix()
        Alpha = self.alpha_matrix()
        Beta = self.beta_matrix()
        mu = self.mu_vec()

        log_lam_list = []
        for n in range(N):
            cn = int(c[n].item())
            lam = mu[cn].clone()
            for m in range(n):
                dt = t[n] - t[m]
                if dt <= 0:
                    continue
                zm = int(c[m].item())
                b = Beta[zm, cn]
                lam = lam + A[zm, cn] * Alpha[zm, cn] * b * torch.exp(-b * dt)
            log_lam_list.append(torch.log(lam.clamp(min=1e-12)))
        log_lam = torch.stack(log_lam_list)

        comp = torch.zeros((), device=device, dtype=dtype)
        comp = comp + mu.sum() * T
        for b in range(self.K):
            for m in range(N):
                zm = int(c[m].item())
                beta = Beta[zm, b]
                comp = comp + A[zm, b] * Alpha[zm, b] * (1.0 - torch.exp(-beta * (T - t[m])))

        return log_lam, comp


def main() -> None:
    p = argparse.ArgumentParser(description="MAP fit: pure temporal Hawkes + LLM cluster prior q(A)")
    p.add_argument("--input", required=True, help="Event CSV (needs time column; text column ignored for likelihood)")
    p.add_argument("--time-col", default="time")
    p.add_argument("--event-col", default="event_id")
    p.add_argument("--text-col", default="Alarm Text", help="Unused in likelihood; still required by CSV loader")
    p.add_argument("--cluster-csv", required=True, help="CSV with cluster_id column, same row order as events after sort")
    p.add_argument("--parse-time", action="store_true")
    p.add_argument("--q-npy", default=None, help="KxK prior matrix (.npy) from LLM pipeline")
    p.add_argument("--q-csv", default=None, help="KxK prior matrix CSV (row index col 0, like q_cluster_prior_*.csv)")
    p.add_argument("--q-default", type=float, default=0.5, help="Fill q_ij when neither --q-npy nor --q-csv is set")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--beta-dag", type=float, default=1.0)
    p.add_argument("--lambda-prior", type=float, default=1.0, help="Weight on semantic log prior term")
    p.add_argument("--lambda-e", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="Random seed (for reproducible randomized init)")
    p.add_argument(
        "--init-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to raw params at init; >0 enables seed-dependent runs",
    )
    p.add_argument("--a-gate-temperature", type=float, default=1.0, help="Sigmoid temperature for relaxed A")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if args.init_std < 0:
        raise ValueError("--init-std must be >= 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
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
    t_np, event_ids, _texts = events_to_arrays(df)
    cl = pd.read_csv(args.cluster_csv)
    if "cluster_id" not in cl.columns:
        raise ValueError("--cluster-csv must contain cluster_id")
    if len(cl) != len(df):
        raise ValueError(f"cluster rows {len(cl)} != event rows {len(df)} after time-sort")
    c_np = cl["cluster_id"].to_numpy(dtype=np.int64)
    if np.any(c_np < 0):
        raise ValueError("cluster_id must be non-negative")
    K = int(c_np.max()) + 1
    if K < 2:
        raise ValueError("Need K>=2 clusters")

    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    c = torch.tensor(c_np, dtype=torch.long, device=device)

    q = load_q_prior_matrix(K, args.q_default, args.q_npy, args.q_csv, device)

    model = PureTemporalMultivariateHawkes(n_types=K, a_gate_temperature=args.a_gate_temperature).to(device)

    # Initialize background rates similarly to train_text_causal.py
    T_obs = float(t_np[-1])
    N_obs = float(len(df))
    mu_target = N_obs / max(1.0, K * T_obs)
    softplus_target = max(mu_target - 1e-5, 1e-8)
    init_log_mu = float(np.log(np.expm1(softplus_target)))
    with torch.no_grad():
        model.log_mu.data.fill_(init_log_mu)
        if args.init_std > 0:
            model.A_raw.add_(torch.randn_like(model.A_raw) * args.init_std)
            model.log_alpha.add_(torch.randn_like(model.log_alpha) * args.init_std)
            model.log_beta.add_(torch.randn_like(model.log_beta) * args.init_std)
            model.log_mu.add_(torch.randn_like(model.log_mu) * args.init_std)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    meta = {
        "model": "PureTemporalMultivariateHawkes",
        "n_events": len(df),
        "K": K,
        "beta_dag": args.beta_dag,
        "lambda_prior": args.lambda_prior,
        "lambda_e": args.lambda_e,
        "seed": args.seed,
        "init_std": args.init_std,
        "q_default": args.q_default,
        "q_npy": args.q_npy,
        "q_csv": args.q_csv,
        "a_gate_temperature": args.a_gate_temperature,
        "input": str(Path(args.input).resolve()),
        "cluster_csv": str(Path(args.cluster_csv).resolve()),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.DataFrame({"event_id": event_ids, "time": t_np, "cluster_id": c_np}).to_csv(
        out / "events_time_cluster.csv", index=False
    )

    for ep in range(args.epochs):
        opt.zero_grad()
        log_lam, comp = model(t, c)
        loglike = log_lam.sum() - comp
        A = model.A_matrix()
        sem = graph_prior.semantic_log_bernoulli(A, q)
        h = graph_prior.dag_penalty_h(A)
        ssum = graph_prior.sparsity_sum(A)
        J = loglike + args.lambda_prior * sem - args.beta_dag * h - args.lambda_e * ssum
        loss = -J
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:4d}")

    with torch.no_grad():
        A_final = model.A_matrix().cpu().numpy()
        Alpha_final = model.alpha_matrix().cpu().numpy()
        Beta_final = model.beta_matrix().cpu().numpy()
        mu_final = model.mu_vec().cpu().numpy()

    np.save(out / "A_matrix.npy", A_final)
    pd.DataFrame(A_final).to_csv(out / "A_matrix.csv", index=True, header=True)
    np.save(out / "Alpha_matrix.npy", Alpha_final)
    pd.DataFrame(Alpha_final).to_csv(out / "Alpha_matrix.csv", index=True, header=True)
    np.save(out / "Beta_matrix.npy", Beta_final)
    pd.DataFrame(Beta_final).to_csv(out / "Beta_matrix.csv", index=True, header=True)
    np.save(out / "mu_vec.npy", mu_final)

    torch.save(model.state_dict(), out / "pure_temporal_hawkes.pt")
    print(f"Done. Wrote: {out}")


if __name__ == "__main__":
    main()
