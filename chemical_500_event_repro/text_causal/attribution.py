"""
Event-level parent responsibilities (soft), after training.

q_{n,0} propto mu_cn * p0(x_n)
q_{n,m} propto A_cm,cn * f(t_n-t_m) * p_phi(x_n | m->cn), m < n
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .marked_hawkes import MarkedHawkes, _log_gaussian_isotropic


@torch.no_grad()
def event_responsibilities(
    model: MarkedHawkes,
    t: torch.Tensor,
    c: torch.Tensor,
    x_in: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    model.eval()
    x = model.proj(x_in)
    A = model.A_matrix()
    Bmat = model.beta_matrix()
    mu = model.mu_vec()
    N = t.shape[0]
    q_bg: List[torch.Tensor] = []
    q_parents: List[torch.Tensor] = []

    for n in range(N):
        cn = int(c[n].item())
        xn = x[n]
        bg_log = torch.log(mu[cn]) + _log_gaussian_isotropic(xn, model.nu[cn], model.log_sig_bg[cn])
        if n == 0:
            q_bg.append(torch.ones((), device=t.device, dtype=bg_log.dtype))
            q_parents.append(torch.zeros(0, device=t.device, dtype=bg_log.dtype))
            continue
        parent_logs = []
        for m in range(n):
            dt = t[n] - t[m]
            if dt <= 0:
                parent_logs.append(torch.tensor(-1e9, device=t.device, dtype=bg_log.dtype))
                continue
            cm = int(c[m].item())
            beta = Bmat[cm, cn]
            a = A[cm, cn]
            log_f = torch.log(beta) - beta * dt
            mean_tr = x[m] + model.b[cm, cn]
            parent_logs.append(
                torch.log(a.clamp(min=1e-12))
                + log_f
                + _log_gaussian_isotropic(xn, mean_tr, model.log_sig_tr[cm, cn])
            )
        pl = torch.stack(parent_logs)
        all_logs = torch.cat([bg_log.unsqueeze(0), pl])
        w = torch.softmax(all_logs, dim=0)
        q_bg.append(w[0])
        q_parents.append(w[1:])

    return torch.stack(q_bg), q_parents


@torch.no_grad()
def multi_parent_edges_from_responsibilities(
    q_bg: torch.Tensor,
    q_parents: List[torch.Tensor],
    event_ids: Sequence[str] | Sequence[int],
    *,
    min_joint_prob: float = 0.0,
    top_k: Optional[int] = None,
    include_argmax_if_empty: bool = True,
) -> List[dict]:
    """
    Same softmax as event_responsibilities: w[0]=background, w[m+1]=P(parent row m).

    Export *multiple* candidate parents per child without changing the generative model.

    - min_joint_prob > 0: drop parents with joint prob below this.
    - top_k: after filtering, keep at most this many (largest prob first).
      If you want "always top-3 parents", set min_joint_prob=0 and top_k=3.
    - If no parent survives and include_argmax_if_empty, keep the single argmax parent.
    """
    out: List[dict] = []
    n_events = len(q_parents)
    for n in range(n_events):
        eid = event_ids[n]
        qp = q_parents[n]
        if qp.numel() == 0:
            continue
        p_bg = float(q_bg[n].cpu())
        cand: List[Tuple[int, float]] = [
            (m, float(qp[m].cpu().item())) for m in range(qp.numel())
        ]
        cand.sort(key=lambda x: -x[1])
        kept = [(m, p) for m, p in cand if p >= min_joint_prob] if min_joint_prob > 0 else cand[:]
        if top_k is not None and top_k > 0:
            kept = kept[:top_k]
        if not kept and include_argmax_if_empty:
            m = int(torch.argmax(qp).item())
            kept = [(m, float(qp[m].cpu().item()))]
        for m, p in kept:
            out.append(
                {
                    "effect_row_idx": n,
                    "effect_event_id": eid,
                    "parent_row_idx": m,
                    "parent_event_id": event_ids[m],
                    "joint_probability": p,
                    "q_background": p_bg,
                }
            )
    return out
