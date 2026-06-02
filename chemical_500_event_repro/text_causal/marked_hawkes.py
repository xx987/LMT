"""
Simplified embedding-marked Hawkes model (v1).

Lambda_{cn}(t_n, x_n) = mu_cn * p0(x_n|cn) + sum_{m<n} A_cm,cn * f(dt) * p_phi(x_n | m->cn)
with isotropic Gaussians in latent space and exponential kernel f(dt)=beta exp(-beta*dt).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_gaussian_isotropic(
    x: torch.Tensor, mean: torch.Tensor, log_sigma: torch.Tensor
) -> torch.Tensor:
    """Scalar log N(x | mean, sigma^2 I). x, mean: (d,). log_sigma: scalar."""
    d = float(x.shape[0])
    sigma = torch.exp(log_sigma).clamp(min=1e-4)
    diff = x - mean
    quad = (diff * diff).sum() / (2.0 * sigma * sigma)
    return -quad - d * torch.log(sigma) - 0.5 * d * math.log(2.0 * math.pi)


class MarkedHawkes(nn.Module):
    def __init__(
        self,
        n_types: int,
        d_in: int,
        d_latent: int = 64,
        a_temperature: float = 1.0,
    ):
        super().__init__()
        self.K = n_types
        self.d_in = d_in
        self.d = d_latent
        if a_temperature <= 0:
            raise ValueError("a_temperature must be > 0")
        self.a_temperature = float(a_temperature)
        self.proj = nn.Linear(d_in, d_latent)
        self.log_mu = nn.Parameter(torch.zeros(n_types))
        self.A_raw = nn.Parameter(torch.zeros(n_types, n_types))
        self.log_beta = nn.Parameter(torch.zeros(n_types, n_types))
        self.nu = nn.Parameter(torch.randn(n_types, d_latent) * 0.02)
        self.log_sig_bg = nn.Parameter(torch.zeros(n_types))
        self.b = nn.Parameter(torch.zeros(n_types, n_types, d_latent))
        self.log_sig_tr = nn.Parameter(torch.zeros(n_types, n_types))

    def A_matrix(self) -> torch.Tensor:
        eye = torch.eye(self.K, device=self.A_raw.device, dtype=self.A_raw.dtype)
        # A = sigmoid(A_raw / tau): tau < 1 sharpens toward 0/1 for the same |A_raw|.
        return torch.sigmoid(self.A_raw / self.a_temperature) * (1.0 - eye)

    def beta_matrix(self) -> torch.Tensor:
        eye = torch.eye(self.K, device=self.log_beta.device, dtype=self.log_beta.dtype)
        return F.softplus(self.log_beta) * (1.0 - eye) + 1e-4

    def mu_vec(self) -> torch.Tensor:
        return F.softplus(self.log_mu) + 1e-5

    def forward(self, t: torch.Tensor, c: torch.Tensor, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        t: (N,)  c: (N,) int64  x_in: (N, d_in)
        Returns log_lambda (N,), compensator (scalar).
        """
        N = t.shape[0]
        device = t.device
        dtype = t.dtype
        x = self.proj(x_in)
        A = self.A_matrix()
        Bmat = self.beta_matrix()
        mu = self.mu_vec()
        T = t[-1]

        log_sig_bg_eff = self.log_sig_bg
        log_sig_tr_eff = self.log_sig_tr

        log_lam_list = []
        for n in range(N):
            cn = int(c[n].item())
            xn = x[n]
            terms = [
                torch.log(mu[cn]) + _log_gaussian_isotropic(xn, self.nu[cn], log_sig_bg_eff[cn])
            ]
            for m in range(n):
                dt = t[n] - t[m]
                if dt <= 0:
                    continue
                cm = int(c[m].item())
                beta = Bmat[cm, cn]
                a = A[cm, cn]
                log_f = torch.log(beta) - beta * dt
                mean_tr = x[m] + self.b[cm, cn]
                terms.append(
                    torch.log(a.clamp(min=1e-12))
                    + log_f
                    + _log_gaussian_isotropic(xn, mean_tr, log_sig_tr_eff[cm, cn])
                )
            log_lam_list.append(torch.logsumexp(torch.stack(terms), dim=0))
        log_lam = torch.stack(log_lam_list)

        comp = torch.zeros((), device=device, dtype=dtype)
        for k in range(self.K):
            comp = comp + mu[k] * T
            for m in range(N):
                cm = int(c[m].item())
                beta = Bmat[cm, k]
                a = A[cm, k]
                comp = comp + a * (1.0 - torch.exp(-beta * (T - t[m])))

        return log_lam, comp
