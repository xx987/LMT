"""
Graph prior terms: NOTEARS-style DAG penalty h(A), semantic Bernoulli term, sparsity.
"""

from __future__ import annotations

import torch


def dag_penalty_h(A: torch.Tensor) -> torch.Tensor:
    """h(G) = tr(exp(A ⊙ A)) - K; h = 0 for DAG under continuous relaxation."""
    M = A * A
    E = torch.linalg.matrix_exp(M)
    k = A.shape[0]
    return E.trace() - float(k)


def semantic_log_bernoulli(A: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Sum_{i!=j} [A_ij log q_ij + (1-A_ij) log(1-q_ij)]."""
    k = A.shape[0]
    q = q.clamp(eps, 1.0 - eps)
    mask = 1.0 - torch.eye(k, device=A.device, dtype=A.dtype)
    term = A * torch.log(q) + (1.0 - A) * torch.log(1.0 - q)
    return (term * mask).sum()


def sparsity_sum(A: torch.Tensor) -> torch.Tensor:
    k = A.shape[0]
    mask = 1.0 - torch.eye(k, device=A.device, dtype=A.dtype)
    return (A * mask).sum()
