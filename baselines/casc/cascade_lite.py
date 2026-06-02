"""
CASCADE (original codebase path) on chemical sim CSVs.

This module **does not reimplement** MDL / gains: it calls the repository’s
`mdl_based_search.topological_search` (or `.search`), `model.Model`,
`score_all_edges`, `get_delays`, geometric delay distribution, etc. — the same
stack as `main.py`.

Chemical CSVs use float `time` and may contain **duplicate timestamps**; the
repo’s `dataloader.transform_from_float` can produce inf/NaN in that case.
Here we convert to **strictly increasing integer** `start_timestamp` while
**preserving time order** (and approximate scale), then run CASCADE unchanged.

If the MDL graph has **very few** edges, optional **DAG-safe fallback** edges
can be added from temporal co-occurrence scores (see ``pad_adjacency_to_min_edges``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _cluster_ids_from_df(df: pd.DataFrame) -> np.ndarray:
    """``cluster_id_true`` column, or parse ``Alarm Text`` like ``alarm_3`` (k8 compat CSV)."""
    if "cluster_id_true" in df.columns:
        return df["cluster_id_true"].astype(int).to_numpy()
    if "Alarm Text" in df.columns:
        z: list[int] = []
        for s in df["Alarm Text"].astype(str):
            m = re.search(r"alarm[_\s]?(\d+)", s, re.I)
            if not m:
                raise ValueError(
                    f"Expected 'cluster_id_true' or Alarm Text like 'alarm_0'; got {s!r}"
                )
            z.append(int(m.group(1)))
        return np.asarray(z, dtype=int)
    raise ValueError(
        f"Need 'cluster_id_true' or 'Alarm Text' (with alarm_<id>); got {list(df.columns)}"
    )


def _offdiag_count(A: np.ndarray) -> int:
    k = A.shape[0]
    m = ~np.eye(k, dtype=bool)
    return int((A[m] >= 0.5).sum())


def temporal_cooccurrence_scores(
    sim_csv: Path, window: float, tau: float
) -> tuple[np.ndarray, int]:
    """Exp-weighted i→j precedence counts (same float `time` as sim CSV)."""
    df = pd.read_csv(sim_csv)
    if "time" not in df.columns:
        raise ValueError(f"{sim_csv}: need column 'time'")
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(np.float64)
    z = _cluster_ids_from_df(df)
    if np.isnan(t).any():
        raise ValueError("non-numeric time")
    K = int(z.max()) + 1
    order = np.argsort(t, kind="stable")
    t, z = t[order], z[order]
    n = len(t)
    S = np.zeros((K, K), dtype=np.float64)
    for e in range(n):
        j = int(z[e])
        te = float(t[e])
        s = e - 1
        while s >= 0:
            dt = te - float(t[s])
            if dt > window:
                break
            if dt <= 0.0:
                s -= 1
                continue
            i = int(z[s])
            if i != j:
                S[i, j] += np.exp(-dt / max(tau, 1e-9))
            s -= 1
    return S, K


def resolve_min_output_edges(K: int, user_cap: int | None) -> int:
    """
    Minimum off-diagonal edges to keep in the saved graph.
    ``user_cap`` ``None`` → auto; ``0`` → disable padding.

    Default auto is **conservative** (sparse graph, lower TPR vs ground truth on
    typical chemical sims). Increase with ``--min-output-edges`` if needed.
    """
    if user_cap is not None:
        return max(0, int(user_cap))
    return max(2, min(K // 5, 4))


def pad_adjacency_to_min_edges(
    A: np.ndarray,
    sim_csv: Path,
    min_edges: int,
    window: float,
    tau: float,
) -> tuple[np.ndarray, int]:
    """
    If CASCADE returns too few edges, greedily add high-scoring i→j pairs from
    temporal co-occurrence while keeping a DAG (baseline-friendly fallback).
    Returns (possibly updated A, number of edges added).
    """
    import networkx as nx

    if min_edges <= 0:
        return A, 0
    A = np.asarray(A, dtype=np.float64).copy()
    K = A.shape[0]
    before = _offdiag_count(A)
    if before >= min_edges:
        return A, 0

    S, K2 = temporal_cooccurrence_scores(sim_csv, window, tau)
    if K2 != K:
        raise ValueError(f"K mismatch A={K} vs scores={K2}")

    pairs = [(float(S[i, j]), i, j) for i in range(K) for j in range(K) if i != j]
    pairs.sort(key=lambda x: -x[0])

    G = nx.DiGraph()
    G.add_nodes_from(range(K))
    for i in range(K):
        for j in range(K):
            if i != j and A[i, j] >= 0.5:
                G.add_edge(i, j)

    added = 0
    for _, i, j in pairs:
        if _offdiag_count(A) >= min_edges:
            break
        if A[i, j] >= 0.5:
            continue
        G.add_edge(i, j)
        if nx.is_directed_acyclic_graph(G):
            A[i, j] = 1.0
            added += 1
        else:
            G.remove_edge(i, j)

    return A, added


def chemical_sim_to_alarms(sim_csv: Path, *, seed: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(sim_csv)
    if "time" not in df.columns:
        raise ValueError(
            f"{sim_csv}: need 'time' and 'cluster_id_true' or 'Alarm Text' (alarm_<id>); "
            f"got {list(df.columns)}"
        )
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(np.float64)
    if np.isnan(t).any():
        raise ValueError("non-numeric time")
    if seed is not None:
        rng = np.random.default_rng(seed)
        dt = np.diff(np.sort(t))
        pos = dt[dt > 0]
        med = float(np.median(pos)) if len(pos) else 1.0
        min_gap = float(np.min(pos)) if len(pos) else 1.0
        # ``med*1e-9`` never reorders events; scale by ``min_gap`` so ordering can
        # change enough that CASCADE may differ across seeds (0.35 was often too weak).
        scale = max(min_gap * 2.0, med * 1e-9)
        t = t + rng.standard_normal(len(t)) * scale
    z = _cluster_ids_from_df(df)
    if z.min() < 0:
        raise ValueError("cluster / alarm id must be non-negative")
    K = int(z.max()) + 1
    if len(np.unique(z)) != K:
        raise ValueError(
            f"cluster ids must be contiguous 0..K-1 (unique={len(np.unique(z))}, max={z.max()})"
        )
    out = pd.DataFrame(
        {
            "alarm_id": z,
            "device_id": 0,
            "start_timestamp": t,
        }
    )
    return out.sort_values("start_timestamp").reset_index(drop=True)


def float_times_to_strict_integer_clock(alarms: pd.DataFrame) -> pd.DataFrame:
    """
    Integer timestamps strictly increasing in global time order, so CASCADE’s
    delay machinery never sees zero/negative diffs from duplicate floats.

    Values are scaled into ``np.int32`` range (``Model.all_alarms`` is int32):
    preserve *relative* float times, then break ties with +1 steps.
    """
    alarms = alarms.copy()
    t = alarms["start_timestamp"].astype(np.float64).to_numpy()
    n = len(t)
    if n == 0:
        alarms["start_timestamp"] = alarms["start_timestamp"].astype(np.int32)
        return alarms
    t0, t1 = float(np.min(t)), float(np.max(t))
    span = max(t1 - t0, 1e-12)
    hi = int(np.iinfo(np.int32).max) - 2 - n - 64
    scaled = (t - t0) / span * float(hi)
    order = np.argsort(scaled, kind="stable")
    new_t = np.zeros(n, dtype=np.int64)
    last = 0
    for idx in order:
        v = int(np.round(float(scaled[idx])))
        last = max(last + 1, v)
        new_t[idx] = last
    if new_t.max() > np.iinfo(np.int32).max:
        raise ValueError("internal: scaled timestamps still overflow int32")
    alarms["start_timestamp"] = new_t.astype(np.int32)
    return alarms


def run_cascade_repo(
    sim_csv: Path,
    *,
    search: str = "topo",
    topology: str = "empty",
    candidate_delay: int = 1000,
    precision: int = 2,
    dst: str = "geometric",
    align_mode: str = "next",
    instant: bool = False,
    instant_idf: bool = True,
    init_with_self_loops: bool = False,
    no_topo: bool = False,
    optimize_dst: bool = False,
    quiet: bool = False,
    min_output_edges: int | None = None,
    fallback_window: float = 250.0,
    fallback_tau: float = 40.0,
    seed: int | None = None,
) -> tuple[np.ndarray, float, int]:
    """
    Run CASCADE (repo implementation).

    ``seed``: tiny jitter on float times so repeated runs differ (CASCADE is
    otherwise deterministic for fixed CSV). ``None`` = no jitter.

    Returns ``(adjacency KxK float64, MDL length from CASCADE, n_fallback_edges_added)``.
    """
    # Import `mdl_based_search` before `distributions` alone — same init order as `main.py`
    # avoids circular import (our_gloabls <-> distributions) on some Python versions.
    import logging

    import dataloader as dl
    import mdl_based_search  # noqa: F401
    import distributions
    import our_gloabls

    if quiet:
        logging.getLogger().setLevel(logging.WARNING)

    our_gloabls.precision = precision
    our_gloabls.max_delay = candidate_delay
    our_gloabls.optimize_dst = optimize_dst
    our_gloabls.align_mode = align_mode
    our_gloabls.instant_effects = instant
    our_gloabls.instant_idf = instant_idf

    if dst == "geometric":
        our_gloabls.distribution = distributions.GeometricDistribution
    elif dst == "poisson":
        our_gloabls.distribution = distributions.PoissonDistribution
    elif dst == "uniform":
        our_gloabls.distribution = distributions.UniformDistribution
    elif dst == "normal":
        our_gloabls.distribution = distributions.DiscreteNormalDistribution
    else:
        raise ValueError(f"invalid dst: {dst}")

    alarms = float_times_to_strict_integer_clock(
        chemical_sim_to_alarms(sim_csv, seed=seed)
    )

    if topology == "full":
        topology_mat = dl.get_full_connected_topology_matrix(alarms)
    elif topology == "empty":
        topology_mat = dl.get_empty_topology_matrix(alarms)
    elif topology == "all-in-one":
        alarms["device_id"] = 0
        topology_mat = dl.get_empty_topology_matrix(alarms)
    else:
        topology_mat = np.load(topology)

    causal_prior = dl.get_empty_causal_prior(alarms, None)

    if search == "greedy":
        m = mdl_based_search.search(alarms, topology_mat, causal_prior, no_topo=no_topo)
    elif search == "topo":
        m = mdl_based_search.topological_search(
            alarms,
            topology_mat,
            causal_prior,
            init_with_self_loops=init_with_self_loops,
        )
    else:
        raise ValueError(f"invalid search: {search}")

    length = float(m.compute_length())
    A = np.vectorize(lambda x: 0 if x is None else 1)(m.edges).astype(np.float64)

    K = A.shape[0]
    target = resolve_min_output_edges(K, min_output_edges)
    A, n_pad = pad_adjacency_to_min_edges(A, sim_csv, target, fallback_window, fallback_tau)
    return A, length, n_pad


def discover_adjacency(
    sim_csv: Path,
    *,
    window: float | None = None,
    tau: float | None = None,
    min_edge_score: float | None = None,
    search: str = "topo",
    topology: str = "empty",
    candidate_delay: int = 1000,
    precision: int = 2,
    dst: str = "geometric",
    align_mode: str = "next",
    instant: bool = False,
    instant_idf: bool = True,
    init_with_self_loops: bool = False,
    no_topo: bool = False,
    quiet: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Backward-compatible entry: ``window`` / ``tau`` / ``min_edge_score`` are ignored
    (kept so older CLI invocations do not break); graph comes from CASCADE MDL search.
    """
    del window, tau, min_edge_score  # legacy no-op args
    A, _length, _pad = run_cascade_repo(
        sim_csv,
        search=search,
        topology=topology,
        candidate_delay=candidate_delay,
        precision=precision,
        dst=dst,
        align_mode=align_mode,
        instant=instant,
        instant_idf=instant_idf,
        init_with_self_loops=init_with_self_loops,
        no_topo=no_topo,
        quiet=quiet,
        seed=seed,
    )
    return A, int(A.shape[0])


def save_outputs(
    A: np.ndarray,
    out_dir: Path,
    *,
    also_npy: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(i) for i in range(A.shape[0])]
    df = pd.DataFrame(A, index=labels, columns=labels)
    df.to_csv(out_dir / "A_matrix.csv")
    if also_npy:
        np.save(out_dir / "A_matrix.npy", A.astype(np.float64))


def run_cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="CASCADE (repo MDL stack) on chemical sim CSV.")
    p.add_argument("--events-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--search", type=str, default="topo", choices=["topo", "greedy"])
    p.add_argument("--topology", type=str, default="empty", choices=["empty", "full", "all-in-one"])
    p.add_argument("--candidate-delay", type=int, default=1000)
    p.add_argument("--precision", type=int, default=2)
    p.add_argument("--dst", type=str, default="geometric")
    p.add_argument("--align", type=str, default="next")
    p.add_argument("--instant", action="store_true")
    p.add_argument("--no-instant-idf", action="store_true")
    p.add_argument("--init-with-self-loops", action="store_true")
    p.add_argument("--no-topo", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument(
        "--min-output-edges",
        type=int,
        default=None,
        help="Minimum off-diagonal edges (0=disable). Default: auto max(2, min(K//5, 4)).",
    )
    p.add_argument("--fallback-window", type=float, default=250.0)
    p.add_argument("--fallback-tau", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=None, help="RNG seed for tiny time jitter (multi-seed runs).")
    args = p.parse_args()

    instant_idf = not args.no_instant_idf
    A, length, n_pad = run_cascade_repo(
        args.events_csv.resolve(),
        search=args.search,
        topology=args.topology,
        candidate_delay=args.candidate_delay,
        precision=args.precision,
        dst=args.dst,
        align_mode=args.align,
        instant=args.instant,
        instant_idf=instant_idf,
        init_with_self_loops=args.init_with_self_loops,
        no_topo=args.no_topo,
        quiet=args.quiet,
        min_output_edges=args.min_output_edges,
        fallback_window=args.fallback_window,
        fallback_tau=args.fallback_tau,
        seed=args.seed,
    )
    save_outputs(A, args.out_dir.resolve())
    print(f"Final model length (CASCADE): {length:.4f}")
    if n_pad:
        print(f"Fallback edges added (temporal DAG pad): {n_pad}")
    print(f"Wrote {args.out_dir.resolve() / 'A_matrix.csv'}")


if __name__ == "__main__":
    run_cli()
