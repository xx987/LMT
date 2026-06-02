#!/usr/bin/env python3
"""
Pure-LLM baseline (k5 / k8 / k16, 500 or 1000 events): OpenAI Chat Completions → K×K cluster matrix A
(P(causal edge i→j)), CASCADE-style CSV, SHD/FDR/TPR vs GT (off-diagonal, pred>=τ).

Bundled CSVs under baselines/pure_llm/data/ (same sims as vocabulary/CASCADE).

Usage::

  export OPENAI_API_KEY=...
  python3 baselines/pure_llm/pure_llm_cluster_graph.py --k 5 --n-events 500
  python3 baselines/pure_llm/pure_llm_cluster_graph.py --k 16 --n-events 1000 --mock
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# (events_csv, cluster_csv, gt_csv) under pure_llm/data/ — keys (k, n_events), same as vocabulary.
_DATA_BUNDLES: dict[tuple[int, int], tuple[str, str, str]] = {
    (5, 500): (
        "sim_chemical_500_k5_uniform_passset.csv",
        "sim_chemical500_k5_uniform_passset_oracle_clusters.csv",
        "sim_chemical_true_graph_k5_uniform_passset.csv",
    ),
    (5, 1000): (
        "sim_chemical_1000_k5_uniform_passset.csv",
        "sim_chemical1000_k5_uniform_passset_oracle_clusters.csv",
        "sim_chemical_true_graph_k5_uniform_passset_1000.csv",
    ),
    (8, 500): (
        "sim_chemical_500_k8_uniform_passset_compat.csv",
        "sim_chemical500_k8_uniform_passset_oracle_clusters.csv",
        "sim_chemical_true_graph_k8_uniform_passset.csv",
    ),
    (8, 1000): (
        "sim_chemical_1000_k8_uniform_passset.csv",
        "sim_chemical1000_k8_uniform_passset_oracle_clusters.csv",
        "sim_chemical_true_graph_k8_uniform_passset_1000.csv",
    ),
    (16, 500): (
        "sim_chemical_500_k16_cascadefriendly.csv",
        "sim_chemical500_k16_cascadefriendly_oracle_clusters.csv",
        "sim_chemical_true_graph_k16_cascadefriendly.csv",
    ),
    (16, 1000): (
        "sim_chemical_1000_k16_uniform_passset.csv",
        "sim_chemical1000_k16_uniform_passset_oracle_clusters.csv",
        "sim_chemical_true_graph_k16_uniform_passset_1000.csv",
    ),
}

_OUT_SUBDIR: dict[tuple[int, int], str] = {
    (5, 500): "k5_500",
    (5, 1000): "k5_1000",
    (8, 500): "k8_500",
    (8, 1000): "k8_1000",
    (16, 500): "k16_500",
    (16, 1000): "k16_1000",
}


def metric(pred_flat: np.ndarray, gt_flat: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    shd = float(fp + fn)
    fdr = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return shd, fdr, tpr


def build_system_prompt(k: int, n_rows: int) -> str:
    km1 = k - 1
    return (
        "You are an analyst for industrial alarm logs. Estimate directed CAUSAL RELATIONSHIPS "
        f"between alarm CLUSTERS (nodes 0–{km1} only).\n\n"
        f"You will receive ~{n_rows} rows. Each row has: event_id, cluster_id, time, alarm_text.\n"
        f"- cluster_id is an integer in {{0,1,…,{km1}}} (oracle cluster label; use as node id).\n"
        "- time is numeric (use temporal order together with alarm_text as evidence).\n"
        "- alarm_text is English.\n\n"
        "Matrix semantics (important):\n"
        f"- Return a {k}×{k} matrix A. Diagonal MUST be 0.\n"
        "- For every ordered pair (i, j) with i != j, A[i][j] is your estimate of the "
        "PROBABILITY that a directed causal influence exists from cluster i to cluster j "
        "(row i, column j). Values are continuous in [0, 1], not only 0/1.\n"
        "- Treat A as a probabilistic adjacency: higher A[i][j] means stronger belief that "
        "i causally drives j in this log. Use the full range when warranted by co-occurrence, "
        "temporal precedence, and text; do not default the whole matrix to zeros unless the "
        "logs genuinely contain no usable signal.\n"
        "- Off-diagonal entries may use several decimal places (like a soft classifier), "
        "similar in spirit to a neural edge-probability matrix.\n\n"
        "Rules:\n"
        "1) Use only the provided rows. Do not invent events.\n"
        "2) Output VALID JSON ONLY (no markdown fences, no extra keys) with this schema:\n"
        f'{{"A":[[{k} rows of {k} floats each]],"notes":"<=240 chars English summary"}}\n'
        f"3) CRITICAL: the JSON value \"A\" MUST be a JSON array of length EXACTLY {k}; each "
        f"element A[i] MUST be an array of length EXACTLY {k} (so the matrix is {k}×{k} numbers). "
        f"len(A) must be {k} and every len(A[i]) must be {k}. Do not return {k-1} or {k+1} rows.\n"
    )


def build_user_table(df: pd.DataFrame) -> str:
    lines = ["event_id\tcluster_id\ttime\talarm_text"]
    for _, r in df.iterrows():
        eid = r["event_id"]
        cid = int(r["cluster_id"])
        t = r["time"]
        txt = str(r["alarm_text"]).replace("\t", " ").replace("\n", " ").strip()
        lines.append(f"{eid}\t{cid}\t{t}\t{txt}")
    return "\n".join(lines)


def call_openai(
    *,
    user_content: str,
    model: str,
    temperature: float,
    k: int,
    n_rows: int,
    repair_prefix: str | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai: pip install openai") from e
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()
    body = (
        f"Infer the {k}×{k} cluster-level matrix A of directed causal probabilities "
        "P(edge i→j) from the following tab-separated data (header row first).\n\n"
        f"{user_content}\n\n"
        "Return JSON only as specified."
    )
    if repair_prefix:
        body = repair_prefix.strip() + "\n\n" + body
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": build_system_prompt(k, n_rows)},
            {"role": "user", "content": body},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _coerce_matrix_to_kxk(a: np.ndarray, k: int, pad: float = 0.5) -> np.ndarray:
    """Trim to at most k×k, then pad missing rows/columns with *pad*, diagonal set to 0."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"A must be 2D, got ndim={a.ndim} shape={getattr(a, 'shape', None)}")
    r, c = a.shape
    if r > k:
        a = a[:k, :].copy()
    if a.shape[1] > k:
        a = a[:, :k].copy()
    r, c = a.shape
    out = np.full((k, k), float(pad), dtype=np.float64)
    out[:r, :c] = a
    np.fill_diagonal(out, 0.0)
    return np.clip(out, 0.0, 1.0)


def parse_a_json(raw: str, k: int) -> np.ndarray:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    obj = json.loads(s)
    if "A" not in obj:
        raise KeyError("JSON missing key 'A'")
    a = np.array(obj["A"], dtype=float)
    if a.shape == (k, k):
        np.fill_diagonal(a, 0.0)
        a = np.clip(a, 0.0, 1.0)
        return a
    if a.ndim != 2:
        raise ValueError(f"A must be 2D, got ndim={a.ndim} shape={a.shape}")
    return _coerce_matrix_to_kxk(a, k, pad=0.5)


def mock_a_matrix(*, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 0.15, size=(k, k)).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    return np.round(a, 3)


def _parse_temperature_list(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --temperatures")
    return [float(p) for p in parts]


def _std_sample(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=float)
    if a.size <= 1:
        return 0.0
    return float(np.std(a, ddof=1))


def _tau_grid() -> list[float]:
    return [round(0.05 + 0.05 * i, 10) for i in range(19)]


def _one_inference(
    *,
    user_table: str,
    model: str,
    temperature: float,
    mock: bool,
    mock_seed: int,
    k: int,
    n_rows: int,
    max_parse_retries: int = 3,
) -> tuple[np.ndarray, str]:
    if mock:
        a = mock_a_matrix(k=k, seed=mock_seed)
        return a, "MOCK (no API call)\n"
    n_try = max(1, int(max_parse_retries))
    raws: list[str] = []
    hint: str | None = None
    last_err: Exception | None = None
    for attempt in range(n_try):
        raw = call_openai(
            user_content=user_table,
            model=model,
            temperature=float(temperature),
            k=k,
            n_rows=n_rows,
            repair_prefix=hint,
        )
        raws.append(raw)
        try:
            a = parse_a_json(raw, k)
            combined = "\n\n".join(
                f"=== API attempt {i + 1} of {len(raws)} ===\n{t}" for i, t in enumerate(raws)
            )
            return a, combined
        except (ValueError, KeyError, json.JSONDecodeError, TypeError) as e:
            last_err = e
            hint = (
                f"VALIDATION FAILED ({type(e).__name__}): {e}\n"
                f"Return ONLY valid JSON with keys exactly \"A\" and \"notes\". "
                f"The value \"A\" must be a JSON array of EXACTLY {k} rows; each row must be "
                f"an array of EXACTLY {k} numbers (matrix shape {k}×{k}). "
                f"Double-check: len(A) should be {k}, not {k - 1} or {k + 1}."
            )
    assert last_err is not None
    raise RuntimeError(
        f"Could not obtain a valid {k}×{k} matrix after {n_try} API attempt(s). Last error: {last_err}"
    ) from last_err


def _default_data_paths(k: int, n_events: int) -> tuple[Path, Path, Path]:
    key = (k, n_events)
    if key not in _DATA_BUNDLES:
        raise ValueError(f"unsupported (k, n_events)={key}; use one of {sorted(_DATA_BUNDLES)}")
    ev, cl, gt = _DATA_BUNDLES[key]
    base = _HERE / "data"
    return base / ev, base / cl, base / gt


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(
        description="OpenAI -> K×K cluster A_matrix + GT metrics (bundled 500- or 1000-event sims)."
    )
    p.add_argument("--k", type=int, choices=(5, 8, 16), required=True, help="Number of clusters / matrix size.")
    p.add_argument(
        "--n-events",
        type=int,
        choices=(500, 1000),
        default=500,
        help="Which bundled dataset (500 vs 1000 rows); selects CSV triple and default output dir.",
    )
    p.add_argument(
        "--events-csv",
        type=Path,
        default=None,
        help="Override events CSV (default: bundled file under pure_llm/data/ for --k and --n-events).",
    )
    p.add_argument(
        "--cluster-csv",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
    )
    p.add_argument("--text-col", default="Alarm Text")
    p.add_argument("--time-col", default="time")
    p.add_argument("--event-col", default="event_id")
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Max rows to send (default: same as --n-events). Use a smaller number for quick tests.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: baselines/pure_llm/output/k{K}_{500|1000}",
    )
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Comma-separated LLM sampling temperatures. One API call per value; writes "
        "metrics_by_tau.csv for τ=0.05..0.95; stdout prints only Result at --result-tau.",
    )
    p.add_argument(
        "--result-tau",
        type=float,
        default=0.4,
        help="With --temperatures: binarization τ for the printed Result line (default 0.4).",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Used with --summary-only.")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print one Result line at τ=--threshold (default is full τ grid + Result at 0.5).",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Skip API; write a random sparse A for pipeline smoke test.",
    )
    p.add_argument(
        "--max-retries-on-bad-shape",
        type=int,
        default=3,
        help="If the model returns malformed JSON or wrong A shape, re-call the API up to "
        "this many times per matrix (default 3).",
    )
    args = p.parse_args(argv)

    k = int(args.k)
    n_events = int(args.n_events)
    row_cap = int(args.max_events) if args.max_events is not None else n_events

    dev, dcl, dgt = _default_data_paths(k, n_events)
    events_csv = Path(args.events_csv).resolve() if args.events_csv is not None else dev
    cluster_csv = Path(args.cluster_csv).resolve() if args.cluster_csv is not None else dcl
    gt_csv = Path(args.gt_csv).resolve() if args.gt_csv is not None else dgt
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else (_HERE / "output" / _OUT_SUBDIR[(k, n_events)])
    )

    events = pd.read_csv(events_csv)
    clusters = pd.read_csv(cluster_csv)
    if "cluster_id" not in clusters.columns:
        raise ValueError("cluster CSV must have column cluster_id")
    n = min(len(events), row_cap)
    events = events.iloc[:n].copy()
    clusters = clusters.iloc[:n].copy()
    if len(events) != len(clusters):
        raise ValueError(f"events {len(events)} != clusters {len(clusters)}")

    df = pd.DataFrame(
        {
            "event_id": events[args.event_col] if args.event_col in events.columns else np.arange(1, n + 1),
            "cluster_id": clusters["cluster_id"].astype(int).to_numpy(),
            "time": pd.to_numeric(events[args.time_col], errors="coerce").fillna(0.0),
            "alarm_text": events[args.text_col].astype(str),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(gt_csv, index_col=0).to_numpy()
    gt = (gt > 0).astype(int)
    if gt.shape != (k, k):
        raise ValueError(f"GT shape {gt.shape} != ({k},{k})")
    mask = ~np.eye(k, dtype=bool)
    gt_flat = gt[mask]

    if args.temperatures:
        temperatures = _parse_temperature_list(args.temperatures)
        n_calls = len(temperatures)
        if n_calls < 1:
            raise ValueError("need at least one value in --temperatures")
        tau_list = _tau_grid()
        user_table = build_user_table(df)
        cfg = {
            "k": k,
            "n_events_preset": n_events,
            "rows_sent": n,
            "llm_temperatures": temperatures,
            "n_api_calls": n_calls,
            "tau_grid": tau_list,
            "result_tau": float(args.result_tau),
            "model": args.model,
            "mock": bool(args.mock),
            "max_retries_on_bad_shape": int(args.max_retries_on_bad_shape),
        }
        (out_dir / "temperature_sweep_config.json").write_text(
            json.dumps(cfg, indent=2),
            encoding="utf-8",
        )
        matrices: list[np.ndarray] = []
        for ri, temp in enumerate(temperatures):
            rdir = out_dir / f"run_{ri:02d}"
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "config.json").write_text(
                json.dumps(
                    {
                        "llm_temperature": float(temp),
                        "run_index": ri,
                        "k": k,
                        "n_events_preset": n_events,
                        "rows_sent": n,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mock_seed = ri * 10_000
            a, raw = _one_inference(
                user_table=user_table,
                model=args.model,
                temperature=temp,
                mock=bool(args.mock),
                mock_seed=mock_seed,
                k=k,
                n_rows=n,
                max_parse_retries=int(args.max_retries_on_bad_shape),
            )
            (rdir / "openai_raw.txt").write_text(raw, encoding="utf-8")
            if not args.mock:
                (rdir / "parsed.json").write_text(
                    json.dumps({"A": a.tolist()}, indent=2),
                    encoding="utf-8",
                )
            idx = list(range(k))
            pd.DataFrame(a, index=idx, columns=idx).to_csv(
                rdir / "A_matrix.csv",
                index=True,
                header=True,
            )
            matrices.append(a)

        agg_rows: list[dict[str, float]] = []
        for tau in tau_list:
            shds: list[float] = []
            fdrs: list[float] = []
            tprs: list[float] = []
            for a in matrices:
                pred_flat = (a >= tau).astype(int)[mask]
                shd, fdr, tpr = metric(pred_flat, gt_flat)
                shds.append(shd)
                fdrs.append(fdr)
                tprs.append(tpr)
            sm, ss = float(np.mean(shds)), _std_sample(shds)
            fm, fs = float(np.mean(fdrs)), _std_sample(fdrs)
            tm, ts = float(np.mean(tprs)), _std_sample(tprs)
            agg_rows.append(
                {
                    "tau": float(tau),
                    "SHD_mean": sm,
                    "SHD_sd": ss,
                    "FDR_mean": fm,
                    "FDR_sd": fs,
                    "TPR_mean": tm,
                    "TPR_sd": ts,
                }
            )
        pd.DataFrame(agg_rows).to_csv(out_dir / "metrics_by_tau.csv", index=False)

        rt = float(args.result_tau)
        shds_r, fdrs_r, tprs_r = [], [], []
        for a in matrices:
            pred_flat = (a >= rt).astype(int)[mask]
            shd, fdr, tpr = metric(pred_flat, gt_flat)
            shds_r.append(shd)
            fdrs_r.append(fdr)
            tprs_r.append(tpr)
        sm, ss = float(np.mean(shds_r)), _std_sample(shds_r)
        fm, fs = float(np.mean(fdrs_r)), _std_sample(fdrs_r)
        tm, ts = float(np.mean(tprs_r)), _std_sample(tprs_r)
        print("Result：")
        print(f"SHD={sm:.3f} ± {ss:.3f} | FDR={fm:.3f} ± {fs:.3f} | TPR={tm:.3f} ± {ts:.3f}")
        return

    raw_path = out_dir / "openai_raw.txt"
    parsed_path = out_dir / "parsed.json"
    a_path = out_dir / "A_matrix.csv"
    idx = list(range(k))

    if args.mock:
        a = mock_a_matrix(k=k, seed=0)
        raw_path.write_text("MOCK (no API call)\n", encoding="utf-8")
    else:
        user_table = build_user_table(df)
        a, raw = _one_inference(
            user_table=user_table,
            model=args.model,
            temperature=float(args.temperature),
            mock=False,
            mock_seed=0,
            k=k,
            n_rows=n,
            max_parse_retries=int(args.max_retries_on_bad_shape),
        )
        raw_path.write_text(raw, encoding="utf-8")
        parsed_path.write_text(
            json.dumps({"A": a.tolist()}, indent=2),
            encoding="utf-8",
        )

    pd.DataFrame(a, index=idx, columns=idx).to_csv(a_path, index=True, header=True)

    print(f"Wrote: {a_path}")
    print(f"GT:    {gt_csv}")

    if args.summary_only:
        th = float(args.threshold)
        pred_flat = (a >= th).astype(int)[mask]
        shd, fdr, tpr = metric(pred_flat, gt_flat)
        print(f"\n(threshold τ={th}, off-diagonal)\n")
        print("Result：")
        print(f"SHD={shd:.3f} | FDR={fdr:.3f} | TPR={tpr:.3f}")
        return

    print("\nOff-diagonal; binarize pred >= τ. (single LLM run → no mean/sd across seeds)\n")
    hdr = "threshold     SHD       FDR       TPR"
    print(hdr)
    print("-" * len(hdr))
    for th in _tau_grid():
        pred_flat = (a >= th).astype(int)[mask]
        shd, fdr, tpr = metric(pred_flat, gt_flat)
        print(f"{th:11.3f}   {shd:5.1f}   {fdr:7.4f}   {tpr:7.4f}")

    th05 = 0.5
    pred_flat = (a >= th05).astype(int)[mask]
    shd, fdr, tpr = metric(pred_flat, gt_flat)
    print(f"\n(threshold τ={th05}, off-diagonal)\n")
    print("Result：")
    print(f"SHD={shd:.3f} | FDR={fdr:.3f} | TPR={tpr:.3f}")


if __name__ == "__main__":
    main()
