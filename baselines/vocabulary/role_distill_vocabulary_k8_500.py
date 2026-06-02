#!/usr/bin/env python3
"""
k8, 500-event chemical sim: vocabulary teacher -> role distillation + cluster A_matrix.

Same pipeline as ``role_distill_vocabulary_k5_500.py``; defaults point at bundled k8 data
(same files as ``baselines/casc/chemical_500_data/output/`` CASCADE 500 bundle).

- ``data/sim_chemical_500_k8_uniform_passset_compat.csv``
- ``data/sim_chemical500_k8_uniform_passset_oracle_clusters.csv``
- ``data/sim_chemical_true_graph_k8_uniform_passset.csv`` (for eval; not read here)

Run from repo root:
  python3 baselines/vocabulary/role_distill_vocabulary_k8_500.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from causal_vocabulary import vocabulary_role_scores

_VOCAB = Path(__file__).resolve().parent
_ROOT = _VOCAB.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from llm_role_distillation_demo import RoleDistillModel, train_role_heads  # noqa: E402


def _default_input_csv() -> Path:
    return _VOCAB / "data" / "sim_chemical_500_k8_uniform_passset_compat.csv"


def _default_cluster_csv() -> Path:
    return _VOCAB / "data" / "sim_chemical500_k8_uniform_passset_oracle_clusters.csv"


def run_cluster_aggregate_to_a_matrix(
    *,
    events_df: pd.DataFrame,
    cluster_csv: Path,
    checkpoint: Path,
    out_csv: Path,
    out_npy: Path,
    text_col: str,
    time_col: str,
    embed_model: str,
    device: str,
) -> None:
    clusters_full = pd.read_csv(cluster_csv)
    if "cluster_id" not in clusters_full.columns:
        raise ValueError(f"{cluster_csv} must contain column 'cluster_id'")
    n = len(events_df)
    if len(clusters_full) < n:
        raise ValueError(f"cluster rows {len(clusters_full)} < event rows {n}")
    clusters_use = clusters_full.iloc[:n].copy()

    build_script = _ROOT / "build_cluster_prior_from_role_distill.py"
    if not build_script.is_file():
        raise FileNotFoundError(f"Missing {build_script}")

    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_events = out_dir / "_vocab_slice_events.csv"
    slice_clusters = out_dir / "_vocab_slice_clusters.csv"
    events_df.to_csv(slice_events, index=False)
    clusters_use.to_csv(slice_clusters, index=False)

    cmd = [
        sys.executable,
        str(build_script),
        "--input",
        str(slice_events.resolve()),
        "--cluster-csv",
        str(slice_clusters.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--text-col",
        text_col,
        "--time-col",
        time_col,
        "--embed-model",
        embed_model,
        "--out-csv",
        str(out_csv.resolve()),
        "--out-npy",
        str(out_npy.resolve()),
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True, cwd=str(_ROOT))


def load_or_compute_vocab_teacher(
    texts: list[str],
    *,
    cache: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache and cache.exists():
        df = pd.read_csv(cache)
        if len(df) != len(texts):
            raise ValueError(f"Teacher cache rows {len(df)} != data rows {len(texts)}")
        if "c_teacher" in df.columns and "e_teacher" in df.columns:
            return df["c_teacher"].to_numpy(float), df["e_teacher"].to_numpy(float)
        if "c_llm" in df.columns and "e_llm" in df.columns:
            return df["c_llm"].to_numpy(float), df["e_llm"].to_numpy(float)
        raise ValueError("Cache needs columns (c_teacher,e_teacher) or (c_llm,e_llm).")

    pairs = [vocabulary_role_scores(tx) for tx in texts]
    c = np.array([p[0] for p in pairs], dtype=np.float32)
    e = np.array([p[1] for p in pairs], dtype=np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"c_teacher": c, "e_teacher": e}).to_csv(cache, index=False)
        print(f"Saved vocabulary teacher scores to {cache}")
    return c, e


def main() -> None:
    p = argparse.ArgumentParser(description="Vocabulary teacher + role distillation (k8, 500 events)")
    p.add_argument(
        "--input",
        type=Path,
        default=_default_input_csv(),
        help="Event CSV (k8 500 sim; compat passset).",
    )
    p.add_argument("--text-col", default="Alarm Text")
    p.add_argument("--time-col", default="time", help="Loaded for parity only.")
    p.add_argument("--event-col", default="event_id")
    p.add_argument("--max-events", type=int, default=500)
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--teacher-cache", type=Path, default=None)
    p.add_argument("--pair-sample", type=int, default=12, help="Used only with --verbose.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--out-pt",
        type=Path,
        default=None,
        help="Default: baselines/vocabulary/output/vocab_role_distill_k8_500.pt",
    )
    p.add_argument("--cluster-csv", type=Path, default=_default_cluster_csv())
    p.add_argument(
        "--out-a-matrix-csv",
        type=Path,
        default=None,
        help="Default: baselines/vocabulary/output/vocab_k8_500_A_matrix.csv",
    )
    p.add_argument(
        "--out-a-matrix-npy",
        type=Path,
        default=None,
        help="Default: baselines/vocabulary/output/vocab_k8_500_A_matrix.npy",
    )
    p.add_argument("--skip-cluster-a", action="store_true")
    args = p.parse_args()

    inp = Path(args.input).resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    df = pd.read_csv(inp)
    if len(df) > args.max_events:
        df = df.iloc[: args.max_events].copy()
    texts = df[args.text_col].astype(str).tolist()
    _ = pd.to_numeric(df[args.time_col], errors="coerce").fillna(0.0).to_numpy()
    event_ids = df[args.event_col].tolist() if args.event_col in df.columns else list(range(1, len(df) + 1))

    here = Path(__file__).resolve().parent
    teacher_cache = args.teacher_cache
    if teacher_cache is None:
        teacher_cache = here / "output" / "vocab_teacher_scores_k8_500.csv"

    c_tgt, e_tgt = load_or_compute_vocab_teacher(texts, cache=Path(teacher_cache).resolve())

    if args.verbose:
        print(f"Loading embeddings: {args.embed_model} ({len(texts)} rows) ...")
    st = SentenceTransformer(args.embed_model)
    g_np = st.encode(texts, convert_to_numpy=True, show_progress_bar=args.verbose)
    d_in = int(g_np.shape[1])

    model = RoleDistillModel(d_in=d_in, d_h=args.d_hidden)
    g_t = torch.from_numpy(g_np.astype(np.float32))
    c_t = torch.from_numpy(c_tgt.astype(np.float32))
    e_t = torch.from_numpy(e_tgt.astype(np.float32))

    if args.verbose:
        print("Training role distillation (MSE to vocabulary teacher targets) ...")
    train_role_heads(
        model,
        g_t,
        c_t,
        e_t,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        verbose=args.verbose,
    )

    if args.verbose:
        model.eval()
        with torch.no_grad():
            hc, he, c_hat, e_hat = model(g_t.to(device))
            hc = hc.cpu()
            he = he.cpu()
            c_hat = c_hat.cpu()
            e_hat = e_hat.cpu()

        print("\nFirst 8 rows: vocabulary teacher vs predicted")
        for i in range(min(8, len(texts))):
            print(
                f"  id={event_ids[i]}  c: {c_tgt[i]:.3f}->{c_hat[i]:.3f}   "
                f"e: {e_tgt[i]:.3f}->{e_hat[i]:.3f}   | {texts[i][:64]}..."
            )

        rng = np.random.default_rng(args.seed)
        n = len(texts)
        print(f"\nSample directed text plausibility r_ij^text (sigmoid of <h_i^(c), h_j^(e)>):")
        for _ in range(args.pair_sample):
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j:
                j = (j + 1) % n
            logit = RoleDistillModel.pair_logits(hc[i], he[j]).item()
            r = float(torch.sigmoid(torch.tensor(logit)))
            print(f"  {event_ids[i]} -> {event_ids[j]}: r_text={r:.4f}  (logit={logit:.3f})")

    out_pt = args.out_pt
    if out_pt is None:
        out_pt = here / "output" / "vocab_role_distill_k8_500.pt"
    out_pt = Path(out_pt).resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "d_in": d_in, "d_h": args.d_hidden}, out_pt)
    print(f"\nSaved trained weights to {out_pt}")

    if not args.skip_cluster_a:
        out_a_csv = args.out_a_matrix_csv or (here / "output" / "vocab_k8_500_A_matrix.csv")
        out_a_npy = args.out_a_matrix_npy or (here / "output" / "vocab_k8_500_A_matrix.npy")
        cluster_p = Path(args.cluster_csv).resolve()
        if not cluster_p.is_file():
            raise FileNotFoundError(f"Cluster CSV not found: {cluster_p}")
        run_cluster_aggregate_to_a_matrix(
            events_df=df,
            cluster_csv=cluster_p,
            checkpoint=out_pt,
            out_csv=Path(out_a_csv).resolve(),
            out_npy=Path(out_a_npy).resolve(),
            text_col=args.text_col,
            time_col=args.time_col,
            embed_model=args.embed_model,
            device=args.device,
        )


if __name__ == "__main__":
    main()
