"""
Text -> embeddings; optional KMeans clusters k_i in {0..K-1} or load cluster_id CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing package 'sentence-transformers'. Install with:\n"
            "  python -m pip install sentence-transformers\n"
            "or: python -m pip install -r requirements-text-causal.txt\n"
            f"(use the same interpreter as this run: {sys.executable})"
        ) from e

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        normalize_embeddings=False,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    return np.asarray(emb, dtype=np.float32)


def cluster_embeddings(emb: np.ndarray, n_clusters: int, random_state: int = 0) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(emb).astype(np.int64)


def load_or_compute_clusters(
    texts: list[str],
    n_clusters: int,
    cluster_csv: Optional[str | Path],
    embed_model: str,
) -> Tuple[np.ndarray, np.ndarray]:
    emb = embed_texts(texts, model_name=embed_model)
    if cluster_csv is None:
        return emb, cluster_embeddings(emb, n_clusters)

    cdf = pd.read_csv(Path(cluster_csv))
    if "cluster_id" not in cdf.columns:
        raise ValueError(f"{cluster_csv} must contain column 'cluster_id'")
    if len(cdf) != len(texts):
        raise ValueError(
            f"cluster_csv rows ({len(cdf)}) must match event rows ({len(texts)}); "
            "use same sort order as input CSV after load_events_csv."
        )
    c = cdf["cluster_id"].to_numpy(dtype=np.int64)
    return emb, c
