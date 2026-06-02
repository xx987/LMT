#!/usr/bin/env python3
"""
End-to-end demo of: LLM soft role scores -> distill W_c, W_e -> text pair score r_ij.

Uses N=50 chemical k=5 CSV by default (see ``output/sim_chemical_50_k5_uniform_passset.csv``).

Modes
-----
1) Real LLM (OpenAI Chat Completions): set OPENAI_API_KEY and pass --use-openai
   (requires: pip install openai)

2) No API: default --mock-llm=heuristic  (deterministic scores from alarm wording only; no time to LLM/mock)

Optional: --llm-cache path.csv  to save / reuse LLM targets so you do not re-query.

Example
-------
  cd chemical_50_event_repro
  python3 llm_role_distillation_demo.py --max-events 100 --epochs 200

  OPENAI_API_KEY=... python3 llm_role_distillation_demo.py --use-openai --max-events 50 \\
      --llm-cache output/llm_role_scores_cache_n50.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# LLM interfaces
# ---------------------------------------------------------------------------


def _stable01(s: str, salt: str) -> float:
    h = hashlib.sha256((salt + s).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / (2**64)


def mock_heuristic_scores(text: str) -> Tuple[float, float]:
    """
    Cheap deterministic stand-in for LLM scores (no network).
    Text-only (no timestamps), to match the LLM interface below.
    """
    t = str(text).lower()
    # crude lexical cues (demo only)
    cause_hints = ("feed", "oscillation", "train", "dosing", "inlet", "suction")
    effect_hints = ("surge", "imbalance", "mismatch", "jitter", "pressure", "column")
    c_raw = sum(1 for w in cause_hints if w in t)
    e_raw = sum(1 for w in effect_hints if w in t)
    c_score = 0.30 + 0.15 * c_raw + 0.30 * _stable01(t, "c")
    e_score = 0.30 + 0.15 * e_raw + 0.30 * _stable01(t, "e")
    return float(np.clip(c_score, 0.0, 1.0)), float(np.clip(e_score, 0.0, 1.0))


def openai_role_scores(
    texts: Sequence[str],
    *,
    model: str,
    temperature: float,
) -> List[Tuple[float, float]]:
    """Call OpenAI once per row. Alarm text only — no timestamps (avoids temporal leakage)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai: pip install openai") from e

    client = OpenAI()
    out: List[Tuple[float, float]] = []

    system = (

        "You are labeling industrial alarm events using ONLY the alarm text provided.\n\n"
        "Task:\n"
        "For each alarm text, output two independent continuous scores in [0,1]:\n"
        "- cause_like: degree that the wording suggests an upstream trigger/root-cause role.\n"
        "- effect_like: degree that the wording suggests a downstream symptom/consequence role.\n\n"
        "Critical constraints:\n"
        "1) Use ONLY the alarm text. Do NOT assume timestamps, order, or external context.\n"
        "2) The two scores are independent evidences, NOT a binary class split.\n"
        "3) Do NOT set effect_like = 1 - cause_like by default.\n"
        "4) The two scores do NOT need to sum to 1.\n"
        "5) If evidence is ambiguous, it is valid for both scores to be near 0.5.\n"
        "6) Avoid default template pairs such as (0.7, 0.3) unless strongly justified by wording.\n"
        "7) Use fine-grained values and the full range when appropriate.\n\n"
        "Scoring anchors (guidance):\n"
        "- 0.1: almost no evidence for that role\n"
        "- 0.3: weak evidence\n"
        "- 0.5: ambiguous / neutral\n"
        "- 0.7: clear evidence\n"
        "- 0.9: very strong evidence\n\n"
        "Output format (strict):\n"
        'Return JSON ONLY, with exactly these keys and 3 decimals: {"cause_like": 0.000, "effect_like": 0.000}\n'
        "No markdown, no explanation text, no extra keys.\n\n"
    )

    for i, text in enumerate(texts):
        user = f"Alarm text:\n{text}\n\nReturn JSON only with cause_like and effect_like (3 decimals)."
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        content = content.strip()
        # strip markdown fences if any
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        obj = json.loads(content)
        c = float(obj["cause_like"])
        e = float(obj["effect_like"])
        c = float(np.clip(c, 0.0, 1.0))
        e = float(np.clip(e, 0.0, 1.0))
        out.append((c, e))
        if (i + 1) % 10 == 0:
            print(f"  LLM queried {i+1}/{len(texts)} ...")
        time.sleep(0.05)  # light throttle
    return out


# ---------------------------------------------------------------------------
# Torch model (two role projections + scalar heads + pair score)
# ---------------------------------------------------------------------------


class RoleDistillModel(nn.Module):
    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.Wc = nn.Linear(d_in, d_h)
        self.We = nn.Linear(d_in, d_h)
        self.head_c = nn.Linear(d_h, 1)
        self.head_e = nn.Linear(d_h, 1)

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        g: (B, d_in)
        returns hc, he, c_hat, e_hat each (B, d_h) or (B,) for hats
        """
        hc = self.Wc(g)
        he = self.We(g)
        c_hat = torch.sigmoid(self.head_c(hc)).squeeze(-1)
        e_hat = torch.sigmoid(self.head_e(he)).squeeze(-1)
        return hc, he, c_hat, e_hat

    @staticmethod
    def pair_logits(hc_i: torch.Tensor, he_j: torch.Tensor) -> torch.Tensor:
        """hc_i: (d_h,), he_j: (d_h,) -> scalar"""
        return (hc_i * he_j).sum()


def train_role_heads(
    model: RoleDistillModel,
    g: torch.Tensor,
    c_tgt: torch.Tensor,
    e_tgt: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = g.to(device)
    c_tgt = c_tgt.to(device)
    e_tgt = e_tgt.to(device)
    model.to(device)
    model.train()
    for ep in range(epochs):
        _, _, c_hat, e_hat = model(g)
        loss = torch.mean((c_hat - c_tgt) ** 2) + torch.mean((e_hat - e_tgt) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"  epoch {ep+1}/{epochs}  L_role={loss.item():.6f}")


@dataclass
class RowBundle:
    event_ids: List
    texts: List[str]
    g: np.ndarray  # (N, d_in)
    c_llm: np.ndarray
    e_llm: np.ndarray


def load_or_compute_llm_scores(
    texts: Sequence[str],
    *,
    use_openai: bool,
    mock_mode: str,
    llm_cache: Path | None,
    openai_temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if llm_cache and llm_cache.exists():
        df = pd.read_csv(llm_cache)
        if len(df) != len(texts):
            raise ValueError(f"Cache rows {len(df)} != data rows {len(texts)}")
        return df["c_llm"].to_numpy(float), df["e_llm"].to_numpy(float)

    if use_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        pairs = openai_role_scores(
            texts,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=openai_temperature,
        )
        c = np.array([p[0] for p in pairs], dtype=np.float32)
        e = np.array([p[1] for p in pairs], dtype=np.float32)
    else:
        if mock_mode != "heuristic":
            raise ValueError("unknown --mock-llm mode")
        pairs = [mock_heuristic_scores(tx) for tx in texts]
        c = np.array([p[0] for p in pairs], dtype=np.float32)
        e = np.array([p[1] for p in pairs], dtype=np.float32)

    if llm_cache:
        llm_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"c_llm": c, "e_llm": e}).to_csv(llm_cache, index=False)
        print(f"Saved LLM/mock scores to {llm_cache}")

    return c, e


def main() -> None:
    p = argparse.ArgumentParser(description="LLM role scores + distill Wc, We + r_ij demo")
    p.add_argument(
        "--input",
        default=str(_HERE / "output/sim_chemical_50_k5_uniform_passset.csv"),
        help="Event CSV (needs text + numeric time columns)",
    )
    p.add_argument("--text-col", default="Alarm Text")
    p.add_argument(
        "--time-col",
        default="time",
        help="CSV time column (loaded for compatibility; never sent to LLM / mock teacher)",
    )
    p.add_argument("--event-col", default="event_id")
    p.add_argument("--max-events", type=int, default=120, help="Cap rows for speed")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-openai", action="store_true", help="Use OpenAI API (needs openai + key)")
    p.add_argument(
        "--openai-temperature",
        type=float,
        default=0.0,
        help="Temperature for OpenAI Chat Completions when --use-openai is set",
    )
    p.add_argument("--mock-llm", default="heuristic", choices=["heuristic"])
    p.add_argument("--llm-cache", default=None, help="CSV with columns c_llm,e_llm to reuse scores")
    p.add_argument("--pair-sample", type=int, default=12, help="How many random directed pairs to print")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    raw = Path(args.input).expanduser()
    if raw.is_file():
        in_path = raw.resolve()
    elif (_HERE / raw).is_file():
        in_path = (_HERE / raw).resolve()
    elif (Path.cwd() / raw).is_file():
        in_path = (Path.cwd() / raw).resolve()
    else:
        raise FileNotFoundError(f"--input not found: {args.input} (also tried {_HERE / raw})")
    df = pd.read_csv(in_path)
    if len(df) > args.max_events:
        df = df.iloc[: args.max_events].copy()
    texts = df[args.text_col].astype(str).tolist()
    # Time is not used for LLM or mock role scores (text-only to avoid temporal leakage).
    _ = pd.to_numeric(df[args.time_col], errors="coerce").fillna(0.0).to_numpy()
    event_ids = df[args.event_col].tolist() if args.event_col in df.columns else list(range(1, len(df) + 1))

    llm_path: Path | None = None
    if args.llm_cache:
        lc = Path(args.llm_cache).expanduser()
        if lc.is_absolute() or lc.is_file():
            llm_path = lc
        elif (_HERE / lc).parent.exists() or (_HERE / lc).exists():
            llm_path = (_HERE / lc).resolve()
        else:
            llm_path = (Path.cwd() / lc).resolve()
    c_llm, e_llm = load_or_compute_llm_scores(
        texts,
        use_openai=args.use_openai,
        mock_mode=args.mock_llm,
        llm_cache=llm_path,
        openai_temperature=args.openai_temperature,
    )

    print(f"Loading embeddings: {args.embed_model} ({len(texts)} rows) ...")
    st = SentenceTransformer(args.embed_model)
    g_np = st.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    d_in = int(g_np.shape[1])

    model = RoleDistillModel(d_in=d_in, d_h=args.d_hidden)
    g_t = torch.from_numpy(g_np.astype(np.float32))
    c_t = torch.from_numpy(c_llm.astype(np.float32))
    e_t = torch.from_numpy(e_llm.astype(np.float32))

    print("Training role distillation (MSE to LLM/mock targets) ...")
    train_role_heads(model, g_t, c_t, e_t, epochs=args.epochs, lr=args.lr, device=device)

    model.eval()
    with torch.no_grad():
        hc, he, c_hat, e_hat = model(g_t.to(device))
        hc = hc.cpu()
        he = he.cpu()
        c_hat = c_hat.cpu()
        e_hat = e_hat.cpu()

    print("\nFirst 8 rows: LLM/mock vs predicted")
    for i in range(min(8, len(texts))):
        print(
            f"  id={event_ids[i]}  c: {c_llm[i]:.3f}->{c_hat[i]:.3f}   "
            f"e: {e_llm[i]:.3f}->{e_hat[i]:.3f}   | {texts[i][:64]}..."
        )

    # random pairs i != j
    rng = np.random.default_rng(args.seed)
    N = len(texts)
    print(f"\nSample directed text plausibility r_ij^text (sigmoid of <h_i^(c), h_j^(e)>):")
    for _ in range(args.pair_sample):
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
        if i == j:
            j = (j + 1) % N
        logit = RoleDistillModel.pair_logits(hc[i], he[j]).item()
        r = float(torch.sigmoid(torch.tensor(logit)))
        print(
            f"  {event_ids[i]} -> {event_ids[j]}: r_text={r:.4f}  "
            f"(logit={logit:.3f})"
        )

    out_pt = _HERE / "output" / "llm_role_distill_demo.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "d_in": d_in, "d_h": args.d_hidden}, out_pt)
    print(f"\nSaved trained weights to {out_pt}")


if __name__ == "__main__":
    main()
