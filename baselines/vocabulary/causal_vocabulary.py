"""
Hand-written lexical prior for cause_like / effect_like (teacher scores).

Replaces OpenAI / mock_heuristic in the role-distillation demo: same [0,1] targets,
text-only, deterministic given the text (plus a tiny hash tie-break like the mock).
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Tuple


def _stable01(s: str, salt: str) -> float:
    h = hashlib.sha256((salt + s).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / (2**64)


# Upstream / trigger-ish cues (industrial alarm wording aligned with chemical_500 sim)
_CAUSE_TERMS: Tuple[Tuple[str, float], ...] = (
    ("feed", 1.0),
    ("inlet", 1.0),
    ("suction", 1.0),
    ("dosing", 1.1),
    ("feeder", 1.0),
    ("catalyst", 0.9),
    ("train", 0.85),
    ("oscillation", 1.0),
    ("upstream", 1.2),
    ("root", 0.9),
    ("trigger", 1.0),
    ("supply", 0.9),
)

# Downstream / symptom-ish cues
_EFFECT_TERMS: Tuple[Tuple[str, float], ...] = (
    ("surge", 1.1),
    ("imbalance", 1.0),
    ("mismatch", 1.0),
    ("jitter", 1.0),
    ("pressure", 1.0),
    ("column", 0.95),
    ("reactor", 0.85),
    ("heat", 0.9),
    ("downstream", 1.2),
    ("symptom", 1.0),
    ("recovery", 0.8),
    ("latch", 0.85),
    ("warning", 0.7),
)


def _score_axis(text_lower: str, weighted_terms: Iterable[Tuple[str, float]]) -> float:
    s = 0.0
    for term, w in weighted_terms:
        if term in text_lower:
            s += w
    return s


def vocabulary_role_scores(text: str) -> Tuple[float, float]:
    """
    Map alarm text to (cause_like, effect_like) in [0, 1].

    Base ~0.35 so ambiguous lines stay mid-range; matched terms add mass;
    small per-text jitter preserves spread without using timestamps.
    """
    t = str(text).lower()
    c_raw = _score_axis(t, _CAUSE_TERMS)
    e_raw = _score_axis(t, _EFFECT_TERMS)
    # saturate contribution from lexicon hits
    c_boost = min(c_raw, 4.0) * 0.11
    e_boost = min(e_raw, 4.0) * 0.11
    c_score = 0.35 + c_boost + 0.22 * _stable01(t, "vocab_c")
    e_score = 0.35 + e_boost + 0.22 * _stable01(t, "vocab_e")
    c_score = float(max(0.0, min(1.0, c_score)))
    e_score = float(max(0.0, min(1.0, e_score)))
    return c_score, e_score
