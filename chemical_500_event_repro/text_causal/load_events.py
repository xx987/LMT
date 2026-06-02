"""
Load event table from CSV: one row per event (id, text, time), sorted by time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_events_csv(
    path: str | Path,
    event_col: str = "event_id",
    text_col: str = "Alarm Text",
    time_col: str = "time",
    parse_time_as_datetime: bool = False,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    for c in (event_col, text_col, time_col):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}. Found: {list(df.columns)}")

    if parse_time_as_datetime:
        dt = pd.to_datetime(df[time_col], errors="coerce")
        if dt.isna().any():
            raise ValueError(f"Failed to parse some datetimes in column '{time_col}'")
        t0 = dt.min()
        t_num = (dt - t0).dt.total_seconds().astype(np.float64)
    else:
        t_num = pd.to_numeric(df[time_col], errors="coerce")
        if t_num.isna().any():
            raise ValueError(f"Non-numeric time in '{time_col}' (use --parse-time if datetimes)")
        t_num = t_num.astype(np.float64)

    out = pd.DataFrame(
        {
            "event_id": df[event_col].astype(str),
            "text": df[text_col].fillna("").astype(str),
            "time": t_num,
        }
    )
    out = out.sort_values("time").reset_index(drop=True)
    return out


def events_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, list[str], list[str]]:
    t = df["time"].to_numpy(dtype=np.float64)
    if len(t) > 1 and np.any(np.diff(t) < 0):
        raise ValueError("times must be non-decreasing")
    return t, df["event_id"].tolist(), df["text"].tolist()
