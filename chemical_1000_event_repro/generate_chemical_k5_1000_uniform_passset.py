

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (3, 4),
]


def build_true_graph() -> np.ndarray:
    A = np.zeros((5, 5), dtype=int)
    for i, j in EDGES:
        A[i, j] = 1
    return A


def text_for_cluster(c: int, rng: random.Random) -> str:
    templates = {
        0: ["ALARM60010-Feed pressure transient train {x}.", "ALARM60011-Feed oscillation train {x}."],
        1: ["ALARM61020-Reactor thermal drift unit {x}.", "ALARM61021-Reactor heat imbalance unit {x}."],
        2: ["ALARM62030-Catalyst dosing mismatch line {x}.", "ALARM62031-Catalyst feeder jitter line {x}."],
        3: ["ALARM63040-Column pressure surge section {x}.", "ALARM63041-Column deltaP anomaly section {x}."],
        4: ["ALARM64050-Recovery interlock sequence state {x}.", "ALARM64051-SIS recovery latch state {x}."],
    }
    return rng.choice(templates[c]).format(x=rng.randint(1, 4))


def generate_events(n_events: int, seed: int, start_dt: datetime) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if n_events != 1000:
        raise ValueError("This generator fixes n_events=1000 for controlled composition.")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    A = build_true_graph()

    episodes = n_events // 2  # 500 two-event episodes
    base = episodes // len(EDGES)
    rem = episodes % len(EDGES)
    edge_list: list[tuple[int, int]] = []
    for idx, e in enumerate(EDGES):
        cnt = base + (1 if idx < rem else 0)
        edge_list.extend([e] * cnt)
    rng.shuffle(edge_list)

    rows = []
    links = []
    t = 0.0
    eid = 1
    for cause, effect in edge_list:
        t += float(np_rng.uniform(180.0, 420.0))
        cause_eid = eid
        rows.append(
            {
                "event_id": cause_eid,
                "Alarm Text": text_for_cluster(cause, rng),
                "Set": (start_dt + timedelta(seconds=t)).strftime("%-m/%-d/%y %-H:%M:%S"),
                "time": round(t, 3),
                "cluster_id_true": cause,
                "cause_event_id_true": -1,
            }
        )
        eid += 1

        dt = float(np_rng.uniform(6.0, 18.0))
        child_t = t + dt
        child_eid = eid
        rows.append(
            {
                "event_id": child_eid,
                "Alarm Text": text_for_cluster(effect, rng),
                "Set": (start_dt + timedelta(seconds=child_t)).strftime("%-m/%-d/%y %-H:%M:%S"),
                "time": round(child_t, 3),
                "cluster_id_true": effect,
                "cause_event_id_true": cause_eid,
            }
        )
        links.append(
            {
                "cause_event_id": cause_eid,
                "cause_cluster": cause,
                "effect_event_id": child_eid,
                "effect_cluster": effect,
            }
        )
        eid += 1
        t = child_t

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    old_to_new = {int(old): i + 1 for i, old in enumerate(df["event_id"].tolist())}
    df["event_id"] = [old_to_new[int(x)] for x in df["event_id"]]
    df["cause_event_id_true"] = [old_to_new.get(int(x), -1) if int(x) != -1 else -1 for x in df["cause_event_id_true"]]

    link_df = pd.DataFrame(
        [
            {
                "cause_event_id": old_to_new[int(r["cause_event_id"])],
                "cause_cluster": int(r["cause_cluster"]),
                "effect_event_id": old_to_new[int(r["effect_event_id"])],
                "effect_cluster": int(r["effect_cluster"]),
            }
            for r in links
        ]
    ).sort_values("effect_event_id")

    return df, link_df, A


def main() -> None:
    p = argparse.ArgumentParser(description="Generate chemical K=5 N=1000 dataset with new DAG")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-graph", required=True)
    p.add_argument("--output-links", required=True)
    p.add_argument("--output-oracle-clusters", required=True)
    p.add_argument("--n-events", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-datetime", default="2026-01-01 08:00:00")
    args = p.parse_args()

    out_csv = Path(args.output_csv)
    out_graph = Path(args.output_graph)
    out_links = Path(args.output_links)
    out_oracle = Path(args.output_oracle_clusters)
    for x in (out_csv, out_graph, out_links, out_oracle):
        x.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(args.start_datetime, "%Y-%m-%d %H:%M:%S")
    df, links, A = generate_events(args.n_events, args.seed, start_dt)

    df.to_csv(out_csv, index=False)
    pd.DataFrame(A).to_csv(out_graph, index=True, header=True)
    links.to_csv(out_links, index=False)
    df[["cluster_id_true"]].rename(columns={"cluster_id_true": "cluster_id"}).to_csv(out_oracle, index=False)

    print(f"Saved events: {out_csv} ({len(df)} rows)")
    print(f"Saved true graph: {out_graph} (K={A.shape[0]}, edges={int(A.sum())})")
    print(f"Saved links: {out_links} ({len(links)} rows)")
    print(f"Saved oracle clusters: {out_oracle}")


if __name__ == "__main__":
    main()
