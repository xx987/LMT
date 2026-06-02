#!/usr/bin/env python3
"""
Run CASCADE (repository MDL implementation via baselines/casc/cascade_lite.py) on
chemical sim CSVs. Outputs A_matrix.csv for eval-style comparisons.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_BASELINE_DIR = Path(__file__).resolve().parent
if str(_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_DIR))

from cascade_lite import run_cascade_repo, save_outputs

# Data lives under baselines/casc/ (mirrors chemical_* repro output filenames).
PRESETS: dict[str, str] = {
    "500-k5": "baselines/casc/chemical_500_data/output/sim_chemical_500_k5_uniform_passset.csv",
    "500-k8": "baselines/casc/chemical_500_data/output/sim_chemical_500_k8_uniform_passset_compat.csv",
    "500-k16": "baselines/casc/chemical_500_data/output/sim_chemical_500_k16_cascadefriendly.csv",
    "1000-k5": "baselines/casc/chemical_1000_data/output/sim_chemical_1000_k5_uniform_passset.csv",
    "1000-k8": "baselines/casc/chemical_1000_data/output/sim_chemical_1000_k8_uniform_passset.csv",
    "1000-k16": "baselines/casc/chemical_1000_data/output/sim_chemical_1000_k16_uniform_passset.csv",
}


def main() -> None:
    here = _BASELINE_DIR
    default_root = here.parent.parent

    p = argparse.ArgumentParser(description="CASCADE (repo MDL) on chemical sim CSVs.")
    p.add_argument("--repo-root", type=Path, default=default_root)
    p.add_argument("--preset", type=str, default=None, help=f"One of: {', '.join(sorted(PRESETS))}, or 'all'.")
    p.add_argument("--events-csv", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (A_matrix.csv). Default: baselines/casc/output/cascade_lite_<preset>/.",
    )
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
    p.add_argument("-q", "--quiet", action="store_true", help="Less CASCADE debug logging.")
    p.add_argument(
        "--min-output-edges",
        type=int,
        default=None,
        help="Min off-diagonal edges in A (0=disable). Default: auto max(2, min(K//5, 4)).",
    )
    p.add_argument("--fallback-window", type=float, default=250.0, help="Lag window for fallback scores (time units of CSV).")
    p.add_argument("--fallback-tau", type=float, default=40.0, help="Exp decay for fallback scores.")
    p.add_argument("--seed", type=int, default=None, help="Tiny time jitter for repeatable multi-seed variation.")
    # Legacy no-ops (ignored; CASCADE uses MDL, not window scores)
    p.add_argument("--window", type=float, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--min-edge-score", type=float, default=None)
    args = p.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    repo_root = args.repo_root.resolve()

    def run_one(preset_name: str, sim_csv: Path, out_dir: Path | None) -> None:
        if not sim_csv.is_file():
            sys.exit(f"Missing sim CSV: {sim_csv}")
        if out_dir is None:
            out_dir = here / "output" / f"cascade_lite_{preset_name.replace('-', '_')}"
        out_dir = out_dir.resolve()
        A, length, n_pad = run_cascade_repo(
            sim_csv,
            search=args.search,
            topology=args.topology,
            candidate_delay=args.candidate_delay,
            precision=args.precision,
            dst=args.dst,
            align_mode=args.align,
            instant=args.instant,
            instant_idf=not args.no_instant_idf,
            init_with_self_loops=args.init_with_self_loops,
            no_topo=args.no_topo,
            quiet=args.quiet,
            min_output_edges=args.min_output_edges,
            fallback_window=args.fallback_window,
            fallback_tau=args.fallback_tau,
            seed=args.seed,
        )
        save_outputs(A, out_dir)
        extra = f" +{n_pad} fallback" if n_pad else ""
        print(f"Done {preset_name}: MDL length={length:.4f}{extra} -> {out_dir / 'A_matrix.csv'}")

    if args.events_csv is not None:
        sim_csv = args.events_csv.resolve()
        tag = sim_csv.stem
        run_one(tag, sim_csv, args.out_dir.resolve() if args.out_dir is not None else None)
        return

    if args.preset is None:
        p.error("Pass --preset, --preset all, or --events-csv")

    if args.preset == "all":
        if args.out_dir is not None:
            sys.exit("--out-dir cannot be used with --preset all.")
        for name in sorted(PRESETS.keys()):
            run_one(name, (repo_root / PRESETS[name]).resolve(), None)
        return

    if args.preset not in PRESETS:
        p.error(f"Unknown preset {args.preset!r}")
    run_one(
        args.preset,
        (repo_root / PRESETS[args.preset]).resolve(),
        args.out_dir.resolve() if args.out_dir is not None else None,
    )


if __name__ == "__main__":
    main()
