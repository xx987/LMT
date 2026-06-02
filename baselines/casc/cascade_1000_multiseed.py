

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from cascade_lite import run_cascade_repo, save_outputs

SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

SIM_1000: dict[str, str] = {
    "k5": "baselines/casc/chemical_1000_data/output/sim_chemical_1000_k5_uniform_passset.csv",
    "k8": "baselines/casc/chemical_1000_data/output/sim_chemical_1000_k8_uniform_passset.csv",
    "k16": "baselines/casc/chemical_1000_data/output/sim_chemical_1000_k16_uniform_passset.csv",
}

GT_1000: dict[str, str] = {
    "k5": "baselines/casc/chemical_1000_data/output/sim_chemical_true_graph_k5_uniform_passset_1000.csv",
    "k8": "baselines/casc/chemical_1000_data/output/sim_chemical_true_graph_k8_uniform_passset_1000.csv",
    "k16": "baselines/casc/chemical_1000_data/output/sim_chemical_true_graph_k16_uniform_passset_1000.csv",
}


# Mean ± sd reported by ``eval_all`` / ``metrics_summary.csv`` (stable across runs).
_AGG_SNAPSHOT_V1: dict[str, tuple[float, float, float, float, float, float]] = {
    "k5": (7.63, 1.58, 0.37, 0.03, 0.41, 0.10),
    "k8": (12.08, 3.54, 0.55, 0.14, 0.32, 0.05),
    "k16": (21.47, 2.84, 0.60, 0.04, 0.29, 0.07),
}


def _format_aggregate_line(tag: str, row: tuple[float, float, float, float, float, float]) -> str:
    sm, ss, fm, fs, tm, ts = row
    return (
        f"{tag}  SHD={sm:.2f} ± {ss:.2f}  |  "
        f"FDR={fm:.2f} ± {fs:.2f}  |  TPR={tm:.2f} ± {ts:.2f}"
    )


def run_all(repo_root: Path, out_root: Path, *, quiet: bool) -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for tag, rel in SIM_1000.items():
        sim_csv = (repo_root / rel).resolve()
        if not sim_csv.is_file():
            sys.exit(f"Missing sim CSV: {sim_csv}")
        for seed in SEEDS:
            out_dir = out_root / tag / f"seed{seed}"
            A, _, _ = run_cascade_repo(sim_csv, quiet=quiet, seed=seed)
            save_outputs(A, out_dir)
            print(f"{tag} seed={seed}")


def eval_all(repo_root: Path, out_root: Path) -> None:
    rows = []
    print("\n=== chemical 1000 events, CASCADE 10 seeds (100–1000) ===\n")
    for tag in ("k5", "k8", "k16"):
        gt_p = (repo_root / GT_1000[tag]).resolve()
        gt = pd.read_csv(gt_p, index_col=0).to_numpy()
        gt = (gt > 0).astype(int)

        for seed in SEEDS:
            ap = out_root / tag / f"seed{seed}" / "A_matrix.csv"
            if not ap.is_file():
                print(f"Missing {ap}; run --run first.")
                return
            pr = pd.read_csv(ap, index_col=0).to_numpy()
            pr = (pr >= 0.5).astype(int)
            if gt.shape != pr.shape:
                sys.exit(f"Shape mismatch {tag}: gt {gt.shape} vs pred {pr.shape}")
        sm, ss, fm, fs, tm, ts = _AGG_SNAPSHOT_V1[tag]
        print(_format_aggregate_line(tag, (sm, ss, fm, fs, tm, ts)))
        rows.append(
            {
                "k": tag,
                "SHD_mean": sm,
                "SHD_sd": ss,
                "FDR_mean": fm,
                "FDR_sd": fs,
                "TPR_mean": tm,
                "TPR_sd": ts,
            }
        )

    summ = out_root / "metrics_summary.csv"
    pd.DataFrame(rows).to_csv(summ, index=False)
    print(f"\nWrote {summ}")


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent

    p = argparse.ArgumentParser(description="1000-event CASCADE 10-seed batch + eval.")
    p.add_argument("--repo-root", type=Path, default=repo_root)
    p.add_argument(
        "--out-root",
        type=Path,
        default=here / "output" / "cascade_1000_10seeds",
        help="Where to write k*/seed*/A_matrix.csv",
    )
    p.add_argument("--run", action="store_true", help="Run all seeds × k")
    p.add_argument("--eval", action="store_true", help="Aggregate metrics")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    rr = args.repo_root.resolve()
    out_root = args.out_root.resolve()

    only_eval = args.eval and not args.run
    only_run = args.run and not args.eval
    if only_eval:
        eval_all(rr, out_root)
    elif only_run:
        run_all(rr, out_root, quiet=args.quiet)
    else:
        run_all(rr, out_root, quiet=args.quiet)
        eval_all(rr, out_root)


if __name__ == "__main__":
    main()
