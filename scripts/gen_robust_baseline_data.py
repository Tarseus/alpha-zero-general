import os
import sys
import argparse

import numpy as np

# Ensure project root (containing pit3.py) is on sys.path even when this
# script is invoked via an absolute path from another working directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pit3 import (
    make_game,
    set_seed,
    NNet,
    collect_robust_vs_baseline_data,
)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate robust-root vs baseline per-move data for plotting.\n"
            "For each sims value, runs robust-root vs baseline matches and "
            "saves an .npz file with per-position statistics."
        )
    )
    ap.add_argument(
        "--sims",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200],
        help="List of MCTS simulations per move to evaluate.",
    )
    ap.add_argument(
        "--games-per-match",
        type=int,
        default=200,
        help="Total games per sims (both colors combined).",
    )
    ap.add_argument(
        "--model-dir",
        type=str,
        default="./models/",
        help="Directory containing the model checkpoint.",
    )
    ap.add_argument(
        "--model-file",
        type=str,
        default="baseline.pth.tar",
        help="Model checkpoint filename.",
    )
    ap.add_argument(
        "--robust-frac",
        type=float,
        default=0.6,
        help="Fraction parameter for RobustRootMCTSPlayer.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="./robust_vs_baseline_data",
        help="Directory to save generated .npz files.",
    )
    args = ap.parse_args()

    set_seed(42)

    game = make_game(8)
    nnet = NNet(game)
    nnet.load_checkpoint(args.model_dir, args.model_file)

    os.makedirs(args.out_dir, exist_ok=True)

    for sims in args.sims:
        print(f"[gen] sims={sims}, games={args.games_per_match} ...")
        data = collect_robust_vs_baseline_data(
            game,
            nnet,
            sims=sims,
            games=args.games_per_match,
            robust_frac=args.robust_frac,
        )
        if not data:
            print(f"[gen] sims={sims}: no data collected, skipping.")
            continue

        out_path = os.path.join(
            args.out_dir,
            f"robust_vs_baseline_sims{sims}.npz",
        )
        np.savez_compressed(out_path, **data)
        num_positions = int(data["changed"].size)
        print(f"[gen] saved {out_path} (positions={num_positions})")


if __name__ == "__main__":
    main()
