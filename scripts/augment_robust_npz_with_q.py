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

from pit3 import (  # type: ignore
    make_game,
    set_seed,
    NNet,
    _extract_root_Qs,
    _compute_q_gap,
)
from MCTS import MCTS  # type: ignore
from utils import dotdict  # type: ignore


def _build_mcts(game, nnet, sims: int) -> MCTS:
    args = dotdict(
        {
            "numMCTSSims": int(sims),
            "cpuct": 1.0,
            "use_dyn_c": False,
            "addRootNoise": False,
            "sym_eval": True,
        }
    )
    return MCTS(game, nnet, args)


def _augment_single_file(
    path: str,
    game,
    nnet,
    sims: int,
    overwrite: bool = False,
) -> None:
    if not os.path.exists(path):
        print(f"[augment] missing file, skip: {path}")
        return

    data = np.load(path)
    files = set(data.files)

    # Already has Q fields and not overwriting -> skip.
    if {"Q_base", "Q_rob", "delta_Q", "Q_gap"}.issubset(files) and not overwrite:
        print(f"[augment] {os.path.basename(path)} already has Q_* fields, skip.")
        return

    boards_before = np.asarray(data["board_before"])
    player_global = np.asarray(data["player_global"], dtype=np.int8)
    baseline_action = np.asarray(data["baseline_action"], dtype=np.int64)
    robust_action = np.asarray(data["robust_action"], dtype=np.int64)

    N = boards_before.shape[0]
    Q_base = np.full(N, np.nan, dtype=np.float32)
    Q_rob = np.full(N, np.nan, dtype=np.float32)
    delta_Q = np.full(N, np.nan, dtype=np.float32)
    Q_gap = np.full(N, np.nan, dtype=np.float32)

    print(f"[augment] {os.path.basename(path)}: positions={N}, sims={sims}")

    # For simplicity and determinism, use a fresh MCTS per position.
    for i in range(N):
        board = boards_before[i]
        cur_player = int(player_global[i])
        a_base = int(baseline_action[i])
        a_rob = int(robust_action[i])

        canonical = game.getCanonicalForm(board, cur_player)

        mcts = _build_mcts(game, nnet, sims)
        _ = mcts.getActionProb(canonical, temp=0)
        Qs = _extract_root_Qs(game, mcts, canonical)
        valids = game.getValidMoves(canonical, 1)

        if Qs is not None:
            if 0 <= a_base < len(Qs):
                Q_base[i] = float(Qs[a_base])
            if 0 <= a_rob < len(Qs):
                Q_rob[i] = float(Qs[a_rob])

            if np.isfinite(Q_base[i]) and np.isfinite(Q_rob[i]):
                delta_Q[i] = float(Q_rob[i] - Q_base[i])

            Q_gap[i] = float(_compute_q_gap(Qs, valids))

        if (i + 1) % 500 == 0 or i + 1 == N:
            print(f"  processed {i+1}/{N} positions", end="\r", flush=True)

    print()  # newline after progress

    # Merge back into a dict and save, preserving existing fields.
    out_dict = {k: data[k] for k in data.files}
    out_dict["Q_base"] = Q_base
    out_dict["Q_rob"] = Q_rob
    out_dict["delta_Q"] = delta_Q
    out_dict["Q_gap"] = Q_gap

    # Overwrite in-place for simplicity.
    np.savez_compressed(path, **out_dict)
    print(f"[augment] updated file with Q_* fields: {path}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Augment existing robust_vs_baseline_sims*.npz files with\n"
            "Q_base, Q_rob, delta_Q and Q_gap, without re-playing games.\n"
            "For each stored root position, a fresh baseline MCTS is run\n"
            "to estimate Q(s,a) under the same sims."
        )
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="./robust_vs_baseline_data",
        help="Directory containing robust_vs_baseline_sims*.npz files.",
    )
    ap.add_argument(
        "--sims",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200],
        help="List of sims values to process (must match filenames).",
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
        "--overwrite",
        action="store_true",
        help="Recompute Q_* fields even if they already exist in the .npz.",
    )
    args = ap.parse_args()

    set_seed(42)

    game = make_game(8)
    nnet = NNet(game)
    nnet.load_checkpoint(args.model_dir, args.model_file)

    for sims in args.sims:
        fname = f"robust_vs_baseline_sims{sims}.npz"
        path = os.path.join(args.data_dir, fname)
        _augment_single_file(path, game, nnet, sims=sims, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

