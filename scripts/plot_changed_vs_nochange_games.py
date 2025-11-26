import os
import argparse
from typing import List

import numpy as np


def _compute_game_level_stats(path: str):
    data = np.load(path)
    game_idx = np.asarray(data["game_index"], dtype=np.int32)
    changed = np.asarray(data["changed"], dtype=bool)
    actor_is_robust = np.asarray(data["actor_is_robust"], dtype=bool)
    outcome = np.asarray(data["outcome"], dtype=np.int8)

    uniq_games = np.unique(game_idx)
    res_changed = []
    res_unchanged = []

    for g in uniq_games:
        mask_g = game_idx == g
        has_change = bool(np.any(changed[mask_g]))

        mask_rob = mask_g & actor_is_robust
        if not np.any(mask_rob):
            continue
        r = int(outcome[mask_rob][0])  # robust side result in this game

        if has_change:
            res_changed.append(r)
        else:
            res_unchanged.append(r)

    def _win_rate(rs: List[int]) -> float:
        if not rs:
            return float("nan")
        arr = np.asarray(rs, dtype=np.int32)
        return float((arr > 0).mean())

    total_games = len(uniq_games)
    n_changed = len(res_changed)
    n_unchanged = len(res_unchanged)
    wr_changed = _win_rate(res_changed)
    wr_unchanged = _win_rate(res_unchanged)

    return total_games, n_changed, n_unchanged, wr_changed, wr_unchanged


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot robust-root win rate on changed vs no-change games, "
            "using robust_vs_baseline_sims*.npz.\n"
            "Requires fields: game_index, changed, actor_is_robust, outcome."
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
        help="List of sims values to include on the x-axis.",
    )
    ap.add_argument(
        "--out-path",
        type=str,
        default="./compare/robust_vs_baseline_changed_vs_nochange_games.png",
        help="Output path for the grouped bar chart PNG.",
    )
    args = ap.parse_args()

    sims_arr = np.asarray(args.sims, dtype=np.int32)
    wr_changed = []
    wr_nochange = []
    frac_changed = []

    for s in sims_arr:
        path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{s}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data file for sims={s}: {path}")

        total_games, n_changed, n_unchanged, w_ch, w_no = _compute_game_level_stats(path)
        frac = n_changed / float(total_games) if total_games > 0 else float("nan")

        wr_changed.append(w_ch)
        wr_nochange.append(w_no)
        frac_changed.append(frac)

    # If matplotlib is unavailable, print summary only.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available; skipping plot.")
        print("sims:", list(map(int, sims_arr)))
        print("win_rate_changed:", [float(x) for x in wr_changed])
        print("win_rate_nochange:", [float(x) for x in wr_nochange])
        print("frac_changed_games:", [float(x) for x in frac_changed])
        return

    x = np.arange(len(sims_arr))
    width = 0.35

    plt.figure(figsize=(6.0, 3.8))
    vals_ch = np.where(np.isfinite(wr_changed), wr_changed, 0.0)
    vals_nc = np.where(np.isfinite(wr_nochange), wr_nochange, 0.0)
    plt.bar(x - width / 2, vals_ch, width=width, label="changed games", color="#4e79a7")
    plt.bar(x + width / 2, vals_nc, width=width, label="no-change games", color="#f28e2b")

    plt.xticks(x, list(map(int, sims_arr)))
    plt.xlabel("MCTS sims")
    plt.ylabel("Robust win rate")
    plt.title("Robust-root win rate: changed vs no-change games")
    plt.ylim(0.0, 1.0)
    plt.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out_path, dpi=600)
    plt.close()

    print("sims:", list(map(int, sims_arr)))
    print("win_rate_changed:", [f"{x:.4f}" if np.isfinite(x) else "nan" for x in wr_changed])
    print("win_rate_nochange:", [f"{x:.4f}" if np.isfinite(x) else "nan" for x in wr_nochange])
    print("frac_changed_games:", [f"{x:.4f}" if np.isfinite(x) else "nan" for x in frac_changed])


if __name__ == "__main__":
    main()

