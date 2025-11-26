import os
import argparse
from typing import List, Tuple

import numpy as np


def _compute_phase_change_ratio(
    path: str,
    phase_bins: List[float],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    按对局阶段（归一化步数）分桶，统计每个桶中的动作改动比例。

    定义：
      - 对每局 g，记该局最大步数为 L_g = max move_index。
      - 对该局中每一步 i，阶段位置 phase_i = move_index_i / L_g ∈ (0,1]。
      - 用 phase_bins 对 phase_i 分桶，例如 [0.0,0.33,0.66,1.01] -> early/mid/late。

    返回:
      labels: 每个阶段桶的字符串标签
      change_ratio: 每个桶中 changed 比例
      counts: 每个桶中的样本数
    """
    data = np.load(path)
    game_idx = np.asarray(data["game_index"], dtype=np.int32)
    move_idx = np.asarray(data["move_index"], dtype=np.int32)
    changed = np.asarray(data["changed"], dtype=bool)

    # 计算每局的最大步数 L_g
    uniq_games = np.unique(game_idx)
    max_move_per_game = {}
    for g in uniq_games:
        mask_g = game_idx == g
        max_move_per_game[int(g)] = int(move_idx[mask_g].max())

    # 为每个位置计算归一化阶段 phase ∈ (0,1]
    phases = np.empty_like(move_idx, dtype=np.float64)
    for g in uniq_games:
        L = max_move_per_game[int(g)]
        if L <= 0:
            continue
        mask_g = game_idx == g
        phases[mask_g] = move_idx[mask_g].astype(np.float64) / float(L)

    bins_arr = np.asarray(phase_bins, dtype=np.float64)
    if bins_arr.ndim != 1 or bins_arr.size < 2:
        raise ValueError("phase_bins must be a 1D array with at least 2 elements")

    nb = bins_arr.size - 1
    labels: List[str] = []
    change_ratio = np.full(nb, np.nan, dtype=np.float64)
    counts = np.zeros(nb, dtype=np.int64)

    for i in range(nb):
        lo, hi = float(bins_arr[i]), float(bins_arr[i + 1])
        labels.append(f"[{lo:.2f},{hi:.2f})")
        mask_bucket = np.isfinite(phases) & (phases >= lo) & (phases < hi)
        cnt = int(mask_bucket.sum())
        counts[i] = cnt
        if cnt == 0:
            continue
        c = changed[mask_bucket].astype(np.float64)
        change_ratio[i] = float(c.mean()) if c.size > 0 else float("nan")

    return labels, change_ratio, counts


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot action-change ratio vs game phase (early/mid/late), "
            "for multiple sims on the same figure.\n"
            "Game phase is defined by normalized move index within each game."
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
        help="List of sims values to process.",
    )
    ap.add_argument(
        "--phase-bins",
        type=float,
        nargs="+",
        default=[0.0, 0.33, 0.66, 1.01],
        help="Phase bucket boundaries on normalized move index (e.g. 0.0 0.33 0.66 1.01).",
    )
    ap.add_argument(
        "--out-path",
        type=str,
        default="./robust_vs_baseline_plots/phase_change_ratio_all_sims.png",
        help="Output path for the grouped bar chart PNG.",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available; skipping plot.")
        for s in args.sims:
            path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{s}.npz")
            if not os.path.exists(path):
                print(f"Missing data file for sims={s}: {path}")
                continue
            labels, change_ratio, counts = _compute_phase_change_ratio(
                path, args.phase_bins
            )
            print(f"[sims={s}] phase buckets: {labels}")
            print(
                f"  change_ratio: "
                f"{[f'{x:.4f}' if np.isfinite(x) else 'nan' for x in change_ratio]}"
            )
            print(f"  counts      : {[int(c) for c in counts]}")
        return

    labels_ref: List[str] = []
    all_ratios = []

    for s in args.sims:
        path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{s}.npz")
        if not os.path.exists(path):
            print(f"Missing data file for sims={s}: {path}")
            continue
        labels, ratios, _ = _compute_phase_change_ratio(path, args.phase_bins)
        if not labels_ref:
            labels_ref = labels
        elif labels != labels_ref:
            raise ValueError("Phase bucket labels differ across sims; check phase-bins.")
        all_ratios.append(ratios)

    if not labels_ref or not all_ratios:
        print("No valid data loaded; nothing to plot.")
        return

    sims_arr = np.asarray(args.sims, dtype=np.int32)
    nb = len(labels_ref)
    idx = np.arange(nb)
    S = len(all_ratios)
    width = 0.8 / max(S, 1)

    all_ratios_arr = np.asarray(all_ratios, dtype=np.float64)  # (S, B)

    plt.figure(figsize=(6.5, 3.8))
    for i, sims in enumerate(sims_arr):
        vals = (
            np.where(np.isfinite(all_ratios_arr[i]), all_ratios_arr[i], 0.0)
            if i < all_ratios_arr.shape[0]
            else np.zeros(nb, dtype=np.float64)
        )
        x_offset = idx + (i - S / 2) * width + width / 2
        plt.bar(x_offset, vals, width=width, label=f"sims={int(sims)}")

    plt.xlabel("Game phase (normalized move index)")
    plt.ylabel("Action change ratio")
    plt.title("Robust-root action change ratio vs game phase (all sims)")
    plt.ylim(0.0, 1.0)
    plt.xticks(idx, labels_ref, rotation=0, ha="center")
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out_path, dpi=600)
    plt.close()

    print(f"Saved phase change-ratio plot: {args.out_path}")


if __name__ == "__main__":
    main()

