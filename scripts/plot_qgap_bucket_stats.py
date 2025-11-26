import os
import argparse
from typing import List

import numpy as np


def _compute_qgap_bucket_stats(
    path: str,
    gap_bins: List[float],
):
    data = np.load(path)
    Q_gap = np.asarray(data["Q_gap"], dtype=np.float64)
    delta_Q = np.asarray(data["delta_Q"], dtype=np.float64)
    changed = np.asarray(data["changed"], dtype=bool)
    actor_is_robust = np.asarray(data["actor_is_robust"], dtype=bool)
    outcome = np.asarray(data["outcome"], dtype=np.int8)

    bins_arr = np.asarray(gap_bins, dtype=np.float64)
    if bins_arr.ndim != 1 or bins_arr.size < 2:
        raise ValueError("gap_bins must be a 1D array with at least 2 elements")

    nb = bins_arr.size - 1
    labels: List[str] = []
    change_ratio = np.full(nb, np.nan, dtype=np.float64)
    mean_delta_q = np.full(nb, np.nan, dtype=np.float64)
    robust_win_rate = np.full(nb, np.nan, dtype=np.float64)

    for i in range(nb):
        lo, hi = float(bins_arr[i]), float(bins_arr[i + 1])
        labels.append(f"[{lo:.3f},{hi:.3f})")

        mask_bucket = np.isfinite(Q_gap) & (Q_gap >= lo) & (Q_gap < hi)
        cnt = int(mask_bucket.sum())
        if cnt == 0:
            continue

        c = changed[mask_bucket].astype(np.float64)
        change_ratio[i] = float(c.mean()) if c.size > 0 else float("nan")

        mask_dq = mask_bucket & changed & np.isfinite(delta_Q)
        if np.any(mask_dq):
            mean_delta_q[i] = float(delta_Q[mask_dq].mean())

        mask_rob = mask_bucket & actor_is_robust
        if np.any(mask_rob):
            wins = (outcome[mask_rob] > 0).astype(np.float64)
            robust_win_rate[i] = float(wins.mean()) if wins.size > 0 else float("nan")

    return labels, change_ratio, mean_delta_q, robust_win_rate


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot Q-gap bucket stats per sims:\n"
            "  (1) action-change ratio per ΔQ_gap bucket,\n"
            "  (2) mean ΔQ in each bucket (on changed roots),\n"
            "  (3) robust local win-rate per bucket.\n"
            "Requires fields: Q_gap, delta_Q, changed, actor_is_robust, outcome."
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
        "--gap-bins",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 1.0],
        help="Q-gap bucket boundaries (e.g. 0.0 0.02 0.05 1.0).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="./robust_vs_baseline_plots",
        help="Directory to save per-sims bar charts.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available; skipping plots.")
        for s in args.sims:
            path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{s}.npz")
            if not os.path.exists(path):
                print(f"Missing data file for sims={s}: {path}")
                continue
            labels, change_ratio, mean_delta_q, robust_wr = _compute_qgap_bucket_stats(
                path, args.gap_bins
            )
            print(f"[sims={s}] Q-gap buckets: {labels}")
            print(
                f"  change_ratio: "
                f"{[f'{x:.4f}' if np.isfinite(x) else 'nan' for x in change_ratio]}"
            )
            print(
                f"  mean_delta_Q: "
                f"{[f'{x:.4f}' if np.isfinite(x) else 'nan' for x in mean_delta_q]}"
            )
            print(
                f"  robust_win_rate: "
                f"{[f'{x:.4f}' if np.isfinite(x) else 'nan' for x in robust_wr]}"
            )
        return

    for sims in args.sims:
        path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{sims}.npz")
        if not os.path.exists(path):
            print(f"Missing data file for sims={sims}: {path}")
            continue

        labels, change_ratio, mean_delta_q, robust_wr = _compute_qgap_bucket_stats(
            path, args.gap_bins
        )
        idx = np.arange(len(labels))

        # 1) 改动频率
        vals_cr = np.where(np.isfinite(change_ratio), change_ratio, 0.0)
        plt.figure(figsize=(5.5, 3.5))
        plt.bar(idx, vals_cr, tick_label=labels, color="#4e79a7")
        plt.xlabel("ΔQ_gap bucket")
        plt.ylabel("Action change ratio")
        plt.title(f"Action change ratio per Q-gap bucket (sims={sims})")
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path_cr = os.path.join(args.out_dir, f"qgap_change_ratio_sims{sims}.png")
        plt.savefig(out_path_cr, dpi=600)
        plt.close()

        # 2) 平均 ΔQ（只在 changed roots 上）
        vals_dq = np.where(np.isfinite(mean_delta_q), mean_delta_q, 0.0)
        plt.figure(figsize=(5.5, 3.5))
        plt.bar(idx, vals_dq, tick_label=labels, color="#f28e2b")
        plt.xlabel("ΔQ_gap bucket")
        plt.ylabel("Mean ΔQ (changed roots)")
        plt.title(f"Mean ΔQ per Q-gap bucket (sims={sims})")
        plt.axhline(0.0, color="#444444", linewidth=1.0)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path_dq = os.path.join(args.out_dir, f"qgap_mean_deltaQ_sims{sims}.png")
        plt.savefig(out_path_dq, dpi=600)
        plt.close()

        # 3) 局部 robust 胜率
        vals_wr = np.where(np.isfinite(robust_wr), robust_wr, 0.0)
        plt.figure(figsize=(5.5, 3.5))
        plt.bar(idx, vals_wr, tick_label=labels, color="#59a14f")
        plt.xlabel("ΔQ_gap bucket")
        plt.ylabel("Robust local win rate")
        plt.title(f"Robust win rate per Q-gap bucket (sims={sims})")
        plt.ylim(0.0, 1.0)
        plt.axhline(0.5, color="#444444", linewidth=1.0, linestyle="--")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path_wr = os.path.join(args.out_dir, f"qgap_local_winrate_sims{sims}.png")
        plt.savefig(out_path_wr, dpi=600)
        plt.close()

        print(f"[sims={sims}] saved Q-gap plots:")
        print(f"  {out_path_cr}")
        print(f"  {out_path_dq}")
        print(f"  {out_path_wr}")


if __name__ == "__main__":
    main()

