import os
import argparse
from typing import List, Tuple

import numpy as np


def _compute_qgap_bucket_stats(
    path: str,
    gap_bins: List[float],
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      labels: 每个桶的字符串标签
      change_ratio: 每个桶中的 changed 比例
      mean_delta_q: 每个桶中（changed 且有 ΔQ）的平均 ΔQ
      robust_win_rate: 每个桶中 robust 执子方的局部胜率
    """
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
        labels.append(f"[{lo:.2f},{hi:.2f})")

        mask_bucket = np.isfinite(Q_gap) & (Q_gap >= lo) & (Q_gap < hi)
        cnt = int(mask_bucket.sum())
        if cnt == 0:
            continue

        # 改动频率
        c = changed[mask_bucket].astype(np.float64)
        change_ratio[i] = float(c.mean()) if c.size > 0 else float("nan")

        # 桶内 ΔQ（只看 changed 且 ΔQ 有效）
        mask_dq = mask_bucket & changed & np.isfinite(delta_Q)
        if np.any(mask_dq):
            mean_delta_q[i] = float(delta_Q[mask_dq].mean())

        # 桶内 robust 执子方的局部胜率
        mask_rob = mask_bucket & actor_is_robust
        if np.any(mask_rob):
            wins = (outcome[mask_rob] > 0).astype(np.float64)
            robust_win_rate[i] = float(wins.mean()) if wins.size > 0 else float("nan")

    return labels, change_ratio, mean_delta_q, robust_win_rate


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot Q-gap bucket stats across sims:\n"
            "  - action-change ratio per ΔQ_gap bucket (still per-sims),\n"
            "  - ONE grouped bar chart for mean ΔQ (buckets × sims),\n"
            "  - ONE grouped bar chart for robust local win-rate (buckets × sims).\n"
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
        help="Directory to save plots.",
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

    # 1) 先逐 sims 计算桶统计，并保留改动频率单独画
    labels_ref: List[str] = []
    all_change_ratio = []
    all_mean_delta_q = []
    all_robust_wr = []

    for sims in args.sims:
        path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{sims}.npz")
        if not os.path.exists(path):
            print(f"Missing data file for sims={sims}: {path}")
            continue

        labels, change_ratio, mean_delta_q, robust_wr = _compute_qgap_bucket_stats(
            path, args.gap_bins
        )
        if not labels_ref:
            labels_ref = labels
        elif labels != labels_ref:
            raise ValueError("Bucket labels differ across sims; check gap_bins settings.")

        all_change_ratio.append(change_ratio)
        all_mean_delta_q.append(mean_delta_q)
        all_robust_wr.append(robust_wr)

        # 单独输出每个 sims 的改动频率图（保持原有功能）
        idx = np.arange(len(labels))
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
        print(f"[sims={sims}] saved change-ratio plot: {out_path_cr}")

    if not labels_ref or not all_mean_delta_q or not all_robust_wr:
        return

    sims_arr = np.asarray(args.sims, dtype=np.int32)
    nb = len(labels_ref)
    idx = np.arange(nb)
    S = len(sims_arr)
    width = 0.8 / max(S, 1)

    all_mean_delta_q_arr = np.asarray(all_mean_delta_q, dtype=np.float64)  # (S, B)
    all_robust_wr_arr = np.asarray(all_robust_wr, dtype=np.float64)  # (S, B)

    # 2) qgap_mean_deltaQ：每个桶画多条柱子，区分不同 sims
    plt.figure(figsize=(6.5, 3.8))
    print("="*80)
    print(all_mean_delta_q_arr)
    print("="*80)
    for i, sims in enumerate(sims_arr):
        vals = np.where(np.isfinite(all_mean_delta_q_arr[i]), all_mean_delta_q_arr[i], 0.0)
        x_offset = idx + (i - S / 2) * width + width / 2
        plt.bar(x_offset, vals, width=width, label=f"sims={int(sims)}")
    plt.xlabel("ΔQ_gap bucket")
    plt.ylabel("Mean ΔQ (changed roots)")
    plt.title("Mean ΔQ per Q-gap bucket (all sims)")
    plt.axhline(0.0, color="#444444", linewidth=1.0)
    plt.xticks(idx, labels_ref, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    out_path_dq_all = os.path.join(args.out_dir, "qgap_mean_deltaQ_all_sims.png")
    plt.savefig(out_path_dq_all, dpi=600)
    plt.close()

    # 3) qgap_local_winrate：每个桶多条柱子，对应不同 sims
    plt.figure(figsize=(6.5, 3.8))
    for i, sims in enumerate(sims_arr):
        vals = np.where(np.isfinite(all_robust_wr_arr[i]), all_robust_wr_arr[i], 0.0)
        x_offset = idx + (i - S / 2) * width + width / 2
        plt.bar(x_offset, vals, width=width, label=f"sims={int(sims)}")
    plt.xlabel("ΔQ_gap bucket")
    plt.ylabel("Robust local win rate")
    plt.title("Robust win rate per Q-gap bucket (all sims)")
    plt.ylim(0.0, 1.0)
    plt.axhline(0.5, color="#444444", linewidth=1.0, linestyle="--")
    plt.xticks(idx, labels_ref, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    out_path_wr_all = os.path.join(args.out_dir, "qgap_local_winrate_all_sims.png")
    plt.savefig(out_path_wr_all, dpi=600)
    plt.close()

    print("Saved combined Q-gap plots:")
    print(f"  {out_path_dq_all}")
    print(f"  {out_path_wr_all}")


if __name__ == "__main__":
    main()

