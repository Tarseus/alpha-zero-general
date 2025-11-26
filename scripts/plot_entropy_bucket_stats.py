import os
import argparse
from typing import List, Tuple

import numpy as np


def _compute_bucket_stats(
    path: str,
    bins: List[float],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    data = np.load(path)
    entropy = np.asarray(data["entropy"], dtype=np.float64)
    changed = np.asarray(data["changed"], dtype=bool)
    outcome = np.asarray(data["outcome"], dtype=np.int8)
    actor_is_robust = np.asarray(data["actor_is_robust"], dtype=bool)

    bins_arr = np.asarray(bins, dtype=np.float64)
    if bins_arr.ndim != 1 or bins_arr.size < 2:
        raise ValueError("bins must be a 1D array with at least 2 elements")

    nb = bins_arr.size - 1
    labels: List[str] = []
    change_ratio = np.full(nb, np.nan, dtype=np.float64)
    improvement = np.full(nb, np.nan, dtype=np.float64)

    for i in range(nb):
        lo, hi = float(bins_arr[i]), float(bins_arr[i + 1])
        labels.append(f"[{lo:.2f},{hi:.2f})")

        mask_bucket = (entropy >= lo) & (entropy < hi)
        cnt = int(mask_bucket.sum())
        if cnt == 0:
            continue

        c = changed[mask_bucket].astype(np.float64)
        change_ratio[i] = float(c.mean()) if c.size > 0 else float("nan")

        robust_mask = mask_bucket & actor_is_robust
        base_mask = mask_bucket & (~actor_is_robust)

        def _win_rate(m: np.ndarray) -> float:
            if not np.any(m):
                return float("nan")
            wins = (outcome[m] > 0).astype(np.float64)
            return float(wins.mean()) if wins.size > 0 else float("nan")

        wr_robust = _win_rate(robust_mask)
        wr_base = _win_rate(base_mask)
        if np.isfinite(wr_robust) and np.isfinite(wr_base):
            improvement[i] = wr_robust - wr_base

    return labels, change_ratio, improvement


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot entropy-bucket stats per sims:\n"
            "  (1) action-change ratio per bucket,\n"
            "  (2) local correctness improvement per bucket."
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
        "--bins",
        type=float,
        nargs="+",
        default=[0.0, 0.33, 0.66, 1.01],
        help="Entropy bucket boundaries (e.g. 0.0 0.33 0.66 1.01).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="./robust_vs_baseline_plots",
        help="Directory to save per-sims bar charts.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Matplotlib may not be available; handle gracefully.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available; skipping plots.", flush=True)
        for s in args.sims:
            path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{s}.npz")
            if not os.path.exists(path):
                print(f"Missing data file for sims={s}: {path}", flush=True)
                continue
            labels, change_ratio, improvement = _compute_bucket_stats(path, args.bins)
            print(f"[sims={s}] entropy buckets: {labels}", flush=True)
            print(f"  change_ratio: {[f'{x:.4f}' if np.isfinite(x) else 'nan' for x in change_ratio]}", flush=True)
            print(f"  improvement : {[f'{x:.4f}' if np.isfinite(x) else 'nan' for x in improvement]}", flush=True)
        return

    for sims in args.sims:
        path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{sims}.npz")
        if not os.path.exists(path):
            print(f"Missing data file for sims={sims}: {path}", flush=True)
            continue

        labels, change_ratio, improvement = _compute_bucket_stats(path, args.bins)
        idx = np.arange(len(labels))

        # 1) 动作改变比例柱状图
        vals_cr = np.where(np.isfinite(change_ratio), change_ratio, 0.0)
        plt.figure(figsize=(5.5, 3.8))
        plt.bar(idx, vals_cr, tick_label=labels, color="#4e79a7")
        plt.xlabel("Entropy bucket")
        plt.ylabel("Action change ratio")
        plt.title(f"Action change ratio per entropy bucket (sims={sims})")
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path_cr = os.path.join(args.out_dir, f"entropy_change_ratio_sims{sims}.png")
        plt.savefig(out_path_cr, dpi=600)
        plt.close()

        # 2) 局部正确率提升柱状图（robust-win-rate - baseline-win-rate）
        vals_impr = np.where(np.isfinite(improvement), improvement, 0.0)
        plt.figure(figsize=(5.5, 3.8))
        plt.bar(idx, vals_impr, tick_label=labels, color="#f28e2b")
        plt.xlabel("Entropy bucket")
        plt.ylabel("Local correctness improvement")
        plt.title(f"Local correctness improvement per bucket (sims={sims})")
        # symmetric y-limits around 0 for easier comparison
        finite_impr = improvement[np.isfinite(improvement)]
        if finite_impr.size > 0:
            max_abs = float(np.max(np.abs(finite_impr)))
            ylim = max(0.05, max_abs * 1.2)
            plt.ylim(-ylim, ylim)
        plt.axhline(0.0, color="#444444", linewidth=1.0)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_path_impr = os.path.join(args.out_dir, f"entropy_local_improvement_sims{sims}.png")
        plt.savefig(out_path_impr, dpi=600)
        plt.close()

        print(f"[sims={sims}] saved:", flush=True)
        print(f"  {out_path_cr}", flush=True)
        print(f"  {out_path_impr}", flush=True)


if __name__ == "__main__":
    main()

