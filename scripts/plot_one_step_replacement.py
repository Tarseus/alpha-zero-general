import os
import argparse
from typing import Dict, Any, Sequence, Tuple

import numpy as np


def _load_one_step_file(
    data_dir: str,
    sims: int,
    eval_sims: int | None,
    num_repeat: int,
) -> Dict[str, Any]:
    """
    读取 one-step replacement 输出文件：
      one_step_replacement_sims{sims}_eval{eval_sims}_rep{num_repeat}.npz
    """
    if eval_sims is None:
        eval_sims = sims
    fname = f"one_step_replacement_sims{sims}_eval{eval_sims}_rep{num_repeat}.npz"
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"one-step file not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def _load_robust_file(data_dir: str, sims: int) -> Dict[str, Any]:
    """
    读取 robust_vs_baseline_sims{sims}.npz（需要已经由 augment_robust_npz_with_q.py 写入 delta_Q）。
    """
    fname = f"robust_vs_baseline_sims{sims}.npz"
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"robust_vs_baseline file not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def _phase_bucket_indices(
    move_index: np.ndarray,
    phase_bins: Sequence[int],
) -> Tuple[np.ndarray, Sequence[str]]:
    """
    按 move_index 把样本分到不同“阶段桶”里，并返回 bucket id 和可读标签。
    """
    bins = np.asarray(phase_bins, dtype=np.int64)
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("phase_bins must be 1D with at least 2 elements")

    # digitize: 返回 1..len(bins)，我们减 1 得到 0..len(bins)-1
    bid = np.digitize(move_index, bins, right=False) - 1
    bid[(bid < 0) | (bid >= bins.size - 1)] = -1

    labels = [f"[{int(bins[i])},{int(bins[i+1])})" for i in range(bins.size - 1)]
    return bid, labels


def _compute_phase_stats(
    move_index: np.ndarray,
    delta_win: np.ndarray,
    phase_bins: Sequence[int],
) -> Tuple[Sequence[str], np.ndarray, np.ndarray]:
    """
    计算每个阶段桶内：
      - 平均 delta_win
      - 样本数量
    """
    bid, labels = _phase_bucket_indices(move_index, phase_bins)
    nb = len(labels)
    mean_dw = np.full(nb, np.nan, dtype=np.float64)
    counts = np.zeros(nb, dtype=np.int64)

    for i in range(nb):
        mask = (bid == i) & np.isfinite(delta_win)
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            mean_dw[i] = float(np.mean(delta_win[mask].astype(np.float64)))

    return labels, mean_dw, counts


def _compute_deltaq_correlation(
    delta_q: np.ndarray,
    delta_win: np.ndarray,
) -> float | float:
    """
    返回 ΔQ 与 delta_win 的皮尔逊相关系数（只在均为有限值的样本上计算）。
    """
    mask = np.isfinite(delta_q) & np.isfinite(delta_win)
    if not np.any(mask):
        return float("nan")
    x = delta_q[mask].astype(np.float64)
    y = delta_win[mask].astype(np.float64)
    if x.size < 2:
        return float("nan")
    C = np.corrcoef(x, y)
    return float(C[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Visualize one-step replacement effects:\n"
            "  1) Which game phase benefits most from action changes;\n"
            "  2) Correlation between ΔQ and win-rate improvement."
        )
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="./robust_vs_baseline_data",
        help="Directory containing robust_vs_baseline_sims*.npz and one_step_replacement_*.npz.",
    )
    ap.add_argument(
        "--sims",
        type=int,
        nargs="+",
        default=[25],
        help="List of sims values to visualize.",
    )
    ap.add_argument(
        "--eval-sims",
        type=int,
        default=None,
        help="eval_sims used in one-step replacement (default: same as sims).",
    )
    ap.add_argument(
        "--num-repeat",
        type=int,
        default=32,
        help="num_repeat used in one-step replacement.",
    )
    ap.add_argument(
        "--phase-bins",
        type=int,
        nargs="+",
        default=[1, 20, 40, 60],
        help="Game phase buckets based on move_index, e.g. 1 20 40 60.",
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
        print("Matplotlib not available; will only print numerical stats.")
        plt = None

    for sims in args.sims:
        print(f"\n=== sims={sims} ===")
        one_step = _load_one_step_file(args.data_dir, sims, args.eval_sims, args.num_repeat)
        robust = _load_robust_file(args.data_dir, sims)

        move_index = np.asarray(one_step["move_index"], dtype=np.int32)
        delta_win = np.asarray(one_step["delta_win"], dtype=np.float64)
        idx_in_file = np.asarray(one_step["idx_in_file"], dtype=np.int64)

        # 与 robust_vs_baseline_sims*.npz 对齐得到 delta_Q
        delta_Q_full = np.asarray(robust.get("delta_Q"), dtype=np.float64)
        if delta_Q_full.ndim != 1:
            raise ValueError("delta_Q in robust_vs_baseline file must be 1D")
        delta_q = delta_Q_full[idx_in_file]

        # 1) 哪个阶段的动作改动对胜率影响最大：看各 phase 的平均 delta_win
        labels, mean_dw, counts = _compute_phase_stats(move_index, delta_win, args.phase_bins)
        print("Phase buckets:", labels)
        print("  counts     :", [int(c) for c in counts])
        print("  mean Δwin  :", [f"{x:.4f}" if np.isfinite(x) else "nan" for x in mean_dw])

        if plt is not None:
            idx = np.arange(len(labels))
            vals = np.where(np.isfinite(mean_dw), mean_dw, 0.0)
            plt.figure(figsize=(5.5, 3.5))
            plt.bar(idx, vals, tick_label=labels, color="#4e79a7")
            plt.axhline(0.0, color="#444444", linewidth=1.0)
            plt.xlabel("Game phase (by move_index)")
            plt.ylabel("Mean win-rate gain (Δwin)")
            plt.title(f"One-step win-rate gain per phase (sims={sims})")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            out_path_phase = os.path.join(args.out_dir, f"one_step_phase_gain_sims{sims}.png")
            plt.savefig(out_path_phase, dpi=600)
            plt.close()
            print(f"  saved phase plot: {out_path_phase}")

        # 2) ΔQ 与胜率提升相关性
        corr = _compute_deltaq_correlation(delta_q, delta_win)
        print(f"  corr(delta_Q, Δwin) = {corr:.4f}" if np.isfinite(corr) else "  corr(delta_Q, Δwin) = nan")

        if plt is not None:
            mask = np.isfinite(delta_q) & np.isfinite(delta_win)
            if np.any(mask):
                plt.figure(figsize=(5.0, 4.0))
                plt.scatter(
                    delta_q[mask],
                    delta_win[mask],
                    s=6,
                    alpha=0.4,
                    edgecolors="none",
                    color="#59a14f",
                )
                plt.axhline(0.0, color="#444444", linewidth=1.0)
                plt.axvline(0.0, color="#444444", linewidth=1.0)
                plt.xlabel("ΔQ (Q_rob - Q_base)")
                plt.ylabel("Δwin (robust - baseline)")
                plt.title(f"ΔQ vs win-rate gain (sims={sims})")
                plt.tight_layout()
                out_path_scatter = os.path.join(args.out_dir, f"one_step_deltaQ_vs_gain_sims{sims}.png")
                plt.savefig(out_path_scatter, dpi=600)
                plt.close()
                print(f"  saved ΔQ–Δwin scatter: {out_path_scatter}")


if __name__ == "__main__":
    main()

