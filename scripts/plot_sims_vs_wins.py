import os
import argparse
import math
from typing import List, Tuple

import numpy as np


def se_binomial(k: int, n: int) -> float:
    if n <= 0:
        return float('nan')
    p = k / n
    return math.sqrt(p * (1.0 - p) / n)


def main():
    ap = argparse.ArgumentParser(description='Plot sims vs wins (and win rate) for two settings.')
    ap.add_argument('--out-dir', type=str, default='./compare')
    # Defaults from the user-provided results
    ap.add_argument('--sims', type=int, nargs='+', default=[25, 50, 100, 200])
    ap.add_argument('--baseline-wld', type=int, nargs='+', default=[136, 64, 0, 163, 37, 0, 177, 23, 0, 169, 31, 0],
                    help='Wins,Losses,Draws repeated for each sims in order')
    ap.add_argument('--ours-wld', type=int, nargs='+', default=[159, 41, 0, 177, 23, 0, 176, 24, 0, 181, 19, 0],
                    help='Wins,Losses,Draws repeated for each sims in order')
    args = ap.parse_args()

    sims = np.asarray(args.sims, dtype=np.int32)

    def parse_wld(arr: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(arr, dtype=np.int32)
        if a.size % 3 != 0:
            raise ValueError('wld array length must be a multiple of 3 (wins,losses,draws per sims)')
        a = a.reshape(-1, 3)
        return a[:, 0], a[:, 1], a[:, 2]

    bw, bl, bd = parse_wld(args.baseline_wld)
    ow, ol, od = parse_wld(args.ours_wld)

    if not (len(sims) == len(bw) == len(ow)):
        raise ValueError('Length mismatch between sims and W/L/D entries')

    # Win rates and 95% CI (normal approx)
    bn = (bw + bl + bd).astype(np.int32)
    on = (ow + ol + od).astype(np.int32)
    bwr = bw / np.maximum(1, bn)
    owr = ow / np.maximum(1, on)
    bse = np.array([se_binomial(int(k), int(n)) for k, n in zip(bw, bn)])
    ose = np.array([se_binomial(int(k), int(n)) for k, n in zip(ow, on)])
    bci = 1.96 * bse
    oci = 1.96 * ose

    os.makedirs(args.out_dir, exist_ok=True)

    # Plot wins
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3.8))
        plt.plot(sims, bw, marker='o', label='baseline')
        plt.plot(sims, ow, marker='o', label='ours')
        plt.xlabel('MCTS sims')
        plt.ylabel('Wins vs Alpha-Beta (d=3)')
        plt.title('Wins vs MCTS sims')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'sims_vs_wins.png'), dpi=600)
        plt.close()

        # Plot win rate with 95% CI
        plt.figure(figsize=(6, 3.8))
        plt.errorbar(sims, bwr, yerr=bci, fmt='-o', label='baseline', capsize=3)
        plt.errorbar(sims, owr, yerr=oci, fmt='-o', label='ours', capsize=3)
        plt.xlabel('MCTS sims')
        plt.ylabel('Win rate vs Alpha-Beta (d=3)')
        plt.title('Win rate vs MCTS sims (95% CI)')
        plt.ylim(0.5, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'sims_vs_winrate.png'), dpi=600)
        plt.close()

        # Print quick summary
        print('Wins (baseline):', list(map(int, bw)))
        print('Wins (ours)    :', list(map(int, ow)))
        print('WinRate (baseline):', [f'{x:.3f}±{e:.3f}' for x, e in zip(bwr, bci)])
        print('WinRate (ours)    :', [f'{x:.3f}±{e:.3f}' for x, e in zip(owr, oci)])
        print(f'Figures saved to: {args.out_dir}')
    except Exception as e:
        # Matplotlib missing; fallback to console-only summary
        print('Matplotlib not available; skipping plots.')
        print('sims:', list(map(int, sims)))
        print('baseline wins:', list(map(int, bw)))
        print('ours wins:', list(map(int, ow)))


if __name__ == '__main__':
    main()

