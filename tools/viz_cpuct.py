import argparse
import os
import math
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)


def plot_timeseries(rows, game_id, outdir):
    data = [r for r in rows if r['game_id'] == game_id and r['logged_side'] == 'dynamic']
    if not data:
        return None
    x = [int(r['turn']) for r in data]
    c = [to_float(r['c']) for r in data]
    B = [to_float(r['B']) for r in data]
    E = [to_float(r['E']) for r in data]
    danger = [to_float(r['danger']) for r in data]
    Hn = [to_float(r['H_norm']) for r in data]
    Hc = [to_float(r['H_counts']) for r in data]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title(f'Game {game_id} dynamics')
    ax1.set_xlabel('Turn (dynamic moves)')
    ax1.set_ylabel('c / entropy')
    ax1.plot(x, c, label='c', color='tab:blue')
    ax1.plot(x, Hn, label='H_norm', color='tab:green', linestyle='--')
    ax1.plot(x, Hc, label='H_counts', color='tab:olive', linestyle=':')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('B / E / danger')
    ax2.plot(x, B, label='B (valid)', color='tab:purple', alpha=0.6)
    ax2.plot(x, E, label='E (empties)', color='tab:red', alpha=0.6)
    ax2.plot(x, danger, label='danger', color='tab:orange', alpha=0.8)
    ax2.legend(loc='upper right')

    out = os.path.join(outdir, f'timeseries_{game_id}.png')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = (~np.isnan(x)) & (~np.isnan(y))
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def scatter(ax, x, y, xlabel, ylabel, color):
    ax.scatter(x, y, s=12, alpha=0.5, color=color)
    # best-fit line
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    m = (~np.isnan(xv)) & (~np.isnan(yv))
    if m.sum() >= 2:
        k, b = np.polyfit(xv[m], yv[m], 1)
        xs = np.linspace(np.nanmin(xv[m]), np.nanmax(xv[m]), 100)
        ys = k * xs + b
        ax.plot(xs, ys, color=color, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_correlations(rows, outdir):
    data = [r for r in rows if r['logged_side'] == 'dynamic' and r['c'] not in ('', None)]
    if not data:
        return None, {}
    c = [to_float(r['c']) for r in data]
    B = [to_float(r['B']) for r in data]
    E = [to_float(r['E']) for r in data]
    rq = [to_float(r['r_q']) for r in data]
    danger = [to_float(r['danger']) for r in data]
    Hn = [to_float(r['H_norm']) for r in data]
    Hc = [to_float(r['H_counts']) for r in data]

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    scatter(axs[0, 0], B, c, 'B (valid moves)', 'c', 'tab:blue')
    scatter(axs[0, 1], E, c, 'E (empties)', 'c', 'tab:red')
    scatter(axs[0, 2], rq, c, 'r_q (Q std tnh)', 'c', 'tab:green')
    scatter(axs[1, 0], danger, c, 'danger ratio', 'c', 'tab:orange')
    scatter(axs[1, 1], Hn, c, 'H_norm (policy)', 'c', 'tab:purple')
    scatter(axs[1, 2], Hc, c, 'H_counts (visits)', 'c', 'tab:olive')
    fig.suptitle('c vs features (dynamic turns)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(outdir, 'correlations.png')
    fig.savefig(out, dpi=160)
    plt.close(fig)

    summary = {
        'corr_c_B': corr(B, c),
        'corr_c_E': corr(E, c),
        'corr_c_rq': corr(rq, c),
        'corr_c_danger': corr(danger, c),
        'corr_c_Hnorm': corr(Hn, c),
        'corr_c_Hcounts': corr(Hc, c),
    }
    return out, summary


def compute_rates(rows):
    data = [r for r in rows if r['logged_side'] == 'dynamic']
    if not data:
        return {}
    total = len(data)
    dang = sum(1 for r in data if str(r['chosen_is_danger']).lower() in ('true', '1'))
    return {
        'danger_choice_rate': dang / total if total else float('nan'),
        'samples': total,
    }


def main():
    ap = argparse.ArgumentParser(description='Visualize cpuct metrics (no heatmaps)')
    ap.add_argument('--csv', required=True, help='Input metrics CSV (from collect_cpuct_metrics.py)')
    ap.add_argument('--outdir', default='temp/viz', help='Output directory for figures')
    ap.add_argument('--example-game', type=str, default=None, help='Game id for time-series plot (e.g., A-1)')
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    rows = load_rows(args.csv)

    # time series for one example game
    if args.example_game is None:
        # pick the first game with dynamic logs
        dyn_games = [r['game_id'] for r in rows if r['logged_side'] == 'dynamic']
        args.example_game = dyn_games[0] if dyn_games else None
    ts_path = None
    if args.example_game:
        ts_path = plot_timeseries(rows, args.example_game, args.outdir)

    # correlations
    corr_path, corr_summary = plot_correlations(rows, args.outdir)
    rates = compute_rates(rows)

    # write summary txt
    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write('Correlation (c vs features)\n')
        for k, v in corr_summary.items():
            f.write(f'{k}: {v}\n')
        f.write('\nDangerous move rate\n')
        for k, v in rates.items():
            f.write(f'{k}: {v}\n')
        if ts_path:
            f.write(f'Example time-series figure: {ts_path}\n')
        if corr_path:
            f.write(f'Correlation figure: {corr_path}\n')

    print(f'Wrote figures to {args.outdir}')


if __name__ == '__main__':
    main()

