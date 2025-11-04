import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np


def _read_reanalyze_csv(dir_path: str, filename: str = 'reanalyze_metrics.csv') -> Dict[str, List[float]]:
    path = os.path.join(dir_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    out: Dict[str, List[float]] = {
        'js_pi': [],
        'top1_agree': [],
        'v_student': [],
        'v_root': [],
    }
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # js_pi
            v = row.get('js_pi')
            if v not in (None, ''):
                try:
                    out['js_pi'].append(float(v))
                except Exception:
                    pass
            # top1_agree
            v = row.get('top1_agree')
            if v not in (None, ''):
                try:
                    out['top1_agree'].append(int(v))
                except Exception:
                    pass
            # v_student
            v = row.get('v_student')
            if v not in (None, ''):
                try:
                    out['v_student'].append(float(v))
                except Exception:
                    pass
            # v_root (may be empty)
            v = row.get('v_root')
            if v not in (None, ''):
                try:
                    out['v_root'].append(float(v))
                except Exception:
                    pass
    return out


def _overlay_hist(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, title: str, out_path: str,
                  bins: int = 50):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 and b.size == 0:
        return
    plt.figure(figsize=(5.5, 3.8))
    if a.size > 0:
        plt.hist(a, bins=bins, alpha=0.6, label=label_a)
    if b.size > 0:
        plt.hist(b, bins=bins, alpha=0.6, label=label_b)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _bar_means(vals_a: np.ndarray, vals_b: np.ndarray, labels: Tuple[str, str], title: str, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    def _mean(x):
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size > 0 else float('nan')
    means = [_mean(vals_a), _mean(vals_b)]
    plt.figure(figsize=(4.5, 3.6))
    plt.bar([0, 1], means, tick_label=list(labels), color=['#4e79a7', '#f28e2b'])
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'count': 0, 'mean': math.nan, 'median': math.nan, 'std': math.nan,
                'p10': math.nan, 'p25': math.nan, 'p75': math.nan, 'p90': math.nan}
    return {
        'count': int(x.size),
        'mean': float(x.mean()),
        'median': float(np.median(x)),
        'std': float(x.std(ddof=0)),
        'p10': float(np.quantile(x, 0.10)),
        'p25': float(np.quantile(x, 0.25)),
        'p75': float(np.quantile(x, 0.75)),
        'p90': float(np.quantile(x, 0.90)),
    }


def main():
    ap = argparse.ArgumentParser(description='Compare two analysis outputs (reanalyze_metrics.csv) with overlaid plots.')
    ap.add_argument('--dir-a', required=True, type=str, help='Directory A containing reanalyze_metrics.csv')
    ap.add_argument('--dir-b', required=True, type=str, help='Directory B containing reanalyze_metrics.csv')
    ap.add_argument('--label-a', type=str, default='A')
    ap.add_argument('--label-b', type=str, default='B')
    ap.add_argument('--file', type=str, default='reanalyze_metrics.csv')
    ap.add_argument('--out-dir', type=str, default='./compare')
    args = ap.parse_args()

    data_a = _read_reanalyze_csv(args.dir_a, args.file)
    data_b = _read_reanalyze_csv(args.dir_b, args.file)

    # Convert to np arrays
    js_a = np.asarray(data_a['js_pi'], dtype=np.float64)
    js_b = np.asarray(data_b['js_pi'], dtype=np.float64)
    top_a = np.asarray(data_a['top1_agree'], dtype=np.float64)
    top_b = np.asarray(data_b['top1_agree'], dtype=np.float64)
    vs_a = np.asarray(data_a['v_student'], dtype=np.float64)
    vs_b = np.asarray(data_b['v_student'], dtype=np.float64)
    vr_a = np.asarray(data_a['v_root'], dtype=np.float64) if len(data_a['v_root']) > 0 else np.asarray([], dtype=np.float64)
    vr_b = np.asarray(data_b['v_root'], dtype=np.float64) if len(data_b['v_root']) > 0 else np.asarray([], dtype=np.float64)

    # Derived: absolute value error |v_student - v_root|
    def _abs_err(vs: np.ndarray, vr: np.ndarray) -> np.ndarray:
        if vs.size == 0 or vr.size == 0:
            return np.asarray([], dtype=np.float64)
        n = min(vs.size, vr.size)
        vs = vs[:n]
        vr = vr[:n]
        mask = np.isfinite(vs) & np.isfinite(vr)
        return np.abs(vs[mask] - vr[mask])

    ve_a = _abs_err(vs_a, vr_a)
    ve_b = _abs_err(vs_b, vr_b)

    os.makedirs(args.out_dir, exist_ok=True)

    # Overlaid histograms
    _overlay_hist(js_a, js_b, args.label_a, args.label_b,
                  title='JS(pi_student || pi_teacher)',
                  out_path=os.path.join(args.out_dir, 'js_overlaid.png'))

    _overlay_hist(vs_a, vs_b, args.label_a, args.label_b,
                  title='v_student distribution',
                  out_path=os.path.join(args.out_dir, 'v_student_overlaid.png'))

    _overlay_hist(vr_a, vr_b, args.label_a, args.label_b,
                  title='v_root distribution',
                  out_path=os.path.join(args.out_dir, 'v_root_overlaid.png'))

    _overlay_hist(ve_a, ve_b, args.label_a, args.label_b,
                  title='|v_student - v_root|',
                  out_path=os.path.join(args.out_dir, 'v_abs_error_overlaid.png'))

    # top1_agree bar means
    _bar_means(top_a, top_b, (args.label_a, args.label_b),
               title='Top-1 agreement rate',
               out_path=os.path.join(args.out_dir, 'top1_agree_bar.png'))

    # Summary CSV
    rows = []
    def add_rows(metric: str, xa: np.ndarray, xb: np.ndarray):
        sa = _summary_stats(xa)
        sb = _summary_stats(xb)
        rows.append({'metric': metric, 'setting': args.label_a, **sa})
        rows.append({'metric': metric, 'setting': args.label_b, **sb})
        # diff of means for quick glance
        try:
            dm = float(sa['mean']) - float(sb['mean'])
        except Exception:
            dm = float('nan')
        rows.append({'metric': metric + '_diff_mean', 'setting': f"{args.label_a}-{args.label_b}", 'count': '',
                     'mean': dm, 'median': '', 'std': '', 'p10': '', 'p25': '', 'p75': '', 'p90': ''})

    add_rows('js_pi', js_a, js_b)
    add_rows('top1_agree_rate', top_a, top_b)
    add_rows('v_student', vs_a, vs_b)
    add_rows('v_root', vr_a, vr_b)
    add_rows('v_abs_error', ve_a, ve_b)

    out_csv = os.path.join(args.out_dir, 'compare_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        cols = ['metric', 'setting', 'count', 'mean', 'median', 'std', 'p10', 'p25', 'p75', 'p90']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Console print quick summary
    print(f"Saved overlaid plots and summary to: {args.out_dir}")
    for r in rows:
        if r['metric'].endswith('_diff_mean'):
            print(f"{r['metric']}: {r['mean']:.4g}")


if __name__ == '__main__':
    main()

