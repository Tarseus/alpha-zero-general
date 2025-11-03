import os
import sys
import csv
import math
import argparse
from typing import List, Tuple, Optional
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import dotdict
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as NNet
from MCTS import MCTS
def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))
    return 0.5 * (_kl(p, m) + _kl(q, m))
def kl_divergence(p_log: np.ndarray, q_prob: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) where p is given as log-probabilities and q as probabilities.
    Useful if you store student outputs as log-softmax. Here we use straight probs.
    """
    p = np.exp(np.asarray(p_log, dtype=np.float64))
    q = np.asarray(q_prob, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))
def build_args(device: str = 'cpu', use_sym_mcts: bool = False, numMCTSSims: int = 25) -> dotdict:
    return dotdict({
        'numMCTSSims': numMCTSSims,
        'cpuct': 1,
        'use_sym_mcts': use_sym_mcts,
        'dirichletAlpha': 0.3,
        'addRootNoise': False,
        'cuda': False if device == 'cpu' else True,
        'device': device,
    })
def load_nnet(game: Game, checkpoint: str, device: str = 'cpu') -> NNet:
    args = dotdict({
        'lr': 0.0003,
        'dropout': 0.3,
        'epochs': 1,
        'batch_size': 256,
        'cuda': False if device == 'cpu' else True,
        'device': device,
        'num_channels': 512,
        'use_sym': False,
        'inv_coef': 0.0,
        'sym_k': 8,
        'sym_strategy': 'cycle',
        'amp': False,
    })
    nnet = NNet(game, args)
    folder, fname = os.path.split(checkpoint)
    if not folder:
        folder = '.'
    nnet.load_checkpoint(folder=folder, filename=fname)
    return nnet
def generate_states(game: Game, nnet: NNet, mcts_args: dotdict, num_states: int = 1000,
                    temp_threshold: int = 15, max_eps: Optional[int] = None,
                    seed: int = 42) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    states = []
    eps_count = 0
    while len(states) < num_states and (max_eps is None or eps_count < max_eps):
        eps_count += 1
        board = game.getInitBoard()
        curPlayer = 1
        step = 0
        mcts = MCTS(game, nnet, mcts_args)
        while True:
            step += 1
            canonical = game.getCanonicalForm(board, curPlayer)
            states.append(np.array(canonical, dtype=np.float32))
            if len(states) >= num_states:
                break
            temp = 1 if step < temp_threshold else 0
            pi = mcts.getActionProb(canonical, temp=temp)
            action = rng.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)
            r = game.getGameEnded(board, curPlayer)
            if r != 0:
                break
    return states[:num_states]
def teacher_mcts_metrics(game: Game, nnet: NNet, boards: List[np.ndarray], sims_teacher: int,
                         use_sym_ens: bool = True) -> List[dict]:
    args_teacher = build_args(device=nnet.args.device, use_sym_mcts=False, numMCTSSims=sims_teacher)
    # sym_eval applies only inside MCTS leaf evaluation
    setattr(args_teacher, 'sym_eval', bool(use_sym_ens))
    setattr(args_teacher, 'addRootNoise', False)
    mcts_t = MCTS(game, nnet, args_teacher)
    out = []
    A = game.getActionSize()
    for b in boards:
        # student direct prediction on canonical board
        pi_s, v_s = nnet.predict(b)
        # teacher MCTS policy on canonical board
        pi_t = mcts_t.getActionProb(b, temp=0)
        # root value estimate from visit-weighted Q (as in reanalyze)
        meta = mcts_t._sym_canonicalize(b)
        s_key = meta['s_key']
        try:
            num = 0.0
            den = 0.0
            for a in range(A):
                key_sa = (s_key, a)
                if key_sa in mcts_t.Nsa:
                    nsa = float(mcts_t.Nsa[key_sa])
                    qsa = float(mcts_t.Qsa.get(key_sa, 0.0))
                    num += nsa * qsa
                    den += nsa
            v_root = (num / den) if den > 0.0 else None
        except Exception:
            v_root = None
        # divergences
        js = js_divergence(pi_s, pi_t)
        top1_agree = int(int(np.argmax(pi_s)) == int(np.argmax(pi_t)))
        out.append({
            'js_pi': js,
            'top1_agree': top1_agree,
            'v_student': float(v_s),
            'v_root': (float(v_root) if v_root is not None else ''),
        })
    return out
def ema_predictive_metrics(game: Game, ckpts: List[str], boards: List[np.ndarray], ema_decay: float = 0.999,
                           device: str = 'cpu') -> List[dict]:
    if len(ckpts) == 0:
        return []
    nets = [load_nnet(game, c, device=device) for c in ckpts]
    latest = nets[-1]
    # normalized geometric weights over checkpoints (older → smaller weight)
    T = len(nets)
    weights = [(1.0 - ema_decay) * (ema_decay ** (T - 1 - t)) for t in range(T)]
    s = sum(weights)
    weights = [w / s for w in weights]
    out = []
    for b in boards:
        # latest student
        pi_s, v_s = latest.predict(b)
        # EMA predictive ensemble over checkpoints (probability-space mixture)
        pis = []
        vs = []
        for net in nets:
            pi_i, v_i = net.predict(b)
            pis.append(np.asarray(pi_i, dtype=np.float64))
            vs.append(float(v_i))
        pis = np.stack(pis, axis=0)  # (T, A)
        vs = np.asarray(vs, dtype=np.float64)  # (T,)
        pi_ema = np.sum(pis * np.asarray(weights)[:, None], axis=0)
        v_ema = float(np.sum(vs * np.asarray(weights)))
        js = js_divergence(pi_s, pi_ema)
        top1_agree = int(int(np.argmax(pi_s)) == int(np.argmax(pi_ema)))
        out.append({
            'js_pi_ema': js,
            'top1_agree_ema': top1_agree,
            'v_student': float(v_s),
            'v_ema': float(v_ema),
        })
    return out
def maybe_plot_hist(values: List[float], title: str, out_path: str):
    try:
        import matplotlib.pyplot as plt
        arr = np.asarray([v for v in values if isinstance(v, (int, float)) and math.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            return
        plt.figure(figsize=(4, 3))
        plt.hist(arr, bins=50, color='#4e79a7', alpha=0.9)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception:
        # Matplotlib not available; skip silently
        pass
def write_csv(rows: List[dict], out_csv: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
def main():
    p = argparse.ArgumentParser(description='Post-hoc analysis for Reanalyze and EMA (offline).')
    p.add_argument('--checkpoint', type=str, default='./baseline/best.pth.tar', help='Primary checkpoint path')
    p.add_argument('--checkpoints', type=str, default='', help='Comma-separated list of checkpoints (for EMA predictive analysis)')
    p.add_argument('--board-size', type=int, default=8)
    p.add_argument('--device', type=str, default='cpu', help="'cpu' or CUDA index (e.g., 0)")
    p.add_argument('--num-states', type=int, default=1000)
    p.add_argument('--gen-sims', type=int, default=25, help='MCTS sims used to generate states')
    p.add_argument('--teacher-sims', type=int, default=200, help='Teacher MCTS sims for reanalyze metrics')
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--out-dir', type=str, default='./analysis_out')
    args = p.parse_args()
    # Game & nets
    game = Game(args.board_size)
    device = 'cpu' if str(args.device).lower() == 'cpu' else int(args.device)
    nnet = load_nnet(game, args.checkpoint, device=device)
    # Sample states via self-play with lightweight MCTS
    gen_args = build_args(device=device, use_sym_mcts=False, numMCTSSims=args.gen_sims)
    boards = generate_states(game, nnet, gen_args, num_states=args.num_states)
    os.makedirs(args.out_dir, exist_ok=True)
    # Reanalyze teacher metrics
    re_rows = teacher_mcts_metrics(game, nnet, boards, sims_teacher=args.teacher_sims, use_sym_ens=True)
    re_csv = os.path.join(args.out_dir, 'reanalyze_metrics.csv')
    write_csv(re_rows, re_csv)
    maybe_plot_hist([r['js_pi'] for r in re_rows], 'JS(pi_student || pi_teacher)', os.path.join(args.out_dir, 'reanalyze_js_hist.png'))
    # EMA predictive metrics (if multiple checkpoints provided)
    ema_rows = []
    ckpts = [c for c in (args.checkpoints.split(',') if args.checkpoints else []) if c.strip()]
    if len(ckpts) >= 2:
        ema_rows = ema_predictive_metrics(game, ckpts, boards, ema_decay=args.ema_decay, device=device)
        ema_csv = os.path.join(args.out_dir, 'ema_metrics.csv')
        write_csv(ema_rows, ema_csv)
        maybe_plot_hist([r['js_pi_ema'] for r in ema_rows], 'JS(pi_student || pi_EMA-ensemble)', os.path.join(args.out_dir, 'ema_js_hist.png'))
    # Quick console summary
    def summarize(tag: str, rows: List[dict], key: str):
        if not rows:
            print(f"[{tag}] no rows")
            return
        vals = np.asarray([float(r[key]) for r in rows if r.get(key, '') != '' and math.isfinite(float(r[key]))], dtype=np.float64)
        if vals.size == 0:
            print(f"[{tag}] {key}: no finite values")
            return
        print(f"[{tag}] {key}: mean={vals.mean():.4g}, median={np.median(vals):.4g}, p90={np.quantile(vals,0.9):.4g}")
    summarize('reanalyze', re_rows, 'js_pi')
    if ema_rows:
        summarize('ema', ema_rows, 'js_pi_ema')
if __name__ == '__main__':
    main()
