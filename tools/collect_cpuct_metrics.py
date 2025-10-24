import argparse
import csv
import os
import time

import numpy as np

from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict


def is_danger_move(n, board, action):
    """Return True if action is an X/C square with its associated corner empty."""
    # pass move
    if action == n * n:
        return False
    r, c = action // n, action % n
    # corners
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    x_squares = {(1, 1), (1, n - 2), (n - 2, 1), (n - 2, n - 2)} if n >= 3 else set()
    c_squares = {
        (0, 1), (1, 0), (0, n - 2), (1, n - 1), (n - 1, 1), (n - 2, 0),
        (n - 2, n - 1), (n - 1, n - 2)
    } if n >= 2 else set()

    if (r, c) in x_squares:
        assoc = {(1, 1): (0, 0), (1, n - 2): (0, n - 1), (n - 2, 1): (n - 1, 0), (n - 2, n - 2): (n - 1, n - 1)}
        cr, cc = assoc[(r, c)]
        return board[cr][cc] == 0
    if (r, c) in c_squares:
        assoc = {
            (0, 1): (0, 0), (1, 0): (0, 0), (0, n - 2): (0, n - 1), (1, n - 1): (0, n - 1),
            (n - 1, 1): (n - 1, 0), (n - 2, 0): (n - 1, 0), (n - 2, n - 1): (n - 1, n - 1),
            (n - 1, n - 2): (n - 1, n - 1)
        }
        cr, cc = assoc[(r, c)]
        return board[cr][cc] == 0
    return False


def compute_metrics(game, mcts, board, mode):
    """Compute root-level metrics after MCTS simulations for the given board.

    Returns a dict with keys: B, E, H_norm, r_q, danger, c, H_counts, breadth, chosen_is_danger, chosen_action.
    """
    n = game.n
    s = game.stringRepresentation(board)
    # valid moves for canonical player
    valids = mcts.Vs.get(s)
    A = game.getActionSize()
    pass_idx = A - 1
    valid_idx = [a for a in range(A) if valids[a] and a != pass_idx]
    B = len(valid_idx)

    # policy after masking and potential root noise
    P = mcts.Ps.get(s)
    if P is None:
        # fallback to network prediction
        pi, _ = mcts.nnet.predict(board)
        vm = game.getValidMoves(board, 1)
        P = pi * vm
        Z = P.sum()
        if Z > 0:
            P = P / Z
    p = [float(max(P[a], 0.0)) for a in valid_idx] if B > 0 else []
    Zp = sum(p)
    if Zp <= 0 and B > 0:
        p = [1.0 / B] * B
    elif Zp > 0 and B > 0:
        p = [pi / Zp for pi in p]

    if B > 1:
        H = -sum(pi * np.log(max(pi, 1e-12)) for pi in p)
        H_norm = float(H / np.log(B))
    else:
        H_norm = 0.0

    # Q dispersion among visited edges
    Qs = [mcts.Qsa[(s, a)] for a in valid_idx if (s, a) in mcts.Qsa]
    r_q = float(np.tanh(np.std(Qs) / max(float(getattr(mcts.args, 'othello_q_kappa', 0.5)), 1e-6))) if len(Qs) >= 2 else 0.0

    # danger ratio across valid moves
    danger_cnt = 0
    for a in valid_idx:
        if is_danger_move(n, board, a):
            danger_cnt += 1
    danger = (danger_cnt / float(B)) if B > 0 else 0.0

    # compute c consistent with mode
    if getattr(mcts.args, 'use_dyn_c', False):
        dyn_mode = getattr(mcts.args, 'dyn_c_mode', 'entropy')
        if mode == 'othello' and hasattr(mcts, '_cpuct_othello'):
            c_val = float(mcts._cpuct_othello(board, s, valids, depth=0))
        elif dyn_mode == 'entropy' and hasattr(mcts, '_cpuct_from_entropy'):
            c_val = float(mcts._cpuct_from_entropy(s, valids))
        else:
            c_val = float(mcts.args.cpuct)
    else:
        c_val = float(mcts.args.cpuct)

    # visit distribution entropy at root
    counts = np.array([mcts.Nsa.get((s, a), 0) for a in valid_idx], dtype=np.float32)
    if counts.sum() > 0 and len(counts) > 1:
        p_counts = counts / counts.sum()
        H_counts = float(-(p_counts * np.log(np.clip(p_counts, 1e-12, 1))).sum())
    else:
        H_counts = 0.0
    breadth = int((counts > 0).sum())

    # empties on board (phase)
    E = int((board == 0).sum())

    return {
        'B': B,
        'E': E,
        'H_norm': H_norm,
        'r_q': r_q,
        'danger': danger,
        'c': c_val,
        'H_counts': H_counts,
        'breadth': breadth,
        # chosen_* filled by caller
    }


def main():
    ap = argparse.ArgumentParser(description='Collect cpuct metrics during Othello play')
    ap.add_argument('--mini', action='store_true', help='Use 6x6 board')
    ap.add_argument('--games', type=int, default=20, help='Number of games to log')
    ap.add_argument('--sims', type=int, default=50, help='MCTS simulations per move')
    ap.add_argument('--mode', choices=['entropy', 'othello', 'fixed'], default='othello')
    ap.add_argument('--cmin', type=float, default=0.8)
    ap.add_argument('--cmax', type=float, default=2.2)
    ap.add_argument('--tau', type=float, default=8.0, help='othello depth tau')
    ap.add_argument('--danger', type=float, default=0.5, help='othello danger weight')
    ap.add_argument('--kappa', type=float, default=0.5, help='othello q kappa')
    ap.add_argument('--out', type=str, default='temp/cpuct_metrics.csv')
    ap.add_argument('--log-both', action='store_true', help='Log both players instead of only dynamic one')
    args = ap.parse_args()

    n = 6 if args.mini else 8
    game = OthelloGame(n)

    # players: baseline vs dynamic
    args1 = dotdict({'numMCTSSims': args.sims, 'cpuct': 1.0, 'use_dyn_c': False})
    args2 = dotdict({'numMCTSSims': args.sims, 'cpuct': 1.0})
    if args.mode == 'fixed':
        args2.use_dyn_c = False
        args2.cpuct = (args.cmin + args.cmax) / 2.0
    else:
        args2.use_dyn_c = True
        args2.cmin = args.cmin
        args2.cmax = args.cmax
        args2.dyn_c_mode = args.mode
        if args.mode == 'othello':
            args2.othello_depth_tau = args.tau
            args2.othello_danger_weight = args.danger
            args2.othello_q_kappa = args.kappa

    n1 = NNet(game)
    n2 = NNet(game)
    # Expect local baseline checkpoint; adjust if necessary
    ckpt_dir = './models/'
    ckpt_file = 'baseline.pth.tar'
    n1.load_checkpoint(ckpt_dir, ckpt_file)
    n2.load_checkpoint(ckpt_dir, ckpt_file)

    mcts1 = MCTS(game, n1, args1)
    mcts2 = MCTS(game, n2, args2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        'game_id', 'turn', 'cur_player', 'logged_side', 'mode', 'sims',
        'c', 'cmin', 'cmax', 'tau', 'danger_w', 'kappa',
        'B', 'E', 'H_norm', 'r_q', 'danger', 'H_counts', 'breadth',
        'chosen_is_danger', 'chosen_action', 'result'
    ]
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        def play_one(game_id, p_first_dynamic):
            board = game.getInitBoard()
            curPlayer = 1
            turn = 0
            res = None
            while game.getGameEnded(board, curPlayer) == 0:
                turn += 1
                canonical = game.getCanonicalForm(board, curPlayer)
                # choose action
                if curPlayer == 1:
                    probs = mcts1.getActionProb(canonical, temp=0)
                    action = int(np.argmax(probs))
                    # log if baseline is set to be logged too
                    if args.log_both and not p_first_dynamic:
                        metrics = compute_metrics(game, mcts1, canonical, 'fixed')
                        row = {
                            'game_id': game_id, 'turn': turn, 'cur_player': curPlayer,
                            'logged_side': 'baseline', 'mode': 'fixed', 'sims': args.sims,
                            'c': 1.0, 'cmin': 1.0, 'cmax': 1.0, 'tau': '', 'danger_w': '', 'kappa': '',
                            **metrics, 'chosen_is_danger': is_danger_move(game.n, canonical, action),
                            'chosen_action': action, 'result': ''
                        }
                        w.writerow(row)
                else:
                    probs = mcts2.getActionProb(canonical, temp=0)
                    action = int(np.argmax(probs))
                    # log dynamic player's turn if needed
                    if p_first_dynamic or args.log_both:
                        metrics = compute_metrics(game, mcts2, canonical, args.mode)
                        row = {
                            'game_id': game_id, 'turn': turn, 'cur_player': curPlayer,
                            'logged_side': 'dynamic', 'mode': args.mode, 'sims': args.sims,
                            'c': metrics.pop('c'), 'cmin': args.cmin, 'cmax': args.cmax,
                            'tau': (args.tau if args.mode == 'othello' else ''),
                            'danger_w': (args.danger if args.mode == 'othello' else ''),
                            'kappa': (args.kappa if args.mode == 'othello' else ''),
                            **metrics, 'chosen_is_danger': is_danger_move(game.n, canonical, action),
                            'chosen_action': action, 'result': ''
                        }
                        w.writerow(row)

                board, curPlayer = game.getNextState(board, curPlayer, action)

            res = game.getGameEnded(board, 1)
            # add final row for result
            w.writerow({
                'game_id': game_id, 'turn': turn, 'cur_player': curPlayer, 'logged_side': 'final',
                'mode': args.mode, 'sims': args.sims, 'c': '', 'cmin': args.cmin, 'cmax': args.cmax,
                'tau': (args.tau if args.mode == 'othello' else ''), 'danger_w': (args.danger if args.mode == 'othello' else ''),
                'kappa': (args.kappa if args.mode == 'othello' else ''), 'B': '', 'E': '', 'H_norm': '', 'r_q': '',
                'danger': '', 'H_counts': '', 'breadth': '', 'chosen_is_danger': '', 'chosen_action': '',
                'result': res,
            })

        # half games starting baseline, then swap
        half = args.games // 2
        for gid in range(1, half + 1):
            play_one(game_id=f'A-{gid}', p_first_dynamic=False)
        # swap roles: dynamic becomes current player 1, baseline player 2
        mcts1, mcts2 = mcts2, mcts1
        for gid in range(1, args.games - half + 1):
            play_one(game_id=f'B-{gid}', p_first_dynamic=True)

    print(f"Saved metrics to {args.out}")


if __name__ == '__main__':
    main()
