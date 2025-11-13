import os
import sys
import argparse
from typing import List, Tuple

import numpy as np

# Ensure repo root is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from othello.OthelloGame import OthelloGame as Game


def _random_play_steps(game: Game, steps: int = 3, seed: int = 42) -> Tuple[List[np.ndarray], List[int]]:
    """Return ([s0..sK], [p0..pK]) starting from initial board with random legal moves.

    s0 是初始棋盘，p0 是初始行动方（1 表示白，-1 表示黑，和 draw_board 配色一致）。
    默认生成 3 步，共 4 帧；若提前结束，会重复最后一帧以补齐长度。
    """
    rng = np.random.RandomState(seed)
    board = game.getInitBoard()
    player = 1
    states = [np.array(board, dtype=np.int8)]
    players = [int(player)]
    n = game.getBoardSize()[0]
    A = game.getActionSize()
    for _ in range(steps):
        valids = game.getValidMoves(board, player)
        idxs = np.where(valids > 0)[0]
        if idxs.size == 0:
            # no moves for either player => terminal
            break
        # Prefer non-pass if available
        non_pass = idxs[idxs != (A - 1)]
        if non_pass.size > 0:
            a = int(rng.choice(non_pass))
        else:
            a = int(A - 1)
        board, player = game.getNextState(board, player, a)
        states.append(np.array(board, dtype=np.int8))
        players.append(int(player))
    # Ensure length 4 by repeating last if game ended early
    while len(states) < (steps + 1):
        states.append(states[-1])
        players.append(players[-1])
    return states[: steps + 1], players[: steps + 1]


def _draw_board(ax, board: np.ndarray, player_to_move: int = None, valid_moves: List[Tuple[int, int]] = None, title: str = ""):
    import matplotlib.pyplot as plt
    n = board.shape[0]
    ax.set_aspect('equal')
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    # Background
    ax.add_patch(plt.Rectangle((0, 0), n, n, color='#2e7d32'))
    # Grid
    for i in range(n + 1):
        ax.plot([i, i], [0, n], color='black', linewidth=1)
        ax.plot([0, n], [i, i], color='black', linewidth=1)
    # Stones
    for y in range(n):
        for x in range(n):
            v = int(board[y, x])
            if v == 0:
                continue
            cx, cy = x + 0.5, n - y - 0.5  # invert y for plotting
            color = 'white' if v == 1 else 'black'
            edge = 'black' if v == 1 else 'white'
            ax.add_patch(plt.Circle((cx, cy), 0.4, facecolor=color, edgecolor=edge, linewidth=1.5))
    # Valid moves (if any): mark empty legal moves with small hollow yellow circles
    if valid_moves:
        for (yy, xx) in valid_moves:
            cx, cy = xx + 0.5, n - yy - 0.5
            ax.add_patch(plt.Circle((cx, cy), 0.15, facecolor='none', edgecolor='#ffeb3b', linewidth=2.0, alpha=0.95))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Title with side-to-move color if provided
    if player_to_move is not None:
        side = 'White' if int(player_to_move) == 1 else 'Black'
        ttl = f"{title} — {side} to move"
    else:
        ttl = title
    ax.set_title(ttl)


def main():
    ap = argparse.ArgumentParser(description='Visualize Othello: initial + 3 random steps (4 panels).')
    ap.add_argument('--board-size', type=int, default=8)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='othello_random_4.png')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    game = Game(args.board_size)
    states, players = _random_play_steps(game, steps=args.steps, seed=args.seed)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('Matplotlib not available; please install matplotlib to produce the image.')
        sys.exit(1)

    titles = ['Init', 'Step 1', 'Step 2', 'Step 3']
    fig, axes = plt.subplots(1, 4, figsize=(4 * 3.2, 3.2))
    n = game.getBoardSize()[0]
    A = game.getActionSize()
    for i in range(4):
        # Compute legal moves for the side-to-move at this frame (exclude pass)
        valids_vec = game.getValidMoves(states[i], players[i])
        idxs = np.where(valids_vec > 0)[0]
        mv = []
        for idx in idxs:
            if int(idx) == (A - 1):
                continue  # skip pass
            yy = int(idx) // n
            xx = int(idx) % n
            mv.append((yy, xx))
        _draw_board(axes[i], states[i], player_to_move=players[i], valid_moves=mv, title=titles[i])
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
