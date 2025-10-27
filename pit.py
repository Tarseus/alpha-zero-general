import time
import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
import os, random, torch

import numpy as np
from utils import *

"""
This script now runs a fixed battery of matches:
- baseline (no dyn_c, no sym_mcts) vs random/greedy/alphabeta
- sym_mcts (no dyn_c, use_sym_mcts) vs random/greedy/alphabeta
  sims: random/greedy -> 25, 50; alphabeta -> 25, 50, 75, 100
- baseline vs sym_mcts with equal sims: 25, 50, 75, 100
Each matchup runs 1000 games (500 as each color via Arena.playGames).
"""


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_game(n: int = 8):
    return OthelloGame(n)


def make_mcts_player(game, nnet, sims: int, use_sym_mcts: bool, use_dyn_c: bool = False,
                     dyn_mode: str = 'entropy', cmin: float = 0.8, cmax: float = 1.3):
    args = dotdict({
        'numMCTSSims': sims,
        'cpuct': 1.0,
        'use_dyn_c': bool(use_dyn_c),
        'dyn_c_mode': dyn_mode,
        'cmin': cmin,
        'cmax': cmax,
        'addRootNoise': False,
        'use_sym_mcts': bool(use_sym_mcts),
    })
    mcts = MCTS(game, nnet, args)
    return lambda x: np.argmax(mcts.getActionProb(x, temp=0))


def run_matchup(label: str, p1, p2, game, games: int = 1000, verbose: bool = False):
    arena = Arena.Arena(p1, p2, game, display=OthelloGame.display)
    start = time.time()
    result = arena.playGames(games, verbose=verbose)
    secs = time.time() - start
    oneWon, twoWon, draws = result
    print(f"{label}: result={result}, time={secs:.2f}s")
    return result


def main():
    set_seed(42)

    g = make_game(8)

    # Opponents
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    mp = AlphaBetaOthelloPlayer(g, depth=3).play

    # Shared network (baseline weights)
    nnet = NNet(g)
    nnet.load_checkpoint('./models/', 'baseline.pth.tar')

    games = 1000

    # 1) baseline/sym_mcts vs random, greedy
    # for label, use_sym in [("baseline", False), ("sym_mcts", True)]:
    label = "baseline"
    use_sym = False
    for sims in [25, 50]:
        p_mcts = make_mcts_player(g, nnet, sims=sims, use_sym_mcts=use_sym)
        run_matchup(f"{label} vs random (sims={sims})", p_mcts, rp, g, games)
        run_matchup(f"{label} vs greedy (sims={sims})", p_mcts, gp, g, games)

    # 2) vs alphabeta: test four configurations
    #    - baseline (dyn=False, sym=False)
    #    - dyn only (dyn=True, sym=False)
    #    - sym only (dyn=False, sym=True)
    #    - dyn+sym (dyn=True, sym=True)
    for sims in [25, 50, 75, 100]:
        cfgs = [
            ("baseline", False, False),
            ("dyn_only", True, False),
            ("sym_only", False, True),
            ("dyn_sym", True, True),
        ]
        for label, use_dyn, use_sym in cfgs:
            p_mcts = make_mcts_player(g, nnet, sims=sims, use_sym_mcts=use_sym, use_dyn_c=use_dyn,
                                      dyn_mode='entropy', cmin=0.8, cmax=1.3)
            run_matchup(f"{label} vs alphabeta(d=3) (sims={sims})", p_mcts, mp, g, games)

    # 3) baseline vs variants (ensure equal sims on both sides): dyn_only, sym_only, dyn_sym
    for sims in [25, 50, 75, 100]:
        p_base = make_mcts_player(g, nnet, sims=sims, use_sym_mcts=False, use_dyn_c=False)
        opponents = [
            ("dyn_only", make_mcts_player(g, nnet, sims=sims, use_sym_mcts=False, use_dyn_c=True,
                                           dyn_mode='entropy', cmin=0.8, cmax=1.3)),
            ("sym_only", make_mcts_player(g, nnet, sims=sims, use_sym_mcts=True, use_dyn_c=False)),
            ("dyn_sym",  make_mcts_player(g, nnet, sims=sims, use_sym_mcts=True, use_dyn_c=True,
                                           dyn_mode='entropy', cmin=0.8, cmax=1.3)),
        ]
        for label, p_var in opponents:
            run_matchup(f"baseline vs {label} (sims={sims})", p_base, p_var, g, games)


if __name__ == "__main__":
    main()
