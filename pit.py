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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_game(n: int = 8):
    return OthelloGame(n)


def make_mcts_player(game, nnet, sims: int,):
    args = dotdict({
        'numMCTSSims': sims,
        'cpuct': 1.0, 
        'use_dyn_c': False, 
        'addRootNoise': False, 
        'sym_eval': True,
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
    nnet_base = NNet(g)
    nnet_base.load_checkpoint('./models/', 'baseline.pth.tar')

    nnet_ours = NNet(g)
    nnet_ours.load_checkpoint('./models/', 'best60.pth.tar')

    games = 200

    # 1) baseline/sym_mcts vs random, greedy
    for sims in [25,50,100,200]:
        cfgs = [
            ("baseline", nnet_base),
            ("ours", nnet_ours)
        ]
        for label, nnet in cfgs:
            p_mcts = make_mcts_player(g, nnet, sims=sims)
            run_matchup(f"{label} vs random (sims={sims})", p_mcts, rp, g, games)
            run_matchup(f"{label} vs greedy (sims={sims})", p_mcts, gp, g, games)

    # 2) vs alphabeta: test four configurations
    #    - baseline (dyn=False, sym=False)
    #    - dyn only (dyn=True, sym=False)
    #    - sym only (dyn=False, sym=True)
    #    - dyn+sym (dyn=True, sym=True)
    for sims in [25,50,100,200]:
        cfgs = [
            ("baseline", nnet_base),
            ("ours", nnet_ours)
        ]
        for label, nnet in cfgs:
            p_mcts = make_mcts_player(g, nnet, sims=sims)
            run_matchup(f"{label} vs alphabeta(d=3) (sims={sims})", p_mcts, mp, g, games)

    # 3) baseline vs variants (ensure equal sims on both sides): dyn_only, sym_only, dyn_sym
    for sims in [25,50,100,200]:
        p_base = make_mcts_player(g, nnet_base, sims=sims,)
        opponents = [
            ("ours", make_mcts_player(g, nnet_ours, sims=sims)),
        ]
        for label, p_var in opponents:
            run_matchup(f"baseline vs {label} (sims={sims})", p_base, p_var, g, games)


if __name__ == "__main__":
    main()
