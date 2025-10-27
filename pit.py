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


def make_mcts_player(game, nnet, sims: int, use_sym_mcts: bool):
    args = dotdict({
        'numMCTSSims': sims,
        'cpuct': 1.0,
        'use_dyn_c': False,
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
    for label, use_sym in [("baseline", False), ("sym_mcts", True)]:
        for sims in [25, 50]:
            p_mcts = make_mcts_player(g, nnet, sims=sims, use_sym_mcts=use_sym)
            run_matchup(f"{label} vs random (sims={sims})", p_mcts, rp, g, games)
            run_matchup(f"{label} vs greedy (sims={sims})", p_mcts, gp, g, games)

    # 2) baseline/sym_mcts vs alphabeta
    for label, use_sym in [("baseline", False), ("sym_mcts", True)]:
        for sims in [25, 50, 75, 100]:
            p_mcts = make_mcts_player(g, nnet, sims=sims, use_sym_mcts=use_sym)
            run_matchup(f"{label} vs alphabeta(d=3) (sims={sims})", p_mcts, mp, g, games)

    # 3) baseline vs sym_mcts (ensure equal sims on both sides)
    for sims in [25, 50, 75, 100]:
        p_base = make_mcts_player(g, nnet, sims=sims, use_sym_mcts=False)
        p_sym = make_mcts_player(g, nnet, sims=sims, use_sym_mcts=True)
        run_matchup(f"baseline vs sym_mcts (sims={sims})", p_base, p_sym, g, games)


if __name__ == "__main__":
    main()
