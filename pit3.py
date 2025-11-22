import time
import os, random, torch
import numpy as np

import Arena
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
from adaptive_budget import AdaptiveMCTSPlayer, FixedMCTSPlayer, RobustRootMCTSPlayer
from utils import *

"""
Adaptive search budget at inference (root-entropy guided):
1) Run a small initial batch of simulations to form an initial policy.
2) Compute root policy entropy to assess uncertainty.
3) Allocate more simulations on uncertain positions and fewer on easy ones,
   while keeping a fixed total per-game budget (approximate) via a rolling
   budget controller.

This script compares the same baseline network with and without adaptive
budget, using sims in {25, 50, 100, 200}. Results are printed for each sims.
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


def make_fixed_player(game, nnet, sims: int):
    return FixedMCTSPlayer(game, nnet, sims=sims, cpuct=1.0, sym_eval=True)


def make_adaptive_player(game, nnet, sims: int):
    # init_frac/min/max may be tuned; defaults are reasonable for Othello

    expected_moves = 30

    return AdaptiveMCTSPlayer(
        game,
        nnet,
        avg_sims=sims,
        cpuct=1.0,
        init_frac=0.25,
        min_boost=0.7,
        max_boost=1.3,
        expected_moves_per_player=expected_moves,
        sym_eval=True,
    )

def make_robust_player(game, nnet, sims:int):
    return RobustRootMCTSPlayer(game, nnet, sims=sims, cpuct=1.0, sym_eval=True, frac=0.6)


def run_matchup(label: str, p1, p2, game, games: int = 400, verbose: bool = False):
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

    # Shared network (baseline weights) used for both players
    nnet = NNet(g)
    nnet.load_checkpoint('./models/', 'baseline.pth.tar')

    games_per_match = 200  # 200 per side

    # Compare dynamic vs fixed at multiple average sims
    for sims in [25, 50, 100, 200]:
        p_fixed = make_fixed_player(g, nnet, sims=sims)
        p_adapt = make_robust_player(g, nnet, sims=sims)
        label = f"adaptive(vs fixed) sims={sims}"
        run_matchup(label, p_adapt, p_fixed, g, games_per_match)

    # Also pit adaptive against random, greedy, and alpha-beta
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    mp = AlphaBetaOthelloPlayer(g, depth=3).play

    for sims in [25, 50, 100, 200]:
        p_adapt = make_robust_player(g, nnet, sims=sims)
        run_matchup(f"adaptive vs random (sims={sims})", p_adapt, rp, g, games_per_match)
        run_matchup(f"adaptive vs greedy (sims={sims})", p_adapt, gp, g, games_per_match)
        run_matchup(f"adaptive vs alphabeta(d=3) (sims={sims})", p_adapt, mp, g, games_per_match)


if __name__ == "__main__":
    main()
