import argparse
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
use this script to play any two agents against each other, or play manually with
any agent.
"""

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# nohup python grid_search.py --mode othello --cmin-list 0.8,0.9 --cmax-list 1.3,1.5 --tau-list 6,8,12 --danger-list 0.3,0.5,0.7 --kappa-list 0.4,0.5 --games 100 --sims 50 > test.log 2>&1 &
parser = argparse.ArgumentParser(description="Grid search dynamic exploration params")
parser.add_argument("--mini", action="store_true", help="Use 6x6 Othello (default 8x8)")
parser.add_argument("--games", type=int, default=100, help="Games per configuration")
parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
parser.add_argument("--mode", choices=["entropy", "othello"], default="entropy",
                    help="Dynamic c mode to evaluate")
parser.add_argument("--cmin-list", type=str, default="0.8,0.9,1.0")
parser.add_argument("--cmax-list", type=str, default="1.2,1.3,1.5")
parser.add_argument("--tau-list", type=str, default="8", help="othello depth tau list (comma sep)")
parser.add_argument("--danger-list", type=str, default="0.5", help="othello danger weight list")
parser.add_argument("--kappa-list", type=str, default="0.5", help="othello q kappa list")
args_cli = parser.parse_args()

mini_othello = bool(args_cli.mini)
human_vs_cpu = False

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play
mp = AlphaBetaOthelloPlayer(g, depth=3).play


# nnet players

# args1 = dotdict({
#     'numMCTSSims': 50, 
#     'cpuct':1.0, 
#     'use_dyn_c': True,
#     'cmin': 0.5,
#     'cmax': 3.0,
#     'kc': 3.0,})

args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0, 'use_dyn_c': False})
n1 = NNet(g)
# n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
n1.load_checkpoint('./models/', 'baseline.pth.tar')
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
# n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
n2.load_checkpoint('./models/', 'baseline.pth.tar')
player1 = n1p
#
# Grid search for dynamic c parameters
# Keeps player1 static; loops player2 with dynamic c.

def parse_float_list(s):
    return [float(x) for x in str(s).split(',') if x != '']

cmin_list = parse_float_list(args_cli.cmin_list)
cmax_list = parse_float_list(args_cli.cmax_list)
tau_list = parse_float_list(args_cli.tau_list)
danger_list = parse_float_list(args_cli.danger_list)
kappa_list = parse_float_list(args_cli.kappa_list)
num_games = int(args_cli.games)

mode = args_cli.mode

best = None  # Will hold (score, params_dict, result, secs)

def run_eval(params):
    a2 = dotdict({
        'numMCTSSims': args_cli.sims,
        'cpuct': 1.0,
        'use_dyn_c': True,
        'cmin': params['cmin'],
        'cmax': params['cmax'],
        'dyn_c_mode': mode,
    })
    if mode == 'othello':
        a2.othello_depth_tau = params['tau']
        a2.othello_danger_weight = params['danger']
        a2.othello_q_kappa = params['kappa']
    m2 = MCTS(g, n2, a2)
    p2 = (lambda m: (lambda x: np.argmax(m.getActionProb(x, temp=0))))(m2)
    arena = Arena.Arena(player1, p2, g, display=OthelloGame.display)
    start = time.time()
    result = arena.playGames(num_games, verbose=False)
    secs = time.time() - start
    oneWon, twoWon, draws = result
    score = twoWon - oneWon
    return score, result, secs

if mode == 'entropy':
    for cmin in cmin_list:
        for cmax in cmax_list:
            if cmin >= cmax:
                continue
            params = {'cmin': cmin, 'cmax': cmax}
            score, result, secs = run_eval(params)
            print(f"mode=entropy cmin={cmin}, cmax={cmax} -> {result}, {secs:.2f}s")
            if best is None or score > best[0]:
                best = (score, params, result, secs)
else:  # othello
    for cmin in cmin_list:
        for cmax in cmax_list:
            if cmin >= cmax:
                continue
            for tau in tau_list:
                for danger in danger_list:
                    for kappa in kappa_list:
                        params = {
                            'cmin': cmin, 'cmax': cmax,
                            'tau': tau, 'danger': danger, 'kappa': kappa,
                        }
                        score, result, secs = run_eval(params)
                        print(
                            f"mode=othello cmin={cmin}, cmax={cmax}, tau={tau}, danger={danger}, kappa={kappa} -> {result}, {secs:.2f}s"
                        )
                        if best is None or score > best[0]:
                            best = (score, params, result, secs)

if best is not None:
    print(f"BEST mode={mode} params={best[1]} -> {best[2]}, {best[3]:.2f}s")
