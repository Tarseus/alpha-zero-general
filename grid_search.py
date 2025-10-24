import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
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

# args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0, 'use_dyn_c': False})
args2 = dotdict({
    'numMCTSSims': 50, 
    'cpuct':1.0, 
    'use_dyn_c': True,
    'cmin': 0.5,
    'cmax': 3.0,
    'kc': 3.0,})
n2 = NNet(g)
# n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
n2.load_checkpoint('./models/', 'baseline.pth.tar')
player1 = n1p
#
# Grid search over (cmin, cmax) for dynamic c_puct
# Keeps player1 static; loops player2 with dynamic c.
# BEST cmin=0.8, cmax=1.3 -> (38, 62, 0), 440.89s
cmin_list = [0.8, 0.9, 1.0]
cmax_list = [1.2, 1.3, 1.5]
num_games = 100

best = None  # (score(twoWon-oneWon), cmin, cmax, result, secs)
for cmin in cmin_list:
    for cmax in cmax_list:
        if cmin >= cmax:
            continue
        args2 = dotdict({
            'numMCTSSims': 50,
            'cpuct': 1.0,
            'use_dyn_c': True,
            'cmin': cmin,
            'cmax': cmax,
            'kc': 3.0,
        })
        mcts2 = MCTS(g, n2, args2)
        n2p = (lambda m: (lambda x: np.argmax(m.getActionProb(x, temp=0))))(mcts2)
        player2 = n2p

        arena = Arena.Arena(player1, player2, g, display=OthelloGame.display)
        start = time.time()
        result = arena.playGames(num_games, verbose=False)
        game_time = time.time() - start
        oneWon, twoWon, draws = result
        print(f"cmin={cmin}, cmax={cmax} -> {result}, {game_time:.2f}s")
        score = twoWon - oneWon
        if best is None or score > best[0]:
            best = (score, cmin, cmax, result, game_time)

if best is not None:
    print(f"BEST cmin={best[1]}, cmax={best[2]} -> {best[3]}, {best[4]:.2f}s")
