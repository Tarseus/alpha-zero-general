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

args1 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0, 'use_dyn_c': False, 'addRootNoise': False, 'use_sym_mcts': True})
n1 = NNet(g)
# n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
n1.load_checkpoint('./models/', 'baseline.pth.tar')
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

args2 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0, 'use_dyn_c': True, 'cmin': 0.8, 'cmax': 1.3, 'addRootNoise': False, 'use_sym_mcts': False})
n2 = NNet(g)
# n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
n2.load_checkpoint('./models/', 'baseline.pth.tar')
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

player1 = mp

player2 = n1p

arena = Arena.Arena(player1, player2, g, display=OthelloGame.display)

start = time.time()
result = arena.playGames(100, verbose=False)
game_time = time.time() - start

print(result, game_time)

# base-dyn: (38, 62, 0) mcts=50
# random-dyn: (0, 100, 0)
# greedy-dyn: (0, 100, 0)
# AlphaBeta(d=3)-base: (39, 61, 0) 962.8510503768921 mcts=50
# AlphaBeta(d=3)-base: (38, 62, 0) 1049.8993735313416 mcts=75
# AlphaBeta(d=3)-base: (29, 71, 0) 1325.0332896709442 mcts=100