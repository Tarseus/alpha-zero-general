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
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0, 'use_dyn_c': False, 'addRootNoise': False, 'sym_eval': True})
n1 = NNet(g)
# n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
n1.load_checkpoint('./models/', 'best60.pth.tar')
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0, 'use_dyn_c': False, 'addRootNoise': False, 'sym_eval': True})
n2 = NNet(g)
# n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
n2.load_checkpoint('./models/', 'baseline.pth.tar')
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

player1 = n1p

player2 = n2p

arena = Arena.Arena(player1, player2, g, display=OthelloGame.display)

start = time.time()
result = arena.playGames(200, verbose=False)
game_time = time.time() - start

print(result, game_time)