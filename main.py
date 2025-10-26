import logging
import os, random, torch
import numpy as np
import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

import argparse

def _device_arg(s: str):
    s = s.strip().lower()
    if s in ("cpu", "-1"):
        return "cpu"
    if s.isdigit():
        return int(s)
    raise argparse.ArgumentTypeError("device 必须是 'cpu' 或非负整数 GPU 编号，例如 0/1/2")

def parse_cli_args():
    p = argparse.ArgumentParser(description="AlphaZero Othello training")
    p.add_argument("--checkpoint", type=str, default="./baseline",
                   help="模型/数据的检查点目录 (默认: ./baseline)")
    p.add_argument("--use-dyn-c", dest="use_dyn_c", action="store_true",
                   help="启用基于策略熵等的动态 c_puct")
    p.add_argument("--no-use-dyn-c", dest="use_dyn_c", action="store_false",
                   help="禁用动态 c_puct（覆盖上面的启用）")
    p.set_defaults(use_dyn_c=False)

    p.add_argument("--use-sym", dest="use_sym", action="store_true",
                   help="训练/推理时使用棋盘 D4 对称增强/不变性")
    p.add_argument("--no-use-sym", dest="use_sym", action="store_false",
                   help="禁用对称增强（覆盖上面的启用）")
    p.set_defaults(use_sym=False)

    p.add_argument("--device", type=_device_arg, default=0,
                   help="计算设备：'cpu' 或 GPU 编号(如 0)。默认 0")

    return p.parse_args()

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    log.info('Seed set to %s...', seed)

args = dotdict({
    'numIters': 100,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    # 'checkpoint': './diy_dyn/',
    'checkpoint': './baseline',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'dirichletAlpha': 0.3,

    'use_dyn_c': False,
    'cmin': 0.5,
    'cmax': 3.0,
    'kc': 3.0,

    'eliteCapacity': 5000,
    'eliteWindow': 200,
    'eliteFrac': 0.1,

    'evalGames': 40,
    'evalNumMCTSSims': 50,
    'logBaselinesToCSV': True,

    'evalABDepth': 3,
    'evalABTimeLimit': None,

    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'device': 0,
    'num_channels': 512,

    'use_sym': True,
    'inv_coef': 0.05,
    'sym_k': 8,              # number of symmetry views per batch (<=8)
    'sym_strategy': 'cycle', # one of: 'cycle', 'random'
    'amp': False,             # mixed precision training
    'sym_anchor_stopgrad': True,
})


def main():
    
    _cli = parse_cli_args()
    args.checkpoint = _cli.checkpoint
    args.use_dyn_c = _cli.use_dyn_c
    args.use_sym = _cli.use_sym
    args.device = _cli.device

    set_seed(42)
    log.info('Loading %s...', Game.__name__)
    g = Game(8)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')
    
    log.info('Loading checkpoint "%s/%s"...', './models', 'sym_iter25.pth.tar')
    nnet.load_checkpoint('./models', 'sym_iter25.pth.tar')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
