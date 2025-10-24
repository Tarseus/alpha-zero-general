import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.nn.functional as F

from .OthelloNNet import OthelloNNet as onnet

class NNetWrapper(NeuralNet):
    def __init__(self, game, args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 256,
            'cuda': torch.cuda.is_available(),
            'device': 1,
            'num_channels': 512,

            'use_sym': False,
            'inv_coef': 0.5,
        })):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.args = args

        if self.args.cuda:
            self.nnet.cuda(self.args.device)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            inv_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(self.args.device), target_pis.contiguous().cuda(self.args.device), target_vs.contiguous().cuda(self.args.device)

                if self.args.use_sym: 
                    boards = self.get_symmetries(boards)

                # compute output
                out_pi, out_pi_sym, out_v, out_v_sym, out_z_sym = self.nnet(boards)

                if out_pi_sym is not None and out_v_sym is not None:
                    l_pi = self.loss_pi_sym(target_pis, out_pi_sym)
                    l_v = self.loss_v_sym(target_vs, out_v_sym)
                else:
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)

                if out_z_sym is not None:
                    l_inv = self.loss_sym_cos(out_z_sym)
                else:
                    l_inv = torch.tensor(0, dtype=torch.float32)
                total_loss = l_pi + l_v + self.args.inv_coef * l_inv

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                inv_losses.update(l_inv.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, Loss_inv=inv_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        board shape: (n, n) (B, n, n) (1, 1, n, n) (B, 1, n, n)
        """
        # timing
        start = time.time()

        # preparing input
        board_np = np.array(board, dtype=np.float32)
        self.nnet.eval()

        if board_np.ndim == 2:
            board_np = board_np[None, None, ...]
        elif board_np.ndim == 3:
            board_np = board_np[:, None, ...]
        elif board_np.ndim == 4:
            pass
        else:
            raise ValueError("Board has incorrect dimensions.")

        with torch.no_grad():
            x = torch.from_numpy(board_np).to(self.args.device)
            pi_logits, _, v, _, _ = self.nnet(x)
            pi = torch.exp(pi_logits)
        
        pi = pi.cpu().numpy()
        v = v.squeeze(-1).data.cpu().numpy()

        if pi.shape[0] == 1:
            return pi[0], float(v[0])
        
        return pi, v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        
    def loss_pi_sym(self, targets, outputs):
        return -torch.sum(targets * outputs[:, -1, :]) / targets.size()[0]
    
    def loss_v_sym(self, targets, outputs):
        return ((outputs - targets.unsqueeze(-1).unsqueeze(1)) ** 2).mean()

    def loss_sym_cos(self, out_z_sym, stopgrad=False):
        z = F.normalize(out_z_sym, p=2, dim=-1)          # (B, 8, D)
        center = z.mean(dim=1, keepdim=True)             # (B, 1, D)
        center = F.normalize(center, p=2, dim=-1)        # (B, 1, D)
        if stopgrad:
            center = center.detach()
        cos = (z * center).sum(dim=-1)                   # (B, 8)
        return (1.0 - cos).mean()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def get_symmetries(self, boards):
        # boards.shape: batchsize x n x n
        assert boards.ndim == 3 and boards.shape[1] == 8 and boards.shape[2] == 8, \
            f"Expected (B, 8, 8), got {tuple(boards.shape)}"

        syms = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = torch.rot90(boards, i, dims=(1, 2))
                if j:
                    newB = torch.flip(newB, dims=(2,))
                syms.append(newB)

        out = torch.stack(syms, dim=1)

        assert out.shape[1] == 8
        return out