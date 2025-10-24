import os
import sys
import time

import numpy as np
import math
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
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

            'use_sym': True,
            'inv_coef': 0.1,
            'sym_k': 8,              # number of symmetry views per batch (<=8)
            'sym_strategy': 'cycle', # one of: 'cycle', 'random'
            'amp': False,             # mixed precision training
        })):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.args = args

        # Precompute symmetry action permutations and register as module buffers
        # so they automatically move with the model between devices.
        self._init_action_perms()

        if self.args.cuda:
            self.nnet.cuda(self.args.device)
            torch.backends.cudnn.benchmark = True

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
                boards = torch.tensor(np.array(boards), dtype=torch.float32)
                target_pis = torch.tensor(np.array(pis), dtype=torch.float32)
                target_vs = torch.tensor(np.array(vs), dtype=torch.float32)

                # move tensors
                if self.args.cuda:
                    boards = boards.contiguous().cuda(self.args.device)
                    target_pis = target_pis.contiguous().cuda(self.args.device)
                    target_vs = target_vs.contiguous().cuda(self.args.device)

                # symmetry subset
                t_idx = None
                if self.args.use_sym:
                    k = max(1, min(int(getattr(self.args, 'sym_k', 8)), 8))
                    t_idx = self._pick_sym_indices(k, getattr(self.args, 'sym_strategy', 'cycle'))
                    boards = self.get_symmetries_subset(boards, t_idx)

                # compute output with AMP
                scaler = getattr(self, '_scaler', None)
                if scaler is None:
                    self._scaler = GradScaler(enabled=(self.args.cuda and bool(getattr(self.args, 'amp', False))))
                    scaler = self._scaler

                with autocast(enabled=(self.args.cuda and bool(getattr(self.args, 'amp', False)))):
                    out_pi, out_pi_sym, out_v, out_v_sym, out_z_sym = self.nnet(boards)

                    if out_pi_sym is not None and out_v_sym is not None:
                        # Use symmetric views properly aligned
                        l_pi = self.loss_pi_sym_aligned(target_pis, out_pi_sym, t_idx)
                        l_v = self.loss_v_sym(target_vs, out_v_sym)
                    else:
                        l_pi = self.loss_pi(target_pis, out_pi)
                        l_v = self.loss_v(target_vs, out_v)

                    if out_z_sym is not None:
                        l_inv = self.loss_sym_cos(out_z_sym)
                    else:
                        l_inv = torch.tensor(0, dtype=torch.float32, device=boards.device)
                    total_loss = l_pi + l_v + self.args.inv_coef * l_inv

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                inv_losses.update(l_inv.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, Loss_inv=inv_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
        # Kept for backward compatibility; uses only the last view.
        return -torch.sum(targets * outputs[:, -1, :]) / targets.size()[0]

    def _init_action_perms(self):
        """Build action index permutations for 8 board symmetries.

        For a base action index a in [0, n*n), perm_fwd[t, a] gives the index of
        the corresponding action in the t-th symmetric view. The pass action
        (index n*n) maps to itself.
        """
        n = int(self.board_x)
        assert n == int(self.board_y), "Only square boards are supported for sym perms"

        A = n * n
        # base index grid 0..A-1
        base = torch.arange(A, dtype=torch.long)
        grid = base.view(n, n)

        perms = []
        for i in range(1, 5):
            for j in (True, False):
                M = torch.rot90(grid, i, dims=(0, 1))
                if j:
                    M = torch.flip(M, dims=(1,))
                perms.append(M.reshape(-1))
        perm_fwd = torch.stack(perms, dim=0)  # (8, A) base -> sym index

        # inverse permutations: sym index -> base
        perm_back = torch.empty_like(perm_fwd)
        for t in range(perm_fwd.size(0)):
            inv = torch.empty(A, dtype=torch.long)
            inv[perm_fwd[t]] = base
            perm_back[t] = inv

        # extend to include pass action
        pass_id = A
        S = perm_fwd.size(0)
        perm_fwd_ext = torch.full((S, A + 1), pass_id, dtype=torch.long)
        perm_back_ext = torch.full((S, A + 1), pass_id, dtype=torch.long)
        perm_fwd_ext[:, :A] = perm_fwd
        perm_back_ext[:, :A] = perm_back

        # Register as buffers on the underlying nn.Module so they move with .cuda()
        # and are saved in checkpoints without contributing gradients.
        # Non-persistent so old checkpoints (without these buffers) load cleanly.
        # Buffers are rebuilt at init time anyway.
        self.nnet.register_buffer('perm_fwd_ext', perm_fwd_ext, persistent=False)
        self.nnet.register_buffer('perm_back_ext', perm_back_ext, persistent=False)
        self._sym_order = list(range(S))
        self._sym_cycle_pos = 0

    def _pick_sym_indices(self, k: int, strategy: str = 'cycle'):
        S = len(self._sym_order)
        if strategy == 'random':
            idx = torch.randperm(S)[:k].tolist()
        else:  # cycle
            start = self._sym_cycle_pos
            idx = [self._sym_order[(start + i) % S] for i in range(k)]
            self._sym_cycle_pos = (start + k) % S
        return idx

    def loss_pi_sym_aligned(self, targets, outputs_sym, t_idx=None):
        """Cross-entropy using all symmetric outputs properly aligned.

        targets: (B, A) in base orientation (probabilities)
        outputs_sym: (B, 8, A) log-probs for each symmetry view

        We align each view's action indices back to base via perm_fwd, then
        aggregate across views by log-mean-exp.
        """
        B, S, A = outputs_sym.shape
        device = outputs_sym.device
        # select subset of perms if provided
        if t_idx is None:
            perm = self.nnet.perm_fwd_ext
        else:
            perm = self.nnet.perm_fwd_ext[torch.as_tensor(t_idx, dtype=torch.long, device=device)]
        # (1,S,A) -> (B,S,A)
        index = perm.to(device).unsqueeze(0).expand(B, -1, -1)
        aligned = torch.gather(outputs_sym, dim=2, index=index)  # (B,S,A)

        # log-mean-exp across S
        lme = torch.logsumexp(aligned, dim=1) - math.log(aligned.size(1))

        return -torch.sum(targets.to(device) * lme) / targets.size(0)
    
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
        result = self.nnet.load_state_dict(checkpoint['state_dict'], strict=False)
        missing = getattr(result, 'missing_keys', [])
        unexpected = getattr(result, 'unexpected_keys', [])
        if missing:
            print(f"[load_checkpoint] missing keys ignored: {missing}")
        if unexpected:
            print(f"[load_checkpoint] unexpected keys ignored: {unexpected}")

    def get_symmetries(self, boards):
        """Vectorized symmetry generation using precomputed index maps.

        boards: (B, n, n) float tensor
        returns: (B, 8, n, n)
        """
        assert boards.ndim == 3 and boards.shape[1] == boards.shape[2], \
            f"Expected (B, n, n) square boards, got {tuple(boards.shape)}"

        B, n, _ = boards.shape
        A = n * n
        device = boards.device

        boards_flat = boards.view(B, A)                      # (B, A)
        perm = self.nnet.perm_back_ext[:, :A]                # (8, A)
        idx = perm.unsqueeze(0).expand(B, -1, -1)            # (B, 8, A)
        src = boards_flat.unsqueeze(1).expand(-1, idx.size(1), -1)  # (B, 8, A)
        out = torch.gather(src, dim=2, index=idx)            # (B, 8, A)
        out = out.view(B, idx.size(1), n, n)
        return out

    def get_symmetries_subset(self, boards: torch.Tensor, t_idx):
        """Return subset of symmetry views specified by indices t_idx (list[int])."""
        assert boards.ndim == 3 and boards.shape[1] == boards.shape[2], \
            f"Expected (B, n, n) square boards, got {tuple(boards.shape)}"
        B, n, _ = boards.shape
        A = n * n
        device = boards.device
        k = len(t_idx)
        perm = self.nnet.perm_back_ext[:, :A][torch.as_tensor(t_idx, dtype=torch.long, device=device)]  # (K,A)
        idx = perm.unsqueeze(0).expand(B, -1, -1)            # (B, K, A)
        src = boards.view(B, A).unsqueeze(1).expand(-1, k, -1)
        out = torch.gather(src, dim=2, index=idx).view(B, k, n, n)
        return out
