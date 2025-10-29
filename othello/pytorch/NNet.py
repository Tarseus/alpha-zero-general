import os
import sys
import time
import copy

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
            'device': 0,
            'num_channels': 512,

            'use_sym': False,
            'inv_coef': 0.0,
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
        # Note: we keep symmetry buffers for encoder invariance loss only.
        self._init_action_perms()

        if self.args.cuda:
            self.nnet.cuda(self.args.device)
            torch.backends.cudnn.benchmark = True
        # EMA teacher placeholder
        self._ema_teacher = None

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
            total_losses = AverageMeter()
            kl_losses = AverageMeter()
            cons_v_losses = AverageMeter()

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

                # compute output with AMP
                scaler = getattr(self, '_scaler', None)
                if scaler is None:
                    self._scaler = GradScaler(enabled=(self.args.cuda and bool(getattr(self.args, 'amp', False))))
                    scaler = self._scaler

                with autocast(enabled=(self.args.cuda and bool(getattr(self.args, 'amp', False)))):
                    out_pi, _, out_v, _, _ = self.nnet(boards)

                    # Train only on base view outputs; no symmetry augmentation
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                    # EMA Teacher consistency losses (optional)
                    if bool(getattr(self.args, 'ema_enable', False)):
                        # init teacher if needed
                        if self._ema_teacher is None:
                            self._ema_teacher = copy.deepcopy(self.nnet).eval()
                            if self.args.cuda:
                                self._ema_teacher.cuda(self.args.device)
                            for p in self._ema_teacher.parameters():
                                p.requires_grad_(False)
                        with torch.no_grad():
                            t_pi_logits, _, t_v, _, _ = self._ema_teacher(boards)
                        # KL(student||teacher): input is log-prob, target is prob
                        consis_pi_coef = float(getattr(self.args, 'consis_pi_coef', 0.05))
                        consis_v_coef = float(getattr(self.args, 'consis_v_coef', 0.01))
                        cur_kl = None
                        cur_cv = None
                        if consis_pi_coef > 0.0:
                            t_pi_prob = torch.exp(t_pi_logits)
                            kl = F.kl_div(out_pi, t_pi_prob, reduction='batchmean')
                            total_loss = total_loss + consis_pi_coef * kl
                            cur_kl = kl
                        if consis_v_coef > 0.0:
                            mse_v = F.mse_loss(out_v.view(-1), t_v.view(-1))
                            total_loss = total_loss + consis_v_coef * mse_v
                            cur_cv = mse_v

                # record loss
                bs = boards.size(0)
                pi_losses.update(l_pi.item(), bs)
                v_losses.update(l_v.item(), bs)
                total_losses.update(total_loss.item(), bs)
                # optional metrics
                if 'cur_kl' in locals() and cur_kl is not None:
                    kl_losses.update(cur_kl.item(), bs)
                if 'cur_cv' in locals() and cur_cv is not None:
                    cons_v_losses.update(cur_cv.item(), bs)
                # update tqdm postfix with compact metrics
                try:
                    lr = optimizer.param_groups[0].get('lr', None)
                except Exception:
                    lr = None
                postfix = {
                    'L_pi': f"{pi_losses.avg:.3e}",
                    'L_v': f"{v_losses.avg:.3e}",
                    'L_tot': f"{total_losses.avg:.3e}",
                }
                if kl_losses.count > 0:
                    postfix['KL'] = f"{kl_losses.avg:.3e}"
                if cons_v_losses.count > 0:
                    postfix['ConsV'] = f"{cons_v_losses.avg:.3e}"
                if lr is not None:
                    postfix['LR'] = f"{lr:.2e}"
                t.set_postfix(**postfix)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                # EMA update after student step
                if bool(getattr(self.args, 'ema_enable', False)) and (self._ema_teacher is not None):
                    with torch.no_grad():
                        m = float(getattr(self.args, 'ema_decay', 0.999))
                        for p_t, p_s in zip(self._ema_teacher.parameters(), self.nnet.parameters()):
                            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))

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

    def loss_sym_to_identity(self, out_z_sym, stopgrad=True):
        """Align symmetric encodings to the identity-view encoding.

        out_z_sym: (B, S, D) encoder features for S symmetry views.
        Assumes the last view (index S-1) is the identity view, as ensured by
        the training batch preparation. Uses cosine similarity.
        """
        if out_z_sym.dim() != 3 or out_z_sym.size(1) <= 1:
            # no symmetric views to compare; return zero loss
            return torch.tensor(0, dtype=torch.float32, device=out_z_sym.device)

        z = F.normalize(out_z_sym, p=2, dim=-1)   # (B, S, D)
        anchor = z[:, -1, :]                      # (B, D)
        if stopgrad:
            anchor = anchor.detach()
        # cosine similarity to identity for all non-identity views
        cos = (z[:, :-1, :] * anchor.unsqueeze(1)).sum(dim=-1)  # (B, S-1)
        return (1.0 - cos).mean()

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

        # Build sym->base mapping first by rotating the index grid.
        # M.reshape(-1)[j] = a means: at symmetric view position j lies base index a.
        perms_sym2base = []
        for i in range(1, 5):
            for j in (True, False):
                M = torch.rot90(grid, i, dims=(0, 1))
                if j:
                    M = torch.flip(M, dims=(1,))
                perms_sym2base.append(M.reshape(-1))
        perm_sym2base = torch.stack(perms_sym2base, dim=0)  # (8, A) sym index -> base index

        # Invert to get base->sym mapping so that for a base action 'a', perm_base2sym[t, a]
        # gives the action index in the t-th symmetric view.
        perm_base2sym = torch.empty_like(perm_sym2base)
        for t in range(perm_sym2base.size(0)):
            inv = torch.empty(A, dtype=torch.long)
            inv[perm_sym2base[t]] = base
            perm_base2sym[t] = inv

        # extend to include pass action
        pass_id = A
        S = perm_base2sym.size(0)
        perm_fwd_ext = torch.full((S, A + 1), pass_id, dtype=torch.long)
        perm_back_ext = torch.full((S, A + 1), pass_id, dtype=torch.long)
        # Note: by convention
        #  - perm_fwd_ext: base -> sym index (forward)
        #  - perm_back_ext: sym index -> base index (backward)
        perm_fwd_ext[:, :A] = perm_base2sym
        perm_back_ext[:, :A] = perm_sym2base

        # Register as buffers on the underlying nn.Module so they move with .cuda()
        # and are saved in checkpoints without contributing gradients.
        # Non-persistent so old checkpoints (without these buffers) load cleanly.
        # Buffers are rebuilt at init time anyway.
        self.nnet.register_buffer('perm_fwd_ext', perm_fwd_ext, persistent=False)
        self.nnet.register_buffer('perm_back_ext', perm_back_ext, persistent=False)
        self._sym_order = list(range(S))
        self._sym_cycle_pos = 0
        # find the identity symmetry index (the one mapping to itself)
        is_identity = (perm_sym2base == base).all(dim=1)
        idxs = torch.nonzero(is_identity, as_tuple=False).view(-1)
        self._sym_identity = int(idxs[0].item()) if idxs.numel() > 0 else S - 1

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
