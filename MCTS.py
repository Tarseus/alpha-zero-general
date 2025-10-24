import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        # debug: count of masked-all-valids logs to avoid huge files
        self._mask_debug_count = 0

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        s0 = self.game.stringRepresentation(canonicalBoard)
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, depth=0)

            if i == 0 and self.args.get('addRootNoise', False):
                P = self.Ps[s0].copy()
                valids = self.Vs[s0]
                mask = (valids > 0)
                eps = 0.25
                alpha = self.args.get('dirichletAlpha', 0.3)
                eta = np.zeros_like(P)
                m = int(mask.sum())
                if m > 0:
                    eta[mask] = np.random.dirichlet([alpha] * m)
                    P = (1 - eps) * P + eps * eta
                    P = P / P.sum()
                    self.Ps[s0] = P

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, depth=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # A = self.game.getActionSize()
            # dummy_pi = np.zeros(A, dtype = np.float32)
            # syms = self.game.getSymmetries(canonicalBoard, dummy_pi)
            # boards_sym = [b for (b, _) in syms]
            # K = len(boards_sym)

            # pis_sym, vs_sym = self.nnet.predict(np.stack(boards_sym, axis=0))  # pis_sym:(K,A), vs_sym:(K,)

            # pis_back = np.zeros((K, A), dtype=np.float32)
            # for i in range(K):
            #     b_sym = boards_sym[i]
            #     pi_sym = pis_sym[i]
            #     back = self.game.getSymmetries(b_sym, pi_sym)
            #     pi_i = None
            #     for (b_j, pi_j) in back:
            #         if np.array_equal(b_j, canonicalBoard):
            #             pi_i = pi_j
            #             break
            #     if pi_i is None:
            #         pi_i = pi_sym
            #     pis_back[i] = pi_i

            # policy = pis_back.mean(axis=0)
            # v = vs_sym.mean()
            # self.Ps[s] = policy

            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            P_raw = self.Ps[s].copy()
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                # debug dump
                try:
                    limit = int(getattr(self.args, 'mask_debug_limit', 200))
                    if self._mask_debug_count < limit:
                        self._mask_debug_count += 1
                        import os, datetime
                        os.makedirs('temp', exist_ok=True)
                        path = os.path.join('temp', 'mcts_debug.log')
                        valid_idx = [i for i in range(self.game.getActionSize()) if valids[i] > 0]
                        mask = np.array(valids, dtype=np.float32) > 0
                        p_raw = np.array(P_raw, dtype=np.float64)
                        p_mask = p_raw * mask
                        def topk_pairs(arr, idxs, k=10):
                            pairs = [(float(arr[i]), int(i)) for i in idxs]
                            pairs.sort(key=lambda x: x[0], reverse=True)
                            return pairs[:k]
                        top_valid = topk_pairs(p_raw, valid_idx, k=10)
                        top_mask = topk_pairs(p_mask, valid_idx, k=10)
                        non_finite = int(np.count_nonzero(~np.isfinite(p_raw)))
                        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        dyn_mode = getattr(self.args, 'dyn_c_mode', None)
                        with open(path, 'a', encoding='utf-8') as f:
                            f.write(f"\n[{ts}] MASKED-ALL at state s(len)={len(s)} A={self.game.getActionSize()}\n")
                            f.write(f"use_dyn_c={getattr(self.args, 'use_dyn_c', False)}, dyn_c_mode={dyn_mode}\n")
                            try:
                                if hasattr(self.game, 'stringRepresentationReadable'):
                                    br = self.game.stringRepresentationReadable(canonicalBoard)
                                    f.write(f"board:\n{br}\n")
                            except Exception:
                                pass
                            f.write(f"valid_count={len(valid_idx)}, valid_idx(sample)={valid_idx[:20]}\n")
                            f.write(f"sum(P_raw)={float(np.nansum(p_raw)):.6g}, sum(P*valids)={float(np.nansum(p_mask)):.6g}, non_finite={non_finite}\n")
                            if len(valid_idx) > 0:
                                f.write(f"top_valid_P_raw (p,idx)={top_valid}\n")
                                f.write(f"top_valid_P_mask (p,idx)={top_mask}\n")
                except Exception as e:
                    log.exception("Failed to write mcts_debug log: %s", e)
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        if self.args.use_dyn_c:
            mode = getattr(self.args, 'dyn_c_mode', 'entropy')
            if mode == 'othello':
                c = self._cpuct_othello(canonicalBoard, s, valids, depth)
            else:
                c = self._cpuct_from_entropy(s, valids)
        else:
            c = self.args.cpuct

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + c * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = c * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, depth + 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def _cpuct_from_entropy(self, s, valids):
        valid_idx = [a for a in range(self.game.getActionSize()) if valids[a]]
        B = len(valid_idx)
        if B <= 1:
            return self.args.cmin

        p = [max(float(self.Ps[s][a]), 0.0) for a in valid_idx]
        Z = sum(p)
        if Z <= 0.0:
            p = [1.0 / B] * B
        else:
            p = [pi / Z for pi in p]

        H = -sum(pi * math.log(max(pi, 1e-12)) for pi in p)
        H_norm = H / math.log(B)

        c = self.args.cmin + (self.args.cmax - self.args.cmin) * H_norm
        return max(self.args.cmin, min(self.args.cmax, c))

    def _cpuct_othello(self, board, s, valids, depth):
        n0, n1 = self.game.getBoardSize()
        n = int(n0)
        A = self.game.getActionSize()
        pass_idx = A - 1

        valid_idx = [a for a in range(A) if valids[a] and a != pass_idx]
        B = len(valid_idx)
        if B <= 1:
            return self.args.cmin

        p = [max(float(self.Ps[s][a]), 0.0) for a in valid_idx]
        Z = sum(p)
        if Z <= 0.0:
            p = [1.0 / B] * B
        else:
            p = [pi / Z for pi in p]
        H = -sum(pi * math.log(max(pi, 1e-12)) for pi in p)
        H_norm = H / math.log(B)

        E = int((board == 0).sum()) if hasattr(board, 'sum') else 0
        denom = max(n * n - 4, 1)
        r_e = min(1.0, max(0.0, E / float(denom)))

        B_ref = max(1.0, 0.22 * n * n)
        r_m = min(1.0, B / B_ref)

        Qs = [self.Qsa[(s, a)] for a in valid_idx if (s, a) in self.Qsa]
        if len(Qs) >= 2:
            std_q = float(np.std(Qs))
            kappa = float(getattr(self.args, 'othello_q_kappa', 0.5))
            r_q = math.tanh(std_q / max(kappa, 1e-6))
        else:
            r_q = 0.0

        if n >= 3:
            x_squares = {(1, 1), (1, n - 2), (n - 2, 1), (n - 2, n - 2)}
        else:
            x_squares = set()
        if n >= 2:
            c_squares = {
                (0, 1), (1, 0), (0, n - 2), (1, n - 1),
                (n - 1, 1), (n - 2, 0), (n - 2, n - 1), (n - 1, n - 2)
            }
        else:
            c_squares = set()

        def assoc_corner(rc):
            r, c = rc
            if (r, c) in x_squares:
                if r == 1 and c == 1:
                    return (0, 0)
                if r == 1 and c == n - 2:
                    return (0, n - 1)
                if r == n - 2 and c == 1:
                    return (n - 1, 0)
                if r == n - 2 and c == n - 2:
                    return (n - 1, n - 1)
            if (r, c) in c_squares:
                if r == 0 and c == 1: return (0, 0)
                if r == 1 and c == 0: return (0, 0)
                if r == 0 and c == n - 2: return (0, n - 1)
                if r == 1 and c == n - 1: return (0, n - 1)
                if r == n - 1 and c == 1: return (n - 1, 0)
                if r == n - 2 and c == 0: return (n - 1, 0)
                if r == n - 2 and c == n - 1: return (n - 1, n - 1)
                if r == n - 1 and c == n - 2: return (n - 1, n - 1)
            return None

        danger_cnt = 0
        for a in valid_idx:
            r = a // n
            c = a % n
            if (r, c) in x_squares or (r, c) in c_squares:
                corner = assoc_corner((r, c))
                if corner is not None:
                    cr, cc = corner
                    if 0 <= cr < n and 0 <= cc < n and board[cr][cc] == 0:
                        danger_cnt += 1
        danger = danger_cnt / float(B) if B > 0 else 0.0

        w_e, w_m, w_h, w_q = 0.40, 0.25, 0.20, 0.15
        score = w_e * r_e + w_m * r_m + w_h * H_norm + w_q * r_q
        score = max(0.0, min(1.0, score))

        cmin = float(getattr(self.args, 'cmin', 0.5))
        cmax = float(getattr(self.args, 'cmax', 3.0))
        base = cmin + (cmax - cmin) * score

        danger_w = float(getattr(self.args, 'othello_danger_weight', 0.5))
        base *= (1.0 - danger_w * danger)

        tau = float(getattr(self.args, 'othello_depth_tau', 8.0))
        c = base / (1.0 + float(depth) / max(tau, 1e-6))

        return max(cmin, min(cmax, c))
