import math
import numpy as np

from MCTS import MCTS
from utils import dotdict


def robust_root_select(Ns: np.ndarray, Qs: np.ndarray, frac: float):
    """
    Given root visit counts Ns and values Qs (current-orientation indexing),
    compute:
      - baseline_action: argmax_a Ns[a] (ties broken uniformly at random)
      - robust_action  : among actions with Ns[a] >= frac * max_a Ns[a],
                         pick the one with highest Q[a] (ties by Ns).

    Returns (baseline_action, robust_action) as ints. If there are no
    meaningful visits (max Ns <= 0), both fall back to argmax Ns.
    """
    Ns = np.asarray(Ns, dtype=np.float32)
    Qs = np.asarray(Qs, dtype=np.float32)
    A = int(Ns.size)
    if A == 0:
        return 0, 0

    bestN = float(Ns.max())
    if not np.isfinite(bestN):
        bestN = 0.0

    if bestN <= 0.0:
        # No visits recorded (e.g., terminal root); fall back to argmax Ns
        bestAs = np.flatnonzero(Ns == bestN)
        if bestAs.size == 0:
            return 0, 0
        a = int(np.random.choice(bestAs))
        return a, a

    bestAs = np.flatnonzero(Ns == bestN)
    baseline_action = int(np.random.choice(bestAs)) if bestAs.size > 0 else int(np.argmax(Ns))

    frac_clamped = float(max(0.0, min(1.0, frac)))
    threshold = frac_clamped * bestN
    candidates = [a for a in range(A) if Ns[a] >= threshold]
    if not candidates:
        # Fallback: at least keep standard argmax-visits actions
        candidates = bestAs.tolist() if bestAs.size > 0 else [baseline_action]

    def score(a: int):
        return (float(Qs[a]), float(Ns[a]))

    robust_action = int(max(candidates, key=score))
    return baseline_action, robust_action


class FixedMCTSPlayer:
    """Fixed per-move simulation budget MCTS player.

    Callable object compatible with Arena: implements __call__(board) -> action
    and optional startGame()/endGame() hooks.
    """

    def __init__(self, game, nnet, sims: int, cpuct: float = 1.0, sym_eval: bool = True):
        self.game = game
        self.nnet = nnet
        self.sims = int(sims)
        self.cpuct = float(cpuct)
        self.sym_eval = bool(sym_eval)
        # persistent MCTS instance reused across moves & games (baseline pit.py style)
        self.mcts = None

    def _new_mcts(self):
        args = dotdict({
            'numMCTSSims': self.sims,
            'cpuct': self.cpuct,
            'use_dyn_c': False,
            'addRootNoise': False,
            'sym_eval': self.sym_eval,
        })
        return MCTS(self.game, self.nnet, args)

    def startGame(self):
        pass

    def endGame(self):
        pass

    def __call__(self, canonical_board):
        # Match pit.py baseline: reuse a single MCTS instance across games.
        if self.mcts is None:
            self.mcts = self._new_mcts()
        probs = self.mcts.getActionProb(canonical_board, temp=0)
        return int(np.argmax(probs))


class RobustRootMCTSPlayer:
    """MCTS player with a more value-aware root move selection.

    Search is identical to the standard FixedMCTSPlayer; only the final root
    move is chosen differently:
      1) Run MCTS and gather root visit counts N(s,a) and values Q(s,a).
      2) Let bestN = max_a N(s,a).
      3) Form a candidate set {a : N(s,a) >= frac * bestN}.
      4) Among candidates, pick the action with the highest Q(s,a).

    This keeps the tree policy unchanged and only tweaks the root choice,
    making it a cheap but often helpful trick.
    """

    def __init__(self, game, nnet, sims: int, cpuct: float = 1.0, sym_eval: bool = True, frac: float = 0.6):
        self.game = game
        self.nnet = nnet
        self.sims = int(sims)
        self.cpuct = float(cpuct)
        self.sym_eval = bool(sym_eval)
        # fraction used to define "well-visited" actions at the root
        self.frac = float(max(0.0, min(1.0, frac)))
        # persistent MCTS instance reused across games (like pit.py)
        self.mcts = None

    def _new_mcts(self):
        args = dotdict({
            'numMCTSSims': self.sims,
            'cpuct': self.cpuct,
            'use_dyn_c': False,
            'addRootNoise': False,
            'sym_eval': self.sym_eval,
        })
        return MCTS(self.game, self.nnet, args)

    def startGame(self):
        # Match pit.py behavior: do not reset tree between games.
        pass

    def endGame(self):
        # Keep tree for subsequent games (full cross-game reuse).
        pass

    def __call__(self, canonical_board):
        # Lazily create MCTS if startGame wasn't called
        if self.mcts is None:
            self.mcts = self._new_mcts()
        mcts = self.mcts

        # Run standard MCTS search (same as pit.py usage, but on a persistent tree)
        _ = mcts.getActionProb(canonical_board, temp=0)

        # Extract root N(s,a) and Q(s,a) in current-orientation indexing
        meta = mcts._sym_canonicalize(canonical_board)
        s_key = meta['s_key']
        perm_cur2can = meta['perm_cur2can']
        A = self.game.getActionSize()

        Ns = np.zeros(A, dtype=np.float32)
        Qs = np.zeros(A, dtype=np.float32)
        for a_cur in range(A):
            a_can = int(perm_cur2can[a_cur])
            key = (s_key, a_can)
            Ns[a_cur] = float(mcts.Nsa.get(key, 0))
            Qs[a_cur] = float(mcts.Qsa.get(key, 0.0))
        _, robust_action = robust_root_select(Ns, Qs, self.frac)
        return int(robust_action)


class AdaptiveMCTSPlayer:
    """Adaptive per-move simulation budget based on root policy entropy.

    Maintains a fixed total budget per game (approximate, based on expected
    moves per player) and redistributes sims: fewer on low-uncertainty roots,
    more on high-uncertainty roots.
    """

    def __init__(
        self,
        game,
        nnet,
        avg_sims: int,
        *,
        cpuct: float = 1.0,
        init_frac: float = 0.2,
        min_boost: float = 0.5,
        max_boost: float = 1.5,
        expected_moves_per_player=None,
        sym_eval: bool = True,
    ):
        self.game = game
        self.nnet = nnet
        self.avg_sims = int(max(1, avg_sims))
        self.cpuct = float(cpuct)
        self.init_frac = float(max(0.0, min(1.0, init_frac)))
        self.min_boost = float(max(0.0, min(10.0, min_boost)))
        self.max_boost = float(max(self.min_boost, min(10.0, max_boost)))
        self.sym_eval = bool(sym_eval)

        # Estimate total turns per player (Othello default: (n*n-4)/2)
        if expected_moves_per_player is None:
            n0, n1 = self.game.getBoardSize()
            n = int(n0)
            expected_moves_per_player = max(1, (n * n - 4) // 2)
        self.expected_moves_per_player = int(expected_moves_per_player)

        # Rolling game state
        self.used_sims = 0
        self.move_count = 0
        self.total_budget = self.avg_sims * self.expected_moves_per_player

    def startGame(self):
        self.used_sims = 0
        self.move_count = 0
        self.total_budget = self.avg_sims * self.expected_moves_per_player

    def endGame(self):
        if self.move_count > 0:
            avg_sims = self.used_sims / float(self.move_count)
            print(
                f"[AdaptiveMCTSPlayer] 平均模拟次数 N = {avg_sims:.2f} "
                f"(moves={self.move_count}, sims={self.used_sims})"
            )

    def _run_simulations(self, mcts: MCTS, canonical_board, num: int):
        # Lightweight driver to run a fixed number of simulations
        for _ in range(int(num)):
            mcts.search(canonical_board, depth=0)

    def _root_counts(self, mcts: MCTS, canonical_board):
        # Mirror logic from MCTS.getActionProb for extracting root visit counts
        meta = mcts._sym_canonicalize(canonical_board)
        s = meta['s_key']
        perm_cur2can = meta['perm_cur2can']
        A = self.game.getActionSize()
        counts = [mcts.Nsa[(s, int(perm_cur2can[a]))] if (s, int(perm_cur2can[a])) in mcts.Nsa else 0 for a in range(A)]
        return s, np.array(counts, dtype=np.float32)

    def _entropy_norm(self, probs: np.ndarray, valids: np.ndarray) -> float:
        # Compute entropy normalized by log(#valid)
        mask = (valids > 0)
        B = int(mask.sum())
        if B <= 1:
            return 0.0
        p = probs[mask].astype(np.float64)
        Z = float(p.sum())
        if Z <= 0:
            p = np.full((B,), 1.0 / B, dtype=np.float64)
        else:
            p = p / Z
        H = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
        return float(H / math.log(B))

    def __call__(self, canonical_board):
        # Remaining aggregate budget and baseline per-move allowance
        moves_left_est = max(1, self.expected_moves_per_player - self.move_count)
        remaining_budget = max(0, self.total_budget - self.used_sims)
        base_allow = remaining_budget / moves_left_est

        # Initial probe
        init_sims = max(1, int(round(base_allow * self.init_frac)))
        args = dotdict({
            'numMCTSSims': init_sims,  # not used because we call search directly
            'cpuct': self.cpuct,
            'use_dyn_c': False,
            'addRootNoise': False,
            'sym_eval': self.sym_eval,
        })
        mcts = MCTS(self.game, self.nnet, args)
        self._run_simulations(mcts, canonical_board, init_sims)

        # Measure uncertainty via root visit distribution entropy
        s_key, counts = self._root_counts(mcts, canonical_board)
        valids = mcts.Vs.get(s_key)
        if valids is None:
            # Ensure valids present (create via one pass if somehow missing)
            self._run_simulations(mcts, canonical_board, 1)
            s_key, counts = self._root_counts(mcts, canonical_board)
            valids = mcts.Vs.get(s_key)
        if valids is None:
            valids = np.ones_like(counts, dtype=np.int32)

        probs_est = counts.astype(np.float64)
        Hn = self._entropy_norm(probs_est, np.array(valids))

        # Target sims multiplier based on entropy
        mult = self.min_boost + (self.max_boost - self.min_boost) * float(max(0.0, min(1.0, Hn)))
        desired = int(round(base_allow * mult))

        # Respect remaining budget and keep a floor for future moves
        min_future = max(1, int(round(self.avg_sims * self.min_boost)))
        reserve = min_future * max(0, moves_left_est - 1)
        max_this_turn = max(1, remaining_budget - reserve)
        target = int(max(init_sims, min(desired, max_this_turn)))

        # Run additional sims if needed
        add_sims = max(0, target - init_sims)
        if add_sims > 0:
            self._run_simulations(mcts, canonical_board, add_sims)

        # Pick action via final visit counts (temp=0 argmax tie-broken randomly)
        _, counts = self._root_counts(mcts, canonical_board)
        # Use current-orientation valids for selection masking
        valids_cur = self.game.getValidMoves(canonical_board, 1)
        counts_sel = counts.astype(np.float64).copy()
        counts_sel = np.where(np.array(valids_cur) > 0, counts_sel, -1.0)
        best = float(np.max(counts_sel)) if counts_sel.size > 0 else -1.0

        if best < 0.0:
            # Fallback to masked network prior
            pi, _ = self.nnet.predict(canonical_board)
            p = np.array(pi, dtype=np.float64) * np.array(valids_cur, dtype=np.float64)
            if np.sum(p) <= 0:
                # Uniform over valid moves
                valid_idx = np.flatnonzero(np.array(valids_cur) > 0)
                a = int(np.random.choice(valid_idx)) if valid_idx.size > 0 else int(np.argmax(pi))
            else:
                a = int(np.argmax(p))
        else:
            bestAs = np.flatnonzero(counts_sel == best)
            a = int(np.random.choice(bestAs)) if bestAs.size > 0 else int(np.argmax(counts_sel))

        # Book-keeping
        self.used_sims += int(init_sims + add_sims)
        self.move_count += 1
        return a
