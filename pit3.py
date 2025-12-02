import time
import os, random, torch
import numpy as np

import Arena
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
from MCTS import MCTS
from adaptive_budget import (
    AdaptiveMCTSPlayer,
    FixedMCTSPlayer,
    RobustRootMCTSPlayer,
    robust_root_select,
)
from utils import *

"""
Adaptive search budget at inference (root-entropy guided):
1) Run a small initial batch of simulations to form an initial policy.
2) Compute root policy entropy to assess uncertainty.
3) Allocate more simulations on uncertain positions and fewer on easy ones,
   while keeping a fixed total per-game budget (approximate) via a rolling
   budget controller.

This script compares the same baseline network with and without adaptive
budget, using sims in {25, 50, 100, 200}. Results are printed for each sims.
"""


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_game(n: int = 8):
    return OthelloGame(n)


def make_fixed_player(game, nnet, sims: int):
    return FixedMCTSPlayer(game, nnet, sims=sims, cpuct=1.0, sym_eval=True)


def make_adaptive_player(game, nnet, sims: int):
    # init_frac/min/max may be tuned; defaults are reasonable for Othello

    expected_moves = 30

    return AdaptiveMCTSPlayer(
        game,
        nnet,
        avg_sims=sims,
        cpuct=1.0,
        init_frac=0.25,
        min_boost=0.7,
        max_boost=1.3,
        expected_moves_per_player=expected_moves,
        sym_eval=True,
    )

def make_robust_player(game, nnet, sims:int):
    return RobustRootMCTSPlayer(game, nnet, sims=sims, cpuct=1.0, sym_eval=True, frac=0.6)


def _root_policy_entropy(game, nnet, canonical_board):
    """
    标准 MCTS 根节点对应的网络 policy 熵（按合法动作归一化后，使用对数底 e，
    再除以 log(B) 得到 [0,1] 的归一化熵），用于不确定性分桶。
    """
    pi, _ = nnet.predict(canonical_board)
    valids = game.getValidMoves(canonical_board, 1)
    mask = (valids > 0)
    B = int(mask.sum())
    if B <= 1:
        return 0.0
    p = np.asarray(pi, dtype=np.float64) * np.asarray(valids, dtype=np.float64)
    Z = float(p.sum())
    if Z <= 0.0:
        p = np.full_like(p, 1.0 / float(B), dtype=np.float64)
    else:
        p = p / Z
    p_valid = p[mask]
    H = -float(np.sum(p_valid * np.log(np.maximum(p_valid, 1e-12))))
    return float(H / np.log(B))


def _extract_root_Qs(game, mcts, canonical_board):
    """
    从给定 MCTS 实例中提取当前根节点在“当前走子方视角”的 Q(s,a) 向量。
    若 mcts 为空，则返回 None。
    """
    if mcts is None:
        return None
    meta = mcts._sym_canonicalize(canonical_board)
    s_key = meta["s_key"]
    perm_cur2can = meta["perm_cur2can"]
    A = game.getActionSize()
    Qs = np.zeros(A, dtype=np.float32)
    valids = game.getValidMoves(canonical_board, 1)
    for a_cur in range(A):
        if valids[a_cur] <= 0:
            continue
        a_can = int(perm_cur2can[a_cur])
        Qs[a_cur] = float(mcts.Qsa.get((s_key, a_can), 0.0))
    return Qs


def _compute_q_gap(Qs, valids):
    """
    基于 baseline 根节点的 Q(s,a) 计算“决策边缘” ΔQ_gap = Q(1) - Q(2)。
    只在合法动作上排序；若合法动作数 < 2，则返回 NaN。
    """
    if Qs is None:
        return float("nan")
    mask = (np.asarray(valids) > 0)
    q_valid = np.asarray(Qs, dtype=np.float64)[mask]
    if q_valid.size < 2:
        return float("nan")
    idx = np.argsort(q_valid)[::-1]
    return float(q_valid[idx[0]] - q_valid[idx[1]])


def collect_robust_vs_baseline_data(
    game,
    nnet,
    sims: int,
    games: int = 200,
    robust_frac: float = 0.6,
):
    """
    在 robust-root 与 baseline（FixedMCTSPlayer）对弈下，收集逐手数据：
    - baseline 根动作 vs robust-root 根动作
    - 是否改变（changed）
    - 根节点网络 policy 熵 H（归一化）
    - 落子前/后的棋盘、双方棋子数、当前执子方的局面分差及其变化
    - 该手对应方在整局中的胜负结果（1 赢, -1 输）

    返回一个 dict，方便后续可视化/分析；不会打印任何统计量。
    """
    p_fixed = FixedMCTSPlayer(game, nnet, sims=sims, cpuct=1.0, sym_eval=True)
    p_robust = RobustRootMCTSPlayer(game, nnet, sims=sims, cpuct=1.0, sym_eval=True, frac=robust_frac)

    records = []

    def play_one_game(game_index: int, robust_as_player1: bool):
        board = game.getInitBoard()
        cur_player = 1
        move_idx = 0
        game_records = []

        while game.getGameEnded(board, cur_player) == 0:
            move_idx += 1
            canonical = game.getCanonicalForm(board, cur_player)

            # baseline / robust-root 在同一局面上的建议动作
            a_base = int(p_fixed(canonical))
            a_robust = int(p_robust(canonical))

            # 根节点 Q 信息：分别从 baseline / robust-root 的 MCTS 中读取
            Qs_base = _extract_root_Qs(game, p_fixed.mcts, canonical)
            Qs_robust = _extract_root_Qs(game, p_robust.mcts, canonical)
            valids = game.getValidMoves(canonical, 1)

            if Qs_base is not None and 0 <= a_base < len(Qs_base):
                Q_base = float(Qs_base[a_base])
            else:
                Q_base = float("nan")

            if Qs_robust is not None and 0 <= a_robust < len(Qs_robust):
                Q_rob = float(Qs_robust[a_robust])
            else:
                Q_rob = float("nan")

            if np.isfinite(Q_base) and np.isfinite(Q_rob):
                delta_Q = float(Q_rob - Q_base)
            else:
                delta_Q = float("nan")

            Q_gap = _compute_q_gap(Qs_base, valids)

            actor_is_robust = (cur_player == 1 and robust_as_player1) or (
                cur_player == -1 and not robust_as_player1
            )
            action = a_robust if actor_is_robust else a_base

            # 根节点网络 policy 熵
            H = _root_policy_entropy(game, nnet, canonical)

            board_before = np.array(board, copy=True)
            p1_before = int((board == 1).sum())
            p2_before = int((board == -1).sum())
            score_before = int(game.getScore(board, cur_player))

            board_after, next_player = game.getNextState(board, cur_player, action)
            board_after = np.array(board_after, copy=True)
            p1_after = int((board_after == 1).sum())
            p2_after = int((board_after == -1).sum())
            score_after = int(game.getScore(board_after, cur_player))

            rec = {
                "game_index": int(game_index),
                "move_index": int(move_idx),
                "sims": int(sims),
                "player_global": int(cur_player),
                "actor_is_robust": bool(actor_is_robust),
                "baseline_action": a_base,
                "robust_action": a_robust,
                "Q_base": Q_base,
                "Q_rob": Q_rob,
                "delta_Q": delta_Q,
                "Q_gap": Q_gap,
                "changed": bool(a_base != a_robust),
                "entropy": float(H),
                "p1_count_before": p1_before,
                "p2_count_before": p2_before,
                "p1_count_after": p1_after,
                "p2_count_after": p2_after,
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": int(score_after - score_before),
                "board_before": board_before,
                "board_after": board_after,
                "outcome": None,  # 游戏结束后再填充
            }
            game_records.append(rec)

            board, cur_player = board_after, next_player

        # 以 player=1 视角得到整局结果，再回写到每一步
        result_p1 = int(game.getGameEnded(board, 1))
        for rec in game_records:
            pg = rec["player_global"]
            if result_p1 == 0:
                outcome = 0
            else:
                outcome = result_p1 if pg == 1 else -result_p1
            rec["outcome"] = int(outcome)

        records.extend(game_records)

    # 仿照 Arena：一半 robust 执黑（player1），一半执白（player2）
    half = games // 2
    for i in range(half):
        play_one_game(i, robust_as_player1=True)
    for i in range(half, games):
        play_one_game(i, robust_as_player1=False)

    # 将 list[dict] 整理成列式 numpy 结构，便于保存/分析
    if not records:
        return {}

    boards_before = np.stack([r["board_before"] for r in records]).astype(np.int8)
    boards_after = np.stack([r["board_after"] for r in records]).astype(np.int8)

    data = {
        "sims": int(sims),
        "game_index": np.array([r["game_index"] for r in records], dtype=np.int32),
        "move_index": np.array([r["move_index"] for r in records], dtype=np.int32),
        "player_global": np.array([r["player_global"] for r in records], dtype=np.int8),
        "actor_is_robust": np.array([r["actor_is_robust"] for r in records], dtype=bool),
        "baseline_action": np.array([r["baseline_action"] for r in records], dtype=np.int16),
        "robust_action": np.array([r["robust_action"] for r in records], dtype=np.int16),
        "changed": np.array([r["changed"] for r in records], dtype=bool),
        "entropy": np.array([r["entropy"] for r in records], dtype=np.float32),
        "Q_base": np.array([r["Q_base"] for r in records], dtype=np.float32),
        "Q_rob": np.array([r["Q_rob"] for r in records], dtype=np.float32),
        "delta_Q": np.array([r["delta_Q"] for r in records], dtype=np.float32),
        "Q_gap": np.array([r["Q_gap"] for r in records], dtype=np.float32),
        "p1_count_before": np.array([r["p1_count_before"] for r in records], dtype=np.int16),
        "p2_count_before": np.array([r["p2_count_before"] for r in records], dtype=np.int16),
        "p1_count_after": np.array([r["p1_count_after"] for r in records], dtype=np.int16),
        "p2_count_after": np.array([r["p2_count_after"] for r in records], dtype=np.int16),
        "score_before": np.array([r["score_before"] for r in records], dtype=np.int16),
        "score_after": np.array([r["score_after"] for r in records], dtype=np.int16),
        "score_delta": np.array([r["score_delta"] for r in records], dtype=np.int16),
        "outcome": np.array([r["outcome"] for r in records], dtype=np.int8),
        "board_before": boards_before,
        "board_after": boards_after,
    }
    return data


def collect_robust_vs_baseline_data(
    game,
    nnet,
    sims: int,
    games: int = 200,
    robust_frac: float = 0.6,
):
    """
    新版数据收集：在同一棵 MCTS 树上同时得到 baseline 根动作和 robust-root 根动作，
    并记录 Q_base、Q_rob 与 ΔQ=Q_rob-Q_base（理论上 ΔQ>=0）。
    其余字段与旧版保持兼容。
    """

    records = []

    def play_one_game(game_index: int, robust_as_player1: bool):
        board = game.getInitBoard()
        cur_player = 1
        move_idx = 0
        game_records = []

        while game.getGameEnded(board, cur_player) == 0:
            move_idx += 1
            canonical = game.getCanonicalForm(board, cur_player)

            args = dotdict({
                "numMCTSSims": int(sims),
                "cpuct": 1.0,
                "use_dyn_c": False,
                "addRootNoise": False,
                "sym_eval": True,
            })
            mcts = MCTS(game, nnet, args)
            _ = mcts.getActionProb(canonical, temp=0)

            meta = mcts._sym_canonicalize(canonical)
            s_key = meta["s_key"]
            perm_cur2can = meta["perm_cur2can"]
            A = game.getActionSize()

            Ns = np.zeros(A, dtype=np.float32)
            Qs = np.zeros(A, dtype=np.float32)
            valids = game.getValidMoves(canonical, 1)
            for a_cur in range(A):
                if valids[a_cur] <= 0:
                    continue
                a_can = int(perm_cur2can[a_cur])
                key = (s_key, a_can)
                Ns[a_cur] = float(mcts.Nsa.get(key, 0))
                Qs[a_cur] = float(mcts.Qsa.get(key, 0.0))

            a_base, a_robust = robust_root_select(Ns, Qs, robust_frac)

            if 0 <= a_base < len(Qs) and valids[a_base] > 0:
                Q_base = float(Qs[a_base])
            else:
                Q_base = float("nan")

            if 0 <= a_robust < len(Qs) and valids[a_robust] > 0:
                Q_rob = float(Qs[a_robust])
            else:
                Q_rob = float("nan")

            if np.isfinite(Q_base) and np.isfinite(Q_rob):
                delta_Q = float(Q_rob - Q_base)
            else:
                delta_Q = float("nan")

            Q_gap = _compute_q_gap(Qs, valids)

            actor_is_robust = (cur_player == 1 and robust_as_player1) or (
                cur_player == -1 and not robust_as_player1
            )
            action = a_robust if actor_is_robust else a_base

            H = _root_policy_entropy(game, nnet, canonical)

            board_before = np.array(board, copy=True)
            p1_before = int((board == 1).sum())
            p2_before = int((board == -1).sum())
            score_before = int(game.getScore(board, cur_player))

            board_after, next_player = game.getNextState(board, cur_player, action)
            board_after = np.array(board_after, copy=True)
            p1_after = int((board_after == 1).sum())
            p2_after = int((board_after == -1).sum())
            score_after = int(game.getScore(board_after, cur_player))

            rec = {
                "game_index": int(game_index),
                "move_index": int(move_idx),
                "sims": int(sims),
                "player_global": int(cur_player),
                "actor_is_robust": bool(actor_is_robust),
                "baseline_action": int(a_base),
                "robust_action": int(a_robust),
                "Q_base": Q_base,
                "Q_rob": Q_rob,
                "delta_Q": delta_Q,
                "Q_gap": Q_gap,
                "changed": bool(a_base != a_robust),
                "entropy": float(H),
                "p1_count_before": p1_before,
                "p2_count_before": p2_before,
                "p1_count_after": p1_after,
                "p2_count_after": p2_after,
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": int(score_after - score_before),
                "board_before": board_before,
                "board_after": board_after,
                "outcome": None,
            }
            game_records.append(rec)

            board, cur_player = board_after, next_player

        result_p1 = int(game.getGameEnded(board, 1))
        for rec in game_records:
            pg = rec["player_global"]
            if result_p1 == 0:
                outcome = 0
            else:
                outcome = result_p1 if pg == 1 else -result_p1
            rec["outcome"] = int(outcome)

        records.extend(game_records)

    half = games // 2
    for i in range(half):
        play_one_game(i, robust_as_player1=True)
    for i in range(half, games):
        play_one_game(i, robust_as_player1=False)

    if not records:
        return {}

    boards_before = np.stack([r["board_before"] for r in records]).astype(np.int8)
    boards_after = np.stack([r["board_after"] for r in records]).astype(np.int8)

    data = {
        "sims": int(sims),
        "game_index": np.array([r["game_index"] for r in records], dtype=np.int32),
        "move_index": np.array([r["move_index"] for r in records], dtype=np.int32),
        "player_global": np.array([r["player_global"] for r in records], dtype=np.int8),
        "actor_is_robust": np.array([r["actor_is_robust"] for r in records], dtype=bool),
        "baseline_action": np.array([r["baseline_action"] for r in records], dtype=np.int16),
        "robust_action": np.array([r["robust_action"] for r in records], dtype=np.int16),
        "changed": np.array([r["changed"] for r in records], dtype=bool),
        "entropy": np.array([r["entropy"] for r in records], dtype=np.float32),
        "Q_base": np.array([r["Q_base"] for r in records], dtype=np.float32),
        "Q_rob": np.array([r["Q_rob"] for r in records], dtype=np.float32),
        "delta_Q": np.array([r["delta_Q"] for r in records], dtype=np.float32),
        "Q_gap": np.array([r["Q_gap"] for r in records], dtype=np.float32),
        "p1_count_before": np.array([r["p1_count_before"] for r in records], dtype=np.int16),
        "p2_count_before": np.array([r["p2_count_before"] for r in records], dtype=np.int16),
        "p1_count_after": np.array([r["p1_count_after"] for r in records], dtype=np.int16),
        "p2_count_after": np.array([r["p2_count_after"] for r in records], dtype=np.int16),
        "score_before": np.array([r["score_before"] for r in records], dtype=np.int16),
        "score_after": np.array([r["score_after"] for r in records], dtype=np.int16),
        "score_delta": np.array([r["score_delta"] for r in records], dtype=np.int16),
        "outcome": np.array([r["outcome"] for r in records], dtype=np.int8),
        "board_before": boards_before,
        "board_after": boards_after,
    }
    return data


def visualize_change_ratio(
    game,
    nnet,
    sims_list = (25, 50, 100, 200),
    games_per_match: int = 200,
    robust_frac: float = 0.6,
):
    """
    可视化需求 1：
    对于每个 sims，统计 robust-root 在所有根节点上对 baseline 动作的“改变比例”。
    """
    for sims in sims_list:
        data = collect_robust_vs_baseline_data(game, nnet, sims=sims, games=games_per_match, robust_frac=robust_frac)
        if not data:
            print(f"[sims={sims}] no data collected.")
            continue
        changed = data["changed"]
        ratio = float(np.mean(changed.astype(np.float32)))
        total = int(changed.size)
        print(f"[sims={sims}] total positions = {total}, changed = {changed.sum()} ({ratio*100:.2f}%)")


def visualize_entropy_buckets(
    game,
    nnet,
    sims_list = (25, 50, 100, 200),
    games_per_match: int = 200,
    robust_frac: float = 0.6,
    bins = (0.0, 0.33, 0.66, 1.01),
):
    """
    可视化需求 2：
    - 计算标准 MCTS 根节点网络 policy 熵 H（归一化）
    - 标记该手是否被 robust-root 改变（changed）
    - 标记该手对应执子方在整局中的结果 outcome（1 赢, -1 输, 0 和）
    - 对熵做分桶，统计每个桶中的 changed 比例，以及
      robust-root / baseline 在这些局面上的“局部胜率”（按该手执子方的最终胜率近似）。
    """
    bins = tuple(bins)
    assert len(bins) >= 2

    for sims in sims_list:
        data = collect_robust_vs_baseline_data(game, nnet, sims=sims, games=games_per_match, robust_frac=robust_frac)
        if not data:
            print(f"[sims={sims}] no data collected.")
            continue

        ent = data["entropy"]
        changed = data["changed"]
        outcome = data["outcome"]
        actor_is_robust = data["actor_is_robust"]

        print(f"\n[sims={sims}] entropy-bucket stats:")
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask_bucket = (ent >= lo) & (ent < hi)
            cnt = int(mask_bucket.sum())
            if cnt == 0:
                print(f"  H in [{lo:.2f}, {hi:.2f}): count = 0")
                continue

            mask = mask_bucket
            changed_ratio = float(np.mean(changed[mask].astype(np.float32)))

            # 近似“局部胜率”：从该手执子方视角看最终 outcome>0 的比例
            robust_mask = mask & actor_is_robust
            base_mask = mask & (~actor_is_robust)

            def win_rate(m):
                if not np.any(m):
                    return None
                return float(np.mean((outcome[m] > 0).astype(np.float32)))

            wr_robust = win_rate(robust_mask)
            wr_base = win_rate(base_mask)

            msg = (
                f"  H in [{lo:.2f}, {hi:.2f}): "
                f"count={cnt}, changed={changed[mask].sum()} ({changed_ratio*100:.2f}%)"
            )
            if wr_robust is not None:
                msg += f", robust-win-rate={wr_robust*100:.2f}%"
            if wr_base is not None:
                msg += f", baseline-win-rate={wr_base*100:.2f}%"
            print(msg)


def analyze_delta_Q(
    game,
    nnet,
    sims_list=(25, 50, 100, 200),
    games_per_match: int = 200,
    robust_frac: float = 0.6,
):
    """
    重点分析 ①：每次改动带来的价值提升 ΔQ = Q_rob - Q_base。
    这里只看发生了改动（changed=True）的根节点，统计每个 sims 下 ΔQ 的均值和标准差。
    """
    import math

    summary = {}
    for sims in sims_list:
        data = collect_robust_vs_baseline_data(
            game, nnet, sims=sims, games=games_per_match, robust_frac=robust_frac
        )
        if not data:
            print(f"[sims={sims}] no data collected.")
            continue

        delta_q = data["delta_Q"].astype(np.float64)
        changed = data["changed"]
        mask = changed & np.isfinite(delta_q)
        n = int(mask.sum())
        if n == 0:
            print(f"[sims={sims}] no changed positions with finite ΔQ.")
            continue

        vals = delta_q[mask]
        mean = float(vals.mean())
        std = float(vals.std(ddof=1 if n > 1 else 0))
        ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        summary[sims] = dict(n=n, mean=mean, std=std, ci95=ci95)

        print(
            f"[sims={sims}] changed positions: n={n}, "
            f"E[ΔQ]={mean:.4f}, std={std:.4f}, 95%CI≈[{mean-ci95:.4f}, {mean+ci95:.4f}]"
        )

    return summary


def analyze_changed_vs_unchanged_games(
    game,
    nnet,
    sims_list=(25, 50, 100, 200),
    games_per_match: int = 200,
    robust_frac: float = 0.6,
):
    """
    重点分析 ②：按对局划分“有改动的局”(changed games) 与 “没改动的局”(no-change games)。    
    对每个 sims 统计：
      - changed games 占比
      - changed games 中 robust‑root 的胜率
      - no-change games 中 robust‑root 的胜率
    """
    summary = {}

    for sims in sims_list:
        data = collect_robust_vs_baseline_data(
            game, nnet, sims=sims, games=games_per_match, robust_frac=robust_frac
        )
        if not data:
            print(f"[sims={sims}] no data collected.")
            continue

        game_idx = data["game_index"]
        changed = data["changed"]
        actor_is_robust = data["actor_is_robust"]
        outcome = data["outcome"]

        uniq_games = np.unique(game_idx)
        res_changed = []
        res_unchanged = []

        for g_id in uniq_games:
            mask_g = (game_idx == g_id)
            has_change = bool(np.any(changed[mask_g]))

            # 该局中 robust 一方的最终结果：找任意一手 actor_is_robust=True 的记录即可
            mask_rob = mask_g & actor_is_robust
            if not np.any(mask_rob):
                continue
            r = int(outcome[mask_rob][0])  # >0 表示 robust 赢，<0 输，0 和

            if has_change:
                res_changed.append(r)
            else:
                res_unchanged.append(r)

        total_games = len(uniq_games)
        n_changed = len(res_changed)
        n_unchanged = len(res_unchanged)

        if total_games == 0:
            print(f"[sims={sims}] no games in data.")
            continue

        def win_rate(results):
            if not results:
                return None
            arr = np.asarray(results, dtype=np.int32)
            return float(np.mean((arr > 0).astype(np.float32)))

        wr_changed = win_rate(res_changed)
        wr_unchanged = win_rate(res_unchanged)

        frac_changed = n_changed / float(total_games)
        summary[sims] = dict(
            total_games=total_games,
            changed_games=n_changed,
            nochange_games=n_unchanged,
            frac_changed=frac_changed,
            win_rate_changed=wr_changed,
            win_rate_nochange=wr_unchanged,
        )

        print(f"\n[sims={sims}] changed-vs-no-change games:")
        print(
            f"  total_games={total_games}, changed_games={n_changed} "
            f"({frac_changed*100:.2f}%), no-change_games={n_unchanged}"
        )
        if wr_changed is not None:
            print(f"  robust win-rate on changed games   = {wr_changed*100:.2f}%")
        if wr_unchanged is not None:
            print(f"  robust win-rate on no-change games = {wr_unchanged*100:.2f}%")

    return summary


def analyze_q_gap_buckets(
    game,
    nnet,
    sims_list=(25, 50, 100, 200),
    games_per_match: int = 200,
    robust_frac: float = 0.6,
    gap_bins=(0.0, 0.02, 0.05, 1.0),
):
    """
    重点分析 ③：用 “Q 差距” ΔQ_gap = Q(1) - Q(2) 分桶来刻画局面难度。
    对每个 sims 和每个 gap 桶，统计：
      - robust-root 的改动频率（changed 比例）
      - 若存在 ΔQ，则给出桶内 ΔQ 的平均值
      - 近似的局部胜率（actor 为 robust 时 outcome>0 的比例）
    """
    gap_bins = tuple(gap_bins)
    assert len(gap_bins) >= 2

    summary = {}

    for sims in sims_list:
        data = collect_robust_vs_baseline_data(
            game, nnet, sims=sims, games=games_per_match, robust_frac=robust_frac
        )
        if not data:
            print(f"[sims={sims}] no data collected.")
            continue

        Q_gap = data["Q_gap"].astype(np.float64)
        delta_q = data["delta_Q"].astype(np.float64)
        changed = data["changed"]
        actor_is_robust = data["actor_is_robust"]
        outcome = data["outcome"]

        print(f"\n[sims={sims}] Q-gap bucket stats:")
        stats_per_bin = []

        for i in range(len(gap_bins) - 1):
            lo, hi = gap_bins[i], gap_bins[i + 1]
            mask_bucket = np.isfinite(Q_gap) & (Q_gap >= lo) & (Q_gap < hi)
            cnt = int(mask_bucket.sum())
            if cnt == 0:
                print(f"  ΔQ_gap in [{lo:.3f}, {hi:.3f}): count = 0")
                stats_per_bin.append(None)
                continue

            mask = mask_bucket
            changed_ratio = float(np.mean(changed[mask].astype(np.float32)))

            # ΔQ 只在发生改动的位置更有意义
            mask_dq = mask & changed & np.isfinite(delta_q)
            if np.any(mask_dq):
                mean_dq = float(delta_q[mask_dq].mean())
            else:
                mean_dq = None

            # 近似局部胜率：只看 robust 执子的一方
            mask_rob = mask & actor_is_robust
            if np.any(mask_rob):
                wr_rob = float(np.mean((outcome[mask_rob] > 0).astype(np.float32)))
            else:
                wr_rob = None

            stats_per_bin.append(
                dict(
                    lo=lo,
                    hi=hi,
                    count=cnt,
                    changed_ratio=changed_ratio,
                    mean_delta_Q=mean_dq,
                    robust_win_rate=wr_rob,
                )
            )

            msg = (
                f"  ΔQ_gap in [{lo:.3f}, {hi:.3f}): "
                f"count={cnt}, changed={changed[mask].sum()} ({changed_ratio*100:.2f}%)"
            )
            if mean_dq is not None:
                msg += f", mean ΔQ(changed)={mean_dq:.4f}"
            if wr_rob is not None:
                msg += f", robust-win-rate={wr_rob*100:.2f}%"
            print(msg)

        summary[sims] = stats_per_bin

    return summary


def run_matchup(label: str, p1, p2, game, games: int = 400, verbose: bool = False):
    arena = Arena.Arena(p1, p2, game, display=OthelloGame.display)
    start = time.time()
    result = arena.playGames(games, verbose=verbose)
    secs = time.time() - start
    oneWon, twoWon, draws = result
    print(f"{label}: result={result}, time={secs:.2f}s")
    return result


def main():
    set_seed(42)

    g = make_game(8)

    # Shared network (baseline weights) used for both players
    nnet = NNet(g)
    nnet.load_checkpoint('./models/', 'baseline.pth.tar')

    games_per_match = 200  # 200 per side

    # Compare dynamic vs fixed at multiple average sims
    for sims in [25, 50, 100, 200]:
        p_fixed = make_fixed_player(g, nnet, sims=sims)
        p_adapt = make_robust_player(g, nnet, sims=sims)
        label = f"adaptive(vs fixed) sims={sims}"
        run_matchup(label, p_adapt, p_fixed, g, games_per_match)

    # Also pit adaptive against random, greedy, and alpha-beta
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    mp = AlphaBetaOthelloPlayer(g, depth=3).play

    for sims in [25, 50, 100, 200]:
        p_adapt = make_robust_player(g, nnet, sims=sims)
        run_matchup(f"adaptive vs random (sims={sims})", p_adapt, rp, g, games_per_match)
        run_matchup(f"adaptive vs greedy (sims={sims})", p_adapt, gp, g, games_per_match)
        run_matchup(f"adaptive vs alphabeta(d=3) (sims={sims})", p_adapt, mp, g, games_per_match)


if __name__ == "__main__":
    main()
