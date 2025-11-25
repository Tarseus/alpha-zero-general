import time
import os, random, torch
import numpy as np

import Arena
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
from adaptive_budget import AdaptiveMCTSPlayer, FixedMCTSPlayer, RobustRootMCTSPlayer
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
