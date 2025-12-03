import os
import sys
import argparse
from typing import List, Dict, Any, Sequence

import numpy as np

# Ensure project root (containing pit3.py, MCTS.py, etc.) is on sys.path when
# this script is invoked from an arbitrary working directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pit3 import (  # type: ignore
    make_game,
    set_seed,
    NNet,
)
from MCTS import MCTS  # type: ignore
from adaptive_budget import robust_root_select  # type: ignore
from utils import dotdict  # type: ignore


def select_action_with_mcts(
    game,
    nnet,
    board,
    player: int,
    sims: int,
    *,
    use_robust: bool,
    robust_frac: float = 0.6,
    cpuct: float = 1.0,
    sym_eval: bool = True,
) -> int:
    """
    From a given (board, player), run a fresh MCTS search and return one move.

    - If use_robust=False: return baseline action (argmax root visit-count N).
    - If use_robust=True : return robust-root action as defined by
      robust_root_select (Ns, Qs, robust_frac).

    The tree policy itself is standard AlphaZero MCTS; only the root decision
    rule changes via use_robust.
    """
    sims = int(sims)
    player = int(player)
    canonical = game.getCanonicalForm(board, player)

    args = dotdict(
        {
            "numMCTSSims": sims,
            "cpuct": float(cpuct),
            "use_dyn_c": False,
            "addRootNoise": False,
            "sym_eval": bool(sym_eval),
        }
    )
    mcts = MCTS(game, nnet, args)
    _ = mcts.getActionProb(canonical, temp=0)

    # Extract root Ns and Qs in "current player" indexing.
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
    return int(a_robust if use_robust else a_base)


def play_from_state_with_forced_first_move(
    game,
    nnet,
    board,
    player: int,
    forced_action: int,
    *,
    sims: int,
    cpuct: float = 1.0,
    robust_frac: float = 0.6,
    sym_eval: bool = True,
    use_robust_for_agent: bool = False,
    use_robust_for_opponent: bool = False,
) -> int:
    """
    从给定中间局面 (board, player) 开始，第一步强制执行 forced_action，
    之后双方都用同一套 MCTS 配置把棋下完，返回最终结果（从当前 player 视角）。

    - game/getNextState/getGameEnded: 使用标准 Game 接口。
    - sims/cpuct/sym_eval: 控制后续每一步的 MCTS 参数。
    - use_robust_for_agent / use_robust_for_opponent:
        决定“后续步”是用 baseline 还是 robust-root 规则选根动作。
        在 one-step replacement 实验中，建议两者设成同一个值，
        这样 baseline 起手组和 robust 起手组的后续策略完全一致。
    """
    player = int(player)
    forced_action = int(forced_action)

    board_cur = np.array(board, copy=True)

    # 检查强制动作在当前局面下是否合法，方便 debug。
    valids_root = game.getValidMoves(board_cur, player)
    if not (0 <= forced_action < valids_root.size) or valids_root[forced_action] <= 0:
        raise ValueError(
            f"Forced action {forced_action} is invalid for current state "
            f"(sum(valid)={int(valids_root.sum())})."
        )

    # 第一步：强制指定动作
    board_cur, cur_player = game.getNextState(board_cur, player, forced_action)
    start_player = player

    # 后续：双方都用相同 MCTS 配置（只是在根节点决策时选择 baseline / robust-root）
    while game.getGameEnded(board_cur, cur_player) == 0:
        use_robust = (
            use_robust_for_agent if cur_player == start_player else use_robust_for_opponent
        )
        action = select_action_with_mcts(
            game,
            nnet,
            board_cur,
            cur_player,
            sims=sims,
            use_robust=use_robust,
            robust_frac=robust_frac,
            cpuct=cpuct,
            sym_eval=sym_eval,
        )
        board_cur, cur_player = game.getNextState(board_cur, cur_player, action)

    # 按“当前样本的走子方”视角返回结果：+1 / -1 / 0
    result = game.getGameEnded(board_cur, start_player)
    return int(result)


def load_samples_from_npz(
    path: str,
    *,
    only_changed: bool = True,
    max_samples: int | None = None,
) -> List[Dict[str, Any]]:
    """
    从 robust_vs_baseline_sims*.npz 中读取样本。

    每个样本包含：
        - board        : numpy 数组（原始棋盘）
        - player       : 当前轮到谁走（+1 或 -1）
        - action_base  : baseline 根动作
        - action_robust: robust-root 根动作
        - move_index   : 全局第几手
        - sims         : 生成该文件时使用的 numMCTSSims
        - idx_in_file  : 在原 npz 中的索引
    """
    data = np.load(path, allow_pickle=True)

    boards = data["board_before"]
    players = data["player_global"]
    a_base = data["baseline_action"]
    a_rob = data["robust_action"]
    move_index = data["move_index"]
    changed = data["changed"].astype(bool)

    # 文件级的 sims 标记（所有样本相同）
    sims_in_file = int(data["sims"])

    if only_changed:
        idx_all = np.nonzero(changed)[0]
    else:
        idx_all = np.arange(boards.shape[0], dtype=np.int64)

    if max_samples is not None:
        idx_all = idx_all[: int(max_samples)]

    samples: List[Dict[str, Any]] = []
    for idx in idx_all:
        samples.append(
            {
                "board": np.array(boards[idx], copy=True),
                "player": int(players[idx]),
                "action_base": int(a_base[idx]),
                "action_robust": int(a_rob[idx]),
                "move_index": int(move_index[idx]),
                "sims": sims_in_file,
                "idx_in_file": int(idx),
            }
        )

    return samples


def run_one_step_experiment(
    game,
    nnet,
    samples: Sequence[Dict[str, Any]],
    *,
    eval_sims: int,
    robust_frac: float = 0.6,
    num_repeat: int = 32,
    after_policy: str = "baseline",
    cpuct: float = 1.0,
    sym_eval: bool = True,
) -> List[Dict[str, Any]]:
    """
    对一批样本做 one-step replacement 实验。

    对于每个样本 (board, player, a_base, a_robust)：
        - 重复 num_repeat 次：
            · 第一组：第一步强制 a_base，再用 after_policy 继续对弈到终局；
            · 第二组：第一步强制 a_robust，再用同样的 after_policy 继续对弈；
        - 统计两组的胜/负/和局次数（从当前 player 视角）。

    after_policy:
        - "baseline": 后续步骤都用 baseline（argmax N 根决策）；
        - "robust"  : 后续步骤都用 robust-root 根决策。
    """
    eval_sims = int(eval_sims)
    num_repeat = int(num_repeat)

    if after_policy not in ("baseline", "robust"):
        raise ValueError(f"after_policy must be 'baseline' or 'robust', got {after_policy!r}")

    use_robust_after = after_policy == "robust"

    results: List[Dict[str, Any]] = []

    for sid, s in enumerate(samples):
        board = s["board"]
        player = int(s["player"])
        a_base = int(s["action_base"])
        a_rob = int(s["action_robust"])
        move_idx = int(s["move_index"])
        sims_from_gen = int(s["sims"])
        idx_in_file = int(s.get("idx_in_file", sid))

        wins_base = 0
        draws_base = 0
        wins_rob = 0
        draws_rob = 0

        for _ in range(num_repeat):
            # baseline 起手
            res_base = play_from_state_with_forced_first_move(
                game,
                nnet,
                board,
                player,
                a_base,
                sims=eval_sims,
                cpuct=cpuct,
                robust_frac=robust_frac,
                sym_eval=sym_eval,
                use_robust_for_agent=use_robust_after,
                use_robust_for_opponent=use_robust_after,
            )
            if res_base > 0:
                wins_base += 1
            elif res_base == 0:
                draws_base += 1

            # robust 起手
            res_rob = play_from_state_with_forced_first_move(
                game,
                nnet,
                board,
                player,
                a_rob,
                sims=eval_sims,
                cpuct=cpuct,
                robust_frac=robust_frac,
                sym_eval=sym_eval,
                use_robust_for_agent=use_robust_after,
                use_robust_for_opponent=use_robust_after,
            )
            if res_rob > 0:
                wins_rob += 1
            elif res_rob == 0:
                draws_rob += 1

        results.append(
            {
                "sample_local_id": int(sid),
                "idx_in_file": idx_in_file,
                "move_index": move_idx,
                "sims_generated": sims_from_gen,
                "sims_eval": eval_sims,
                "num_repeat": num_repeat,
                "wins_base": int(wins_base),
                "draws_base": int(draws_base),
                "wins_rob": int(wins_rob),
                "draws_rob": int(draws_rob),
            }
        )

    return results


def _summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute overall win/draw rates from a list of per-sample results."""
    total_base = 0
    wins_base = 0
    draws_base = 0

    total_rob = 0
    wins_rob = 0
    draws_rob = 0

    for r in results:
        nb = int(r["num_repeat"])
        wb = int(r["wins_base"])
        db = int(r["draws_base"])
        nr = int(r["num_repeat"])
        wr = int(r["wins_rob"])
        dr = int(r["draws_rob"])

        total_base += nb
        wins_base += wb
        draws_base += db

        total_rob += nr
        wins_rob += wr
        draws_rob += dr

    def _rate(win: int, total: int) -> float:
        return float(win) / float(total) if total > 0 else float("nan")

    return {
        "base_win_rate": _rate(wins_base, total_base),
        "base_draw_rate": _rate(draws_base, total_base),
        "rob_win_rate": _rate(wins_rob, total_rob),
        "rob_draw_rate": _rate(draws_rob, total_rob),
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run one-step replacement experiments on robust_vs_baseline_sims*.npz.\n"
            "For each sampled root position where baseline and robust-root actions "
            "differ, repeatedly play out games with the first move forced to either "
            "baseline or robust-root, and compare win rates."
        )
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="./robust_vs_baseline_data",
        help="Directory containing robust_vs_baseline_sims*.npz files.",
    )
    ap.add_argument(
        "--sims",
        type=int,
        nargs="+",
        default=[25],
        help="List of `sims` values to evaluate (matching robust_vs_baseline_sims*.npz).",
    )
    ap.add_argument(
        "--eval-sims",
        type=int,
        default=None,
        help=(
            "Number of MCTS simulations per move used during continuation.\n"
            "If omitted, defaults to the same value as `sims` for each file."
        ),
    )
    ap.add_argument(
        "--num-repeat",
        type=int,
        default=32,
        help="Number of repetitions per sample for each of (baseline, robust) starts.",
    )
    ap.add_argument(
        "--after-policy",
        type=str,
        choices=["baseline", "robust"],
        default="baseline",
        help=(
            "Policy used for moves after the forced first step for both sides:\n"
            "  - baseline: argmax root visit-count N(s,a)\n"
            "  - robust  : robust-root selection based on Ns and Qs"
        ),
    )
    ap.add_argument(
        "--robust-frac",
        type=float,
        default=0.6,
        help="Fraction parameter for robust-root selection (same as in robust_root_select).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of changed samples per sims value (for quick tests).",
    )
    ap.add_argument(
        "--model-dir",
        type=str,
        default="./models/",
        help="Directory containing the model checkpoint.",
    )
    ap.add_argument(
        "--model-file",
        type=str,
        default="baseline.pth.tar",
        help="Model checkpoint filename.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="./robust_vs_baseline_data",
        help="Directory to save one-step replacement result .npz files.",
    )
    args = ap.parse_args()

    set_seed(42)

    game = make_game(8)
    nnet = NNet(game)
    nnet.load_checkpoint(args.model_dir, args.model_file)

    os.makedirs(args.out_dir, exist_ok=True)

    for sims in args.sims:
        data_path = os.path.join(args.data_dir, f"robust_vs_baseline_sims{sims}.npz")
        if not os.path.isfile(data_path):
            print(f"[one-step] sims={sims}: file not found: {data_path}, skipping.")
            continue

        print(f"[one-step] sims={sims}: loading samples from {data_path}")
        samples = load_samples_from_npz(
            data_path, only_changed=True, max_samples=args.max_samples
        )
        if not samples:
            print(f"[one-step] sims={sims}: no changed samples found, skipping.")
            continue

        eval_sims = int(args.eval_sims) if args.eval_sims is not None else int(sims)
        print(
            f"[one-step] sims={sims}: running experiments on {len(samples)} samples "
            f"(eval_sims={eval_sims}, repeats={args.num_repeat}, "
            f"after_policy={args.after_policy})"
        )

        results = run_one_step_experiment(
            game,
            nnet,
            samples,
            eval_sims=eval_sims,
            robust_frac=float(args.robust_frac),
            num_repeat=int(args.num_repeat),
            after_policy=args.after_policy,
        )

        summary = _summarize_results(results)
        print(
            f"[one-step] sims={sims}: "
            f"baseline win={summary['base_win_rate']:.4f}, "
            f"robust win={summary['rob_win_rate']:.4f}, "
            f"baseline draw={summary['base_draw_rate']:.4f}, "
            f"robust draw={summary['rob_draw_rate']:.4f}"
        )

        # Save per-sample stats for downstream analysis / plotting.
        out_path = os.path.join(
            args.out_dir,
            f"one_step_replacement_sims{sims}_eval{eval_sims}_rep{args.num_repeat}.npz",
        )
        np.savez_compressed(
            out_path,
            sample_local_id=np.array(
                [r["sample_local_id"] for r in results], dtype=np.int32
            ),
            idx_in_file=np.array(
                [r["idx_in_file"] for r in results], dtype=np.int32
            ),
            move_index=np.array(
                [r["move_index"] for r in results], dtype=np.int32
            ),
            sims_generated=np.array(
                [r["sims_generated"] for r in results], dtype=np.int32
            ),
            sims_eval=np.array(
                [r["sims_eval"] for r in results], dtype=np.int32
            ),
            num_repeat=np.array(
                [r["num_repeat"] for r in results], dtype=np.int32
            ),
            wins_base=np.array(
                [r["wins_base"] for r in results], dtype=np.int32
            ),
            draws_base=np.array(
                [r["draws_base"] for r in results], dtype=np.int32
            ),
            wins_rob=np.array(
                [r["wins_rob"] for r in results], dtype=np.int32
            ),
            draws_rob=np.array(
                [r["draws_rob"] for r in results], dtype=np.int32
            ),
            base_win_rate=np.array(summary["base_win_rate"], dtype=np.float64),
            base_draw_rate=np.array(summary["base_draw_rate"], dtype=np.float64),
            rob_win_rate=np.array(summary["rob_win_rate"], dtype=np.float64),
            rob_draw_rate=np.array(summary["rob_draw_rate"], dtype=np.float64),
        )
        print(f"[one-step] sims={sims}: saved results to {out_path}")


if __name__ == "__main__":
    main()

