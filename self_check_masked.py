"""
自检脚本：定位 “All valid moves were masked” 的根因。

功能：
- 采样若干 Othello 随机局面（按规则随机走子生成），
- 用当前神经网络预测策略分布 pi，
- 统计 sum(pi*valids)==0 的比例，并打印详细诊断：
  - 合法着法个数/索引，
  - 非法着法中最高概率的若干个及坐标，
  - pass 动作概率与是否合法，
  - argmax 是否落在合法动作上。

用法示例：
  python self_check_masked.py --n 8 --steps 300 \
      --model-folder ./models --model-file baseline.pth.tar --topk 10

备注：仅依赖本仓库内容（othello + pytorch 版本 NNet），默认 CPU 推理。
"""

import argparse
import os
import numpy as np

from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet


def sample_random_positions(game: OthelloGame, steps=200, seed=42):
    rng = np.random.default_rng(seed)
    boards = []
    b = game.getInitBoard()
    player = 1
    for _ in range(steps):
        boards.append(b.copy())
        valids = np.asarray(game.getValidMoves(b, player), dtype=np.int8)
        A = game.getActionSize()
        moves = np.where(valids == 1)[0]
        if moves.size == 0:
            a = A - 1  # pass
        else:
            a = int(rng.choice(moves))
        nb, np_player = game.getNextState(b, player, a)
        # 规范到当前执子方为 1
        b = game.getCanonicalForm(nb, np_player)
        player = 1
    return boards


def idx_to_rc(a: int, n: int):
    return a // n, a % n


def topk_pairs(arr, idxs, k=10):
    pairs = [(float(arr[i]), int(i)) for i in idxs]
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:k]


def run_check(n: int, steps: int, seed: int, model_folder: str, model_file: str, topk: int):
    game = OthelloGame(n)
    nnet = NNet(game)  # 默认 args（通常为 CPU）
    # 加载权重（可选）
    if model_folder and model_file:
        try:
            nnet.load_checkpoint(model_folder, model_file)
        except Exception as e:
            print(f"[warn] 未能加载 checkpoint {os.path.join(model_folder, model_file)}: {e}")

    boards = sample_random_positions(game, steps=steps, seed=seed)
    total = 0
    masked = 0
    stats_by_B = {}  # 按合法数 B 统计

    for b in boards:
        total += 1
        valids = np.asarray(game.getValidMoves(b, 1), dtype=np.int8)
        A = game.getActionSize()
        try:
            pi, v = nnet.predict(b)
            pi = np.asarray(pi, dtype=np.float64)
        except Exception as e:
            print(f"[error] 预测失败: {e}")
            continue

        B = int(valids.sum())
        stats_by_B[B] = stats_by_B.get(B, 0) + 1

        mass_valid = float(np.sum(pi * (valids > 0)))
        if mass_valid <= 1e-15:
            masked += 1
            print("\n===== MASKED-ALL (sum(pi*valids)=0) =====")
            # 棋盘可读
            try:
                br = game.stringRepresentationReadable(b)
                print(f"board:\n{br}")
            except Exception:
                pass

            valid_idx = [i for i in range(A) if valids[i] > 0]
            invalid_idx = [i for i in range(A) if not (valids[i] > 0)]
            top_valid = topk_pairs(pi, valid_idx, k=topk)
            top_invalid = topk_pairs(pi, invalid_idx, k=topk)
            argmax_idx = int(np.argmax(pi)) if pi.size > 0 else -1
            argmax_is_valid = int(valids[argmax_idx] > 0) if 0 <= argmax_idx < len(valids) else 0
            pass_idx = A - 1
            pass_prob = float(pi[pass_idx]) if 0 <= pass_idx < len(pi) else float("nan")
            pass_is_valid = int(valids[pass_idx] > 0) if 0 <= pass_idx < len(valids) else 0

            print(f"A={A}, valid_count={len(valid_idx)}, valid_idx(sample)={valid_idx[:20]}")
            print(f"sum(pi)={float(np.nansum(pi)):.6g}, sum(pi*valids)={mass_valid:.6g}")
            print(f"argmax_idx={argmax_idx}, argmax_is_valid={argmax_is_valid}")
            print(f"pass_idx={pass_idx}, pass_is_valid={pass_is_valid}, pass_prob={pass_prob:.6g}")

            if len(valid_idx) > 0:
                print(f"top_valid_pi (p,idx)={top_valid}")
            print(f"top_invalid_pi (p,idx)={top_invalid}")

            # 可选：打印坐标版（便于肉眼检查是否整齐地集中在非法格）
            try:
                print("top_invalid (p, r, c):", [ (p,)+idx_to_rc(a, n) for (p,a) in top_invalid ])
            except Exception:
                pass

    ratio = (masked / total) if total > 0 else 0.0
    print("\n===== SUMMARY =====")
    print(f"total={total}, masked={masked}, ratio={ratio:.2%}")
    # 按合法数分布
    print("by valid_count (B):", dict(sorted(stats_by_B.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=8, help='棋盘规模（6 或 8）')
    parser.add_argument('--steps', type=int, default=200, help='随机采样步数（越大越全面）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model-folder', type=str, default='./models', help='checkpoint 目录')
    parser.add_argument('--model-file', type=str, default='baseline.pth.tar', help='checkpoint 文件名')
    parser.add_argument('--topk', type=int, default=10, help='打印前 K 个最大概率项')
    args = parser.parse_args()

    run_check(
        n=args.n,
        steps=args.steps,
        seed=args.seed,
        model_folder=args.model_folder,
        model_file=args.model_file,
        topk=args.topk,
    )


if __name__ == '__main__':
    main()

