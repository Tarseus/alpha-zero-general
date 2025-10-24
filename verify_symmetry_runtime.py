"""
运行时对称性自检（针对最近的对称增强改动）：

1) 置换一致性：
   - 用 OthelloGame.getSymmetries() 得到 8 个视图下的 pi 变换结果，
     与 NNetWrapper._init_action_perms() 生成的 perm_fwd_ext 比对，
     确认 base->sym 的动作索引映射一致（包括 pass 固定）。

2) 合法着法等变：
   - 随机抽样若干棋盘 b，计算 valids(b)；
     对每个视图 k：用 game.getSymmetries(b, pi) 取到变换后的棋盘 b_k，
     计算 valids(b_k)，再用 perm_fwd[k] 拉回 base 坐标，对比 valids(b)。

3) 模型等变（可选，不是严格要求）：
   - 用当前模型对 b 与 b_k 预测 pi，
     将 pi(b_k) 用 perm_fwd[k] 拉回，与 pi(b) 对比。

用法：
  python verify_symmetry_runtime.py --n 8 --num 50 --model-folder ./models --model-file baseline.pth.tar
"""

import argparse
import numpy as np

from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet


def check_perm_vs_game(game: OthelloGame, nnet: NNet):
    n = game.n
    A = game.getActionSize()
    ok = True
    fails = []
    # 构造一个任意棋盘与 one-hot pi 来跟踪动作索引映射
    b0 = game.getInitBoard()
    pi0_full = np.zeros(A, dtype=np.float32)
    pi0_full[-1] = 0.0  # pass 不置 1，这样 newPi 的 argmax 落在 0..A-1

    for a in range(A - 1):  # 仅检查 0..A-2，对应格子动作
        pi = pi0_full.copy()
        pi[a] = 1.0
        sym = game.getSymmetries(b0, pi)  # 顺序与实现保持一致
        # 提取每个视图中的 one-hot 位置
        p_fwd = nnet.nnet.perm_fwd_ext.cpu().numpy()  # (8, A+1)
        for k, (_, pi_k) in enumerate(sym):
            # pi_k 最后一个是 pass，前 A-1 为格子动作
            arr = np.asarray(pi_k, dtype=np.float32)
            arg = int(np.argmax(arr[:-1]))
            expect = int(p_fwd[k, a])
            if arg != expect:
                ok = False
                fails.append((a, k, arg, expect))
                if len(fails) < 10:
                    print(f"[perm mismatch] a={a}, k={k}, game idx={arg}, perm_fwd={expect}")
                break
        if not ok:
            break

    # 检查 pass 固定
    p_fwd = nnet.nnet.perm_fwd_ext.cpu().numpy()
    pass_id = A - 1
    fixed_ok = True
    for k in range(p_fwd.shape[0]):
        if p_fwd[k, pass_id] != pass_id:
            fixed_ok = False
            print(f"[pass not fixed] k={k}, got={p_fwd[k, pass_id]}, expect={pass_id}")

    print("perm_vs_game:", "OK" if (ok and fixed_ok) else "MISMATCH")
    return ok and fixed_ok


def gather_by_perm(arr: np.ndarray, perm: np.ndarray):
    # 返回 arr[perm]，保持 dtype
    return arr[perm]


def check_valids_equivariance(game: OthelloGame, nnet: NNet, num=50, seed=0):
    rng = np.random.default_rng(seed)
    n = game.n
    A = game.getActionSize()
    ok = True
    p_fwd = nnet.nnet.perm_fwd_ext.cpu().numpy()

    # 随机从自对弈序列采样局面
    def sample_positions(steps=50):
        b = game.getInitBoard()
        player = 1
        boards = []
        for _ in range(steps):
            boards.append(b.copy())
            valids = np.asarray(game.getValidMoves(b, player), dtype=np.int8)
            moves = np.where(valids == 1)[0]
            a = int(rng.choice(moves)) if moves.size > 0 else (A - 1)
            nb, np_player = game.getNextState(b, player, a)
            b = game.getCanonicalForm(nb, np_player)
            player = 1
        return boards

    boards = sample_positions(steps=num)
    for b in boards:
        val0 = np.asarray(game.getValidMoves(b, 1), dtype=np.int8)
        # 用 Game 的对称变换拿到棋盘视图
        sym = game.getSymmetries(b, [0]*(A-1) + [0])
        for k, (bk, _) in enumerate(sym):
            val_k = np.asarray(game.getValidMoves(bk, 1), dtype=np.int8)
            back = gather_by_perm(val_k, p_fwd[k, :A].astype(np.int64))
            if not np.array_equal(back, val0):
                ok = False
                print(f"[valids equiv] mismatch at k={k}")
                break
        if not ok:
            break

    print("valids_equivariance:", "OK" if ok else "MISMATCH")
    return ok


def check_model_equivariance(game: OthelloGame, nnet: NNet, num=20, tol=1e-4):
    n = game.n
    A = game.getActionSize()
    p_fwd = nnet.nnet.perm_fwd_ext.cpu().numpy()
    ok = True

    def sample_positions(steps=20):
        rng = np.random.default_rng(123)
        b = game.getInitBoard()
        player = 1
        boards = []
        for _ in range(steps):
            boards.append(b.copy())
            valids = np.asarray(game.getValidMoves(b, player), dtype=np.int8)
            moves = np.where(valids == 1)[0]
            a = int(rng.choice(moves)) if moves.size > 0 else (A - 1)
            nb, np_player = game.getNextState(b, player, a)
            b = game.getCanonicalForm(nb, np_player)
            player = 1
        return boards

    boards = sample_positions(num)
    for b in boards:
        pi0, _ = nnet.predict(b)
        pi0 = np.asarray(pi0, dtype=np.float64)
        sym = game.getSymmetries(b, [0]*(A-1) + [0])
        for k, (bk, _) in enumerate(sym):
            pik, _ = nnet.predict(bk)
            pik = np.asarray(pik, dtype=np.float64)
            back = gather_by_perm(pik, p_fwd[k, :A].astype(np.int64))
            if np.max(np.abs(back - pi0)) > 1e-2:  # 模型本身非严格等变，宽松些
                ok = False
                break
        if not ok:
            break

    print("model_equivariance (loose):", "OK" if ok else "DIFFERS")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--num', type=int, default=50, help='随机抽样局面数量')
    ap.add_argument('--model-folder', type=str, default='./models')
    ap.add_argument('--model-file', type=str, default='baseline.pth.tar')
    args = ap.parse_args()

    game = OthelloGame(args.n)
    nnet = NNet(game)
    try:
        nnet.load_checkpoint(args.model_folder, args.model_file)
    except Exception as e:
        print(f"[warn] load checkpoint failed: {e}")

    ok_perm = check_perm_vs_game(game, nnet)
    ok_valid = check_valids_equivariance(game, nnet, num=args.num)
    ok_model = check_model_equivariance(game, nnet, num=min(20, args.num))

    print("==== SUMMARY ====")
    print(f"perm_vs_game={ok_perm}, valids_equiv={ok_valid}, model_equiv(loose)={ok_model}")


if __name__ == '__main__':
    main()

