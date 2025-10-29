import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)

from Arena import Arena
import numpy as np
import copy, os, csv
from othello.OthelloPlayers import RandomPlayer, GreedyOthelloPlayer, AlphaBetaOthelloPlayer

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(self.game, self.args)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            # self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            # Reanalyze: recompute stronger policy targets with frozen teacher (pnet)
            if bool(getattr(self.args, 'reanalyze_enable', False)) and len(trainExamples) > 0:
                trainExamples = self._reanalyze_examples(trainExamples, teacher=self.pnet)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST BASELINES')
            self._eval_vs_baselines(i)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def _reanalyze_examples(self, examples, teacher):
        """Recompute policy targets with a stronger teacher MCTS.

        examples: list[(board, pi, z)] with canonical boards
        teacher: NNetWrapper-compatible net (frozen)
        """
        # Build teacher MCTS args
        import copy as _cpy, sys, time as _time, random as _rnd
        args_re = _cpy.deepcopy(self.args)
        sims_mult = float(getattr(self.args, 'reanalyze_sims_mult', 4.0))
        args_re.numMCTSSims = max(1, int(self.args.numMCTSSims * sims_mult))
        # Ensure deterministic strong target at root
        setattr(args_re, 'addRootNoise', False)
        # If available, enable symmetry ensembling at leaf evaluation
        setattr(args_re, 'sym_eval', True)

        mcts_t = MCTS(self.game, teacher, args_re)
        alpha = float(getattr(self.args, 'reanalyze_pi_alpha', 1.0))
        lam_v = float(getattr(self.args, 'reanalyze_v_lambda', 0.0))
        frac = float(getattr(self.args, 'reanalyze_fraction', 1.0))
        use_cache = bool(getattr(self.args, 'reanalyze_cache', True))

        # Group by canonical key to avoid recomputing symmetric duplicates
        groups = {}
        for idx, (board, pi, z) in enumerate(examples):
            meta = mcts_t._sym_canonicalize(board)
            s_key = meta['s_key']
            if s_key not in groups:
                groups[s_key] = {
                    'rep_board_can': meta['board_can'],
                    'items': []
                }
            groups[s_key]['items'].append((idx, board, pi, z, meta))

        # Order selection and optional cap
        keys = list(groups.keys())
        if bool(getattr(self.args, 'reanalyze_shuffle', True)):
            _rnd.shuffle(keys)
        max_u = int(getattr(self.args, 'reanalyze_max_unique', 0) or 0)
        if max_u > 0:
            keys = keys[:max_u]

        # Progress mode
        mode = str(getattr(self.args, 'reanalyze_progress', 'auto')).lower()
        use_tqdm = (mode == 'tqdm') or (mode == 'auto' and sys.stderr.isatty())
        if use_tqdm:
            try:
                from tqdm import tqdm
                it_keys = tqdm(keys, desc='Reanalyze (unique roots)')
            except Exception:
                it_keys = keys
                use_tqdm = False
        else:
            it_keys = keys
        log_interval = float(getattr(self.args, 'reanalyze_log_interval_sec', 60.0))
        t0 = _time.time()
        next_log = t0 + log_interval

        cache = {}
        A = self.game.getActionSize()
        total_samples = len(examples)
        total_unique = len(keys)
        new_examples = list(examples)
        processed = 0
        applied_samples = 0
        cache_hits = 0
        lookups = 0

        for s_key in it_keys:
            bundle = groups[s_key]
            rep_board_can = bundle['rep_board_can']

            # Check if any item in the group passes the fraction gate
            need_re = any(_rnd.random() <= frac for _ in bundle['items'])
            if not need_re:
                processed += 1
                if not use_tqdm and _time.time() >= next_log:
                    elapsed = _time.time() - t0
                    done = processed / max(total_unique, 1)
                    eta = elapsed * (1.0 - done) / max(done, 1e-8)
                    print(f"[Reanalyze] unique {processed}/{total_unique} ({done*100:.1f}%), applied {applied_samples}/{total_samples} ({(applied_samples/max(total_samples,1))*100:.1f}%), cache_hit {cache_hits}/{lookups} ({(cache_hits/max(lookups,1))*100:.1f}%), elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m")
                    next_log = _time.time() + log_interval
                continue

            # Resolve canonical policy once
            if use_cache and s_key in cache:
                pi_can, v_root = cache[s_key]
                cache_hits += 1
                lookups += 1
            else:
                pi_rep = mcts_t.getActionProb(rep_board_can, temp=0)
                lookups += 1
                meta_rep = mcts_t._sym_canonicalize(rep_board_can)
                perm_rep_can2cur = meta_rep['perm_can2cur']
                pi_can = np.array(pi_rep, dtype=np.float32)[perm_rep_can2cur]
                v_root = None
                if lam_v > 0.0:
                    try:
                        num = 0.0
                        den = 0.0
                        for a in range(A):
                            key_sa = (s_key, a)
                            if key_sa in mcts_t.Nsa:
                                nsa = float(mcts_t.Nsa[key_sa])
                                qsa = float(mcts_t.Qsa.get(key_sa, 0.0))
                                num += nsa * qsa
                                den += nsa
                        if den > 0.0:
                            v_root = num / den
                    except Exception:
                        v_root = None
                if use_cache:
                    cache[s_key] = (pi_can, v_root)

            # Apply to each item in this group
            for (idx, board, pi, z, meta) in bundle['items']:
                if _rnd.random() > frac:
                    continue
                perm_cur2can = meta['perm_cur2can']
                pi_new_np = np.array(pi_can, dtype=np.float32)[perm_cur2can]
                pi_np = np.array(pi, dtype=np.float32)
                if alpha >= 1.0:
                    pi_star = pi_new_np
                elif alpha <= 0.0:
                    pi_star = pi_np
                else:
                    pi_star = (1.0 - alpha) * pi_np + alpha * pi_new_np
                    s = float(pi_star.sum())
                    if s > 0:
                        pi_star = pi_star / s
                if v_root is not None and lam_v > 0.0:
                    v_star = float((1.0 - lam_v) * float(z) + lam_v * float(v_root))
                else:
                    v_star = z
                new_examples[idx] = (board, pi_star, v_star)
                applied_samples += 1

            processed += 1
            if use_tqdm:
                try:
                    it_keys.set_postfix(applied=applied_samples, cache=f"{cache_hits}/{lookups}")
                except Exception:
                    pass
            else:
                if _time.time() >= next_log:
                    elapsed = _time.time() - t0
                    done = processed / max(total_unique, 1)
                    eta = elapsed * (1.0 - done) / max(done, 1e-8)
                    print(f"[Reanalyze] unique {processed}/{total_unique} ({done*100:.1f}%), applied {applied_samples}/{total_samples} ({(applied_samples/max(total_samples,1))*100:.1f}%), cache_hit {cache_hits}/{lookups} ({(cache_hits/max(lookups,1))*100:.1f}%), elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m")
                    next_log = _time.time() + log_interval

        return new_examples

    def _mcts_player_with(self, nnet, sims):
        eval_args = copy.deepcopy(self.args)
        eval_args.numMCTSSims = sims
        mcts = MCTS(self.game, nnet, eval_args)
        return lambda board: np.argmax(mcts.getActionProb(board, temp=0))

    def _eval_vs_baselines(self, iter_idx):
        if getattr(self.args, 'evalGames', 0) <= 0:
            return
        nnet_player = self._mcts_player_with(self.nnet, getattr(self.args, 'evalNumMCTSSims', self.args.numMCTSSims))

        rows = []

        def eval_one(baseline_play, name):
            arena = Arena(nnet_player, baseline_play, self.game)
            w, l, d = arena.playGames(self.args.evalGames, verbose=False)
            wr = w / max(1, (w + l))
            print(f"[Iter {iter_idx}] vs {name}: W={w}, L={l}, D={d}, WR={wr:.3f}")
            return name, w, l, d, wr

        # random
        if RandomPlayer is not None:
            rp = RandomPlayer(self.game).play
            rows.append(eval_one(rp, 'random'))
        else:
            rows.append(eval_one(self._generic_random_player(), 'random(generic)'))

        # greedy
        if GreedyOthelloPlayer is not None:
            gp = GreedyOthelloPlayer(self.game).play
            rows.append(eval_one(gp, 'greedy'))

        # alphabeta
        if AlphaBetaOthelloPlayer is not None:
            abp = AlphaBetaOthelloPlayer(self.game, depth=self.args.evalABDepth,
                                         time_limit=self.args.evalABTimeLimit).play
            rows.append(eval_one(abp, f'alphabeta(d={self.args.evalABDepth})'))

        if getattr(self.args, 'logBaselinesToCSV', False):
            out_dir = os.path.join(self.args.checkpoint, 'metrics')
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, 'baseline_eval.csv')
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                wtr = csv.writer(f)
                if write_header:
                    wtr.writerow(['iter', 'opponent', 'wins', 'losses', 'draws', 'win_rate'])
                for name, w, l, d, wr in rows:
                    wtr.writerow([iter_idx, name, w, l, d, f"{wr:.4f}"])
